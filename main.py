"""
Main Entry Point for Chess Puzzle Evaluation

This script provides the main interface for evaluating chess puzzles using
OpenAI models and Glicko-2 rating system.

Usage:
    python main.py --help
    python main.py --csv-file puzzles.csv --max-puzzles 100
    python main.py --evaluate --model gpt-4-turbo
"""

import argparse
import os
import sys
import json
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

# Import our modules
from csv_reader import read_chess_puzzles_csv, sample_puzzles, save_puzzles_csv, get_puzzle_stats
from model_interface import ChessModelInterface
from chess_utils import build_chess_prompts, get_partial_pgn_from_url, extract_predicted_move, extract_plan_sans, san_to_uci
from glicko_rating import update_agent_rating_from_puzzles, Rating

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAD_DIR = os.path.join(CURRENT_DIR, "MAD")
if MAD_DIR not in sys.path:
    sys.path.insert(0, MAD_DIR)

from utils.agent import Agent
from utils.openai_utils import num_tokens_from_string, model2max_context
import chess
import chess.pgn
import io

# Import new debate system
from chess_debate import ChessDebateV2, save_debate_history_v2


class ChessDebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature: float, openai_api_key: str, sleep_time: float = 0, base_url: str = None) -> None:
        """Create a chess debate player"""
        super(ChessDebatePlayer, self).__init__(model_name, name, temperature, sleep_time)
        self.openai_api_key = openai_api_key
        self.model_interface = ChessModelInterface(
            api_key=openai_api_key,
            model_name=model_name,
            base_url=base_url,
            max_completion_tokens=640,
            default_temperature=temperature,
            retry_attempts=2,
        )
        self.last_token_info: Optional[dict] = None

    def ask(self, temperature: float=None):
        """Query for answer using our model interface instead of the old API"""
        # Build combined prompt from memory
        combined_prompt = ""
        for msg in self.memory_lst:
            if msg["role"] == "system":
                combined_prompt += msg["content"] + "\n"
            elif msg["role"] == "user":
                combined_prompt += msg["content"] + "\n"
            elif msg["role"] == "assistant":
                combined_prompt += msg["content"] + "\n"
        
        print(f"\n<debug> : ChessDebatePlayer '{self.name}' ask() method:")
        print(f"<debug> : combined_prompt: {repr(combined_prompt[:200])}...")
        
        # Use our model interface instead of the old API
        response, token_info = self.model_interface.query_model_for_move_with_tokens(
            system_prompt="",  # Already included in combined_prompt
            user_prompt=combined_prompt,
            max_tokens=self.model_interface.max_completion_tokens,
            temperature=temperature if temperature else self.temperature,
            top_p=self.model_interface.default_top_p
        )
        self.last_token_info = token_info
        
        print(f"<debug> : response: {repr(response)}")
        return response or ""


class ChessSelfConsistency:
    def __init__(self, 
                 model_name: str = 'gpt-3.5-turbo-instruct',
                 temperature: float = 0.1,
                 openai_api_key: str = None,
                 max_rounds: int = 2,
                 sleep_time: float = 0.1,
                 plan_plies: int = 0,
                 base_url: str = None):
        """Create a chess self-consistency system"""
        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = openai_api_key
        self.max_rounds = max_rounds
        self.sleep_time = sleep_time
        self.plan_plies = max(0, plan_plies)
        
        # Initialize players
        self.aggressive_gm = ChessDebatePlayer(
            model_name=model_name,
            name="Mikhail Tal (Aggressive)",
            temperature=temperature,
            openai_api_key=openai_api_key,
            sleep_time=sleep_time,
            base_url=base_url
        )
        
        self.positional_gm = ChessDebatePlayer(
            model_name=model_name,
            name="Magnus Carlsen (Positional)",
            temperature=temperature,
            openai_api_key=openai_api_key,
            sleep_time=sleep_time,
            base_url=base_url
        )
        
        self.neutral_gm = ChessDebatePlayer(
            model_name=model_name,
            name="Neutral GM",
            temperature=temperature,
            openai_api_key=openai_api_key,
            sleep_time=sleep_time,
            base_url=base_url
        )
        
        # Set up personas
        self.setup_personas()
        
        # Debate results
        self.debate_history = []
        self.final_move = None

    def setup_personas(self):
        """Set up the grandmaster personas"""
        num_future_moves = self.plan_plies + 1 if self.plan_plies > 0 else 1
        
        if num_future_moves == 1:
            move_instruction = "give ONE new move."
        else:
            move_instruction = f"give the next {num_future_moves} moves."

        base_prompt = (
            "You are a chess grandmaster.\n"
            "\n"
            "You will be given a partially completed game.\n"
            "\n"
            f"After seeing it, you should repeat the ENTIRE GAME and then {move_instruction}\n"
            "\n"
            "Use standard algebraic notation, e.g. \"e4\" or \"Rdf8\" or \"R1a3\".\n"
            "\n"
            "ALWAYS repeat the entire representation of the game so far.\n"
            "\n"
            "NEVER explain your choice.\n"
        )

        aggressive_prompt = base_prompt
        positional_prompt = base_prompt
        neutral_prompt = base_prompt
        
        self.aggressive_gm.set_meta_prompt(aggressive_prompt)
        self.positional_gm.set_meta_prompt(positional_prompt)
        self.neutral_gm.set_meta_prompt(neutral_prompt)

    def get_board_fen_from_pgn(self, pgn_string: str) -> str:
        """Get the board FEN from a PGN string"""
        try:
            # Parse the PGN using io.StringIO
            pgn_io = io.StringIO(pgn_string)
            game = chess.pgn.read_game(pgn_io)
            
            if game is None:
                return chess.Board().fen()
            
            # Replay the game to get the final position
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            
            return board.fen()
        except Exception as e:
            print(f"Error parsing PGN: {e}")
            return chess.Board().fen()

    def clear_memory(self):
        """Clear memory between puzzles to prevent token buildup"""
        self.aggressive_gm.memory_lst = []
        self.positional_gm.memory_lst = []
        self.neutral_gm.memory_lst = []
        # Re-setup personas after clearing
        self.setup_personas()

    def run_debate(self, user_prompt: str, expected_uci: str = None, played_plies: int = 0, board_fen: str = None, board = None) -> tuple:
        """Run a chess self-consistency evaluation using 3 independent queries"""
        print(f"\n<self-consistency> : Starting self-consistency evaluation")
        token_events = []
        num_future_moves = self.plan_plies + 1 if self.plan_plies > 0 else 3
        def record_token_event(agent_label: str, token_info: Optional[dict], response_source: str) -> None:
            info = token_info or {}
            token_events.append({
                "paradigm": "self_consistency",
                "agent": agent_label,
                "ply_index": played_plies,
                "turn_number": current_turn_number,
                "round": 1,
                "response_source": response_source,
                "prompt_tokens": info.get("prompt_tokens", 0) or 0,
                "completion_tokens": info.get("completion_tokens", 0) or 0,
                "total_tokens": info.get("total_tokens", 0) or ((info.get("prompt_tokens", 0) or 0) + (info.get("completion_tokens", 0) or 0)),
                "model": info.get("model", self.model_name),
                "finish_reason": info.get("finish_reason", ""),
            })
        
        # Build system prompt
        system_prompt = (
            "You are a chess grandmaster.\n"
            "You will be given a partially completed game.\n"
            f"Complete the algebraic notation by repeating the ENTIRE GAME and then giving the next {num_future_moves} moves.\n"
            "After repeating the game, immediately continue by listing those moves in order on a single line, separated by spaces, starting with the side to move now.\n"
            "Use standard algebraic notation, e.g. 'e4' or 'Rdf8'.\n"
            "ALWAYS repeat the entire representation of the game so far.\n"
            "NO other explanations. Just complete the algebraic notation."
        )

        agg_system_prompt = (
            "You are an aggressive chess grandmaster.\n"
            "You will be given a partially completed game.\n"
            f"Complete the algebraic notation by repeating the ENTIRE GAME and then giving the next {num_future_moves} moves.\n"
            "After repeating the game, immediately continue by listing those moves in order on a single line, separated by spaces, starting with the side to move now.\n"
            "Use standard algebraic notation, e.g. 'e4' or 'Rdf8'.\n"
            "ALWAYS repeat the entire representation of the game so far.\n"
            "NO other explanations. Just complete the algebraic notation."
        )

        pos_system_prompt = (
            "You are a positional chess grandmaster.\n"
            "You will be given a partially completed game.\n"
            f"Complete the algebraic notation by repeating the ENTIRE GAME and then giving the next {num_future_moves} moves.\n"
            "After repeating the game, immediately continue by listing those moves in order on a single line, separated by spaces, starting with the side to move now.\n"
            "Use standard algebraic notation, e.g. 'e4' or 'Rdf8'.\n"
            "ALWAYS repeat the entire representation of the game so far.\n"
            "NO other explanations. Just complete the algebraic notation."
        )

        
        print(f"\n<debug> : Self-consistency system prompt building:")
        print(f"<debug> : user_prompt: {repr(user_prompt)}")
        
        if expected_uci:
            print(f"Expected UCI: {expected_uci}")
        
        # Query 1-3: Independent analysis (3 independent queries)
        print(f"\n<self-consistency> : Running 3 independent queries")
        
        # Aggressive GM analysis
        print(f"\n<debug> : Aggressive GM - using query_model_for_move:")
        print(f"<debug> : agg_system_prompt: {repr(agg_system_prompt)}")
        print(f"<debug> : user_prompt: {repr(user_prompt)}")
        
        # Use board object if available, otherwise fall back to played_plies calculation
        if board is not None:
            is_white_to_move = board.turn
            current_turn_number = played_plies // 2 + 1
        else:
            is_white_to_move = (played_plies % 2 == 0)
            current_turn_number = played_plies // 2 + 1
            
        agg_response, aggressive_san, agg_token_info = self.aggressive_gm.model_interface.get_move_with_extraction(
            agg_system_prompt,
            user_prompt,
            current_turn_number=current_turn_number,
            is_white_to_move=is_white_to_move,
            max_tokens=self.aggressive_gm.model_interface.max_completion_tokens,
            temperature=self.aggressive_gm.temperature,
            retry_attempts=self.aggressive_gm.model_interface.retry_attempts,
        )
        print(f"<debug> : Aggressive GM response: {repr(agg_response)}")
        print(f"<debug> : Aggressive GM move: {aggressive_san}")
        record_token_event("aggressive", agg_token_info, "aggressive_gm")

        aggressive_plan_full = extract_plan_sans(agg_response, primary_san=aggressive_san)
        aggressive_plan_execute = aggressive_plan_full[:self.plan_plies] if self.plan_plies > 0 else []
        if self.plan_plies > 0:
            print(f"<self-consistency> : Aggressive plan (full): {aggressive_plan_full}")
            print(f"<self-consistency> : Aggressive plan (first {self.plan_plies} plies): {aggressive_plan_execute}")

        self.aggressive_gm.add_memory(f"Aggressive GM move: {aggressive_san}")
        
        # Positional GM analysis
        print(f"\n<debug> : Positional GM - using query_model_for_move:")
        print(f"<debug> : pos_system_prompt: {repr(pos_system_prompt)}")
        print(f"<debug> : user_prompt: {repr(user_prompt)}")
        
        pos_response, positional_san, pos_token_info = self.positional_gm.model_interface.get_move_with_extraction(
            pos_system_prompt,
            user_prompt,
            current_turn_number=current_turn_number,
            is_white_to_move=is_white_to_move,
            max_tokens=self.positional_gm.model_interface.max_completion_tokens,
            temperature=self.positional_gm.temperature,
            retry_attempts=self.positional_gm.model_interface.retry_attempts,
        )

        print(f"<debug> : Positional GM response: {repr(pos_response)}")
        print(f"<debug> : Positional GM move: {positional_san}")
        record_token_event("positional", pos_token_info, "positional_gm")
        
        positional_plan_full = extract_plan_sans(pos_response, primary_san=positional_san)
        positional_plan_execute = positional_plan_full[:self.plan_plies] if self.plan_plies > 0 else []
        if self.plan_plies > 0:
            print(f"<self-consistency> : Positional plan (full): {positional_plan_full}")
            print(f"<self-consistency> : Positional plan (first {self.plan_plies} plies): {positional_plan_execute}")

        self.positional_gm.add_memory(f"Positional GM move: {positional_san}")
        
        # Neutral GM analysis
        print(f"\n<debug> : Neutral GM - using query_model_for_move:")
        print(f"<debug> : system_prompt: {repr(system_prompt)}")
        print(f"<debug> : user_prompt: {repr(user_prompt)}")
        
        neutral_response, neutral_san, neutral_token_info = self.neutral_gm.model_interface.get_move_with_extraction(
            system_prompt,
            user_prompt,
            current_turn_number=current_turn_number,
            is_white_to_move=is_white_to_move,
            max_tokens=self.neutral_gm.model_interface.max_completion_tokens,
            temperature=self.neutral_gm.temperature,
            retry_attempts=self.neutral_gm.model_interface.retry_attempts,
        )

        print(f"<debug> : Neutral GM response: {repr(neutral_response)}")
        print(f"<debug> : Neutral GM move: {neutral_san}")
        record_token_event("neutral", neutral_token_info, "neutral_gm")
        
        neutral_plan_full = extract_plan_sans(neutral_response, primary_san=neutral_san)
        neutral_plan_execute = neutral_plan_full[:self.plan_plies] if self.plan_plies > 0 else []
        if self.plan_plies > 0:
            print(f"<self-consistency> : Neutral plan (full): {neutral_plan_full}")
            print(f"<self-consistency> : Neutral plan (first {self.plan_plies} plies): {neutral_plan_execute}")

        self.neutral_gm.add_memory(f"Neutral GM move: {neutral_san}")
        
        # Convert all moves to UCI
        agg_move_uci = san_to_uci(board_fen, aggressive_san) if aggressive_san else None
        pos_move_uci = san_to_uci(board_fen, positional_san) if positional_san else None
        neutral_move_uci = san_to_uci(board_fen, neutral_san) if neutral_san else None
        
        print(f"\n<self-consistency> : All UCI moves:")
        print(f"<self-consistency> : Aggressive GM: {agg_move_uci}")
        print(f"<self-consistency> : Positional GM: {pos_move_uci}")
        print(f"<self-consistency> : Neutral GM: {neutral_move_uci}")
        
        # Voting system: most frequent, then neutral, aggressive, positional
        all_moves = [agg_move_uci, pos_move_uci, neutral_move_uci]
        valid_moves = [move for move in all_moves if move is not None]
        
        if not valid_moves:
            self.final_move = None
            print(f"<self-consistency> : No valid moves extracted")
            # Still return a complete structure even when all moves fail
            self_consistency_history = {
                "query1": {
                    "aggressive_move": aggressive_san,
                    "aggressive_uci": agg_move_uci,
                    "aggressive_response": agg_response,
                    "aggressive_tokens": agg_token_info,
                    "plan_full": aggressive_plan_full,
                    "plan_execute": aggressive_plan_execute
                },
                "query2": {
                    "positional_move": positional_san,
                    "positional_uci": pos_move_uci,
                    "positional_response": pos_response,
                    "positional_tokens": pos_token_info,
                    "plan_full": positional_plan_full,
                    "plan_execute": positional_plan_execute
                },
                "query3": {
                    "neutral_move": neutral_san,
                    "neutral_uci": neutral_move_uci,
                    "neutral_response": neutral_response,
                    "neutral_tokens": neutral_token_info,
                    "plan_full": neutral_plan_full,
                    "plan_execute": neutral_plan_execute
                },
                "final_moves": {
                    "aggressive_uci": agg_move_uci,
                    "positional_uci": pos_move_uci,
                    "neutral_uci": neutral_move_uci,
                    "consensus_move": None,
                    "source_agents": [],
                    "selected_agent": None
                },
                "plans": {
                    "aggressive": {
                        "full": aggressive_plan_full,
                        "execute": aggressive_plan_execute
                    },
                    "positional": {
                        "full": positional_plan_full,
                        "execute": positional_plan_execute
                    },
                    "neutral": {
                        "full": neutral_plan_full,
                        "execute": neutral_plan_execute
                    }
                },
                "final_plan": {
                    "moves_for_execution": [],
                    "full_moves": [],
                    "source_agent": None
                },
                "total_tokens": {
                    "aggressive": agg_token_info,
                    "positional": pos_token_info,
                    "neutral": neutral_token_info,
                    "total_prompt_tokens": (agg_token_info.get("prompt_tokens", 0) if agg_token_info else 0) + 
                                          (pos_token_info.get("prompt_tokens", 0) if pos_token_info else 0) + 
                                          (neutral_token_info.get("prompt_tokens", 0) if neutral_token_info else 0),
                    "total_completion_tokens": (agg_token_info.get("completion_tokens", 0) if agg_token_info else 0) + 
                                             (pos_token_info.get("completion_tokens", 0) if pos_token_info else 0) + 
                                             (neutral_token_info.get("completion_tokens", 0) if neutral_token_info else 0),
                    "total_tokens": (agg_token_info.get("total_tokens", 0) if agg_token_info else 0) + 
                                   (pos_token_info.get("total_tokens", 0) if pos_token_info else 0) + 
                                   (neutral_token_info.get("total_tokens", 0) if neutral_token_info else 0)
                },
                "token_events": token_events
            }
            return self.final_move, self_consistency_history
        
        # Count frequency of moves
        from collections import Counter
        move_counts = Counter(valid_moves)
        most_frequent_moves = move_counts.most_common()
        
        print(f"\n<self-consistency> : Move frequency: {dict(move_counts)}")
        
        # Priority order: most frequent, then neutral, aggressive, positional
        priority_order = [neutral_move_uci, agg_move_uci, pos_move_uci]
        
        # First try most frequent move
        if most_frequent_moves:
            most_frequent_move = most_frequent_moves[0][0]
            if move_counts[most_frequent_move] > 1:  # If there's a majority
                print(f"<self-consistency> : Most frequent move: {most_frequent_move} (count: {move_counts[most_frequent_move]})")
                self.final_move = most_frequent_move
            else:
                # No consensus, use priority order
                for move in priority_order:
                    if move in valid_moves:
                        self.final_move = move
                        print(f"<self-consistency> : Using priority move: {move}")
                        break
        else:
            # Fallback to priority order
            for move in priority_order:
                if move in valid_moves:
                    self.final_move = move
                    print(f"<self-consistency> : Using fallback priority move: {move}")
                    break
        
        print(f"<self-consistency> : Final consensus move: {self.final_move}")
        
        # Collect self-consistency history
        move_sources = {
            "aggressive": agg_move_uci,
            "positional": pos_move_uci,
            "neutral": neutral_move_uci,
        }
        final_source_agents = []
        if self.final_move:
            final_source_agents = [
                agent for agent, uci in move_sources.items()
                if uci and uci == self.final_move
            ]
        selected_agent = None
        for agent in ["neutral", "aggressive", "positional"]:
            if agent in final_source_agents:
                selected_agent = agent
                break
        if not selected_agent and final_source_agents:
            selected_agent = final_source_agents[0]

        # Determine consensus plan
        from collections import Counter
        plan_map = {
            "aggressive": {
                "full": aggressive_plan_full,
                "execute": aggressive_plan_execute
            },
            "positional": {
                "full": positional_plan_full,
                "execute": positional_plan_execute
            },
            "neutral": {
                "full": neutral_plan_full,
                "execute": neutral_plan_execute
            },
        }

        self.final_plan_moves = []
        self.final_plan_full = []
        self.final_plan_source = None

        if self.plan_plies > 0 and self.final_move:
            candidate_agents = [agent for agent in ["neutral", "aggressive", "positional"]
                                if move_sources.get(agent) == self.final_move]
            plan_counter = Counter(
                tuple(plan_map[agent]["execute"])
                for agent in candidate_agents
                if plan_map[agent]["execute"]
            )

            selected_plan_tuple = None
            if plan_counter:
                selected_plan_tuple, count = plan_counter.most_common(1)[0]
                if selected_plan_tuple:
                    for agent in candidate_agents:
                        if tuple(plan_map[agent]["execute"]) == selected_plan_tuple:
                            self.final_plan_moves = list(selected_plan_tuple)
                            self.final_plan_full = list(plan_map[agent]["full"])
                            self.final_plan_source = agent
                            break
            if not self.final_plan_moves:
                for agent in ["neutral", "aggressive", "positional"]:
                    if agent in candidate_agents and plan_map[agent]["execute"]:
                        self.final_plan_moves = list(plan_map[agent]["execute"])
                        self.final_plan_full = list(plan_map[agent]["full"])
                        self.final_plan_source = agent
                        break

            if self.final_plan_moves:
                print(f"<self-consistency> : Selected plan source: {self.final_plan_source}")
                print(f"<self-consistency> : Selected plan (first {self.plan_plies} plies): {self.final_plan_moves}")
            else:
                print(f"<self-consistency> : No consensus plan available")
        else:
            self.final_plan_moves = []
            self.final_plan_full = []
            self.final_plan_source = None

        self_consistency_history = {
            "query1": {
                "aggressive_move": aggressive_san,
                "aggressive_uci": agg_move_uci,
                "aggressive_response": agg_response,
                "aggressive_tokens": agg_token_info,
                "plan_full": aggressive_plan_full,
                "plan_execute": aggressive_plan_execute
            },
            "query2": {
                "positional_move": positional_san,
                "positional_uci": pos_move_uci,
                "positional_response": pos_response,
                "positional_tokens": pos_token_info,
                "plan_full": positional_plan_full,
                "plan_execute": positional_plan_execute
            },
            "query3": {
                "neutral_move": neutral_san,
                "neutral_uci": neutral_move_uci,
                "neutral_response": neutral_response,
                "neutral_tokens": neutral_token_info,
                "plan_full": neutral_plan_full,
                "plan_execute": neutral_plan_execute
            },
            "final_moves": {
                "aggressive_uci": agg_move_uci,
                "positional_uci": pos_move_uci,
                "neutral_uci": neutral_move_uci,
                "consensus_move": self.final_move,
                "source_agents": final_source_agents,
                "selected_agent": selected_agent
            },
            "plans": {
                "aggressive": {
                    "full": aggressive_plan_full,
                    "execute": aggressive_plan_execute
                },
                "positional": {
                    "full": positional_plan_full,
                    "execute": positional_plan_execute
                },
                "neutral": {
                    "full": neutral_plan_full,
                    "execute": neutral_plan_execute
                }
            },
            "final_plan": {
                "moves_for_execution": self.final_plan_moves,
                "full_moves": self.final_plan_full,
                "source_agent": self.final_plan_source
            },
            "total_tokens": {
                "aggressive": agg_token_info,
                "positional": pos_token_info,
                "neutral": neutral_token_info,
                "total_prompt_tokens": (agg_token_info.get("prompt_tokens", 0) if agg_token_info else 0) + 
                                      (pos_token_info.get("prompt_tokens", 0) if pos_token_info else 0) + 
                                      (neutral_token_info.get("prompt_tokens", 0) if neutral_token_info else 0),
                "total_completion_tokens": (agg_token_info.get("completion_tokens", 0) if agg_token_info else 0) + 
                                         (pos_token_info.get("completion_tokens", 0) if pos_token_info else 0) + 
                                         (neutral_token_info.get("completion_tokens", 0) if neutral_token_info else 0),
                "total_tokens": (agg_token_info.get("total_tokens", 0) if agg_token_info else 0) + 
                               (pos_token_info.get("total_tokens", 0) if pos_token_info else 0) + 
                               (neutral_token_info.get("total_tokens", 0) if neutral_token_info else 0)
            },
            "token_events": token_events
        }
        
        return self.final_move, self_consistency_history


def save_debate_history(debate_history, puzzle_idx, output_dir="debate_history"):
    """Save debate history to a JSON file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"puzzle_{puzzle_idx}_debate.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(debate_history, f, indent=2)
    
    print(f"Debate history saved to {filepath}")


def load_environment():
    """Load environment variables from .env file if it exists."""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")
    else:
        print("No .env file found, using system environment variables")


def _is_open_source_model(model_interface: Optional[ChessModelInterface] = None, 
                         debate: Optional[ChessSelfConsistency] = None,
                         debate_v2: Optional[ChessDebateV2] = None) -> bool:
    """Detect if we're using an open source model (Anannas API)."""
    if model_interface:
        return model_interface.base_url and "anannas" in (model_interface.base_url or "").lower()
    elif debate:
        # Check if debate uses Anannas
        return hasattr(debate, 'base_url') and debate.base_url and "anannas" in debate.base_url.lower()
    elif debate_v2:
        # Check if debate_v2 uses Anannas
        return hasattr(debate_v2, 'base_url') and debate_v2.base_url and "anannas" in debate_v2.base_url.lower()
    return False

def _get_system_prompt_for_model(num_future_moves: int) -> str:
    """Get the system prompt (same for all models)."""
    if num_future_moves == 1:
        move_instruction = "give ONE new move."
    else:
        move_instruction = f"give the next {num_future_moves} moves."
    
    return (
        "You are a chess grandmaster.\n"
        "\n"
        "You will be given a partially completed game.\n"
        "\n"
        f"After seeing it, you should repeat the ENTIRE GAME and then {move_instruction}\n"
        "\n"
        "Use standard algebraic notation, e.g. \"e4\" or \"Rdf8\" or \"R1a3\".\n"
        "\n"
        "ALWAYS repeat the entire representation of the game so far.\n"
        "\n"
        "NEVER explain your choice.\n"
    )

def evaluate_puzzles(df: pd.DataFrame, model_interface: ChessModelInterface = None, 
                    debate: ChessSelfConsistency = None, debate_v2: ChessDebateV2 = None, 
                    max_puzzles: int = 5, start_puzzle: int = 0, planning_plies: int = 0,
                    api_delay: float = 0.0) -> pd.DataFrame:
    """
    Evaluate puzzles using either model interface, self-consistency system, or debate system.
    Faithful to the provided evaluation logic:
    - Automatically skips the first move after PGN_partial (the puzzle start move)
    - Tests every other move (model's turns)
    - Automatically plays the intermediate opponent moves
    - Uses correct ply alignment and alternates between model/reference turns
    
    Args:
        df: DataFrame with puzzle data (must have 'PGN_partial' and 'Moves' columns)
        model_interface: ChessModelInterface instance (for single model)
        debate: ChessSelfConsistency instance (for self-consistency mode)
        debate_v2: ChessDebateV2 instance (for debate mode with moderator/judge)
        max_puzzles: Maximum number of puzzles to evaluate
        
    Returns:
        DataFrame with evaluation results
    """
    df_eval = df.copy()
    df_eval["correct_moves"] = 0
    df_eval["puzzle_solved"] = False
    df_eval["error"] = ""
    df_eval["aggressive_move"] = ""
    df_eval["positional_move"] = ""
    df_eval["neutral_move"] = ""
    df_eval["final_consensus_move"] = ""
    df_eval["debate_history"] = ""
    df_eval["single_model_response"] = ""
    df_eval["single_model_move"] = ""
    df_eval["single_model_prompt_tokens"] = 0
    df_eval["single_model_completion_tokens"] = 0
    df_eval["single_model_total_tokens"] = 0
    df_eval["single_model_model"] = ""
    df_eval["single_model_finish_reason"] = ""
    # New V2 debate columns
    df_eval["moderator_decision"] = ""
    df_eval["judge_decision"] = ""
    df_eval["debate_v2_history"] = ""
    
    # Comprehensive token and cost tracking
    df_eval["aggressive_prompt_tokens"] = 0
    df_eval["aggressive_completion_tokens"] = 0
    df_eval["aggressive_total_tokens"] = 0
    df_eval["aggressive_model"] = ""
    df_eval["aggressive_finish_reason"] = ""
    df_eval["aggressive_response"] = ""
    
    df_eval["positional_prompt_tokens"] = 0
    df_eval["positional_completion_tokens"] = 0
    df_eval["positional_total_tokens"] = 0
    df_eval["positional_model"] = ""
    df_eval["positional_finish_reason"] = ""
    df_eval["positional_response"] = ""
    
    df_eval["neutral_prompt_tokens"] = 0
    df_eval["neutral_completion_tokens"] = 0
    df_eval["neutral_total_tokens"] = 0
    df_eval["neutral_model"] = ""
    df_eval["neutral_finish_reason"] = ""
    df_eval["neutral_response"] = ""
    
    df_eval["moderator_prompt_tokens"] = 0
    df_eval["moderator_completion_tokens"] = 0
    df_eval["moderator_total_tokens"] = 0
    df_eval["moderator_model"] = ""
    df_eval["moderator_finish_reason"] = ""
    df_eval["moderator_response"] = ""
    
    df_eval["judge_prompt_tokens"] = 0
    df_eval["judge_completion_tokens"] = 0
    df_eval["judge_total_tokens"] = 0
    df_eval["judge_model"] = ""
    df_eval["judge_finish_reason"] = ""
    df_eval["judge_response"] = ""
    
    df_eval["total_prompt_tokens"] = 0
    df_eval["total_completion_tokens"] = 0
    df_eval["total_tokens"] = 0
    df_eval["estimated_cost_usd"] = 0.0
    df_eval["self_consistency_total_prompt_tokens"] = 0
    df_eval["self_consistency_total_completion_tokens"] = 0
    df_eval["self_consistency_total_tokens"] = 0
    df_eval["debate_total_prompt_tokens"] = 0
    df_eval["debate_total_completion_tokens"] = 0
    df_eval["debate_total_tokens"] = 0
    df_eval["single_model_total_prompt_tokens"] = 0
    df_eval["single_model_total_completion_tokens"] = 0
    df_eval["prompt_token_log"] = ""
    df_eval["self_consistency_prompt_log"] = ""
    df_eval["debate_prompt_log"] = ""
    df_eval["single_model_prompt_log"] = ""
    df_eval["planned_sequences"] = ""
    
    # Debate process information
    df_eval["debate_rounds"] = 0
    df_eval["early_consensus"] = False
    df_eval["single_model_fallback"] = False
    df_eval["both_models_failed"] = False
    df_eval["debate_success"] = False
    df_eval["final_reason"] = ""
    df_eval["supported_side"] = ""
    
    # Input information
    df_eval["board_fen"] = ""
    df_eval["played_plies"] = 0
    df_eval["current_turn"] = 0
    df_eval["is_white_to_move"] = True
    df_eval["user_prompt"] = ""
    df_eval["system_prompt"] = ""

    total_moves = 0
    total_correct_moves = 0
    puzzles_solved = 0

    # Select the range of puzzles to evaluate
    puzzle_range = df_eval.iloc[start_puzzle:start_puzzle + max_puzzles]
    for idx, row in puzzle_range.iterrows():
        print(f"\n=== Evaluating puzzle {idx} ===")

        try:
            per_prompt_events = []
            prompt_logs_by_paradigm = {
                "self_consistency": [],
                "debate": [],
                "single_model": []
            }
            token_totals = {
                "self_consistency": {"prompt": 0, "completion": 0, "total": 0},
                "debate": {"prompt": 0, "completion": 0, "total": 0},
                "single_model": {"prompt": 0, "completion": 0, "total": 0},
            }

            def accumulate_token_event(event: Optional[dict], paradigm_override: Optional[str] = None) -> None:
                if not event:
                    return
                event_copy = dict(event)
                if paradigm_override:
                    event_copy["paradigm"] = paradigm_override
                paradigm = event_copy.get("paradigm", "single_model")
                event_copy.setdefault("puzzle_index", int(idx))
                event_copy.setdefault("role", "prediction")
                event_copy["prompt_tokens"] = event_copy.get("prompt_tokens", 0) or 0
                event_copy["completion_tokens"] = event_copy.get("completion_tokens", 0) or 0
                event_copy["total_tokens"] = event_copy.get("total_tokens", 0) or (
                    event_copy["prompt_tokens"] + event_copy["completion_tokens"]
                )
                # Normalize booleans
                if "is_white_to_move" in event_copy:
                    event_copy["is_white_to_move"] = bool(event_copy["is_white_to_move"])
                per_prompt_events.append(event_copy)
                if paradigm not in prompt_logs_by_paradigm:
                    prompt_logs_by_paradigm[paradigm] = []
                    token_totals[paradigm] = {"prompt": 0, "completion": 0, "total": 0}
                prompt_logs_by_paradigm[paradigm].append(event_copy)
                totals = token_totals[paradigm]
                totals["prompt"] += event_copy["prompt_tokens"]
                totals["completion"] += event_copy["completion_tokens"]
                totals["total"] += event_copy["total_tokens"]

            current_plan_moves: list[str] = []
            current_plan_actor: Optional[str] = None
            current_plan_paradigm: Optional[str] = None
            current_plan_origin: Optional[str] = None
            current_plan_log_entry: Optional[dict] = None
            planned_sequences_log: list[dict] = []

            def reset_plan(status: str) -> None:
                nonlocal current_plan_moves, current_plan_actor, current_plan_paradigm, current_plan_origin, current_plan_log_entry
                if current_plan_log_entry and current_plan_log_entry.get("status", "active") == "active":
                    current_plan_log_entry["status"] = status
                current_plan_moves = []
                current_plan_actor = None
                current_plan_paradigm = None
                current_plan_origin = None
                current_plan_log_entry = None

            def set_plan(paradigm: str, source_agent: str, response_text: Optional[str], *, selected_move: Optional[str] = None, selected_san: Optional[str] = None) -> None:
                nonlocal current_plan_moves, current_plan_actor, current_plan_paradigm, current_plan_origin, current_plan_log_entry
                plan_full = extract_plan_sans(response_text, primary_san=selected_san)
                active_plan_moves = plan_full[:planning_plies] if planning_plies > 0 else []

                if current_plan_moves:
                    reset_plan("replaced")

                plan_entry = {
                    "paradigm": paradigm,
                    "source_agent": source_agent,
                    "moves_for_execution": active_plan_moves,
                    "full_moves": plan_full,
                    "status": "active" if active_plan_moves else ("logged" if plan_full else "empty")
                }
                if selected_move:
                    plan_entry["selected_move"] = selected_move
                if selected_san:
                    plan_entry["selected_san"] = selected_san

                planned_sequences_log.append(plan_entry)

                if active_plan_moves:
                    current_plan_moves = active_plan_moves.copy()
                    current_plan_actor = "opponent"
                    current_plan_paradigm = paradigm
                    current_plan_origin = source_agent
                    current_plan_log_entry = plan_entry
                else:
                    current_plan_moves = []
                    current_plan_actor = None
                    current_plan_paradigm = None
                    current_plan_origin = None
                    current_plan_log_entry = None

            # --- Build starting board from PGN partial
            pgn_io = io.StringIO(row["PGN_partial"])
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                print("Invalid PGN_partial -- skipping")
                df_eval.loc[idx, "error"] = "Invalid PGN_partial"
                continue
                
            board = game.board()
            played_plies = 0
            for mv in game.mainline_moves():
                board.push(mv)
                played_plies += 1

            solution_moves = row["Moves"].split() if pd.notna(row["Moves"]) else []
            if len(solution_moves) == 0:
                print("No moves to play in solution")
                df_eval.loc[idx, "error"] = "No moves to play in solution"
                continue

            current_board = board.copy()
            initial_model_side = current_board.turn
            correct_for_puzzle = 0
            error = ""

            ply = 1  # start from the second solution move (first is automatically played in PGN)

            while ply < len(solution_moves):
                expected_uci = solution_moves[ply]

                if current_board.turn == initial_model_side:
                    # Model's turn to play
                    if planning_plies > 0 and current_plan_moves and current_plan_actor == "model":
                        planned_san = current_plan_moves[0]
                        planned_uci = san_to_uci(current_board.fen(), planned_san)
                        if planned_uci and planned_uci == expected_uci:
                            print(f"<planning> : Executing planned model move {planned_san} ({planned_uci})")
                            current_board.push_uci(expected_uci)
                            correct_for_puzzle += 1
                            total_correct_moves += 1
                            total_moves += 1
                            ply += 1
                            played_plies += 1
                            current_plan_moves.pop(0)
                            if current_plan_moves:
                                current_plan_actor = "opponent"
                            else:
                                if current_plan_log_entry and current_plan_log_entry.get("status") == "active":
                                    current_plan_log_entry["status"] = "completed"
                                current_plan_actor = None
                                current_plan_paradigm = None
                                current_plan_origin = None
                                current_plan_log_entry = None
                            continue
                        else:
                            print(f"<planning> : Planned model move {planned_san} did not match expected {expected_uci}; discarding plan.")
                            reset_plan("aborted")

                    move_paradigm: Optional[str] = None
                    move_source_agent: Optional[str] = None
                    move_response_text: Optional[str] = None
                    move_selected_san: Optional[str] = None

                    num_future_moves = planning_plies + 1 if planning_plies > 0 else 1

                    # Use the same prompt for all models (OpenAI and open-source)
                    system_prompt = _get_system_prompt_for_model(num_future_moves)

                    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
                    current_game = chess.pgn.Game.from_board(current_board)
                    user_prompt = current_game.accept(exporter)

                    print("\n--- Model Turn ---")
                    print("Board:")
                    print(current_board)
                    print("Expected UCI:", expected_uci)

                    if debate:
                        # Use old debate system
                        predicted_uci, debate_history = debate.run_debate(user_prompt, expected_uci, played_plies, current_board.fen())
                        print(f"Debate predicted UCI: {predicted_uci}")
                        
                        # Check if debate_history has the expected structure
                        if not debate_history or "query1" not in debate_history:
                            error_msg = "No legal move generated"
                            print(f"Error: debate_history missing expected structure: {error_msg}")
                            df_eval.loc[idx, "error"] = error_msg
                            df_eval.loc[idx, "final_consensus_move"] = predicted_uci if predicted_uci else ""
                            continue
                        
                        # Save self-consistency data to DataFrame
                        df_eval.loc[idx, "aggressive_move"] = debate_history.get("query1", {}).get("aggressive_move", "")
                        df_eval.loc[idx, "positional_move"] = debate_history.get("query2", {}).get("positional_move", "")
                        df_eval.loc[idx, "neutral_move"] = debate_history.get("query3", {}).get("neutral_move", "")
                        df_eval.loc[idx, "final_consensus_move"] = debate_history.get("final_moves", {}).get("consensus_move", predicted_uci if predicted_uci else "")
                        df_eval.loc[idx, "debate_history"] = str(debate_history)
                        
                        token_events = debate_history.get("token_events") or []
                        if not token_events:
                            fallback_events = []
                            fallback_map = [
                                ("aggressive", debate_history["query1"].get("aggressive_tokens") if debate_history.get("query1") else None),
                                ("positional", debate_history["query2"].get("positional_tokens") if debate_history.get("query2") else None),
                                ("neutral", debate_history["query3"].get("neutral_tokens") if debate_history.get("query3") else None),
                            ]
                            current_turn_number = played_plies // 2 + 1
                            is_white = bool(current_board.turn) if current_board else (played_plies % 2 == 0)
                            for agent_label, token_info in fallback_map:
                                if not token_info:
                                    continue
                                fallback_events.append({
                                    "paradigm": "self_consistency",
                                    "agent": agent_label,
                                    "role": f"{agent_label}_query",
                                    "ply_index": played_plies,
                                    "turn_number": current_turn_number,
                                    "is_white_to_move": is_white,
                                    "prompt_tokens": token_info.get("prompt_tokens", 0) or 0,
                                    "completion_tokens": token_info.get("completion_tokens", 0) or 0,
                                    "total_tokens": token_info.get("total_tokens", 0) or (
                                        (token_info.get("prompt_tokens", 0) or 0) + (token_info.get("completion_tokens", 0) or 0)
                                    ),
                                    "model": token_info.get("model", debate.model_name if debate else ""),
                                    "finish_reason": token_info.get("finish_reason", ""),
                                })
                            token_events = fallback_events
                        for event in token_events:
                            accumulate_token_event(event, paradigm_override="self_consistency")

                        move_paradigm = "self_consistency"
                        final_moves_info = debate_history.get("final_moves", {})
                        selected_agent = final_moves_info.get("selected_agent")
                        source_agents = list(final_moves_info.get("source_agents", []) or [])
                        if selected_agent and selected_agent not in source_agents:
                            source_agents.insert(0, selected_agent)
                        uci_agent_map = {
                            final_moves_info.get("aggressive_uci"): "aggressive",
                            final_moves_info.get("positional_uci"): "positional",
                            final_moves_info.get("neutral_uci"): "neutral",
                        }
                        if not source_agents and predicted_uci in uci_agent_map:
                            source_agents.append(uci_agent_map[predicted_uci])
                        response_lookup = {
                            "aggressive": debate_history.get("query1", {}).get("aggressive_response"),
                            "positional": debate_history.get("query2", {}).get("positional_response"),
                            "neutral": debate_history.get("query3", {}).get("neutral_response"),
                        }
                        san_lookup = {
                            "aggressive": debate_history.get("query1", {}).get("aggressive_move"),
                            "positional": debate_history.get("query2", {}).get("positional_move"),
                            "neutral": debate_history.get("query3", {}).get("neutral_move"),
                        }
                        for candidate_agent in source_agents:
                            response_candidate = response_lookup.get(candidate_agent)
                            if response_candidate:
                                move_source_agent = candidate_agent
                                move_response_text = response_candidate
                                move_selected_san = san_lookup.get(candidate_agent)
                                break
                        if not move_source_agent and predicted_uci in uci_agent_map:
                            candidate_agent = uci_agent_map[predicted_uci]
                            move_source_agent = candidate_agent
                            move_response_text = response_lookup.get(candidate_agent)
                            move_selected_san = san_lookup.get(candidate_agent)
                        if not move_source_agent:
                            move_source_agent = selected_agent
                            move_response_text = response_lookup.get(move_source_agent or "", "")
                            move_selected_san = san_lookup.get(move_source_agent or "")

                        # Save token information for self-consistency
                        # Accumulate tokens across all moves (don't overwrite, add to existing)
                        # Always save tokens, even if no valid move was found
                        if "total_tokens" in debate_history:
                            token_info = debate_history["total_tokens"]
                            
                            # Aggressive tokens - accumulate (always save, even if None/empty)
                            aff_tokens = token_info.get("aggressive") or {}
                            current_agg_p = df_eval.loc[idx, "aggressive_prompt_tokens"] or 0
                            current_agg_c = df_eval.loc[idx, "aggressive_completion_tokens"] or 0
                            current_agg_t = df_eval.loc[idx, "aggressive_total_tokens"] or 0
                            df_eval.loc[idx, "aggressive_prompt_tokens"] = current_agg_p + (aff_tokens.get("prompt_tokens", 0) or 0)
                            df_eval.loc[idx, "aggressive_completion_tokens"] = current_agg_c + (aff_tokens.get("completion_tokens", 0) or 0)
                            df_eval.loc[idx, "aggressive_total_tokens"] = current_agg_t + (aff_tokens.get("total_tokens", 0) or 0)
                            # Only set model/finish_reason if not already set (use first move's info)
                            if not df_eval.loc[idx, "aggressive_model"]:
                                df_eval.loc[idx, "aggressive_model"] = aff_tokens.get("model", "")
                            if not df_eval.loc[idx, "aggressive_finish_reason"]:
                                df_eval.loc[idx, "aggressive_finish_reason"] = aff_tokens.get("finish_reason", "")
                            
                            # Positional tokens - accumulate (always save, even if None/empty)
                            pos_tokens = token_info.get("positional") or {}
                            current_pos_p = df_eval.loc[idx, "positional_prompt_tokens"] or 0
                            current_pos_c = df_eval.loc[idx, "positional_completion_tokens"] or 0
                            current_pos_t = df_eval.loc[idx, "positional_total_tokens"] or 0
                            df_eval.loc[idx, "positional_prompt_tokens"] = current_pos_p + (pos_tokens.get("prompt_tokens", 0) or 0)
                            df_eval.loc[idx, "positional_completion_tokens"] = current_pos_c + (pos_tokens.get("completion_tokens", 0) or 0)
                            df_eval.loc[idx, "positional_total_tokens"] = current_pos_t + (pos_tokens.get("total_tokens", 0) or 0)
                            # Only set model/finish_reason if not already set
                            if not df_eval.loc[idx, "positional_model"]:
                                df_eval.loc[idx, "positional_model"] = pos_tokens.get("model", "")
                            if not df_eval.loc[idx, "positional_finish_reason"]:
                                df_eval.loc[idx, "positional_finish_reason"] = pos_tokens.get("finish_reason", "")
                            
                            # Neutral tokens - accumulate (always save, even if None/empty)
                            neu_tokens = token_info.get("neutral") or {}
                            current_neu_p = df_eval.loc[idx, "neutral_prompt_tokens"] or 0
                            current_neu_c = df_eval.loc[idx, "neutral_completion_tokens"] or 0
                            current_neu_t = df_eval.loc[idx, "neutral_total_tokens"] or 0
                            df_eval.loc[idx, "neutral_prompt_tokens"] = current_neu_p + (neu_tokens.get("prompt_tokens", 0) or 0)
                            df_eval.loc[idx, "neutral_completion_tokens"] = current_neu_c + (neu_tokens.get("completion_tokens", 0) or 0)
                            df_eval.loc[idx, "neutral_total_tokens"] = current_neu_t + (neu_tokens.get("total_tokens", 0) or 0)
                            # Only set model/finish_reason if not already set
                            if not df_eval.loc[idx, "neutral_model"]:
                                df_eval.loc[idx, "neutral_model"] = neu_tokens.get("model", "")
                            if not df_eval.loc[idx, "neutral_finish_reason"]:
                                df_eval.loc[idx, "neutral_finish_reason"] = neu_tokens.get("finish_reason", "")
                            
                            # Cost calculation will be done later
                            df_eval.loc[idx, "estimated_cost_usd"] = 0.0
                        
                        # Save detailed debate history to JSON file
                        save_debate_history(debate_history, idx)
                    elif debate_v2:
                        # Use new debate system with moderator and judge
                        predicted_uci, debate_history = debate_v2.run_debate(user_prompt, expected_uci, played_plies, current_board.fen())
                        print(f"Debate V2 predicted UCI: {predicted_uci}")
                        
                        # Save comprehensive debate V2 data to DataFrame
                        df_eval.loc[idx, "aggressive_move"] = debate_history["round1"]["affirmative_move"]
                        df_eval.loc[idx, "positional_move"] = debate_history["round1"]["negative_move"]
                        df_eval.loc[idx, "moderator_decision"] = str(debate_history["round1"]["moderator_response"])
                        df_eval.loc[idx, "judge_decision"] = debate_history["final_result"]["reason"]
                        df_eval.loc[idx, "final_consensus_move"] = debate_history["final_result"]["final_move_uci"]
                        df_eval.loc[idx, "debate_v2_history"] = str(debate_history)
                        
                        debate_token_events = debate_history.get("token_events") or []
                        if not debate_token_events:
                            token_info = debate_history.get("total_tokens", {})
                            fallback_events = []
                            current_turn_number = played_plies // 2 + 1
                            is_white = bool(current_board.turn) if current_board else (played_plies % 2 == 0)
                            for agent_label, key in (("affirmative", "affirmative"), ("negative", "negative"), ("moderator", "moderator"), ("judge", "judge")):
                                token_info_entry = token_info.get(key)
                                if not token_info_entry:
                                    continue
                                fallback_events.append({
                                    "paradigm": "debate",
                                    "agent": agent_label,
                                    "role": f"{agent_label}_round",
                                    "ply_index": played_plies,
                                    "turn_number": current_turn_number,
                                    "is_white_to_move": is_white,
                                    "prompt_tokens": token_info_entry.get("prompt_tokens", 0) or 0,
                                    "completion_tokens": token_info_entry.get("completion_tokens", 0) or 0,
                                    "total_tokens": token_info_entry.get("total_tokens", 0) or (
                                        (token_info_entry.get("prompt_tokens", 0) or 0) + (token_info_entry.get("completion_tokens", 0) or 0)
                                    ),
                                    "model": token_info_entry.get("model", debate_v2.model_name if debate_v2 else ""),
                                    "finish_reason": token_info_entry.get("finish_reason", ""),
                                })
                            debate_token_events = fallback_events
                        for event in debate_token_events:
                            accumulate_token_event(event, paradigm_override="debate")

                        move_paradigm = "debate"
                        final_result_info = debate_history.get("final_result", {})
                        move_source_agent = final_result_info.get("source_agent")
                        move_selected_san = final_result_info.get("final_move_san")
                        response_lookup_debate = {
                            "affirmative": debate_history.get("round1", {}).get("affirmative_response"),
                            "negative": debate_history.get("round1", {}).get("negative_response"),
                            "moderator": debate_history.get("moderator", {}).get("raw_response"),
                            "judge": debate_history.get("judge", {}).get("final_response") if isinstance(debate_history.get("judge"), dict) else None,
                        }
                        move_response_text = response_lookup_debate.get(move_source_agent or "", None)
                        if not move_response_text and move_source_agent in ("affirmative", "negative"):
                            move_response_text = response_lookup_debate.get(move_source_agent)
                        if not move_response_text and move_source_agent == "moderator":
                            move_response_text = response_lookup_debate.get("moderator") or str(debate_history.get("round1", {}).get("moderator_response"))
                        if not move_response_text and move_source_agent == "judge":
                            move_response_text = response_lookup_debate.get("judge")

                        # Token information
                        # Always save tokens, even if no valid move was found
                        if "total_tokens" in debate_history:
                            token_info = debate_history["total_tokens"]
                            
                            # Aggressive tokens (always save, even if None/empty)
                            aff_tokens = token_info.get("affirmative") or {}
                            df_eval.loc[idx, "aggressive_prompt_tokens"] = aff_tokens.get("prompt_tokens", 0) or 0
                            df_eval.loc[idx, "aggressive_completion_tokens"] = aff_tokens.get("completion_tokens", 0) or 0
                            df_eval.loc[idx, "aggressive_total_tokens"] = aff_tokens.get("total_tokens", 0) or 0
                            df_eval.loc[idx, "aggressive_model"] = aff_tokens.get("model", "")
                            df_eval.loc[idx, "aggressive_finish_reason"] = aff_tokens.get("finish_reason", "")
                            
                            # Positional tokens (always save, even if None/empty)
                            neg_tokens = token_info.get("negative") or {}
                            df_eval.loc[idx, "positional_prompt_tokens"] = neg_tokens.get("prompt_tokens", 0) or 0
                            df_eval.loc[idx, "positional_completion_tokens"] = neg_tokens.get("completion_tokens", 0) or 0
                            df_eval.loc[idx, "positional_total_tokens"] = neg_tokens.get("total_tokens", 0) or 0
                            df_eval.loc[idx, "positional_model"] = neg_tokens.get("model", "")
                            df_eval.loc[idx, "positional_finish_reason"] = neg_tokens.get("finish_reason", "")
                            
                            # Moderator tokens (always save, even if None/empty)
                            mod_tokens = token_info.get("moderator") or {}
                            df_eval.loc[idx, "moderator_prompt_tokens"] = mod_tokens.get("prompt_tokens", 0) or 0
                            df_eval.loc[idx, "moderator_completion_tokens"] = mod_tokens.get("completion_tokens", 0) or 0
                            df_eval.loc[idx, "moderator_total_tokens"] = mod_tokens.get("total_tokens", 0) or 0
                            df_eval.loc[idx, "moderator_model"] = mod_tokens.get("model", "")
                            df_eval.loc[idx, "moderator_finish_reason"] = mod_tokens.get("finish_reason", "")
                            
                            # Judge tokens (always save, even if None/empty)
                            judge_tokens = token_info.get("judge") or {}
                            df_eval.loc[idx, "judge_prompt_tokens"] = judge_tokens.get("prompt_tokens", 0) or 0
                            df_eval.loc[idx, "judge_completion_tokens"] = judge_tokens.get("completion_tokens", 0) or 0
                            df_eval.loc[idx, "judge_total_tokens"] = judge_tokens.get("total_tokens", 0) or 0
                            df_eval.loc[idx, "judge_model"] = judge_tokens.get("model", "")
                            df_eval.loc[idx, "judge_finish_reason"] = judge_tokens.get("finish_reason", "")
                            
                        # Response information
                        df_eval.loc[idx, "aggressive_response"] = debate_history["round1"]["affirmative_response"]
                        df_eval.loc[idx, "positional_response"] = debate_history["round1"]["negative_response"]
                        df_eval.loc[idx, "moderator_response"] = str(debate_history["round1"]["moderator_response"])
                        
                        # Debate process information
                        df_eval.loc[idx, "early_consensus"] = debate_history["round1"].get("early_consensus", False)
                        df_eval.loc[idx, "single_model_fallback"] = debate_history["round1"].get("single_model", False)
                        df_eval.loc[idx, "both_models_failed"] = debate_history["round1"].get("failure", False)
                        df_eval.loc[idx, "debate_success"] = debate_history["final_result"]["success"]
                        df_eval.loc[idx, "final_reason"] = debate_history["final_result"]["reason"]
                        df_eval.loc[idx, "supported_side"] = debate_history["final_result"]["supported_side"]
                        
                        # Input information
                        df_eval.loc[idx, "board_fen"] = current_board.fen()
                        df_eval.loc[idx, "played_plies"] = played_plies
                        df_eval.loc[idx, "current_turn"] = played_plies // 2 + 1
                        df_eval.loc[idx, "is_white_to_move"] = current_board.turn
                        df_eval.loc[idx, "user_prompt"] = user_prompt
                        df_eval.loc[idx, "system_prompt"] = system_prompt
                        
                        # Estimate cost (rough calculation - adjust rates as needed)
                        total_tokens = df_eval.loc[idx, "total_tokens"]
                        # Cost calculation will be done later
                        df_eval.loc[idx, "estimated_cost_usd"] = 0.0
                        
                        # Save detailed debate history to JSON file
                        save_debate_history_v2(debate_history, idx)
                    else:
                        # Use single model
                        raw_response, predicted_san, token_info = model_interface.get_move_with_extraction(
                            system_prompt,
                            user_prompt,
                            current_turn_number=played_plies // 2 + 1,
                            is_white_to_move=current_board.turn,
                            max_tokens=model_interface.max_completion_tokens,
                            temperature=model_interface.default_temperature,
                            retry_attempts=model_interface.retry_attempts,
                            api_delay=api_delay,
                        )
                        print("Predicted Response:", raw_response)
                        print("Turn number:", played_plies // 2 + 1)
                        print("Is white to move:", current_board.turn)
                        
                        if not predicted_san:
                            error = f"Failed to extract SAN at puzzle {idx}, ply {ply}"
                            print(f"❌ Failed to extract SAN at puzzle {idx}, ply {ply}")
                            break
                            
                        predicted_uci = san_to_uci(current_board.fen(), predicted_san)
                        print(f"Predicted UCI: {predicted_uci}")
                        move_paradigm = "single_model"
                        move_source_agent = "single"
                        move_response_text = raw_response
                        move_selected_san = predicted_san
                        
                        # Save single model data to DataFrame
                        df_eval.loc[idx, "single_model_response"] = raw_response
                        df_eval.loc[idx, "single_model_move"] = predicted_san
                        
                        # Save input information for single model
                        df_eval.loc[idx, "board_fen"] = current_board.fen()
                        df_eval.loc[idx, "played_plies"] = played_plies
                        df_eval.loc[idx, "current_turn"] = played_plies // 2 + 1
                        df_eval.loc[idx, "is_white_to_move"] = current_board.turn
                        df_eval.loc[idx, "user_prompt"] = user_prompt
                        df_eval.loc[idx, "system_prompt"] = system_prompt
                        
                        # Save token information for single model
                        if token_info:
                            df_eval.loc[idx, "single_model_prompt_tokens"] = token_info.get("prompt_tokens", 0)
                            df_eval.loc[idx, "single_model_completion_tokens"] = token_info.get("completion_tokens", 0)
                            df_eval.loc[idx, "single_model_total_tokens"] = token_info.get("total_tokens", 0)
                            df_eval.loc[idx, "single_model_model"] = token_info.get("model", "")
                            df_eval.loc[idx, "single_model_finish_reason"] = token_info.get("finish_reason", "")
                            accumulate_token_event({
                                "paradigm": "single_model",
                                "agent": "single",
                                "role": "move_prediction",
                                "ply_index": played_plies,
                                "turn_number": played_plies // 2 + 1,
                                "is_white_to_move": bool(current_board.turn),
                                "prompt_tokens": token_info.get("prompt_tokens", 0) or 0,
                                "completion_tokens": token_info.get("completion_tokens", 0) or 0,
                                "total_tokens": token_info.get("total_tokens", 0) or (
                                    (token_info.get("prompt_tokens", 0) or 0) + (token_info.get("completion_tokens", 0) or 0)
                                ),
                                "model": token_info.get("model", ""),
                                "finish_reason": token_info.get("finish_reason", ""),
                            })
                            
                            # Cost calculation will be done later
                            df_eval.loc[idx, "estimated_cost_usd"] = 0.0

                    if predicted_uci == expected_uci:
                        print("✅ Correct move!")
                        if not move_selected_san and predicted_uci:
                            try:
                                move_obj_for_san = chess.Move.from_uci(predicted_uci)
                                move_selected_san = current_board.san(move_obj_for_san)
                            except Exception:
                                move_selected_san = None
                        set_plan(
                            move_paradigm or "unknown",
                            move_source_agent or "unknown",
                            move_response_text,
                            selected_move=predicted_uci,
                            selected_san=move_selected_san,
                        )
                        current_board.push_uci(expected_uci)
                        correct_for_puzzle += 1
                        total_correct_moves += 1
                        total_moves += 1
                    else:
                        error = f"Mismatch: expected {expected_uci} but got {predicted_uci}"
                        print(f"❌ Mismatch: expected {expected_uci} but got {predicted_uci}")
                        if planning_plies > 0:
                            reset_plan("aborted")
                        break
                else:
                    if planning_plies > 0 and current_plan_moves and current_plan_actor == "opponent":
                        try:
                            move_obj = chess.Move.from_uci(expected_uci)
                            expected_san = current_board.san(move_obj)
                        except Exception:
                            expected_san = None
                        if expected_san and current_plan_moves[0] == expected_san:
                            print(f"<planning> : Opponent move matched plan {expected_san}")
                            current_plan_moves.pop(0)
                            if current_plan_moves:
                                current_plan_actor = "model"
                            else:
                                if current_plan_log_entry and current_plan_log_entry.get("status") == "active":
                                    current_plan_log_entry["status"] = "completed"
                                current_plan_actor = None
                                current_plan_paradigm = None
                                current_plan_origin = None
                                current_plan_log_entry = None
                        else:
                            if expected_san:
                                print(f"<planning> : Opponent move {expected_san} diverged from plan {current_plan_moves[0] if current_plan_moves else 'None'}, clearing plan.")
                            reset_plan("aborted")
                    # Opponent's turn, automatically play this move (no testing or scoring)
                    try:
                        current_board.push_uci(expected_uci)
                        print(f"Auto-played opponent move: {expected_uci}")
                    except Exception as e:
                        print(f"Error applying opponent move {expected_uci}: {e}")
                        error = f"Error applying opponent move {expected_uci}: {e}"
                        break

                ply += 1
                played_plies += 1

            df_eval.loc[idx, "correct_moves"] = correct_for_puzzle
            df_eval.loc[idx, "puzzle_solved"] = (correct_for_puzzle == (len(solution_moves) // 2))
            df_eval.loc[idx, "error"] = error
            if current_plan_moves:
                reset_plan("incomplete")
            df_eval.loc[idx, "self_consistency_total_prompt_tokens"] = token_totals["self_consistency"]["prompt"]
            df_eval.loc[idx, "self_consistency_total_completion_tokens"] = token_totals["self_consistency"]["completion"]
            df_eval.loc[idx, "self_consistency_total_tokens"] = token_totals["self_consistency"]["total"]
            df_eval.loc[idx, "debate_total_prompt_tokens"] = token_totals["debate"]["prompt"]
            df_eval.loc[idx, "debate_total_completion_tokens"] = token_totals["debate"]["completion"]
            df_eval.loc[idx, "debate_total_tokens"] = token_totals["debate"]["total"]
            df_eval.loc[idx, "single_model_total_prompt_tokens"] = token_totals["single_model"]["prompt"]
            df_eval.loc[idx, "single_model_total_completion_tokens"] = token_totals["single_model"]["completion"]
            df_eval.loc[idx, "single_model_total_tokens"] = token_totals["single_model"]["total"]
            overall_prompt_tokens = (
                token_totals["self_consistency"]["prompt"] +
                token_totals["debate"]["prompt"] +
                token_totals["single_model"]["prompt"]
            )
            overall_completion_tokens = (
                token_totals["self_consistency"]["completion"] +
                token_totals["debate"]["completion"] +
                token_totals["single_model"]["completion"]
            )
            overall_total_tokens = (
                token_totals["self_consistency"]["total"] +
                token_totals["debate"]["total"] +
                token_totals["single_model"]["total"]
            )
            df_eval.loc[idx, "total_prompt_tokens"] = overall_prompt_tokens
            df_eval.loc[idx, "total_completion_tokens"] = overall_completion_tokens
            df_eval.loc[idx, "total_tokens"] = overall_total_tokens
            df_eval.loc[idx, "prompt_token_log"] = json.dumps(per_prompt_events)
            df_eval.loc[idx, "self_consistency_prompt_log"] = json.dumps(prompt_logs_by_paradigm["self_consistency"])
            df_eval.loc[idx, "debate_prompt_log"] = json.dumps(prompt_logs_by_paradigm["debate"])
            df_eval.loc[idx, "single_model_prompt_log"] = json.dumps(prompt_logs_by_paradigm["single_model"])
            df_eval.loc[idx, "planned_sequences"] = json.dumps(planned_sequences_log)
            
            # Number of moves model should play (half the moves after skipping the first)
            print("Model solved everything:", correct_for_puzzle == (len(solution_moves) // 2))
            
            if debate:
                # Clear memory between puzzles
                debate.clear_memory()
            elif debate_v2:
                # Clear memory between puzzles
                debate_v2.clear_memory()
                
        except Exception as e:
            print(f"Error processing puzzle {idx}: {e}")
            df_eval.loc[idx, "error"] = str(e)
    
    return df_eval


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chess Puzzle Evaluator")
    
    parser.add_argument("--csv-file", default="data/input/lichess_puzzles_with_pgn_1000.csv",
                       help="Path to CSV file with puzzle data")
    parser.add_argument("--max-puzzles", type=int, default=10,
                       help="Maximum number of puzzles to evaluate")
    parser.add_argument("--start-puzzle", type=int, default=0,
                       help="Starting puzzle index (0-based)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-instruct",
                       help="Model to use (e.g., 'gpt-3.5-turbo-instruct', 'qwen/qwen3-4b:free')")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run puzzle evaluation")
    parser.add_argument("--sample", type=int, default=None,
                       help="Sample N puzzles randomly")
    parser.add_argument("--stats", action="store_true",
                       help="Show puzzle statistics")
    parser.add_argument("--rating", action="store_true",
                       help="Calculate Glicko-2 rating")
    parser.add_argument("--self-consistency", action="store_true",
                       help="Use self-consistency system (3 independent queries with majority vote)")
    parser.add_argument("--debate", action="store_true",
                       help="Use multi-agent debate system with moderator and judge")
    parser.add_argument("--output", default=None,
                       help="Output file for results")
    parser.add_argument("--plan-plies", type=int, default=0,
                       help="Number of future plies to request in planning prompts (0 disables planning)")
    parser.add_argument("--use-anannas", action="store_true",
                       help="Use Anannas API (requires ANANNAS_API_KEY in .env file)")
    parser.add_argument("--anannas-base-url", type=str, default=None,
                       help="Anannas API base URL (defaults to https://api.anannas.ai/v1, or ANANNAS_API_URL from .env)")
    
    args = parser.parse_args()
    
    # Load environment
    load_environment()
    
    # Determine which API to use and get the key from .env
    use_anannas = args.use_anannas
    api_key = None
    base_url = None
    
    if use_anannas:
        # Use Anannas API
        api_key = os.getenv("ANANNAS_API_KEY")
        base_url = args.anannas_base_url or os.getenv("ANANNAS_API_URL", "https://api.anannas.ai/v1")
        if not api_key:
            print("Error: ANANNAS_API_KEY not found in .env file")
            print("Please add ANANNAS_API_KEY=your-key-here to your .env file")
            sys.exit(1)
        
        # Warn if using OpenAI model name with Anannas may not exist
    else:
        # Use OpenAI API (default)
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        if not api_key:
            # Fallback: check if Anannas key is available
            api_key = os.getenv("ANANNAS_API_KEY")
            base_url = os.getenv("ANANNAS_API_URL", "https://api.anannas.ai/v1")
            if api_key:
                print("Warning: OPENAI_API_KEY not found in .env, using ANANNAS_API_KEY instead")
                print("Tip: Use --use-anannas flag to explicitly use Anannas API")
                use_anannas = True
            else:
                print("Error: Neither OPENAI_API_KEY nor ANANNAS_API_KEY found in .env file")
                print("Please add one of these to your .env file:")
                print("  OPENAI_API_KEY=your-openai-key-here")
                print("  ANANNAS_API_KEY=your-anannas-key-here")
                sys.exit(1)
    
    # Read CSV file
    try:
        df = read_chess_puzzles_csv(args.csv_file)
        print(f"Loaded {len(df)} puzzles from {args.csv_file}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    
    # Sample puzzles if requested
    if args.sample:
        df = sample_puzzles(df, n=args.sample)
        print(f"Sampled {len(df)} puzzles")
    
    # Show statistics
    if args.stats:
        stats = get_puzzle_stats(df)
        print("\nPuzzle Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Run evaluation
    if args.evaluate:
        if args.self_consistency:
            print(f"\nEvaluating puzzles with self-consistency system...")
            self_consistency = ChessSelfConsistency(
                model_name=args.model,
                temperature=0.1,
                openai_api_key=api_key,
                base_url=base_url,
                max_rounds=2,
                sleep_time=0.1,
                plan_plies=args.plan_plies
            )
            # Evaluate puzzles with self-consistency system
            df_results = evaluate_puzzles(
                df,
                debate=self_consistency,
                max_puzzles=args.max_puzzles,
                start_puzzle=args.start_puzzle,
                planning_plies=args.plan_plies,
            )
        elif args.debate:
            print(f"\nEvaluating puzzles with new multi-agent debate system (moderator + judge)...")
            debate_v2 = ChessDebateV2(
                model_name=args.model,
                temperature=0.1,
                openai_api_key=api_key,
                base_url=base_url,
                max_rounds=3,
                sleep_time=0.1,
                plan_plies=args.plan_plies
            )
            # Evaluate puzzles with new debate system
            df_results = evaluate_puzzles(
                df,
                debate_v2=debate_v2,
                max_puzzles=args.max_puzzles,
                start_puzzle=args.start_puzzle,
                planning_plies=args.plan_plies,
            )
        else:
            print(f"\nEvaluating puzzles with {args.model}...")
            model_interface = ChessModelInterface(
                api_key=api_key,
                model_name=args.model,
                base_url=base_url,
                max_completion_tokens=640,
                default_temperature=0.1,
                retry_attempts=2,
            )
            # Evaluate puzzles with single model
            df_results = evaluate_puzzles(
                df,
                model_interface=model_interface,
                max_puzzles=args.max_puzzles,
                start_puzzle=args.start_puzzle,
                planning_plies=args.plan_plies,
            )
        
        # Save results
        if args.output:
            save_puzzles_csv(df_results, args.output)
            print(f"Results saved to {args.output}")
        
        # Show summary
        solved = df_results["puzzle_solved"].sum()
        total = len(df_results)
        print(f"\nEvaluation Summary:")
        print(f"  Puzzles solved: {solved}/{total} ({solved/total*100:.1f}%)")
        
        # Show individual results
        for idx, row in df_results.head(args.max_puzzles).iterrows():
            if row["error"] == "":
                status = "✓" if row["puzzle_solved"] else "✗"
                expected_moves = row["Moves"].split() if pd.notna(row["Moves"]) else []
                expected_move = expected_moves[1] if len(expected_moves) > 1 else "N/A"
                print(f"Puzzle {idx}: {status} Expected: {expected_move}, Correct moves: {row['correct_moves']}")
            else:
                print(f"Puzzle {idx}: ERROR - {row['error']}")
    
    # Calculate rating
    if args.rating:
        print("\nCalculating Glicko-2 rating...")
        if "puzzle_solved" in df.columns:
            agent_rating = update_agent_rating_from_puzzles(df)
            print(f"Agent rating: {agent_rating}")
        else:
            print("Error: No puzzle_solved column found. Run evaluation first.")


if __name__ == "__main__":
    main()
