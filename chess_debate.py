"""
Chess Debate System with Moderator and Judge

This module provides a debate system specifically designed for chess moves
using the MAD (Multi-Agent Debate) framework with moderator and judge roles.
"""

import os
import json
import sys
import re
import chess
import chess.pgn
import io
from typing import Optional, Tuple, Dict, Any

# Add the MAD utils path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAD_DIR = os.path.join(CURRENT_DIR, "MAD")
if MAD_DIR not in sys.path:
    sys.path.insert(0, MAD_DIR)
from utils.agent import Agent
from model_interface import ChessModelInterface
from chess_utils import extract_predicted_move, san_to_uci
from MAD.utils.openai_utils import num_tokens_from_string, model2max_context


class ChessDebatePlayer(Agent):
    """Chess-specific debate player that extends the base Agent class"""
    
    def __init__(self, model_name: str, name: str, temperature: float, 
                 openai_api_key: str, sleep_time: float = 0, base_url: str = None) -> None:
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

    def _build_combined_prompt(self) -> str:
        return "\n".join(msg["content"] for msg in self.memory_lst if msg.get("content"))

    def _ensure_memory_budget(self) -> Tuple[str, int, int]:
        max_context = model2max_context.get(self.model_name, 8192)
        safety_margin = 32
        while True:
            combined_prompt = self._build_combined_prompt()
            prompt_tokens = num_tokens_from_string(combined_prompt, self.model_name)
            available_completion = max_context - prompt_tokens - safety_margin
            if available_completion >= safety_margin or len(self.memory_lst) <= 1:
                return combined_prompt, prompt_tokens, max_context
            # prune the oldest non-system message
            for idx, msg in enumerate(self.memory_lst):
                if msg.get("role") != "system":
                    del self.memory_lst[idx]
                    break
            else:
                return combined_prompt, prompt_tokens, max_context

    def ask(self, temperature: float = None):
        """Query for answer using our model interface instead of the old API"""
        combined_prompt, prompt_tokens, max_context = self._ensure_memory_budget()
        available_completion = max_context - prompt_tokens - 32
        max_tokens = max(16, min(self.model_interface.max_completion_tokens, available_completion))
        print(f"\n<debug> : ChessDebatePlayer '{self.name}' ask() method:")
        print(f"<debug> : combined_prompt: {repr(combined_prompt[:200])}...")
        print(f"<debug> : prompt_tokens={prompt_tokens}, max_tokens={max_tokens}")
        
        response, token_info = self.model_interface.query_model_for_move_with_tokens(
            system_prompt="",  # Already included in combined_prompt
            user_prompt=combined_prompt,
            max_tokens=max_tokens,
            temperature=temperature if temperature else self.temperature,
            top_p=self.model_interface.default_top_p
        )
        self.last_token_info = token_info
        
        print(f"<debug> : response: {repr(response)}")
        return response or ""


class ChessDebateV2:
    """Chess Debate System V2 with Moderator and Judge"""
    
    def __init__(self, 
                 model_name: str = 'gpt-3.5-turbo-instruct',
                 temperature: float = 0.1,
                 openai_api_key: str = None,
                 max_rounds: int = 3,
                 sleep_time: float = 0.1,
                 plan_plies: int = 0,
                 base_url: str = None):
        """Create a chess debate with moderator and judge"""
        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = openai_api_key
        self.max_rounds = max_rounds
        self.sleep_time = sleep_time
        self.plan_plies = max(0, plan_plies)
        
        # Load chess debate configuration
        current_script_path = os.path.abspath(__file__)
        config_path = os.path.join(os.path.dirname(current_script_path), 
                                  'MAD/utils/config4chess.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.base_player_meta_prompt = self.config['player_meta_prompt']
        self.base_moderator_meta_prompt = self.config['moderator_meta_prompt']
        
        # Initialize players
        self.affirmative = ChessDebatePlayer(
            model_name=model_name,
            name="Affirmative side",
            temperature=temperature,
            openai_api_key=openai_api_key,
            sleep_time=sleep_time,
            base_url=base_url
        )
        
        self.negative = ChessDebatePlayer(
            model_name=model_name,
            name="Negative side",
            temperature=temperature,
            openai_api_key=openai_api_key,
            sleep_time=sleep_time,
            base_url=base_url
        )
        
        self.moderator = ChessDebatePlayer(
            model_name=model_name,
            name="Moderator",
            temperature=temperature,
            openai_api_key=openai_api_key,
            sleep_time=sleep_time,
            base_url=base_url
        )
        
        # Debate results
        self.debate_history = []
        self.final_move = None
        self.config['success'] = False

    def init_prompt(self, debate_topic: str):
        """Initialize prompts with the debate topic"""
        self.config['debate_topic'] = debate_topic
        self.config['player_meta_prompt'] = self._apply_plan_instruction(self.base_player_meta_prompt)
        self.config['moderator_meta_prompt'] = self._apply_plan_instruction(self.base_moderator_meta_prompt)
        
        # Replace placeholders in prompts
        def prompt_replace(key):
            self.config[key] = self.config[key].replace("##debate_topic##", self.config["debate_topic"])
        
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("affirmative_prompt")
        prompt_replace("judge_prompt_last2")

    def round_dct(self, num: int):
        """Convert round number to word"""
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 
            6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]

    def clear_memory(self):
        """Clear memory between puzzles to prevent token buildup"""
        self.affirmative.memory_lst = []
        self.negative.memory_lst = []
        self.moderator.memory_lst = []

    def _apply_plan_instruction(self, prompt: str) -> str:
        """Apply plan instruction based on plan_plies.
        - If plan_plies == 0: use "ONE new move"
        - If plan_plies > 0: use "the next {num_future_moves} moves"
        """
        num_future_moves = self.plan_plies + 1 if self.plan_plies > 0 else 1
        
        if num_future_moves == 1:
            # Replace "the next ... moves" with "ONE new move"
            pattern = re.compile(r"(give )the next [^\s]+ moves", flags=re.IGNORECASE)
            def _repl_one(match: re.Match) -> str:
                return f"{match.group(1)}ONE new move"
            updated_prompt, count = pattern.subn(_repl_one, prompt, count=1)
            # If no match, check if "ONE new move" is already there
            if count == 0 and "ONE new move" not in prompt:
                # Try to replace any "the next ... moves" pattern
                pattern2 = re.compile(r"the next [^\s]+ moves", flags=re.IGNORECASE)
                updated_prompt, count = pattern2.subn("ONE new move", prompt, count=1)
        else:
            # Replace "ONE new move" with "the next {num_future_moves} moves"
            pattern = re.compile(r"ONE new move", flags=re.IGNORECASE)
            def _repl_multi(match: re.Match) -> str:
                return f"the next {num_future_moves} moves"
            updated_prompt, count = pattern.subn(_repl_multi, prompt, count=1)
            # Also handle existing "the next ... moves" pattern
            if count == 0:
                pattern2 = re.compile(r"(the next) ([^\s]+) (moves)", flags=re.IGNORECASE)
                def _repl_existing(match: re.Match) -> str:
                    return f"{match.group(1)} {num_future_moves} {match.group(3)}"
                updated_prompt, count = pattern2.subn(_repl_existing, prompt, count=1)
        
        return updated_prompt if count > 0 else prompt


    def run_debate(self, user_prompt: str, expected_uci: str = None, 
                   played_plies: int = 0, board_fen: str = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """Run a chess debate on a position using the new moderator/judge system"""
        print(f"\n<debate_v2> : Starting chess debate with moderator and judge")
        print(f"<debug> : run_debate called with played_plies={played_plies}")
        token_events: list[Dict[str, Any]] = []
        turn_number = played_plies // 2 + 1
        is_white_to_move = (played_plies % 2 == 0)
        final_source_agent: Optional[str] = None
        final_source_agents: list[str] = []
        judge_details: Dict[str, Any] = {}
        moderator_raw_response: Optional[str] = None

        def record_token_event(agent_label: str, role: str, round_number: int, token_info: Optional[dict]) -> None:
            info = token_info or {}
            prompt_tokens = info.get("prompt_tokens", 0) or 0
            completion_tokens = info.get("completion_tokens", 0) or 0
            total_tokens = info.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
            token_events.append({
                "paradigm": "debate",
                "agent": agent_label,
                "role": role,
                "round": round_number,
                "ply_index": played_plies,
                "turn_number": turn_number,
                "is_white_to_move": is_white_to_move,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model": info.get("model", self.model_name),
                "finish_reason": info.get("finish_reason", ""),
            })
        
        # Create debate topic
        debate_topic = f"What is the best move in this chess position?\n\nPosition: {user_prompt}"
        self.init_prompt(debate_topic)
        
        # Override prompts to be chess-specific
        self.config['affirmative_prompt'] = user_prompt
        
        # Initialize agents with meta prompts
        self.affirmative.set_meta_prompt(self.config['player_meta_prompt'])
        self.negative.set_meta_prompt(self.config['player_meta_prompt'])
        self.moderator.set_meta_prompt(self.config['moderator_meta_prompt'])
        
        # Round 1: Initial positions
        print(f"===== Debate Round-1 =====")
        
        # Aggressive side presents initial move using proper extraction
        aff_response, aff_move_san, aff_token_info = self.affirmative.model_interface.get_move_with_extraction(
            self.config['player_meta_prompt'],
            user_prompt,
            current_turn_number=turn_number,
            is_white_to_move=is_white_to_move,
            max_tokens=self.affirmative.model_interface.max_completion_tokens,
            temperature=self.affirmative.temperature,
            retry_attempts=self.affirmative.model_interface.retry_attempts,
        )
        record_token_event("affirmative", "affirmative_round1", 1, aff_token_info)
        # Store token info for final case
        self.aff_token_info = aff_token_info
        
        # Handle empty response
        if not aff_response:
            aff_response = "No response from model"
            aff_move_san = None
            
        self.affirmative.add_memory(aff_response)
        self.aff_ans = aff_response
        self.config['base_answer'] = aff_response
        print(f"<debate_v2> : Aggressive move: {aff_move_san}")
        
        # Positional side responds using proper extraction
        neg_response, neg_move_san, neg_token_info = self.negative.model_interface.get_move_with_extraction(
            self.config['player_meta_prompt'],
            user_prompt,
            current_turn_number=turn_number,
            is_white_to_move=is_white_to_move,
            max_tokens=self.negative.model_interface.max_completion_tokens,
            temperature=self.negative.temperature,
            retry_attempts=self.negative.model_interface.retry_attempts,
        )
        record_token_event("negative", "negative_round1", 1, neg_token_info)
        # Store token info for final case
        self.neg_token_info = neg_token_info
        
        # Handle empty response
        if not neg_response:
            neg_response = "No response from model"
            neg_move_san = None
            
        self.negative.add_memory(neg_response)
        self.neg_ans = neg_response
        print(f"<debate_v2> : Positional move: {neg_move_san}")
        
        # Check if both models failed to provide moves
        if not aff_move_san and not neg_move_san:
            print(f"<debate_v2> : Both models failed to provide moves")
            self.config['debate_answer'] = None
            self.config['success'] = False
            self.config['Reason'] = "Both models failed to provide moves"
            self.config['Supported Side'] = "None"
            
            # Return failure result
            debate_history = {
                "round1": {
                    "affirmative_move": None,
                    "negative_move": None,
                    "affirmative_response": aff_response,
                    "negative_response": neg_response,
                    "affirmative_tokens": aff_token_info,
                    "negative_tokens": neg_token_info,
                    "moderator_response": "Both models failed",
                    "early_consensus": False,
                    "failure": True
                },
                "final_result": {
                    "final_move_san": None,
                    "final_move_uci": None,
                    "success": False,
                    "reason": "Both models failed to provide moves",
                    "supported_side": "None",
                    "source_agent": final_source_agent,
                    "source_agents": final_source_agents
                },
                "total_tokens": {
                    "affirmative": aff_token_info,
                    "negative": neg_token_info,
                    "moderator": None,
                    "judge": None,
                    "total_prompt_tokens": (aff_token_info.get("prompt_tokens", 0) if aff_token_info else 0) + (neg_token_info.get("prompt_tokens", 0) if neg_token_info else 0),
                    "total_completion_tokens": (aff_token_info.get("completion_tokens", 0) if aff_token_info else 0) + (neg_token_info.get("completion_tokens", 0) if neg_token_info else 0),
                    "total_tokens": (aff_token_info.get("total_tokens", 0) if aff_token_info else 0) + (neg_token_info.get("total_tokens", 0) if neg_token_info else 0)
                },
                "token_events": list(token_events),
                "judge": judge_details,
                "moderator": {
                    "raw_response": moderator_raw_response
                }
            }
            
            return None, debate_history
        
        # Check if only one model provided a move - use that move
        if aff_move_san and not neg_move_san:
            print(f"<debate_v2> : Only aggressive model provided move: {aff_move_san}")
            self.config['debate_answer'] = aff_move_san
            self.config['success'] = True
            self.config['Reason'] = "Only aggressive model provided a move"
            self.config['Supported Side'] = "Aggressive"
            final_source_agent = "affirmative"
            final_source_agents = ["affirmative"]
            
            # Convert to UCI
            if board_fen:
                self.final_move = san_to_uci(board_fen, aff_move_san)
            else:
                self.final_move = None
            
            print(f"<debate_v2> : Final move (SAN): {aff_move_san}")
            print(f"<debate_v2> : Final move (UCI): {self.final_move}")
            
            # Return single model result
            debate_history = {
                "round1": {
                    "affirmative_move": aff_move_san,
                    "negative_move": None,
                    "affirmative_response": aff_response,
                    "negative_response": neg_response,
                    "affirmative_tokens": aff_token_info,
                    "negative_tokens": neg_token_info,
                    "moderator_response": "Only aggressive model responded",
                    "early_consensus": False,
                    "single_model": True
                },
                "final_result": {
                    "final_move_san": aff_move_san,
                    "final_move_uci": self.final_move,
                    "success": True,
                    "reason": "Only aggressive model provided a move",
                    "supported_side": "Aggressive",
                    "source_agent": final_source_agent,
                    "source_agents": final_source_agents
                },
                "total_tokens": {
                    "affirmative": aff_token_info,
                    "negative": neg_token_info,
                    "moderator": None,
                    "judge": None,
                    "total_prompt_tokens": (aff_token_info.get("prompt_tokens", 0) if aff_token_info else 0) + (neg_token_info.get("prompt_tokens", 0) if neg_token_info else 0),
                    "total_completion_tokens": (aff_token_info.get("completion_tokens", 0) if aff_token_info else 0) + (neg_token_info.get("completion_tokens", 0) if neg_token_info else 0),
                    "total_tokens": (aff_token_info.get("total_tokens", 0) if aff_token_info else 0) + (neg_token_info.get("total_tokens", 0) if neg_token_info else 0)
                },
                "token_events": list(token_events),
                "judge": judge_details,
                "moderator": {
                    "raw_response": moderator_raw_response
                }
            }
            
            return self.final_move, debate_history
            
        elif neg_move_san and not aff_move_san:
            print(f"<debate_v2> : Only positional model provided move: {neg_move_san}")
            self.config['debate_answer'] = neg_move_san
            self.config['success'] = True
            self.config['Reason'] = "Only positional model provided a move"
            self.config['Supported Side'] = "Positional"
            final_source_agent = "negative"
            final_source_agents = ["negative"]
            
            # Convert to UCI
            if board_fen:
                self.final_move = san_to_uci(board_fen, neg_move_san)
            else:
                self.final_move = None
            
            print(f"<debate_v2> : Final move (SAN): {neg_move_san}")
            print(f"<debate_v2> : Final move (UCI): {self.final_move}")
            
            # Return single model result
            debate_history = {
                "round1": {
                    "affirmative_move": None,
                    "negative_move": neg_move_san,
                    "affirmative_response": aff_response,
                    "negative_response": neg_response,
                    "affirmative_tokens": aff_token_info,
                    "negative_tokens": neg_token_info,
                    "moderator_response": "Only positional model responded",
                    "early_consensus": False,
                    "single_model": True
                },
                "final_result": {
                    "final_move_san": neg_move_san,
                    "final_move_uci": self.final_move,
                    "success": True,
                    "reason": "Only positional model provided a move",
                    "supported_side": "Positional",
                    "source_agent": final_source_agent,
                    "source_agents": final_source_agents
                },
                "total_tokens": {
                    "affirmative": aff_token_info,
                    "negative": neg_token_info,
                    "moderator": None,
                    "judge": None,
                    "total_prompt_tokens": (aff_token_info.get("prompt_tokens", 0) if aff_token_info else 0) + (neg_token_info.get("prompt_tokens", 0) if neg_token_info else 0),
                    "total_completion_tokens": (aff_token_info.get("completion_tokens", 0) if aff_token_info else 0) + (neg_token_info.get("completion_tokens", 0) if neg_token_info else 0),
                    "total_tokens": (aff_token_info.get("total_tokens", 0) if aff_token_info else 0) + (neg_token_info.get("total_tokens", 0) if neg_token_info else 0)
                },
                "token_events": list(token_events),
                "judge": judge_details,
                "moderator": {
                    "raw_response": moderator_raw_response
                }
            }
            
            return self.final_move, debate_history
        
        # Check if both models agree on the same move - early consensus
        if aff_move_san and neg_move_san and aff_move_san == neg_move_san:
            print(f"<debate_v2> : Early consensus! Both models agree on move: {aff_move_san}")
            self.config['debate_answer'] = aff_move_san
            self.config['success'] = True
            self.config['Reason'] = "Both debaters agreed on the same move"
            self.config['Supported Side'] = "Both"
            final_source_agent = "affirmative"
            final_source_agents = ["affirmative", "negative"]
            
            # Convert to UCI
            if board_fen:
                self.final_move = san_to_uci(board_fen, aff_move_san)
            else:
                self.final_move = None
            
            print(f"<debate_v2> : Final move (SAN): {aff_move_san}")
            print(f"<debate_v2> : Final move (UCI): {self.final_move}")
            
            # Return early consensus result
            debate_history = {
                "round1": {
                    "affirmative_move": aff_move_san,
                    "negative_move": neg_move_san,
                    "affirmative_response": aff_response,
                    "negative_response": neg_response,
                    "affirmative_tokens": aff_token_info,
                    "negative_tokens": neg_token_info,
                    "moderator_response": "Early consensus - no moderator needed",
                    "early_consensus": True
                },
                "final_result": {
                    "final_move_san": aff_move_san,
                    "final_move_uci": self.final_move,
                    "success": True,
                    "reason": "Early consensus - both models agreed",
                    "supported_side": "Both",
                    "source_agent": final_source_agent,
                    "source_agents": final_source_agents
                },
                "total_tokens": {
                    "affirmative": aff_token_info,
                    "negative": neg_token_info,
                    "moderator": None,
                    "judge": None,
                    "total_prompt_tokens": (aff_token_info.get("prompt_tokens", 0) if aff_token_info else 0) + (neg_token_info.get("prompt_tokens", 0) if neg_token_info else 0),
                    "total_completion_tokens": (aff_token_info.get("completion_tokens", 0) if aff_token_info else 0) + (neg_token_info.get("completion_tokens", 0) if neg_token_info else 0),
                    "total_tokens": (aff_token_info.get("total_tokens", 0) if aff_token_info else 0) + (neg_token_info.get("total_tokens", 0) if neg_token_info else 0)
                },
                "token_events": list(token_events),
                "judge": judge_details,
                "moderator": {
                    "raw_response": moderator_raw_response
                }
            }
            
            return self.final_move, debate_history
        
        # Moderator evaluates first round
        self.moderator.add_event(self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans)
                                .replace('##neg_ans##', self.neg_ans).replace('##round##', 'first'))
        self.mod_ans = self.moderator.ask()
        if not self.mod_ans:
            self.mod_ans = "No response from moderator"
        moderator_raw_response = self.mod_ans
        self.moderator.add_memory(self.mod_ans)
        record_token_event("moderator", "moderator_round1", 1, getattr(self.moderator, "last_token_info", None))
        next_round_index = 2
        
        try:
            self.mod_ans = eval(self.mod_ans)
        except:
            self.mod_ans = {"Whether there is a preference": "No", "debate_answer": ""}
        
        # Continue debate rounds if no consensus
        for round_num in range(self.max_rounds - 1):
            if self.mod_ans.get("debate_answer", "") != "":
                break
                
            print(f"===== Debate Round-{next_round_index} =====")
            round_index = next_round_index
            
            # Affirmative responds to negative using proper extraction
            aff_response, aff_move_san, aff_token_info = self.affirmative.model_interface.get_move_with_extraction(
                self.config['player_meta_prompt'],
                user_prompt,
                current_turn_number=turn_number,
                is_white_to_move=is_white_to_move,
                max_tokens=self.affirmative.model_interface.max_completion_tokens,
                temperature=self.affirmative.temperature,
                retry_attempts=self.affirmative.model_interface.retry_attempts,
            )
            if not aff_response:
                aff_response = "No response from model"
                aff_move_san = None
            self.affirmative.add_memory(aff_response)
            self.aff_ans = aff_response
            record_token_event("affirmative", f"affirmative_round{round_index}", round_index, aff_token_info)
            
            # Negative responds to affirmative using proper extraction
            neg_response, neg_move_san, neg_token_info = self.negative.model_interface.get_move_with_extraction(
                self.config['player_meta_prompt'],
                user_prompt,
                current_turn_number=turn_number,
                is_white_to_move=is_white_to_move,
                max_tokens=self.negative.model_interface.max_completion_tokens,
                temperature=self.negative.temperature,
                retry_attempts=self.negative.model_interface.retry_attempts,
            )
            if not neg_response:
                neg_response = "No response from model"
                neg_move_san = None
            self.negative.add_memory(neg_response)
            self.neg_ans = neg_response
            record_token_event("negative", f"negative_round{round_index}", round_index, neg_token_info)
            
            # Moderator evaluates
            self.moderator.add_event(self.config['moderator_prompt']
                                   .replace('##aff_ans##', self.aff_ans)
                                   .replace('##neg_ans##', self.neg_ans)
                                   .replace('##round##', self.round_dct(round_num + 2)))
            self.mod_ans = self.moderator.ask()
            if not self.mod_ans:
                self.mod_ans = "No response from moderator"
            moderator_raw_response = self.mod_ans
            self.moderator.add_memory(self.mod_ans)
            record_token_event("moderator", f"moderator_round{round_index}", round_index, getattr(self.moderator, "last_token_info", None))
            
            try:
                self.mod_ans = eval(self.mod_ans)
            except:
                self.mod_ans = {"Whether there is a preference": "No", "debate_answer": ""}
            next_round_index += 1
        
        # Check if moderator reached consensus
        if self.mod_ans.get("debate_answer", "") != "":
            self.config.update(self.mod_ans)
            self.config['success'] = True
            # Extract move from moderator's final answer
            final_move_san = extract_predicted_move(
                self.mod_ans.get("debate_answer", ""),
                current_turn_number=turn_number,
                is_white_to_move=is_white_to_move
            )
            final_source_agent = "moderator"
            final_source_agents = ["moderator"]
        else:
            # Use judge as final arbiter
            print(f"===== Judge Round =====")
            judge_player = ChessDebatePlayer(
                model_name=self.model_name,
                name='Judge',
                temperature=self.temperature,
                openai_api_key=self.openai_api_key,
                sleep_time=self.sleep_time
            )
            
            # Get final answers from both sides
            aff_final = self.affirmative.memory_lst[-1]['content'] if self.affirmative.memory_lst else self.aff_ans
            neg_final = self.negative.memory_lst[-1]['content'] if self.negative.memory_lst else self.neg_ans
            
            judge_player.set_meta_prompt(self.config['moderator_meta_prompt'])
            
            # Extract answer candidates - use simple prompt
            simple_judge_prompt = f"Affirmative side: {aff_final}\n\nNegative side: {neg_final}\n\nWhat are the move candidates? List them simply."
            judge_player.add_event(simple_judge_prompt)
            judge_response1 = judge_player.ask()
            if not judge_response1:
                judge_response1 = "No response from judge"
            judge_player.add_memory(judge_response1)
            judge_round_index = next_round_index
            record_token_event("judge", "judge_candidates", judge_round_index, getattr(judge_player, "last_token_info", None))
            
            # Final decision - use simple prompt
            final_judge_prompt = f"Based on the debate, what is the best move? Give just the move in standard algebraic notation."
            judge_player.add_event(final_judge_prompt)
            judge_response2 = judge_player.ask()
            if not judge_response2:
                judge_response2 = "No response from judge"
            judge_player.add_memory(judge_response2)
            record_token_event("judge", "judge_final", judge_round_index, getattr(judge_player, "last_token_info", None))
            judge_details = {
                "candidates": judge_response1,
                "final_response": judge_response2
            }
            
            # Extract move from judge response (should be simple move now)
            final_move_san = extract_predicted_move(
                judge_response2,
                current_turn_number=turn_number,
                is_white_to_move=is_white_to_move
            )
            if final_move_san:
                self.config['debate_answer'] = final_move_san
                self.config['success'] = True
                self.config['Reason'] = "Judge selected move"
                final_source_agent = "judge"
                final_source_agents = ["judge"]
            else:
                # Fallback: use the most recent affirmative move
                final_move_san = aff_move_san
                self.config['debate_answer'] = aff_move_san
                self.config['Reason'] = "Judge evaluation failed, using affirmative move"
                final_source_agent = "affirmative"
                final_source_agents = ["affirmative"]
        
        # Convert to UCI
        if final_move_san and board_fen:
            self.final_move = san_to_uci(board_fen, final_move_san)
        else:
            self.final_move = None
        
        print(f"<debate_v2> : Final move (SAN): {final_move_san}")
        print(f"<debate_v2> : Final move (UCI): {self.final_move}")
        
        # Collect debate history
        debate_history = {
            "round1": {
                "affirmative_move": aff_move_san,
                "negative_move": neg_move_san,
                "affirmative_response": self.aff_ans,
                "negative_response": self.neg_ans,
                "moderator_response": self.mod_ans
            },
            "final_result": {
                "final_move_san": final_move_san,
                "final_move_uci": self.final_move,
                "success": self.config.get('success', False),
                "reason": self.config.get('Reason', ''),
                "supported_side": self.config.get('Supported Side', ''),
                "source_agent": final_source_agent,
                "source_agents": final_source_agents
            },
            "total_tokens": {
                "affirmative": self.aff_token_info,
                "negative": self.neg_token_info,
                "moderator": None,
                "judge": None,
                "total_prompt_tokens": (self.aff_token_info.get("prompt_tokens", 0) if self.aff_token_info else 0) + (self.neg_token_info.get("prompt_tokens", 0) if self.neg_token_info else 0),
                "total_completion_tokens": (self.aff_token_info.get("completion_tokens", 0) if self.aff_token_info else 0) + (self.neg_token_info.get("completion_tokens", 0) if self.neg_token_info else 0),
                "total_tokens": (self.aff_token_info.get("total_tokens", 0) if self.aff_token_info else 0) + (self.neg_token_info.get("total_tokens", 0) if self.neg_token_info else 0)
            },
            "token_events": list(token_events),
            "judge": judge_details,
            "moderator": {
                "raw_response": moderator_raw_response
            }
        }
        
        return self.final_move, debate_history


def save_debate_history_v2(debate_history, puzzle_idx, output_dir="debate_history_v2"):
    """Save debate history to a JSON file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"puzzle_{puzzle_idx}_debate_v2.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(debate_history, f, indent=2)
    
    print(f"Debate history V2 saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        sys.exit(1)
    
    # Test the debate system
    debate = ChessDebateV2(
        model_name='gpt-3.5-turbo-instruct',
        temperature=0.1,
        openai_api_key=api_key,
        max_rounds=3,
        sleep_time=0.1,
        plan_plies=0
    )
    
    # Example chess position
    test_pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5"
    test_board_fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    
    final_move, history = debate.run_debate(
        user_prompt=test_pgn,
        board_fen=test_board_fen
    )
    
    print(f"\nFinal result: {final_move}")
    print(f"Debate history: {json.dumps(history, indent=2)}")
