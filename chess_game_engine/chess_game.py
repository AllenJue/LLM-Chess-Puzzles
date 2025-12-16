#!/usr/bin/env python3
"""
Chess Game Engine - Play against Stockfish using LLM models
Supports single model and self-consistency approaches
"""

import chess
import chess.engine
import chess.pgn
import sys
import os
import argparse
import json
import random
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), '.env'))  # Load from chess_puzzles directory
except ImportError:
    pass  # dotenv not available, continue without it

# Add parent directory to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
MAD_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'MAD'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
if MAD_DIR not in sys.path:
    sys.path.append(MAD_DIR)

from model_interface import ChessModelInterface
from main import ChessSelfConsistency
from chess_debate import ChessDebateV2
from chess_utils import extract_predicted_move, san_to_uci, extract_plan_sans


class ChessGameEngine:
    def __init__(self, model_name: str = 'gpt-3.5-turbo-instruct', 
                 stockfish_path: str = 'stockfish', 
                 stockfish_time: float = 1.0,
                 stockfish_skill: int = 5,
                 use_self_consistency: bool = False,
                 use_debate: bool = False,
                 use_random_opponent: bool = False,
                 plan_plies: int = 0,
                 stockfish_depth: Optional[int] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize the chess game engine
        
        Args:
            model_name: OpenAI model to use
            stockfish_path: Path to Stockfish executable
            stockfish_time: Time limit for Stockfish moves (seconds)
            stockfish_skill: Stockfish skill level (0-20, where 20 is maximum strength)
            use_self_consistency: Whether to use self-consistency approach
            use_random_opponent: Whether to use random legal moves instead of Stockfish
        """
        self.model_name = model_name
        self.stockfish_path = stockfish_path
        self.stockfish_time = stockfish_time
        self.stockfish_skill = stockfish_skill
        self.use_self_consistency = use_self_consistency
        self.use_debate = use_debate
        self.use_random_opponent = use_random_opponent
        self.plan_plies = max(0, plan_plies)
        self.stockfish_depth = stockfish_depth
        self.single_model_token_log: list[dict] = []
        self.self_consistency_token_log: list[dict] = []
        self.single_model_plan_log: list[dict] = []
        self.self_consistency_plan_log: list[dict] = []
        self.active_plan: Optional[dict] = None

        # Get API key if not provided
        if api_key is None:
            api_key = os.getenv('ANANNAS_API_KEY') or os.getenv('OPENAI_API_KEY')
        if base_url is None:
            base_url = os.getenv('ANANNAS_API_URL') or os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_BASE')
        
        # Core model interface for single-model play
        self.model_interface = ChessModelInterface(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            max_completion_tokens=640,
            default_temperature=0.1,
            retry_attempts=2,
        )
        
        # Initialize strategy-specific helpers
        if use_self_consistency:
            self.self_consistency = ChessSelfConsistency(
                model_name=model_name,
                temperature=0.1,
                openai_api_key=api_key,
                base_url=base_url,
                max_rounds=2,
                sleep_time=0.1,
                plan_plies=self.plan_plies
            )
        elif use_debate:
            self.debate_v2 = ChessDebateV2(
                model_name=model_name,
                temperature=0.1,
                openai_api_key=api_key,
                base_url=base_url,
                max_rounds=3,
                sleep_time=0.1,
                plan_plies=self.plan_plies
            )

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def _log_plan_event(self, source: str, event: str, **data) -> None:
        entry = {
            "event": event,
            "source": source,
            **data
        }
        print(f"<planning> : {entry}")
        if source == "single_model":
            self.single_model_plan_log.append(entry)
        elif source == "self_consistency":
            self.self_consistency_plan_log.append(entry)

    def _discard_active_plan(self, reason: str) -> None:
        if self.active_plan:
            self._log_plan_event(
                self.active_plan.get("source", "unknown"),
                "plan_discarded",
                reason=reason,
                remaining_plan=self.active_plan.get("remaining_san", [])
            )
        self.active_plan = None

    def _set_active_plan(self, source: str, plan_full: List[str], plan_execute: List[str], *, turn: int, attempt: int, selected_san: Optional[str]) -> None:
        if self.active_plan and self.active_plan.get("source") == source:
            self._log_plan_event(source, "plan_replaced", turn=turn, attempt=attempt)
            self.active_plan = None

        self._log_plan_event(
            source,
            "plan_generated",
            turn=turn,
            attempt=attempt,
            plan_full=plan_full,
            plan_execute=plan_execute,
            selected_move=selected_san
        )

        if not plan_execute:
            return

        self.active_plan = {
            "source": source,
            "remaining_san": list(plan_execute),
            "plan_full": list(plan_full) if plan_full else list(plan_execute),
            "turn": turn,
            "attempt": attempt,
            "selected_move": selected_san
        }

    def _attempt_planned_move(self, board: chess.Board) -> Optional[str]:
        if not self.active_plan:
            return None
        if not self.active_plan.get("remaining_san"):
            self._discard_active_plan("no_remaining_moves")
            return None

        next_san = self.active_plan["remaining_san"][0]
        try:
            move = board.parse_san(next_san)
        except ValueError:
            # Not a legal move for the current side (likely waiting for opponent)
            return None

        if move not in board.legal_moves:
            self._discard_active_plan(f"planned move {next_san} not legal")
            return None

        self.active_plan["remaining_san"].pop(0)
        move_uci = move.uci()
        self._log_plan_event(
            self.active_plan.get("source", "unknown"),
            "plan_move_used",
            planned_san=next_san,
            move_uci=move_uci
        )

        if not self.active_plan["remaining_san"]:
            self._log_plan_event(
                self.active_plan.get("source", "unknown"),
                "plan_completed",
                reason="plan fully executed"
            )
            self.active_plan = None

        return move_uci

    def _handle_opponent_plan(self, move_san: str) -> None:
        if not self.active_plan:
            return
        if not self.active_plan.get("remaining_san"):
            self._discard_active_plan("no_remaining_moves_for_opponent")
            return

        expected_san = self.active_plan["remaining_san"][0]
        source = self.active_plan.get("source", "unknown")
        if move_san == expected_san:
            self.active_plan["remaining_san"].pop(0)
            self._log_plan_event(source, "plan_opponent_matched", move_san=move_san)
            if not self.active_plan["remaining_san"]:
                self._log_plan_event(source, "plan_completed", reason="opponent finished plan")
                self.active_plan = None
        else:
            self._log_plan_event(source, "plan_aborted", expected_san=expected_san, actual_san=move_san)
            self.active_plan = None
    
    def get_random_legal_move(self, board: chess.Board) -> str:
        """
        Get a random legal move as fallback
        
        Args:
            board: Current chess board position
            
        Returns:
            UCI move string
        """
        legal_moves = list(board.legal_moves)
        if legal_moves:
            random_move = random.choice(legal_moves)
            return random_move.uci()
        else:
            # This shouldn't happen in normal play, but just in case
            return None
    
    def _get_random_opponent_move(self, board: chess.Board) -> str:
        """
        Get a random legal move for the random opponent
        
        Args:
            board: Current chess board position
            
        Returns:
            UCI move string
        """
        legal_moves = list(board.legal_moves)
        if legal_moves:
            random_move = random.choice(legal_moves)
            move_san = board.san(random_move)
            print(f"🎲 Random opponent plays: {move_san} ({random_move.uci()})")
            return random_move.uci()
        else:
            # This shouldn't happen in normal play, but just in case
            return None
        
    def _use_fallback_move(self, board: chess.Board, reason: str, model_response: str = None) -> str:
        """
        Use a random legal move as fallback and record it
        
        Args:
            board: Current chess board position
            reason: Reason for using fallback
            model_response: Original model response (if any)
            
        Returns:
            UCI move string
        """
        print(f"🎲 Using random legal move as fallback...")
        self._discard_active_plan("fallback_move_used")
        fallback_move = self.get_random_legal_move(board)
        
        if fallback_move:
            self.fallback_moves.append({
                "move_number": len(board.move_stack) + 1,
                "reason": reason,
                "model_response": model_response,
                "fallback_move": fallback_move
            })
        
        return fallback_move

    def get_model_move(self, board: chess.Board) -> Optional[str]:
        """
        Get a move from the LLM model
        
        Args:
            board: Current chess board position
            
        Returns:
            UCI move string or None if failed
        """
        planned_move = self._attempt_planned_move(board)
        if planned_move:
            move_obj = chess.Move.from_uci(planned_move)
            move_san = board.san(move_obj)
            print(f"🤖 Using planned move without prompting: {move_san} ({planned_move})")
            return planned_move
        
        # Use the same infrastructure as chess_puzzles
        exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
        current_game = chess.pgn.Game.from_board(board)
        user_prompt = current_game.accept(exporter)
        
        num_future_moves = self.plan_plies + 1 if self.plan_plies > 0 else 3
        system_prompt = (
            "You are a chess grandmaster.\n"
            "You will be given a partially completed game.\n"
            f"After seeing it, you should repeat the ENTIRE GAME and then give the next {num_future_moves} moves.\n"
            "After repeating the game, immediately continue by listing those moves in order on a single line, separated by spaces, starting with the side to move now.\n"
            "Use standard algebraic notation, e.g. 'e4' or 'Rdf8'.\n"
            "ALWAYS repeat the entire representation of the game so far.\n"
            "NEVER explain your choice."
        )
        
        try:
            if self.use_self_consistency:
                # Use self-consistency approach with retry logic
                print(f"🤖 Self-consistency model thinking...")
                print(f"<debug> : Self-consistency system prompt building:")
                print(f"<debug> : user_prompt: {repr(user_prompt)}")
                
                # Try up to 3 times for self-consistency
                for attempt in range(3):
                    if attempt > 0:
                        print(f"🔄 Self-consistency retry attempt {attempt + 1}/3...")
                    
                    predicted_uci, debate_history = self.self_consistency.run_debate(
                        user_prompt, 
                        played_plies=len(board.move_stack),
                        board_fen=board.fen(),
                        board=board
                    )
                    print(f"<debug> : Self-consistency predicted UCI (attempt {attempt + 1}): {repr(predicted_uci)}")
                    print(f"<debug> : Self-consistency debate history (attempt {attempt + 1}): {repr(debate_history)}")
                    
                    if not predicted_uci:
                        print(f"❌ Self-consistency attempt {attempt + 1}: Failed to generate move")
                        if attempt == 2:  # Last attempt
                            return self._use_fallback_move(board, "Self-consistency failed to generate move after 3 attempts", str(debate_history))
                        continue
                    
                    # Convert and validate move
                    try:
                        move = chess.Move.from_uci(predicted_uci)
                        if move not in board.legal_moves:
                            print(f"❌ Self-consistency attempt {attempt + 1}: Generated illegal move: {predicted_uci}")
                            if attempt == 2:  # Last attempt
                                return self._use_fallback_move(board, f"Self-consistency generated illegal move after 3 attempts: {predicted_uci}", str(debate_history))
                            continue
                        
                        plan_info = debate_history.get("final_plan", {}) if isinstance(debate_history, dict) else {}
                        plan_execute = plan_info.get("moves_for_execution") or []
                        if self.plan_plies > 0:
                            plan_execute = plan_execute[:self.plan_plies]
                        plan_full = plan_info.get("full_moves") or list(plan_execute)
                        if self.plan_plies > 0:
                            print(f"<self-consistency> : plan (full): {plan_full}")
                            print(f"<self-consistency> : plan (first {self.plan_plies} plies): {plan_execute}")
                        token_totals = (debate_history or {}).get("total_tokens", {}) if isinstance(debate_history, dict) else {}
                        if token_totals:
                            prompt_tokens = token_totals.get("total_prompt_tokens", 0)
                            completion_tokens = token_totals.get("total_completion_tokens", 0)
                            total_tokens = token_totals.get("total_tokens", prompt_tokens + completion_tokens)
                            print(f"<self-consistency> : token usage -> prompt {prompt_tokens}, completion {completion_tokens}, total {total_tokens}")
                        else:
                            prompt_tokens = completion_tokens = total_tokens = 0
                        self.self_consistency_token_log.append({
                            "turn": len(board.move_stack) + 1,
                            "attempt": attempt + 1,
                            "token_events": (debate_history or {}).get("token_events", []) if isinstance(debate_history, dict) else [],
                            "total_prompt_tokens": prompt_tokens,
                            "total_completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        })
                        move_san = board.san(move)
                        self._set_active_plan(
                            "self_consistency",
                            plan_full,
                            plan_execute,
                            turn=len(board.move_stack) + 1,
                            attempt=attempt + 1,
                            selected_san=move_san
                        )
                        
                        print(f"✅ Self-consistency attempt {attempt + 1}: Valid move generated: {predicted_uci}")
                        return predicted_uci
                        
                    except Exception as e:
                        print(f"❌ Self-consistency attempt {attempt + 1}: Error validating move: {e}")
                        if attempt == 2:  # Last attempt
                            return self._use_fallback_move(board, f"Self-consistency move validation error after 3 attempts: {e}", str(debate_history))
                        continue
                
            elif self.use_debate:
                print(f"🧠 Debate (multi-agent) thinking...")
                for attempt in range(3):
                    if attempt > 0:
                        print(f"🔄 Debate retry attempt {attempt + 1}/3...")

                    predicted_uci, debate_history = self.debate_v2.run_debate(
                        user_prompt,
                        played_plies=len(board.move_stack),
                        board_fen=board.fen()
                    )
                    print(f"<debug> : Debate predicted UCI (attempt {attempt + 1}): {repr(predicted_uci)}")

                    if not predicted_uci:
                        print(f"❌ Debate attempt {attempt + 1}: Failed to generate move")
                        if attempt == 2:
                            return self._use_fallback_move(board, "Debate failed to generate move after 3 attempts", str(debate_history))
                        continue

                    try:
                        move = chess.Move.from_uci(predicted_uci)
                        if move not in board.legal_moves:
                            print(f"❌ Debate attempt {attempt + 1}: Generated illegal move: {predicted_uci}")
                            if attempt == 2:
                                return self._use_fallback_move(board, f"Debate generated illegal move after 3 attempts: {predicted_uci}", str(debate_history))
                            continue

                        print(f"✅ Debate attempt {attempt + 1}: Valid move generated: {predicted_uci}")
                        return predicted_uci

                    except Exception as e:
                        print(f"❌ Debate attempt {attempt + 1}: Error validating move: {e}")
                        if attempt == 2:
                            return self._use_fallback_move(board, f"Debate move validation error after 3 attempts: {e}", str(debate_history))
                        continue
                
            else:
                # Use single model with retry logic
                print(f"🤖 Single model thinking...")
                print(f"<debug> : Single model system prompt building:")
                print(f"<debug> : system_prompt: {repr(system_prompt)}")
                print(f"<debug> : user_prompt: {repr(user_prompt)}")
                
                # Try up to 3 times for single model
                for attempt in range(3):
                    if attempt > 0:
                        print(f"🔄 Retry attempt {attempt + 1}/3...")
                    
                    raw_response, predicted_san, token_info = self.model_interface.get_move_with_extraction(
                        system_prompt,
                        user_prompt,
                        current_turn_number=len(board.move_stack) // 2 + 1,
                        is_white_to_move=board.turn,
                        max_tokens=self.model_interface.max_completion_tokens,
                        temperature=self.model_interface.default_temperature,
                        retry_attempts=self.model_interface.retry_attempts,
                    )
                    print(f"<debug> : Model response (attempt {attempt + 1}): {repr(raw_response)}")
                    print(f"<debug> : Extracted move (attempt {attempt + 1}): {repr(predicted_san)}")
                    if token_info:
                        print(f"<single-model> : token usage -> prompt {token_info.get('prompt_tokens', 0)}, completion {token_info.get('completion_tokens', 0)}, total {token_info.get('total_tokens', 0)}")
                        self.single_model_token_log.append({
                            "turn": len(board.move_stack) + 1,
                            "attempt": attempt + 1,
                            "prompt_tokens": token_info.get("prompt_tokens", 0),
                            "completion_tokens": token_info.get("completion_tokens", 0),
                            "total_tokens": token_info.get("total_tokens", 0)
                        })

                    plan_full = extract_plan_sans(raw_response, primary_san=predicted_san)
                    plan_execute = plan_full[:self.plan_plies] if self.plan_plies > 0 else []
                    if self.plan_plies > 0:
                        print(f"<single-model> : plan (full): {plan_full}")
                        print(f"<single-model> : plan (first {self.plan_plies} plies): {plan_execute}")
                    
                    if not predicted_san:
                        print(f"❌ Attempt {attempt + 1}: Failed to extract move from response")
                        if attempt == 2:  # Last attempt
                            return self._use_fallback_move(board, "Failed to extract move from model response after 3 attempts", raw_response)
                        continue
                    
                    # Convert and validate move
                    try:
                        predicted_uci = san_to_uci(board.fen(), predicted_san)
                        move = chess.Move.from_uci(predicted_uci)
                        if move not in board.legal_moves:
                            print(f"❌ Attempt {attempt + 1}: Generated illegal move: {predicted_san} -> {predicted_uci}")
                            if attempt == 2:  # Last attempt
                                return self._use_fallback_move(board, f"Model generated illegal move after 3 attempts: {predicted_san}", raw_response)
                            continue
                        
                        print(f"✅ Attempt {attempt + 1}: Valid move generated: {predicted_san} -> {predicted_uci}")
                        self._set_active_plan(
                            "single_model",
                            plan_full,
                            plan_execute,
                            turn=len(board.move_stack) + 1,
                            attempt=attempt + 1,
                            selected_san=predicted_san
                        )
                        return predicted_uci
                        
                    except Exception as e:
                        print(f"❌ Attempt {attempt + 1}: Error converting move {predicted_san}: {e}")
                        if attempt == 2:  # Last attempt
                            return self._use_fallback_move(board, f"Error converting move after 3 attempts: {predicted_san}", raw_response)
                        continue
                
                # This shouldn't be reached, but just in case
                return self._use_fallback_move(board, "Unexpected error in retry logic", None)
                    
        except Exception as e:
            return self._use_fallback_move(board, f"Exception in model move generation: {e}", None)
    
    def get_stockfish_move(self, board: chess.Board) -> Optional[str]:
        """
        Get a move from Stockfish or random opponent
        
        Args:
            board: Current chess board position
            
        Returns:
            UCI move string or None if failed
        """
        if self.use_random_opponent:
            # Use random legal move instead of Stockfish
            return self._get_random_opponent_move(board)
        
        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                # Configure Stockfish skill level
                engine.configure({"Skill Level": self.stockfish_skill})
                
                limit_kwargs = {}
                if self.stockfish_depth is not None:
                    limit = chess.engine.Limit(depth=self.stockfish_depth)
                else:
                    limit = chess.engine.Limit(time=self.stockfish_time)
                result = engine.play(board, limit)
                return result.move.uci()
        except Exception as e:
            print(f"❌ Error getting Stockfish move: {e}")
            return None
    
    def play_game(self, model_plays_white: bool = True) -> Dict[str, Any]:
        """
        Play a full chess game
        
        Args:
            model_plays_white: Whether the model plays as white
            
        Returns:
            Game result dictionary
        """
        print(f"🎮 Starting chess game!")
        print(f"🤖 Model plays: {'White' if model_plays_white else 'Black'}")
        print(f"🐟 Stockfish plays: {'Black' if model_plays_white else 'White'} (Skill Level: {self.stockfish_skill}/20)")
        if self.use_self_consistency:
            approach_label = 'Self-consistency'
        elif self.use_debate:
            approach_label = 'Debate (multi-agent)'
        else:
            approach_label = 'Single model'
        print(f"🧠 Model approach: {approach_label}")
        print("=" * 60)
        
        self.board = chess.Board()
        self.game_moves = []
        self.game_history = []
        self.fallback_moves = []  # Reset fallback moves for new game
        self.single_model_token_log = []
        self.self_consistency_token_log = []
        self.single_model_plan_log = []
        self.self_consistency_plan_log = []
        self.active_plan = None
        
        move_number = 1
        
        while not self.board.is_game_over():
            print(f"\n--- Move {move_number} ---")
            print(f"Board:\n{self.board}")
            print(f"FEN: {self.board.fen()}")
            print(f"Turn: {'White' if self.board.turn else 'Black'}")
            
            # Determine whose turn it is
            is_model_turn = (self.board.turn == chess.WHITE) == model_plays_white
            
            if is_model_turn:
                # Model's turn
                move_uci = self.get_model_move(self.board)
                player_name = "Model"
            else:
                # Stockfish's turn
                move_uci = self.get_stockfish_move(self.board)
                player_name = "Stockfish"
            
            if move_uci is None:
                print(f"❌ {player_name} failed to make a move!")
                break
            
            # Validate and make the move
            try:
                move = chess.Move.from_uci(move_uci)
                if move in self.board.legal_moves:
                    # Get SAN notation before pushing the move
                    move_san = self.board.san(move)
                    if not is_model_turn:
                        self._handle_opponent_plan(move_san)
                    
                    self.board.push(move)
                    self.game_moves.append(move_uci)
                    
                    # Check if this was a fallback move
                    is_fallback = any(fb["fallback_move"] == move_uci and fb["move_number"] == move_number 
                                    for fb in self.fallback_moves)
                    
                    # Record move history
                    move_info = {
                        "move_number": move_number,
                        "player": player_name,
                        "move_uci": move_uci,
                        "move_san": move_san,
                        "fen_after": self.board.fen(),
                        "is_model_turn": is_model_turn,
                        "is_fallback_move": is_fallback
                    }
                    self.game_history.append(move_info)
                    
                    if is_fallback:
                        print(f"🎲 {player_name} plays (FALLBACK): {move_san} ({move_uci})")
                    else:
                        print(f"✅ {player_name} plays: {move_san} ({move_uci})")
                    
                    # Check for game end conditions
                    if self.board.is_checkmate():
                        winner = "Model" if is_model_turn else "Stockfish"
                        print(f"🏆 {winner} wins by checkmate!")
                        break
                    elif self.board.is_stalemate():
                        print(f"🤝 Game ends in stalemate!")
                        break
                    elif self.board.is_insufficient_material():
                        print(f"🤝 Game ends due to insufficient material!")
                        break
                    elif self.board.is_seventyfive_moves():
                        print(f"🤝 Game ends due to 75-move rule!")
                        break
                    elif self.board.is_fivefold_repetition():
                        print(f"🤝 Game ends due to fivefold repetition!")
                        break
                    
                    move_number += 1
                else:
                    print(f"❌ {player_name} made illegal move: {move_uci}")
                    break
                    
            except Exception as e:
                print(f"❌ Error making move {move_uci}: {e}")
                break
        
        # Determine final result
        if self.board.is_checkmate():
            winner = "Model" if is_model_turn else "Stockfish"
            result = f"{winner} wins by checkmate"
        elif self.board.is_stalemate():
            result = "Draw by stalemate"
        elif self.board.is_insufficient_material():
            result = "Draw by insufficient material"
        elif self.board.is_seventyfive_moves():
            result = "Draw by 75-move rule"
        elif self.board.is_fivefold_repetition():
            result = "Draw by fivefold repetition"
        else:
            result = "Game ended unexpectedly"
        
        print(f"\n🎯 Final Result: {result}")
        print(f"📊 Total moves: {len(self.game_moves)}")
        
        # Report fallback moves if any were used
        if self.fallback_moves:
            print(f"🎲 Fallback moves used: {len(self.fallback_moves)}")
            for fallback in self.fallback_moves:
                print(f"   Move {fallback['move_number']}: {fallback['reason']} -> {fallback['fallback_move']}")
        else:
            print(f"✅ No fallback moves needed - all model moves were valid")
        
        single_model_total_prompt = sum(entry["prompt_tokens"] for entry in self.single_model_token_log)
        single_model_total_completion = sum(entry["completion_tokens"] for entry in self.single_model_token_log)
        single_model_total_tokens = sum(entry["total_tokens"] for entry in self.single_model_token_log)
        if single_model_total_tokens:
            print(f"📊 Single-model token usage -> prompt {single_model_total_prompt}, completion {single_model_total_completion}, total {single_model_total_tokens}")

        sc_total_prompt = sum(entry["total_prompt_tokens"] for entry in self.self_consistency_token_log)
        sc_total_completion = sum(entry["total_completion_tokens"] for entry in self.self_consistency_token_log)
        sc_total_tokens = sum(entry["total_tokens"] for entry in self.self_consistency_token_log)
        if sc_total_tokens:
            print(f"📊 Self-consistency token usage -> prompt {sc_total_prompt}, completion {sc_total_completion}, total {sc_total_tokens}")
        
        return {
            "result": result,
            "total_moves": len(self.game_moves),
            "model_plays_white": model_plays_white,
            "model_approach": "self_consistency" if self.use_self_consistency else ("debate" if self.use_debate else "single_model"),
            "model_name": self.model_name,
            "opponent_type": "random" if self.use_random_opponent else "stockfish",
            "moves": self.game_moves,
            "game_history": self.game_history,
            "fallback_moves": self.fallback_moves,
            "fallback_count": len(self.fallback_moves),
            "final_fen": self.board.fen(),
            "timestamp": datetime.now().isoformat(),
            "single_model_token_log": self.single_model_token_log,
            "self_consistency_token_log": self.self_consistency_token_log,
            "single_model_total_prompt_tokens": single_model_total_prompt,
            "single_model_total_completion_tokens": single_model_total_completion,
            "single_model_total_tokens": single_model_total_tokens,
            "self_consistency_total_prompt_tokens": sc_total_prompt,
            "self_consistency_total_completion_tokens": sc_total_completion,
            "self_consistency_total_tokens": sc_total_tokens,
            "single_model_plan_log": self.single_model_plan_log,
            "self_consistency_plan_log": self.self_consistency_plan_log
        }
    
    def save_game(self, game_result: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save the game result to a JSON file with comprehensive metadata
        
        Args:
            game_result: Game result dictionary
            filename: Optional filename, will generate if not provided
            
        Returns:
            Path to saved file
        """
        # Create data/games directory if it doesn't exist
        games_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'games')
        os.makedirs(games_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            approach = "self_consistency" if self.use_self_consistency else ("debate" if self.use_debate else "single_model")
            color = "white" if game_result["model_plays_white"] else "black"
            filename = f"chess_game_{approach}_{color}_{timestamp}.json"
        
        filepath = os.path.join(games_dir, filename)
        
        # Add comprehensive metadata
        enhanced_result = game_result.copy()
        enhanced_result.update({
            "metadata": {
                "model_name": self.model_name,
                "model_approach": "self_consistency" if self.use_self_consistency else ("debate" if self.use_debate else "single_model"),
                "stockfish_time": self.stockfish_time,
                "stockfish_skill": self.stockfish_skill,
                "total_fallback_moves": len(self.fallback_moves),
                "fallback_details": self.fallback_moves,
                "single_model_total_prompt_tokens": game_result.get("single_model_total_prompt_tokens", 0),
                "single_model_total_completion_tokens": game_result.get("single_model_total_completion_tokens", 0),
                "single_model_total_tokens": game_result.get("single_model_total_tokens", 0),
                "self_consistency_total_prompt_tokens": game_result.get("self_consistency_total_prompt_tokens", 0),
                "self_consistency_total_completion_tokens": game_result.get("self_consistency_total_completion_tokens", 0),
                "self_consistency_total_tokens": game_result.get("self_consistency_total_tokens", 0),
                "game_engine_version": "1.0",
                "saved_at": datetime.now().isoformat()
            }
        })
        
        with open(filepath, 'w') as f:
            json.dump(enhanced_result, f, indent=2)
        
        print(f"💾 Game saved to: {filepath}")
        return filepath
    
    def save_pgn(self, game_result: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save the game as a PGN file
        
        Args:
            game_result: Game result dictionary
            filename: Optional filename, will generate if not provided
            
        Returns:
            Path to saved PGN file
        """
        # Create data/games directory if it doesn't exist
        games_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'games')
        os.makedirs(games_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            approach = "self_consistency" if self.use_self_consistency else ("debate" if self.use_debate else "single_model")
            color = "white" if game_result["model_plays_white"] else "black"
            filename = f"chess_game_{approach}_{color}_{timestamp}.pgn"
        
        filepath = os.path.join(games_dir, filename)
        
        # Create PGN game
        game = chess.pgn.Game()
        opponent_name = "Random" if self.use_random_opponent else "Stockfish"
        game.headers["Event"] = f"LLM vs {opponent_name}"
        game.headers["Site"] = "Chess Game Engine"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = "Model" if game_result["model_plays_white"] else opponent_name
        game.headers["Black"] = opponent_name if game_result["model_plays_white"] else "Model"
        game.headers["Result"] = self._pgn_result(game_result["result"])
        game.headers["Model"] = self.model_name
        game.headers["Approach"] = "self_consistency" if self.use_self_consistency else ("debate" if self.use_debate else "single_model")
        game.headers["Opponent"] = opponent_name
        game.headers["FallbackMoves"] = str(len(self.fallback_moves))
        game.headers["TotalMoves"] = str(game_result["total_moves"])
        if not self.use_random_opponent:
            game.headers["StockfishSkill"] = str(self.stockfish_skill)
        
        # Add moves
        node = game
        temp_board = chess.Board()
        
        for move_uci in game_result["moves"]:
            move = chess.Move.from_uci(move_uci)
            node = node.add_variation(move)
            temp_board.push(move)
        
        # Save PGN
        with open(filepath, 'w') as f:
            print(game, file=f)
        
        print(f"📄 PGN saved to: {filepath}")
        return filepath
    
    def _pgn_result(self, result: str) -> str:
        """Convert result string to PGN result format"""
        if "Model wins" in result:
            return "1-0"
        elif "Stockfish wins" in result:
            return "0-1"
        else:
            return "1/2-1/2"


def main():
    parser = argparse.ArgumentParser(description="Play chess against Stockfish using LLM models")
    parser.add_argument("--model", default="gpt-3.5-turbo-instruct", 
                       help="OpenAI model to use")
    parser.add_argument("--stockfish", default="stockfish", 
                       help="Path to Stockfish executable")
    parser.add_argument("--time", type=float, default=1.0, 
                       help="Time limit for Stockfish moves (seconds)")
    parser.add_argument("--skill", type=int, default=5, 
                       help="Stockfish skill level (0-20, where 20 is maximum strength)")
    parser.add_argument("--stockfish-depth", type=int, default=None,
                       help="Optional fixed search depth for Stockfish (overrides time limit if set)")
    parser.add_argument("--self-consistency", action="store_true", 
                       help="Use self-consistency approach instead of single model")
    parser.add_argument("--debate", action="store_true",
                       help="Use multi-agent debate approach instead of single model")
    parser.add_argument("--random-opponent", action="store_true",
                       help="Use random legal moves instead of Stockfish")
    parser.add_argument("--model-color", choices=["white", "black"], default="white",
                       help="Color for the model to play")
    parser.add_argument("--plan-plies", type=int, default=0,
                       help="Number of future plies to plan (debate/self-consistency)")
    parser.add_argument("--save-json", action="store_true",
                       help="Save game result as JSON")
    parser.add_argument("--save-pgn", action="store_true", 
                       help="Save game as PGN file")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save any files")
    parser.add_argument("--use-anannas", action="store_true",
                       help="Use Anannas API (requires ANANNAS_API_KEY in .env file)")
    parser.add_argument("--anannas-base-url", type=str, default=None,
                       help="Anannas API base URL (defaults to https://api.anannas.ai/v1, or ANANNAS_API_URL from .env)")
    
    args = parser.parse_args()
    
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
        
        # Warn if using OpenAI model name with Anannas
        openai_models = ["gpt-3.5-turbo-instruct", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
        if args.model in openai_models:
            print(f"\n⚠️  WARNING: You're using Anannas API with OpenAI model name '{args.model}'")
            print("   Anannas uses different model names. Examples:")
            print("   - qwen/qwen3-4b:free")
            print("   - google/gemma-3-4b-it:free")
            print("   - microsoft/phi-3-mini-128k-instruct:free")
            print("   - meta-llama/llama-3.3-8b-instruct:free")
            print("   This will likely result in a 404 error.\n")
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
    
    # Initialize game engine
    engine = ChessGameEngine(
        model_name=args.model,
        stockfish_path=args.stockfish,
        stockfish_time=args.time,
        stockfish_skill=args.skill,
        use_self_consistency=args.self_consistency,
        use_debate=args.debate,
        use_random_opponent=args.random_opponent,
        plan_plies=args.plan_plies,
        stockfish_depth=args.stockfish_depth,
        api_key=api_key,
        base_url=base_url
    )
    
    # Play the game
    model_plays_white = (args.model_color == "white")
    game_result = engine.play_game(model_plays_white=model_plays_white)
    
    # Save results (default behavior unless --no-save is specified)
    if not args.no_save:
        if args.save_json or not args.save_pgn:  # Save JSON by default unless only PGN is requested
            engine.save_game(game_result)
        
        if args.save_pgn:
            engine.save_pgn(game_result)
    
    print(f"\n🎉 Game completed!")
    print(f"📊 Result: {game_result['result']}")
    print(f"📈 Total moves: {game_result['total_moves']}")


if __name__ == "__main__":
    main()
