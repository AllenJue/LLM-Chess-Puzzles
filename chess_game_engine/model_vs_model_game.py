#!/usr/bin/env python3
"""
Model vs Model Chess Game Engine

Supports:
- LLM vs LLM games
- Human vs LLM games (human inputs SAN moves)
- Stockfish as a baseline player
- Opening variations
"""

import chess
import chess.engine
import chess.pgn
import sys
import os
import json
import random
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List, Callable
from enum import Enum

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from chess_puzzles directory (parent of chess_game_engine)
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
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
from chess_utils import san_to_uci, extract_plan_sans


class PlayerType(Enum):
    """Types of players in a chess game"""
    LLM = "llm"
    HUMAN = "human"
    STOCKFISH = "stockfish"
    RANDOM = "random"


class Player:
    """Represents a player in a chess game"""
    def __init__(
        self,
        name: str,
        player_type: PlayerType,
        model_name: Optional[str] = None,
        use_self_consistency: bool = False,
        use_debate: bool = False,
        plan_plies: int = 0,
        stockfish_path: Optional[str] = None,
        stockfish_time: float = 1.0,
        stockfish_skill: int = 5,
        stockfish_depth: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        human_move_callback: Optional[Callable[[chess.Board], str]] = None,
    ):
        """
        Initialize a player
        
        Args:
            name: Player name/identifier
            player_type: Type of player (LLM, HUMAN, STOCKFISH, RANDOM)
            model_name: Model name for LLM players
            use_self_consistency: Use self-consistency for LLM
            use_debate: Use debate for LLM
            stockfish_path: Path to Stockfish executable
            stockfish_time: Time limit for Stockfish moves
            stockfish_skill: Stockfish skill level (0-20)
            stockfish_depth: Fixed depth for Stockfish (overrides time)
            api_key: API key for LLM
            base_url: Base URL for LLM API
            human_move_callback: Function to get human move (board -> SAN string)
        """
        self.name = name
        self.player_type = player_type
        self.model_name = model_name
        self.use_self_consistency = use_self_consistency
        self.use_debate = use_debate
        self.plan_plies = plan_plies
        
        # Set Stockfish path - try to find it if not provided or if default doesn't work
        if player_type == PlayerType.STOCKFISH:
            if stockfish_path and stockfish_path != 'stockfish' and os.path.exists(stockfish_path):
                self.stockfish_path = stockfish_path
            else:
                # Try to find Stockfish in PATH or common locations
                import shutil
                possible_paths = [
                    shutil.which('stockfish'),
                    shutil.which('Stockfish'),
                    '/usr/local/bin/stockfish',
                    '/opt/homebrew/bin/stockfish',
                    '/usr/bin/stockfish',
                ]
                found_path = None
                for path in possible_paths:
                    if path and os.path.exists(path):
                        found_path = path
                        break
                
                if found_path:
                    self.stockfish_path = found_path
                else:
                    self.stockfish_path = stockfish_path or 'stockfish'
        else:
            self.stockfish_path = stockfish_path or 'stockfish'
        
        self.stockfish_time = stockfish_time
        self.stockfish_skill = stockfish_skill
        self.stockfish_depth = stockfish_depth
        self.human_move_callback = human_move_callback
        
        # Initialize LLM components if needed
        if player_type == PlayerType.LLM:
            if not model_name:
                raise ValueError("model_name required for LLM players")
            
            # Pass None for api_key and base_url to let ChessModelInterface
            # automatically select the correct API based on model type
            # (OpenAI models -> OPENAI_API_KEY, open-source -> ANANNAS_API_KEY)
            self.model_interface = ChessModelInterface(
                model_name=model_name,
                api_key=api_key,  # If None, ChessModelInterface will auto-select based on model
                base_url=base_url,  # If None, ChessModelInterface will auto-select based on model
                max_completion_tokens=640,
                default_temperature=0.1,
                retry_attempts=2,
            )
            
            if use_self_consistency:
                self.self_consistency = ChessSelfConsistency(
                    model_name=model_name,
                    temperature=0.1,
                    openai_api_key=api_key,
                    base_url=base_url,
                    max_rounds=2,
                    sleep_time=0.1,
                    plan_plies=plan_plies
                )
            elif use_debate:
                self.debate_v2 = ChessDebateV2(
                    model_name=model_name,
                    temperature=0.1,
                    openai_api_key=api_key,
                    base_url=base_url,
                    max_rounds=3,
                    sleep_time=0.1,
                    plan_plies=plan_plies
                )
        
        # Token tracking and response tracking
        self.token_log: List[Dict[str, Any]] = []
        self.response_log: List[Dict[str, Any]] = []  # Track model responses
        self.error_log: List[Dict[str, Any]] = []  # Track errors
        
        # Planning state (matching puzzle implementation)
        self.active_plan: Optional[Dict[str, Any]] = None  # Current active plan
        self.plan_log: List[Dict[str, Any]] = []  # Log of all plan events
    
    def get_move(self, board: chess.Board, verbose: bool = False) -> Optional[str]:
        """
        Get a move from this player
        
        Args:
            board: Current chess board position
            verbose: Whether to print debug information
            
        Returns:
            UCI move string or None if failed
        """
        if self.player_type == PlayerType.HUMAN:
            if not self.human_move_callback:
                raise ValueError("human_move_callback required for HUMAN players")
            try:
                san_move = self.human_move_callback(board)
                move = board.parse_san(san_move)
                if move not in board.legal_moves:
                    if verbose:
                        print(f"❌ {self.name}: Illegal move: {san_move}")
                    return None
                return move.uci()
            except Exception as e:
                if verbose:
                    print(f"❌ {self.name}: Error parsing move: {e}")
                return None
        
        elif self.player_type == PlayerType.STOCKFISH:
            return self._get_stockfish_move(board, verbose)
        
        elif self.player_type == PlayerType.RANDOM:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                if verbose:
                    print(f"🎲 {self.name} plays random: {board.san(move)} ({move.uci()})")
                return move.uci()
            return None
        
        elif self.player_type == PlayerType.LLM:
            return self._get_llm_move(board, verbose)
        
        return None
    
    def _attempt_planned_move(self, board: chess.Board, verbose: bool = False) -> Optional[str]:
        """
        Attempt to use a planned move if available (matching chess_game.py implementation exactly)
        
        Returns:
            UCI move string if planned move is available and legal, None otherwise
        """
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

        if verbose:
            print(f"🤖 Using planned move without prompting: {board.san(move)} ({move_uci})")
        
        return move_uci
    
    def _set_active_plan(self, source: str, plan_full: List[str], plan_execute: List[str], *, turn: int, attempt: int, selected_san: Optional[str] = None) -> None:
        """
        Set an active plan (matching chess_game.py implementation exactly)
        
        Args:
            source: Source of the plan (e.g., "single_model", "self_consistency", "aggressive", "positional", "neutral")
            plan_full: Full plan extracted from response
            plan_execute: List of SAN moves to execute (first N plies)
            turn: Current turn number
            attempt: Current attempt number
            selected_san: Selected move in SAN notation
        """
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
    
    def _discard_active_plan(self, reason: str) -> None:
        """
        Discard/clear the active plan (matching chess_game.py implementation exactly)
        
        Args:
            reason: Reason for discarding (e.g., "replaced", "aborted", "completed")
        """
        if self.active_plan:
            source = self.active_plan.get("source", "unknown")
            self._log_plan_event(source, "plan_discarded", reason=reason)
        
        self.active_plan = None
    
    def _handle_opponent_plan(self, move_san: str, verbose: bool = False) -> None:
        """
        Handle opponent move - check if it matches the plan (matching chess_game.py implementation exactly)
        
        Args:
            move_san: Opponent's move in SAN notation
            verbose: Whether to print debug information
        """
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
            if verbose:
                print(f"<planning> : Opponent move {move_san} diverged from plan {expected_san}, clearing plan.")
            self.active_plan = None
    
    def _log_plan_event(self, source: str, event: str, **data) -> None:
        """
        Log a plan event for tracking (matching chess_game.py implementation exactly)
        
        Args:
            source: Source of the plan (e.g., "single_model", "self_consistency")
            event: Type of event (e.g., "plan_generated", "plan_move_used", "plan_discarded")
            **data: Additional event data
        """
        entry = {
            "event": event,
            "source": source,
            **data
        }
        print(f"<planning> : {entry}")
        self.plan_log.append(entry)
    
    def _get_stockfish_move(self, board: chess.Board, verbose: bool = False) -> Optional[str]:
        """Get move from Stockfish"""
        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                engine.configure({"Skill Level": self.stockfish_skill})
                
                if self.stockfish_depth is not None:
                    limit = chess.engine.Limit(depth=self.stockfish_depth)
                else:
                    limit = chess.engine.Limit(time=self.stockfish_time)
                
                result = engine.play(board, limit)
                move_uci = result.move.uci()
                if verbose:
                    print(f"🐟 {self.name} (Stockfish): {board.san(result.move)} ({move_uci})")
                return move_uci
        except Exception as e:
            if verbose:
                print(f"❌ {self.name}: Stockfish error: {e}")
            return None
    
    def _get_llm_move(self, board: chess.Board, verbose: bool = False) -> Optional[str]:
        """Get move from LLM with retry logic"""
        # Try to use planned move first (matching puzzle implementation)
        if self.plan_plies > 0:
            planned_move = self._attempt_planned_move(board, verbose=verbose)
            if planned_move:
                return planned_move
        
        exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
        current_game = chess.pgn.Game.from_board(board)
        user_prompt = current_game.accept(exporter)
        
        # Check if this is the starting position (empty game) or first move
        is_starting_position = len(board.move_stack) == 0 or user_prompt.strip() in ['*', '']
        is_first_move = len(board.move_stack) <= 1
        
        # Check if this is a confused model that needs special handling
        # GPT-3.5 works fine with the original prompt, so keep it unchanged
        confused_models = [
            'meta-llama/llama-3.3-70b-instruct',
            'mistralai/mistral-small-24b-instruct-2501',
            'deepseek-ai/deepseek-v3',
        ]
        is_confused_model = self.model_name in confused_models if self.model_name else False
        
        # Adjust prompt for starting position vs in-progress game
        # Only use simplified prompts for confused models, keep original for GPT-3.5
        if is_starting_position and is_confused_model:
            color = "White" if board.turn == chess.WHITE else "Black"
            system_prompt = (
                f"You are a chess grandmaster.\n"
                f"You are playing {color} in a new chess game starting from the initial position.\n"
                f"Provide your first move in standard algebraic notation (e.g., \"e4\", \"d4\", \"Nf3\").\n"
                f"Output only the move, nothing else.\n"
            )
            # For starting position, use empty user prompt
            simplified_user_prompt = ""
        elif is_first_move and board.turn == chess.BLACK and is_confused_model:
            # Black's first move after White has moved - simplified for confused models
            system_prompt = (
                "You are a chess grandmaster playing Black.\n"
                "White has just played their first move.\n"
                "Provide your response move in standard algebraic notation (e.g., \"e5\", \"c5\", \"Nf6\").\n"
                "Output only your move, nothing else.\n"
            )
            # Extract just the move notation, not the full PGN
            simplified_user_prompt = user_prompt.replace('*', '').strip()
        else:
            # Original prompt (works well for GPT-3.5 and in-progress games)
            system_prompt = (
                "You are a chess grandmaster.\n"
                "You will be given a partially completed game.\n"
                "After seeing it, you should repeat the ENTIRE GAME and then give ONE new move.\n"
                "After repeating the game, immediately continue by listing that move on a single line, starting with the side to move now.\n"
                "Use standard algebraic notation, e.g. \"e4\" or \"Rdf8\" or \"R1a3\".\n"
                "ALWAYS repeat the entire representation of the game so far.\n"
                "NEVER explain your choice.\n"
            )
            simplified_user_prompt = user_prompt
        
        # Try up to 3 times
        for attempt in range(3):
            if attempt > 0 and verbose:
                print(f"🔄 {self.name}: Retry attempt {attempt + 1}/3...")
            
            try:
                if self.use_self_consistency:
                    predicted_uci, debate_history = self.self_consistency.run_debate(
                        user_prompt,
                        played_plies=len(board.move_stack),
                        board_fen=board.fen(),
                        board=board
                    )
                    
                    # Extract and set plan from self-consistency (matching chess_game.py implementation exactly)
                    if predicted_uci:
                        move = chess.Move.from_uci(predicted_uci)
                        if move in board.legal_moves:
                            plan_info = debate_history.get("final_plan", {}) if isinstance(debate_history, dict) else {}
                            plan_execute = plan_info.get("moves_for_execution") or []
                            if self.plan_plies > 0:
                                plan_execute = plan_execute[:self.plan_plies]
                            plan_full = plan_info.get("full_moves") or list(plan_execute)
                            if self.plan_plies > 0:
                                print(f"<self-consistency> : plan (full): {plan_full}")
                                print(f"<self-consistency> : plan (first {self.plan_plies} plies): {plan_execute}")
                            
                            move_san = board.san(move)
                            if plan_execute:
                                self._set_active_plan(
                                    "self_consistency",
                                    plan_full,
                                    plan_execute,
                                    turn=len(board.move_stack) + 1,
                                    attempt=attempt + 1,
                                    selected_san=move_san
                                )
                    
                    # Save full debate history (individual agent responses, plans, tokens)
                    if debate_history:
                        self.response_log.append({
                            "turn": len(board.move_stack) + 1,
                            "attempt": attempt + 1,
                            "debate_history": debate_history,
                            "board_fen": board.fen(),
                        })
                    
                    if predicted_uci:
                        move = chess.Move.from_uci(predicted_uci)
                        if move in board.legal_moves:
                            if verbose:
                                print(f"✅ {self.name}: {board.san(move)} ({predicted_uci})")
                            # Track tokens (both individual and total)
                            if debate_history and isinstance(debate_history, dict):
                                token_totals = debate_history.get("total_tokens", {})
                                # Also save individual agent tokens
                                individual_tokens = {
                                    "aggressive": debate_history.get("query1", {}).get("aggressive_tokens", {}),
                                    "positional": debate_history.get("query2", {}).get("positional_tokens", {}),
                                    "neutral": debate_history.get("query3", {}).get("neutral_tokens", {}),
                                }
                                self.token_log.append({
                                    "turn": len(board.move_stack) + 1,
                                    "attempt": attempt + 1,
                                    "tokens": token_totals,
                                    "individual_agent_tokens": individual_tokens
                                })
                            return predicted_uci
                
                elif self.use_debate:
                    predicted_uci, debate_history = self.debate_v2.run_debate(
                        user_prompt,
                        played_plies=len(board.move_stack),
                        board_fen=board.fen()
                    )
                    
                    # Save full debate history
                    if debate_history:
                        self.response_log.append({
                            "turn": len(board.move_stack) + 1,
                            "attempt": attempt + 1,
                            "debate_history": debate_history,
                            "board_fen": board.fen(),
                        })
                    
                    if predicted_uci:
                        move = chess.Move.from_uci(predicted_uci)
                        if move in board.legal_moves:
                            if verbose:
                                print(f"✅ {self.name}: {board.san(move)} ({predicted_uci})")
                            return predicted_uci
                
                else:
                    # Single model
                    raw_response, predicted_san, token_info = self.model_interface.get_move_with_extraction(
                        system_prompt,
                        simplified_user_prompt,
                        current_turn_number=len(board.move_stack) // 2 + 1,
                        is_white_to_move=board.turn,
                        max_tokens=self.model_interface.max_completion_tokens,
                        temperature=self.model_interface.default_temperature,
                        retry_attempts=self.model_interface.retry_attempts,
                    )
                    
                    # Extract planning information if plan_plies > 0 (matching chess_game.py implementation exactly)
                    plan_full = extract_plan_sans(raw_response, primary_san=predicted_san)
                    plan_execute = plan_full[:self.plan_plies] if self.plan_plies > 0 else []
                    if self.plan_plies > 0:
                        print(f"<single-model> : plan (full): {plan_full}")
                        print(f"<single-model> : plan (first {self.plan_plies} plies): {plan_execute}")
                        # Set active plan for future moves (matching chess_game.py signature)
                        if plan_execute and predicted_san:
                            self._set_active_plan(
                                "single_model",
                                plan_full,
                                plan_execute,
                                turn=len(board.move_stack) + 1,
                                attempt=attempt + 1,
                                selected_san=predicted_san
                            )
                    
                    # Track response with planning information
                    response_entry = {
                        "turn": len(board.move_stack) + 1,
                        "attempt": attempt + 1,
                        "raw_response": raw_response,
                        "predicted_san": predicted_san,
                        "board_fen": board.fen(),
                    }
                    if self.plan_plies > 0:
                        response_entry["plan_full"] = plan_full
                        response_entry["plan_execute"] = plan_execute
                    self.response_log.append(response_entry)
                    
                    if predicted_san:
                        predicted_uci = san_to_uci(board.fen(), predicted_san)
                        move = chess.Move.from_uci(predicted_uci)
                        if move in board.legal_moves:
                            if verbose:
                                print(f"✅ {self.name}: {predicted_san} ({predicted_uci})")
                            # Track tokens
                            if token_info:
                                self.token_log.append({
                                    "turn": len(board.move_stack) + 1,
                                    "attempt": attempt + 1,
                                    "tokens": token_info
                                })
                            return predicted_uci
                        else:
                            # Illegal move error
                            self.error_log.append({
                                "turn": len(board.move_stack) + 1,
                                "attempt": attempt + 1,
                                "error_type": "illegal_move",
                                "predicted_san": predicted_san,
                                "predicted_uci": predicted_uci,
                                "raw_response": raw_response,
                            })
                            if verbose:
                                print(f"❌ {self.name}: Illegal move: {predicted_san}")
                    else:
                        # Failed to extract move
                        self.error_log.append({
                            "turn": len(board.move_stack) + 1,
                            "attempt": attempt + 1,
                            "error_type": "failed_extraction",
                            "raw_response": raw_response,
                        })
            
            except Exception as e:
                # Track exception error
                self.error_log.append({
                    "turn": len(board.move_stack) + 1,
                    "attempt": attempt + 1,
                    "error_type": "exception",
                    "error_message": str(e),
                })
                if verbose:
                    print(f"❌ {self.name}: Error on attempt {attempt + 1}: {e}")
                if attempt == 2:  # Last attempt
                    break
        
        # Failed after all retries - disqualify player
        if verbose:
            print(f"❌ {self.name}: DISQUALIFIED - Failed to produce a legal move after 3 attempts")
        return None


class ModelVsModelGame:
    """Game engine for model vs model (or human/stockfish) matches"""
    
    def __init__(
        self,
        white_player: Player,
        black_player: Player,
        initial_fen: Optional[str] = None,
        max_moves: int = 200,
        verbose: bool = True,
    ):
        """
        Initialize a model vs model game
        
        Args:
            white_player: Player for white pieces
            black_player: Player for black pieces
            initial_fen: Starting FEN position (None for standard start)
            max_moves: Maximum number of moves before draw
            verbose: Whether to print game progress
        """
        self.white_player = white_player
        self.black_player = black_player
        self.max_moves = max_moves
        self.verbose = verbose
        
        # Initialize board
        if initial_fen:
            self.board = chess.Board(fen=initial_fen)
        else:
            self.board = chess.Board()
        
        self.game_history: List[Dict[str, Any]] = []
        self.moves: List[str] = []
    
    def play_game(self) -> Dict[str, Any]:
        """
        Play a complete game
        
        Returns:
            Dictionary with game result and metadata
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎮 Starting Game: {self.white_player.name} (White) vs {self.black_player.name} (Black)")
            print(f"{'='*60}\n")
        
        move_number = 1
        
        while not self.board.is_game_over() and len(self.moves) < self.max_moves:
            is_white_turn = self.board.turn == chess.WHITE
            current_player = self.white_player if is_white_turn else self.black_player
            
            if self.verbose:
                print(f"\n--- Move {move_number} ({'White' if is_white_turn else 'Black'}) ---")
                print(f"FEN: {self.board.fen()}")
            
            # Get move from current player
            move_uci = current_player.get_move(self.board, verbose=self.verbose)
            
            if move_uci is None:
                # Player failed to make a move - disqualified
                winner = self.black_player.name if is_white_turn else self.white_player.name
                result = f"{winner} wins by disqualification ({current_player.name} failed to produce a legal move)"
                if self.verbose:
                    print(f"❌ {current_player.name} DISQUALIFIED - failed to produce a legal move after 3 attempts!")
                break
            
            # Validate and make move
            try:
                move = chess.Move.from_uci(move_uci)
                if move not in self.board.legal_moves:
                    # This shouldn't happen if get_move() is working correctly, but handle it anyway
                    winner = self.black_player.name if is_white_turn else self.white_player.name
                    result = f"{winner} wins by disqualification ({current_player.name} made illegal move: {move_uci})"
                    if self.verbose:
                        print(f"❌ {current_player.name} DISQUALIFIED - made illegal move: {move_uci}")
                    break
                
                move_san = self.board.san(move)
                
                # Check if this was a planned move by checking if active_plan had remaining moves
                # and if the move matches what was expected (matching chess_game.py)
                was_planned = False
                plan_source = None
                if current_player.active_plan and current_player.active_plan.get("remaining_san"):
                    # Check if this move matches the first planned move
                    planned_san = current_player.active_plan["remaining_san"][0] if current_player.active_plan["remaining_san"] else None
                    if planned_san and move_san == planned_san:
                        was_planned = True
                        plan_source = current_player.active_plan.get("source", "unknown")
                
                # Handle opponent planning BEFORE pushing move (matching chess_game.py)
                # The opponent's plan expects the move we're about to make
                opponent_player = self.black_player if is_white_turn else self.white_player
                if not is_white_turn:  # Black just moved, so white (opponent) should check their plan
                    opponent_player._handle_opponent_plan(move_san, verbose=self.verbose)
                
                self.board.push(move)
                self.moves.append(move_uci)
                
                # Record move
                move_info = {
                    "move_number": move_number,
                    "player": current_player.name,
                    "color": "white" if is_white_turn else "black",
                    "move_uci": move_uci,
                    "move_san": move_san,
                    "fen_after": self.board.fen(),
                }
                
                # Track if move was from plan (matching chess_game.py)
                if was_planned and current_player.plan_plies > 0:
                    move_info["from_plan"] = True
                    move_info["plan_source"] = plan_source
                
                self.game_history.append(move_info)
                
                if self.verbose:
                    if was_planned and current_player.plan_plies > 0:
                        print(f"✅ {current_player.name} plays: {move_san} ({move_uci}) [from plan]")
                    else:
                        print(f"✅ {current_player.name} plays: {move_san} ({move_uci})")
                
                # Check game end conditions
                if self.board.is_checkmate():
                    winner = current_player.name
                    result = f"{winner} wins by checkmate"
                    if self.verbose:
                        print(f"🏆 {winner} wins by checkmate!")
                    break
                elif self.board.is_stalemate():
                    result = "Draw by stalemate"
                    if self.verbose:
                        print(f"🤝 Game ends in stalemate!")
                    break
                elif self.board.is_insufficient_material():
                    result = "Draw by insufficient material"
                    if self.verbose:
                        print(f"🤝 Game ends due to insufficient material!")
                    break
                elif self.board.is_seventyfive_moves():
                    result = "Draw by 75-move rule"
                    if self.verbose:
                        print(f"🤝 Game ends due to 75-move rule!")
                    break
                elif self.board.is_fivefold_repetition():
                    result = "Draw by fivefold repetition"
                    if self.verbose:
                        print(f"🤝 Game ends due to fivefold repetition!")
                    break
                
                move_number += 1
            
            except Exception as e:
                winner = self.black_player.name if is_white_turn else self.white_player.name
                result = f"{winner} wins by error"
                if self.verbose:
                    print(f"❌ Error making move {move_uci}: {e}")
                break
        
        # Determine final result
        if len(self.moves) >= self.max_moves:
            result = "Draw by move limit"
        elif self.board.is_checkmate():
            winner = self.white_player.name if not self.board.turn else self.black_player.name
            result = f"{winner} wins by checkmate"
        elif self.board.is_stalemate():
            result = "Draw by stalemate"
        elif self.board.is_insufficient_material():
            result = "Draw by insufficient material"
        elif self.board.is_seventyfive_moves():
            result = "Draw by 75-move rule"
        elif self.board.is_fivefold_repetition():
            result = "Draw by fivefold repetition"
        elif "wins" not in result and "Draw" not in result:
            result = "Draw by agreement"  # Fallback
        
        # Calculate game outcome for rating purposes
        if "wins" in result:
            # Extract winner name from result string (format: "PlayerName wins by...")
            winner_name = result.split(" wins")[0].strip()
            if winner_name == self.white_player.name:
                white_score = 1.0
                black_score = 0.0
            elif winner_name == self.black_player.name:
                white_score = 0.0
                black_score = 1.0
            else:
                # Fallback: if winner name doesn't match exactly, log warning and use old logic
                # This shouldn't happen, but handle edge cases
                if self.verbose:
                    print(f"⚠️ Warning: Could not determine winner from result: {result}")
                white_score = 1.0 if self.white_player.name in result.split(" wins")[0] else 0.0
                black_score = 1.0 if self.black_player.name in result.split(" wins")[0] else 0.0
        else:
            white_score = 0.5
            black_score = 0.5
        
        if self.verbose:
            print(f"\n🎯 Final Result: {result}")
            print(f"📊 Total moves: {len(self.moves)}\n")
        
        # Collect token usage
        white_tokens = self._sum_tokens(self.white_player.token_log)
        black_tokens = self._sum_tokens(self.black_player.token_log)
        
        # Collect detailed token logs
        white_token_log = self.white_player.token_log
        black_token_log = self.black_player.token_log
        
        # Collect response logs
        white_responses = self.white_player.response_log if hasattr(self.white_player, 'response_log') else []
        black_responses = self.black_player.response_log if hasattr(self.black_player, 'response_log') else []
        
        # Collect error logs
        white_errors = self.white_player.error_log if hasattr(self.white_player, 'error_log') else []
        black_errors = self.black_player.error_log if hasattr(self.black_player, 'error_log') else []
        
        # Collect plan logs (matching puzzle implementation)
        white_plan_log = self.white_player.plan_log if hasattr(self.white_player, 'plan_log') else []
        black_plan_log = self.black_player.plan_log if hasattr(self.black_player, 'plan_log') else []
        
        return {
            "result": result,
            "white_player": self.white_player.name,
            "black_player": self.black_player.name,
            "white_score": white_score,
            "black_score": black_score,
            "total_moves": len(self.moves),
            "moves": self.moves,
            "game_history": self.game_history,
            "final_fen": self.board.fen(),
            "white_tokens": white_tokens,
            "black_tokens": black_tokens,
            "white_token_log": white_token_log,  # Detailed per-turn token usage
            "black_token_log": black_token_log,  # Detailed per-turn token usage
            "white_responses": white_responses,  # Model responses/outputs
            "black_responses": black_responses,  # Model responses/outputs
            "white_errors": white_errors,  # Error log
            "black_errors": black_errors,  # Error log
            "white_plan_log": white_plan_log,  # Planning events and metadata
            "black_plan_log": black_plan_log,  # Planning events and metadata
            "timestamp": datetime.now().isoformat(),
        }
    
    def _sum_tokens(self, token_log: List[Dict[str, Any]]) -> Dict[str, int]:
        """Sum up token usage from log"""
        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        
        for entry in token_log:
            tokens = entry.get("tokens", {})
            if isinstance(tokens, dict):
                total_prompt += tokens.get("prompt_tokens", 0) or tokens.get("total_prompt_tokens", 0)
                total_completion += tokens.get("completion_tokens", 0) or tokens.get("total_completion_tokens", 0)
                total_tokens += tokens.get("total_tokens", 0)
        
        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
        }
    
    def save_pgn(self, game_result: Dict[str, Any], filepath: str) -> str:
        """Save game as PGN file"""
        game = chess.pgn.Game()
        game.headers["Event"] = "Model vs Model Tournament"
        game.headers["Site"] = "Chess Tournament Engine"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = self.white_player.name
        game.headers["Black"] = self.black_player.name
        game.headers["Result"] = self._pgn_result(game_result["result"])
        
        # Add moves - reconstruct board from moves
        node = game
        temp_board = chess.Board()
        
        # Use initial FEN if available
        if hasattr(self, 'board') and self.board.move_stack:
            # Reconstruct from move_stack if available
            for move in self.board.move_stack:
                node = node.add_variation(move)
        else:
            # Reconstruct from UCI moves
            for move_uci in self.moves:
                move = chess.Move.from_uci(move_uci)
                node = node.add_variation(move)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            print(game, file=f)
        
        return filepath
    
    def _pgn_result(self, result: str) -> str:
        """Convert result string to PGN result format"""
        if self.white_player.name in result and "wins" in result:
            return "1-0"
        elif self.black_player.name in result and "wins" in result:
            return "0-1"
        else:
            return "1/2-1/2"

