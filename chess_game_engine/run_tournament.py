#!/usr/bin/env python3
"""
Tournament Runner CLI

Run chess tournaments between LLMs, Stockfish, and humans
"""

import sys
import os
import argparse
import json
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from chess_puzzles directory
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")
    else:
        print(f"No .env file found at {env_file}, using system environment variables")
except ImportError:
    pass  # dotenv not available, continue without it

# Add parent directory to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from model_vs_model_game import Player, PlayerType
from tournament_manager import TournamentManager, create_default_players
from opening_variations import get_all_openings


def create_human_player(name: str = "Human") -> Player:
    """
    Create a human player that prompts for moves
    
    Args:
        name: Name for the human player
        
    Returns:
        Player object for human
    """
    def get_human_move(board):
        """Prompt human for move"""
        print(f"\n{'='*60}")
        print(f"Current position:")
        print(board)
        print(f"\nFEN: {board.fen()}")
        print(f"Turn: {'White' if board.turn else 'Black'}")
        print(f"{'='*60}")
        
        # Show legal moves
        legal_moves = list(board.legal_moves)
        legal_sans = [board.san(move) for move in legal_moves]
        print(f"\nLegal moves: {', '.join(legal_sans[:20])}")
        if len(legal_sans) > 20:
            print(f"... and {len(legal_sans) - 20} more")
        
        # Get move from user
        while True:
            try:
                san_move = input(f"\nEnter your move (SAN, e.g., 'e4'): ").strip()
                if not san_move:
                    print("Please enter a move")
                    continue
                
                # Try to parse
                move = board.parse_san(san_move)
                if move in board.legal_moves:
                    return san_move
                else:
                    print(f"❌ Illegal move: {san_move}")
            except ValueError as e:
                print(f"❌ Invalid move format: {e}")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                sys.exit(0)
    
    return Player(
        name=name,
        player_type=PlayerType.HUMAN,
        human_move_callback=get_human_move,
    )


def run_human_vs_llm(
    model_name: str,
    model_display_name: Optional[str] = None,
    human_plays_white: bool = True,
    use_openings: bool = False,
    opening_key: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    Run a single game between human and LLM
    
    Args:
        model_name: Model identifier
        model_display_name: Display name for model
        human_plays_white: Whether human plays white
        use_openings: Whether to use opening variation
        opening_key: Specific opening key to use
        api_key: API key
        base_url: Base URL
    """
    from model_vs_model_game import ModelVsModelGame
    from opening_variations import get_opening_variation
    
    # Create players
    human = create_human_player()
    
    llm_player = Player(
        name=model_display_name or model_name,
        player_type=PlayerType.LLM,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    
    # Set up colors
    white_player = human if human_plays_white else llm_player
    black_player = llm_player if human_plays_white else human
    
    # Get opening if specified
    initial_fen = None
    if use_openings:
        if opening_key:
            opening = get_opening_variation(opening_key)
            if opening:
                initial_fen = opening['fen']
                print(f"Starting from opening: {opening['name']}")
        else:
            from opening_variations import get_random_opening
            opening = get_random_opening()
            initial_fen = opening['fen']
            print(f"Starting from random opening: {opening['name']}")
    
    # Create and play game
    game = ModelVsModelGame(
        white_player=white_player,
        black_player=black_player,
        initial_fen=initial_fen,
        verbose=True,
    )
    
    result = game.play_game()
    
    print(f"\n{'='*60}")
    print(f"Game Result: {result['result']}")
    print(f"{'='*60}\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run chess tournaments between LLMs, Stockfish, and humans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full tournament with default models
  python run_tournament.py --tournament

  # Run tournament with openings
  python run_tournament.py --tournament --use-openings

  # Play against an LLM
  python run_tournament.py --human-vs-llm --model gpt-3.5-turbo-instruct

  # Run tournament with specific models
  python run_tournament.py --tournament --models gpt-3.5-turbo-instruct deepseek-v3

  # Run tournament with Stockfish baseline
  python run_tournament.py --tournament --include-stockfish --stockfish-skill 10
        """
    )
    
    # Tournament mode
    parser.add_argument("--tournament", action="store_true",
                       help="Run full tournament")
    
    # Human vs LLM mode
    parser.add_argument("--human-vs-llm", action="store_true",
                       help="Play a single game against an LLM")
    parser.add_argument("--model", type=str,
                       help="Model to play against (for --human-vs-llm)")
    parser.add_argument("--human-black", action="store_true",
                       help="Human plays black (default: white)")
    parser.add_argument("--model-name", type=str,
                       help="Display name for model")
    
    # Tournament options
    parser.add_argument("--games-per-matchup", type=int, default=2,
                       help="Number of games per matchup (default: 2)")
    parser.add_argument("--use-self-consistency", action="store_true",
                       help="Use self-consistency approach for all LLM players")
    parser.add_argument("--plan-plies", type=int, default=0,
                       help="Number of future plies to plan (for self-consistency and single model)")
    parser.add_argument("--use-openings", action="store_true",
                       help="Use opening variations")
    parser.add_argument("--opening", type=str,
                       help="Specific opening key (e.g., 'e4_e5')")
    parser.add_argument("--openings", nargs="+",
                       help="List of opening keys to use")
    
    # Player selection
    parser.add_argument("--models", nargs="+",
                       help="Specific models to include (model identifiers)")
    parser.add_argument("--model-names", nargs="+",
                       help="Display names for models (must match --models)")
    parser.add_argument("--pair", nargs=2, metavar=("MODEL1", "MODEL2"),
                       help="Run a single matchup between two models (e.g., --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3)")
    parser.add_argument("--include-stockfish", action="store_true",
                       help="Include Stockfish as baseline")
    parser.add_argument("--stockfish-skill", type=int, default=5,
                       help="Stockfish skill level 0-20 (default: 5)")
    parser.add_argument("--stockfish-path", type=str, default="stockfish",
                       help="Path to Stockfish executable")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="data/tournaments",
                       help="Output directory for results")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    # API options
    parser.add_argument("--api-key", type=str,
                       help="API key (uses env if not provided)")
    parser.add_argument("--base-url", type=str,
                       help="Base URL (uses env if not provided)")
    
    args = parser.parse_args()
    
    # Get API credentials (will be selected per-model based on model type)
    # For now, get both - ChessModelInterface will select the right one
    api_key = args.api_key  # If provided, use it; otherwise let ChessModelInterface choose
    base_url = args.base_url  # If provided, use it; otherwise let ChessModelInterface choose
    
    # Human vs LLM mode
    if args.human_vs_llm:
        if not args.model:
            print("Error: --model required for --human-vs-llm")
            parser.print_help()
            sys.exit(1)
        
        run_human_vs_llm(
            model_name=args.model,
            model_display_name=args.model_name,
            human_plays_white=not args.human_black,
            use_openings=args.use_openings,
            opening_key=args.opening,
            api_key=api_key,
            base_url=base_url,
        )
        return
    
    # Single pair mode
    if args.pair:
        if len(args.pair) != 2:
            print("Error: --pair requires exactly 2 models")
            sys.exit(1)
        
        from model_vs_model_game import ModelVsModelGame
        
        # Helper to create player from name (handles Stockfish and GPT configs)
        def create_player_from_name(name: str) -> Player:
            if name.lower() == "stockfish":
                return Player(
                    name="Stockfish",
                    player_type=PlayerType.STOCKFISH,
                    stockfish_path=args.stockfish_path,
                    stockfish_skill=args.stockfish_skill,
                    plan_plies=args.plan_plies,  # Pass plan_plies for consistency
                )
            else:
                # Parse configuration from name (e.g., "gpt-3.5-turbo-instruct_SC_plan3")
                # Extract base model name and config
                base_name = name
                use_sc = args.use_self_consistency
                plan_plies = args.plan_plies
                
                # Check for configuration suffixes in name
                if "_SC_plan3" in name:
                    base_name = name.replace("_SC_plan3", "")
                    use_sc = True
                    plan_plies = 3
                elif "_plan3" in name:
                    base_name = name.replace("_plan3", "")
                    use_sc = False
                    plan_plies = 3
                elif "_SC" in name:
                    base_name = name.replace("_SC", "")
                    use_sc = True
                    plan_plies = 0
                
                return Player(
                    name=name,  # Keep full name for identification
                    player_type=PlayerType.LLM,
                    model_name=base_name,  # Use base model name for API
                    use_self_consistency=use_sc,
                    plan_plies=plan_plies,
                    api_key=api_key,
                    base_url=base_url,
                )
        
        # Create players for the pair
        player1 = create_player_from_name(args.pair[0])
        player2 = create_player_from_name(args.pair[1])
        
        # Play games
        print(f"\nPlaying {args.games_per_matchup} game(s) between {args.pair[0]} and {args.pair[1]}")
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine configuration suffix for game ID
        config_parts = []
        if args.use_self_consistency:
            config_parts.append("SC")
        if args.plan_plies and args.plan_plies > 0:
            config_parts.append(f"plan{args.plan_plies}")
        config_suffix = "_".join(config_parts) if config_parts else "single"
        
        # Sanitize names for file paths
        white_safe = player1.name.replace('/', '_').replace('\\', '_')
        black_safe = player2.name.replace('/', '_').replace('\\', '_')
        
        # Find the highest existing game number to avoid overwriting
        # Check both orderings and with/without config suffix
        base_pattern1 = f"{white_safe}_vs_{black_safe}_"
        base_pattern2 = f"{black_safe}_vs_{white_safe}_"
        max_game_num = 0
        
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if filename.endswith('.json'):
                    # Check if this file matches our configuration
                    # Pattern: {white}_vs_{black}_{config}_{num}.json or {white}_vs_{black}_{num}.json
                    for base_pattern in [base_pattern1, base_pattern2]:
                        if filename.startswith(base_pattern):
                            # Try to extract number - could be after config suffix or directly
                            remaining = filename[len(base_pattern):-5]  # Remove .json
                            
                            # Check if it has config suffix
                            if config_suffix and remaining.startswith(config_suffix + "_"):
                                try:
                                    num = int(remaining[len(config_suffix) + 1:])
                                    max_game_num = max(max_game_num, num)
                                except ValueError:
                                    pass
                            elif config_suffix == "single" and "_" in remaining:
                                # Old format without config - skip for new configs
                                pass
                            else:
                                # No config suffix (old format) - only count if we're also single
                                if config_suffix == "single":
                                    try:
                                        num = int(remaining)
                                        max_game_num = max(max_game_num, num)
                                    except ValueError:
                                        pass
        
        # Start from the next available number
        start_game_num = max_game_num + 1
        
        for game_num in range(args.games_per_matchup):
            white_player = player1 if game_num % 2 == 0 else player2
            black_player = player2 if game_num % 2 == 0 else player1
            
            # Use the actual player names for the game_id (not the sanitized base)
            white_safe_current = white_player.name.replace('/', '_').replace('\\', '_')
            black_safe_current = black_player.name.replace('/', '_').replace('\\', '_')
            current_game_num = start_game_num + game_num
            
            print(f"\n--- Game {game_num + 1}/{args.games_per_matchup} (Game ID: {current_game_num}, Config: {config_suffix}) ---")
            game = ModelVsModelGame(
                white_player=white_player,
                black_player=black_player,
                verbose=not args.quiet,
            )
            
            result = game.play_game()
            
            # Save game with unique game number and configuration
            if config_suffix != "single":
                game_id = f"{white_safe_current}_vs_{black_safe_current}_{config_suffix}_{current_game_num}"
            else:
                game_id = f"{white_safe_current}_vs_{black_safe_current}_{current_game_num}"
            result['game_id'] = game_id
            result['configuration'] = config_suffix  # Store config in result
            
            json_file = os.path.join(output_dir, f"{game_id}.json")
            with open(json_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ Saved to: {json_file}")
            
            pgn_file = os.path.join(output_dir, f"{game_id}.pgn")
            game.save_pgn(result, pgn_file)
            print(f"✓ PGN saved to: {pgn_file}")
        
        return
    
    # Tournament mode
    if not args.tournament and not args.human_vs_llm:
        parser.print_help()
        return
    
    # Create players
    players = []
    
    if args.models:
        # Use specified models
        model_names = args.model_names or args.models
        for i, model_id in enumerate(args.models):
            display_name = model_names[i] if i < len(model_names) else model_id
            player = Player(
                name=display_name,
                player_type=PlayerType.LLM,
                model_name=model_id,
                use_self_consistency=args.use_self_consistency,
                plan_plies=args.plan_plies,
                api_key=api_key,
                base_url=base_url,
            )
            players.append(player)
    else:
        # Use default models
        players = create_default_players(
            include_stockfish=False,  # Add separately if requested
        )
    
    # Add Stockfish if requested
    if args.include_stockfish:
        stockfish = Player(
            name="Stockfish",
            player_type=PlayerType.STOCKFISH,
            stockfish_path=args.stockfish_path,
            stockfish_skill=args.stockfish_skill,
        )
        players.append(stockfish)
    
    if not players:
        print("Error: No players specified")
        sys.exit(1)
    
    # Setup openings
    opening_keys = None
    if args.use_openings:
        if args.opening:
            opening_keys = [args.opening]
        elif args.openings:
            opening_keys = args.openings
        # Otherwise use all openings
    
    # Create tournament manager
    manager = TournamentManager(
        players=players,
        games_per_matchup=args.games_per_matchup,
        use_openings=args.use_openings,
        opening_keys=opening_keys,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    
    # Run tournament
    print(f"\nStarting tournament with {len(players)} players...")
    summary = manager.run_tournament()
    
    print(f"\n✅ Tournament complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Leaderboard:")
    manager.ratings.print_leaderboard()


if __name__ == "__main__":
    main()

