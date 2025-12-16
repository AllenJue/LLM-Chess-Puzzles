#!/usr/bin/env python3
"""
Minimal Test Examples

Quick tests for puzzles (1 puzzle) and games (1 game)
"""

import sys
import os

# Load environment
try:
    from dotenv import load_dotenv
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"✓ Loaded .env from {env_file}")
except ImportError:
    pass

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_reader import read_chess_puzzles_csv
from model_interface import ChessModelInterface
from main import evaluate_puzzles


def test_one_puzzle():
    """Test evaluating a single puzzle"""
    print("\n" + "="*60)
    print("TEST: Single Puzzle Evaluation")
    print("="*60)
    
    # Load puzzles
    csv_file = os.path.join("data", "input", "lichess_puzzles_with_pgn_1000.csv")
    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file not found: {csv_file}")
        return False
    
    df = read_chess_puzzles_csv(csv_file)
    print(f"✓ Loaded {len(df)} puzzles")
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANANNAS_API_KEY')
    if not api_key:
        print("❌ Error: No API key found. Set OPENAI_API_KEY or ANANNAS_API_KEY")
        return False
    
    base_url = os.getenv('ANANNAS_API_URL') or os.getenv('OPENAI_BASE_URL')
    
    # Create model interface
    model_interface = ChessModelInterface(
        api_key=api_key,
        model_name='gpt-3.5-turbo-instruct',
        base_url=base_url,
        max_completion_tokens=640,
        default_temperature=0.1,
        retry_attempts=2,
    )
    
    print(f"✓ Model interface created: {model_interface.model_name}")
    
    # Evaluate just 1 puzzle
    print("\nEvaluating puzzle 0...")
    df_results = evaluate_puzzles(
        df=df,
        model_interface=model_interface,
        max_puzzles=1,
        start_puzzle=0,
        planning_plies=0,
        api_delay=0.0,
    )
    
    # Show results
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    for idx, row in df_results.iterrows():
        print(f"Puzzle {idx}:")
        print(f"  Solved: {row['puzzle_solved']}")
        print(f"  Correct moves: {row['correct_moves']}")
        if row['error']:
            print(f"  Error: {row['error']}")
        if row['single_model_move']:
            print(f"  Model move: {row['single_model_move']}")
    
    return True


def test_one_game():
    """Test playing a single game"""
    print("\n" + "="*60)
    print("TEST: Single Game (Model vs Stockfish)")
    print("="*60)
    
    try:
        from chess_game_engine.model_vs_model_game import Player, PlayerType, ModelVsModelGame
        
        # Get API key
        api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANANNAS_API_KEY')
        if not api_key:
            print("❌ Error: No API key found. Set OPENAI_API_KEY or ANANNAS_API_KEY")
            return False
        
        base_url = os.getenv('ANANNAS_API_URL') or os.getenv('OPENAI_BASE_URL')
        
        # Create players
        print("Creating players...")
        white_player = Player(
            name="GPT-3.5",
            player_type=PlayerType.LLM,
            model_name="gpt-3.5-turbo-instruct",
            api_key=api_key,
            base_url=base_url,
        )
        
        black_player = Player(
            name="Stockfish",
            player_type=PlayerType.STOCKFISH,
            stockfish_skill=5,
        )
        
        print("✓ Players created")
        
        # Create and play game
        print("\nPlaying game...")
        game = ModelVsModelGame(
            white_player=white_player,
            black_player=black_player,
            verbose=True,
            max_moves=50,  # Limit for testing
        )
        
        result = game.play_game()
        
        # Show results
        print("\n" + "="*60)
        print("GAME RESULT:")
        print("="*60)
        print(f"Result: {result['result']}")
        print(f"Total moves: {result['total_moves']}")
        print(f"White score: {result['white_score']}")
        print(f"Black score: {result['black_score']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run minimal tests"""
    print("\n" + "="*60)
    print("MINIMAL TEST SUITE")
    print("="*60)
    
    # Test 1: One puzzle
    print("\n[1/2] Testing single puzzle evaluation...")
    puzzle_ok = test_one_puzzle()
    
    # Test 2: One game
    print("\n[2/2] Testing single game...")
    game_ok = test_one_game()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Puzzle test: {'✓ PASSED' if puzzle_ok else '✗ FAILED'}")
    print(f"Game test:   {'✓ PASSED' if game_ok else '✗ FAILED'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

