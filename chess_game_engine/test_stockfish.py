#!/usr/bin/env python3
"""
Test Stockfish with each model - run 1 game each
"""

import sys
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
except ImportError:
    pass

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from model_vs_model_game import Player, PlayerType, ModelVsModelGame
import shutil

def test_stockfish_vs_model(model_name: str, stockfish_plays_white: bool = True):
    """Test Stockfish vs a model"""
    print(f"\n{'='*80}")
    print(f"Testing: {'Stockfish' if stockfish_plays_white else model_name} (White) vs {'Stockfish' if not stockfish_plays_white else model_name} (Black)")
    print(f"{'='*80}\n")
    
    # Find Stockfish - try multiple methods
    stockfish_path = None
    possible_paths = [
        shutil.which('stockfish'),
        shutil.which('Stockfish'),
        '/usr/local/bin/stockfish',
        '/opt/homebrew/bin/stockfish',
        '/usr/bin/stockfish',
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            stockfish_path = path
            break
    
    if not stockfish_path:
        print("❌ Stockfish not found! Tried:")
        for path in possible_paths:
            if path:
                print(f"   - {path}")
        return None
    
    print(f"✅ Using Stockfish at: {stockfish_path}")
    
    # Create players
    if stockfish_plays_white:
        white_player = Player(
            name="Stockfish",
            player_type=PlayerType.STOCKFISH,
            stockfish_path=stockfish_path,
            stockfish_skill=5,
            stockfish_time=1.0,
        )
        black_player = Player(
            name=model_name,
            player_type=PlayerType.LLM,
            model_name=model_name,
        )
    else:
        white_player = Player(
            name=model_name,
            player_type=PlayerType.LLM,
            model_name=model_name,
        )
        black_player = Player(
            name="Stockfish",
            player_type=PlayerType.STOCKFISH,
            stockfish_path=stockfish_path,
            stockfish_skill=5,
            stockfish_time=1.0,
        )
    
    # Create and play game
    game = ModelVsModelGame(
        white_player=white_player,
        black_player=black_player,
        verbose=True,
        max_moves=200,
    )
    
    result = game.play_game()
    
    print(f"\n{'='*80}")
    print(f"Game Result: {result['result']}")
    print(f"White ({result['white_player']}): {result['white_score']}")
    print(f"Black ({result['black_player']}): {result['black_score']}")
    print(f"Total moves: {result['total_moves']}")
    print(f"{'='*80}\n")
    
    # Check for errors
    if result['white_errors']:
        print(f"⚠️  White errors: {len(result['white_errors'])}")
        for err in result['white_errors'][:3]:
            print(f"   - {err.get('error_type', 'unknown')}: {err.get('error_message', '')}")
    
    if result['black_errors']:
        print(f"⚠️  Black errors: {len(result['black_errors'])}")
        for err in result['black_errors'][:3]:
            print(f"   - {err.get('error_type', 'unknown')}: {err.get('error_message', '')}")
    
    return result


if __name__ == "__main__":
    models = [
        "gpt-3.5-turbo-instruct",
        "deepseek-ai/deepseek-v3",
        "mistralai/mistral-small-24b-instruct-2501",
        "meta-llama/llama-3.3-70b-instruct",
    ]
    
    print("Testing Stockfish vs each model (1 game each)")
    print(f"Models to test: {', '.join(models)}\n")
    
    results = []
    for model in models:
        try:
            # Test Stockfish as White
            result = test_stockfish_vs_model(model, stockfish_plays_white=True)
            if result:
                results.append({
                    "model": model,
                    "stockfish_color": "white",
                    "result": result['result'],
                    "moves": result['total_moves'],
                    "white_score": result['white_score'],
                    "black_score": result['black_score'],
                })
        except Exception as e:
            print(f"❌ Error testing {model}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for r in results:
        print(f"{r['model']} vs Stockfish (Stockfish as White):")
        print(f"  Result: {r['result']}")
        print(f"  Moves: {r['moves']}")
        print(f"  Score: White {r['white_score']} - {r['black_score']} Black")
        print()

