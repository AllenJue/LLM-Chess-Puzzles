#!/usr/bin/env python3
"""
Reconstruct ratings from saved game results using Bradley-Terry model

This script loads all game JSON files from a tournament directory and
reconstructs ratings using the Bradley-Terry model, which provides more stable
ratings than Elo because it doesn't weigh recent games more heavily.

Reference: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
"""

import sys
import os
import json
import glob
from typing import List, Dict, Tuple
from collections import defaultdict
from datetime import datetime

# Add parent directory to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from bradley_terry import BradleyTerry


def reconstruct_ratings_from_directory(tournament_dir: str) -> Tuple[BradleyTerry, List[Dict]]:
    """
    Reconstruct ratings from all game JSON files in a tournament directory using Bradley-Terry model
    
    Args:
        tournament_dir: Directory containing game JSON files (or organized subdirectory)
        
    Returns:
        Tuple of (BradleyTerry model, game_history)
    """
    # Check if organized directory exists, use it if available
    organized_dir = os.path.join(tournament_dir, "organized")
    if os.path.isdir(organized_dir):
        # Collect all JSON files from organized subdirectories
        game_files = []
        for root, dirs, files in os.walk(organized_dir):
            for f in files:
                if f.endswith('.json'):
                    game_files.append(os.path.join(root, f))
    else:
        # Fall back to original directory structure
        json_files = glob.glob(os.path.join(tournament_dir, "*.json"))
        game_files = [f for f in json_files if not os.path.basename(f) in ['ratings.json', 'tournament_summary.json']]
    
    # Sort by filename to ensure consistent ordering
    game_files.sort()
    
    print(f"Found {len(game_files)} game files")
    
    # Collect all games for Bradley-Terry (batch method)
    games: List[Tuple[str, str, float, float]] = []
    game_history: List[Dict] = []
    
    # Process each game file
    for game_file in game_files:
        try:
            with open(game_file, 'r') as f:
                game_data = json.load(f)
            
            # Extract game information
            white_player = game_data.get('white_player')
            black_player = game_data.get('black_player')
            white_score = game_data.get('white_score', 0.0)
            black_score = game_data.get('black_score', 0.0)
            game_id = game_data.get('game_id', os.path.basename(game_file).replace('.json', ''))
            configuration = game_data.get('configuration', 'single')  # Default to 'single' for old games
            
            # Check if both players are GPT-3.5 (GPT vs GPT game)
            both_gpt = (white_player and white_player.startswith('gpt-3.5-turbo-instruct') and
                       black_player and black_player.startswith('gpt-3.5-turbo-instruct'))
            
            if both_gpt:
                # For GPT vs GPT games, player names already have configuration suffixes
                # Use them directly (e.g., 'gpt-3.5-turbo-instruct_SC' vs 'gpt-3.5-turbo-instruct_plan3')
                white_player_with_config = white_player
                black_player_with_config = black_player
            else:
                # For GPT vs Other games, extract configuration from filename/game_id if not in JSON
                if configuration == 'single':
                    filename = os.path.basename(game_file)
                    # Check for longer patterns first (SC_plan3 contains both SC and plan3)
                    if '_SC_plan3_' in filename:
                        configuration = 'SC_plan3'
                    elif '_SC_plan3_' in game_id:
                        configuration = 'SC_plan3'
                    elif '_plan3_' in filename:
                        configuration = 'plan3'
                    elif '_plan3_' in game_id:
                        configuration = 'plan3'
                    elif '_SC_' in filename:
                        configuration = 'SC'
                    elif '_SC_' in game_id:
                        configuration = 'SC'
                
                # Add configuration suffix to GPT-3.5 player name (if it's the base name)
                def add_config_to_player(player_name: str, config: str) -> str:
                    """Add configuration suffix to GPT-3.5 player name"""
                    if player_name == 'gpt-3.5-turbo-instruct':
                        if config == 'single':
                            return player_name  # Keep as-is for single model
                        else:
                            return f"{player_name}_{config}"
                    return player_name  # Already has suffix or not GPT-3.5
                
                white_player_with_config = add_config_to_player(white_player, configuration)
                black_player_with_config = add_config_to_player(black_player, configuration)
            
            if white_player and black_player:
                # Add to games list for Bradley-Terry
                # Format: (player1, player2, score1, score2)
                games.append((white_player_with_config, black_player_with_config, white_score, black_score))
                
                # Record game history (with configuration-aware player names)
                game_history.append({
                    "game_id": game_id,
                    "white_player": white_player_with_config,
                    "black_player": black_player_with_config,
                    "white_score": white_score,
                    "black_score": black_score,
                    "configuration": configuration,  # Store original config for reference
                })
                print(f"✓ Processed: {game_id}")
            else:
                print(f"⚠ Skipped {game_file}: missing player info")
                
        except Exception as e:
            print(f"❌ Error processing {game_file}: {e}")
    
    # Fit Bradley-Terry model with all games
    print(f"\nFitting Bradley-Terry model with {len(games)} games...")
    bt = BradleyTerry()
    ratings = bt.fit(games, initial_rating=1500.0)
    
    print(f"✅ Fitted ratings for {len(ratings)} players")
    
    return bt, game_history


def save_ratings(bt: BradleyTerry, game_history: List[Dict], output_file: str):
    """Save Bradley-Terry ratings to JSON file"""
    data = {
        "timestamp": datetime.now().isoformat(),
        "method": "Bradley-Terry",
        "ratings": bt.get_all_ratings(),
        "beta": bt.beta,  # Log-ratings for reference
        "game_history": game_history,
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def print_leaderboard(bt: BradleyTerry, game_history: List[Dict], title: str = "Tournament Leaderboard"):
    """Print formatted leaderboard"""
    ratings = bt.get_all_ratings()
    
    # Count games per player
    games_count: Dict[str, int] = defaultdict(int)
    for game in game_history:
        games_count[game["white_player"]] += 1
        games_count[game["black_player"]] += 1
    
    # Build leaderboard
    leaderboard = [
        (player, rating, games_count.get(player, 0))
        for player, rating in ratings.items()
    ]
    
    # Sort by rating (descending)
    leaderboard.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Player':<40} {'Rating':<12} {'Games':<8}")
    print(f"{'-'*80}")
    
    for rank, (player_name, rating, games) in enumerate(leaderboard, 1):
        print(f"{rank:<6} {player_name:<40} {rating:>10.1f}  {games:>6}")
    
    print(f"{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reconstruct tournament ratings from saved games using Bradley-Terry model"
    )
    parser.add_argument("tournament_dir", help="Directory containing game JSON files")
    parser.add_argument("--output", "-o", help="Output file for ratings (default: tournament_dir/ratings.json)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.tournament_dir):
        print(f"Error: {args.tournament_dir} is not a directory")
        sys.exit(1)
    
    print(f"Reconstructing ratings from: {args.tournament_dir}")
    print(f"Using Bradley-Terry model (more stable than Elo)\n")
    
    # Reconstruct ratings
    bt, game_history = reconstruct_ratings_from_directory(args.tournament_dir)
    
    # Save ratings
    output_file = args.output or os.path.join(args.tournament_dir, "ratings.json")
    save_ratings(bt, game_history, output_file)
    
    print(f"\n✅ Ratings reconstructed and saved to: {output_file}")
    print_leaderboard(bt, game_history)


if __name__ == "__main__":
    main()

