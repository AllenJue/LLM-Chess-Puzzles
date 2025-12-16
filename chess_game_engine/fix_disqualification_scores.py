#!/usr/bin/env python3
"""
Fix scores in tournament JSON files where disqualification results have incorrect scores.
The bug was that scores were set based on whether a player name appeared in the result string,
but the loser's name also appears in disqualification messages, causing incorrect scores.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any


def fix_game_result(game_data: Dict[str, Any]) -> bool:
    """
    Fix scores in a game result if they're incorrect due to disqualification bug.
    
    Returns:
        True if scores were fixed, False otherwise
    """
    result = game_data.get("result", "")
    white_player = game_data.get("white_player", "")
    black_player = game_data.get("black_player", "")
    white_score = game_data.get("white_score", 0.0)
    black_score = game_data.get("black_score", 0.0)
    
    # Only fix files with disqualification results
    if "wins by disqualification" not in result:
        return False
    
    # Extract winner name from result string (format: "PlayerName wins by disqualification...")
    if " wins" not in result:
        return False
    
    winner_name = result.split(" wins")[0].strip()
    
    # Determine correct scores
    if winner_name == white_player:
        correct_white_score = 1.0
        correct_black_score = 0.0
    elif winner_name == black_player:
        correct_white_score = 0.0
        correct_black_score = 1.0
    else:
        # Winner name doesn't match either player - skip this file
        print(f"⚠️  Warning: Winner '{winner_name}' doesn't match white '{white_player}' or black '{black_player}'")
        return False
    
    # Check if scores need fixing
    if white_score == correct_white_score and black_score == correct_black_score:
        return False  # Scores are already correct
    
    # Fix scores
    game_data["white_score"] = correct_white_score
    game_data["black_score"] = correct_black_score
    
    return True


def fix_tournament_directory(tournament_dir: str) -> None:
    """Fix all JSON game files in a tournament directory"""
    tournament_path = Path(tournament_dir)
    
    if not tournament_path.exists():
        print(f"❌ Tournament directory does not exist: {tournament_dir}")
        return
    
    json_files = list(tournament_path.glob("*.json"))
    # Exclude metadata files
    json_files = [f for f in json_files if f.name not in ["ratings.json", "tournament_summary.json"]]
    
    if not json_files:
        print(f"⚠️  No game JSON files found in {tournament_dir}")
        return
    
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"📁 Processing {len(json_files)} game files in {tournament_dir}...")
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                game_data = json.load(f)
            
            if fix_game_result(game_data):
                # Save fixed file
                with open(json_file, 'w') as f:
                    json.dump(game_data, f, indent=2)
                fixed_count += 1
                print(f"✅ Fixed: {json_file.name}")
            else:
                skipped_count += 1
                
        except Exception as e:
            error_count += 1
            print(f"❌ Error processing {json_file.name}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Fixed: {fixed_count} files")
    print(f"   ⏭️  Skipped: {skipped_count} files")
    print(f"   ❌ Errors: {error_count} files")


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_disqualification_scores.py <tournament_directory>")
        print("Example: python fix_disqualification_scores.py data/tournaments/round_robin_single_10games")
        sys.exit(1)
    
    tournament_dir = sys.argv[1]
    fix_tournament_directory(tournament_dir)


if __name__ == "__main__":
    main()

