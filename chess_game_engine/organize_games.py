#!/usr/bin/env python3
"""
Organize tournament game files into directories by matchup
"""

import os
import sys
import json
import shutil
from collections import defaultdict
from pathlib import Path


def get_matchup_name(white_player: str, black_player: str) -> str:
    """Generate a consistent matchup name from two player names"""
    # Normalize player names for directory names
    def normalize_name(name: str) -> str:
        # Replace slashes with underscores
        name = name.replace('/', '_')
        # Remove any other problematic characters
        return name
    
    def extract_base_name(name: str) -> str:
        """Extract base model name (without config suffix)"""
        if name.startswith('gpt-3.5-turbo-instruct'):
            # Check for config suffixes
            if name.endswith('_SC_plan3'):
                return 'gpt-3.5-turbo-instruct_SC_plan3'
            elif name.endswith('_plan3'):
                return 'gpt-3.5-turbo-instruct_plan3'
            elif name.endswith('_SC'):
                return 'gpt-3.5-turbo-instruct_SC'
            else:
                return 'gpt-3.5-turbo-instruct'
        return name
    
    white_norm = normalize_name(white_player)
    black_norm = normalize_name(black_player)
    
    # For GPT vs GPT, keep base name first, then config variants
    # For other matchups, sort alphabetically
    white_base = extract_base_name(white_norm)
    black_base = extract_base_name(black_norm)
    
    # Check if both are GPT-3.5
    both_gpt = (white_base.startswith('gpt-3.5-turbo-instruct') and 
                black_base.startswith('gpt-3.5-turbo-instruct'))
    
    if both_gpt:
        # For GPT vs GPT: put base name first, then variants
        if white_base == 'gpt-3.5-turbo-instruct' and black_base != 'gpt-3.5-turbo-instruct':
            return f"{white_norm}_vs_{black_norm}"
        elif black_base == 'gpt-3.5-turbo-instruct' and white_base != 'gpt-3.5-turbo-instruct':
            return f"{black_norm}_vs_{white_norm}"
        else:
            # Both have configs, sort alphabetically
            players = sorted([white_norm, black_norm])
            return f"{players[0]}_vs_{players[1]}"
    else:
        # Sort alphabetically for non-GPT matchups
        players = sorted([white_norm, black_norm])
        return f"{players[0]}_vs_{players[1]}"


def organize_games(tournament_dir: str, dry_run: bool = False):
    """Organize game files into directories by matchup"""
    
    if not os.path.isdir(tournament_dir):
        print(f"Error: {tournament_dir} is not a directory")
        sys.exit(1)
    
    # Create organized subdirectory
    organized_dir = os.path.join(tournament_dir, "organized")
    
    if not dry_run:
        os.makedirs(organized_dir, exist_ok=True)
    
    # Find all JSON files
    all_files = os.listdir(tournament_dir)
    json_files = [f for f in all_files if f.endswith('.json') and 
                  f not in ['ratings.json', 'tournament_summary.json']]
    
    # Group files by matchup
    matchup_files = defaultdict(list)
    matchup_info = {}
    
    print(f"Processing {len(json_files)} game files...")
    
    for filename in json_files:
        filepath = os.path.join(tournament_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            white_player = data.get('white_player', '')
            black_player = data.get('black_player', '')
            
            if not white_player or not black_player:
                print(f"⚠ Skipping {filename}: missing player info")
                continue
            
            # Get matchup name
            matchup = get_matchup_name(white_player, black_player)
            matchup_files[matchup].append(filename)
            
            # Store matchup info (first game we see)
            if matchup not in matchup_info:
                matchup_info[matchup] = {
                    'white': white_player,
                    'black': black_player,
                }
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    # Create directories and organize files
    print(f"\n{'DRY RUN: ' if dry_run else ''}Organizing into {len(matchup_files)} matchups...")
    print("=" * 90)
    
    total_moved = 0
    for matchup, files in sorted(matchup_files.items()):
        matchup_dir = os.path.join(organized_dir, matchup)
        file_count = len(files)
        
        info = matchup_info[matchup]
        print(f"\n{matchup}:")
        print(f"  Players: {info['white']} vs {info['black']}")
        print(f"  Games: {file_count}")
        
        if not dry_run:
            os.makedirs(matchup_dir, exist_ok=True)
            
            # Copy files to matchup directory
            for filename in sorted(files):
                src = os.path.join(tournament_dir, filename)
                dst = os.path.join(matchup_dir, filename)
                shutil.copy2(src, dst)
                total_moved += 1
                
                # Also copy corresponding PGN file if it exists
                pgn_src = src.replace('.json', '.pgn')
                pgn_dst = dst.replace('.json', '.pgn')
                if os.path.exists(pgn_src):
                    shutil.copy2(pgn_src, pgn_dst)
        
        print(f"  {'Would copy' if dry_run else 'Copied'} {file_count} game file(s)")
    
    print("\n" + "=" * 90)
    if not dry_run:
        print(f"✅ Organized {total_moved} game files into {len(matchup_files)} directories")
        print(f"📁 Organized games are in: {organized_dir}")
        print(f"📁 Original files remain in: {tournament_dir}")
    else:
        print(f"✅ Would organize {total_moved} game files into {len(matchup_files)} directories")
        print(f"📁 Would create organized games in: {organized_dir}")
    
    # Print summary
    print("\n" + "=" * 90)
    print("Matchup Summary:")
    print("=" * 90)
    for matchup, files in sorted(matchup_files.items()):
        info = matchup_info[matchup]
        print(f"  {matchup:60} {len(files):3} games")
    
    return organized_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Organize tournament game files into directories by matchup"
    )
    parser.add_argument(
        "tournament_dir",
        nargs="?",
        default="data/tournaments/round_robin_single_10games",
        help="Tournament directory (default: data/tournaments/round_robin_single_10games)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually organizing files"
    )
    
    args = parser.parse_args()
    
    organize_games(args.tournament_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

