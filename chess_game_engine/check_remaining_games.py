#!/usr/bin/env python3
"""
Check remaining games needed for tournament completion
"""

import os
import sys
import glob
import json
from collections import defaultdict

def check_remaining_games(tournament_dir: str):
    """Check how many games are remaining"""
    
    if not os.path.isdir(tournament_dir):
        print(f"Error: {tournament_dir} is not a directory")
        sys.exit(1)
    
    # Check if organized directory exists, use it if available
    organized_dir = os.path.join(tournament_dir, "organized")
    json_files = []
    
    if os.path.isdir(organized_dir):
        # Collect all JSON files from organized subdirectories
        for root, dirs, files in os.walk(organized_dir):
            for f in files:
                if f.endswith('.json'):
                    json_files.append(os.path.join(root, f))
    
    # Also check original directory for any games not yet organized (for robustness)
    # This helps catch games that were created after organization
    original_files = [f for f in os.listdir(tournament_dir) 
                     if f.endswith('.json') and f not in ['ratings.json', 'tournament_summary.json']]
    for f in original_files:
        filepath = os.path.join(tournament_dir, f)
        # Only add if not already in organized (check by filename)
        filename = os.path.basename(f)
        already_organized = any(os.path.basename(jf) == filename for jf in json_files)
        if not already_organized:
            json_files.append(filepath)
    
    expected_opponents = [
        'Stockfish',
        'deepseek-ai/deepseek-v3',
        'mistralai/mistral-small-24b-instruct-2501',
        'meta-llama/llama-3.3-70b-instruct'
    ]
    configs = ['single', 'SC', 'plan3', 'SC_plan3']
    
    # Count GPT vs Other Models
    actual_games = defaultdict(int)
    
    for filepath in json_files:
        filename = os.path.basename(filepath)
        if 'gpt-3.5-turbo-instruct' not in filename:
            continue
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            white_player = data.get('white_player', '')
            black_player = data.get('black_player', '')
            
            # Skip GPT vs GPT games for this count
            both_gpt = (white_player and white_player.startswith('gpt-3.5-turbo-instruct') and
                       black_player and black_player.startswith('gpt-3.5-turbo-instruct'))
            if both_gpt:
                continue
            
            # Extract config from player name, JSON configuration field, or filename
            def get_config(player_name: str, json_config: str, filename: str) -> str:
                # First check if player name has config suffix (for GPT vs GPT games)
                if player_name == 'gpt-3.5-turbo-instruct_SC_plan3':
                    return 'SC_plan3'
                elif player_name == 'gpt-3.5-turbo-instruct_plan3':
                    return 'plan3'
                elif player_name == 'gpt-3.5-turbo-instruct_SC':
                    return 'SC'
                elif player_name == 'gpt-3.5-turbo-instruct':
                    # For base name, check JSON configuration field or filename
                    if json_config and json_config != 'single':
                        return json_config
                    # Check filename for config suffix
                    if '_SC_plan3' in filename:
                        return 'SC_plan3'
                    elif '_plan3' in filename and '_SC_' not in filename:
                        return 'plan3'
                    elif '_SC' in filename and '_plan3' not in filename:
                        return 'SC'
                    return 'single'
                return 'single'
            
            # Get configuration from JSON if available
            json_config = data.get('configuration', 'single')
            
            # Determine which player is GPT-3.5
            if white_player.startswith('gpt-3.5-turbo-instruct'):
                config = get_config(white_player, json_config, filename)
                opponent = black_player
            elif black_player.startswith('gpt-3.5-turbo-instruct'):
                config = get_config(black_player, json_config, filename)
                opponent = white_player
            else:
                continue
            
            # Match opponent to expected list
            for opp in expected_opponents:
                if opponent == opp or opponent.replace('/', '_') == opp.replace('/', '_'):
                    key = f'{config}_vs_{opp}'
                    actual_games[key] += 1
                    break
        except Exception as e:
            # Skip files that can't be read
            pass
    
    # Count GPT vs GPT games by matchup (read JSON to get actual player names)
    # Note: Keys are sorted tuples, so ('single', 'SC') and ('SC', 'single') both become ('SC', 'single')
    # When sorting alphabetically: 'SC' < 'plan3' < 'SC_plan3' < 'single'
    # So sorted(['SC', 'plan3']) = ['SC', 'plan3']
    # And sorted(['plan3', 'SC_plan3']) = ['plan3', 'SC_plan3'] (because 'p' < 'S' is False, so 'plan3' < 'SC_plan3' is False)
    # Wait, let me check: 'S' (83) < 'p' (112), so 'SC_plan3' < 'plan3' is True
    # So sorted(['plan3', 'SC_plan3']) = ['SC_plan3', 'plan3']
    gpt_vs_gpt_matchups = {
        ('SC', 'single'): 0,  # sorted(['SC', 'single']) = ['SC', 'single']
        ('plan3', 'single'): 0,  # sorted(['plan3', 'single']) = ['plan3', 'single']
        ('SC_plan3', 'single'): 0,  # sorted(['SC_plan3', 'single']) = ['SC_plan3', 'single']
        ('SC', 'plan3'): 0,  # sorted(['SC', 'plan3']) = ['SC', 'plan3']
        ('SC', 'SC_plan3'): 0,  # sorted(['SC', 'SC_plan3']) = ['SC', 'SC_plan3']
        ('SC_plan3', 'plan3'): 0,  # sorted(['plan3', 'SC_plan3']) = ['SC_plan3', 'plan3'] (S < p)
    }
    
    def extract_config_from_player_name(player_name: str) -> str:
        """Extract configuration from player name"""
        if player_name == 'gpt-3.5-turbo-instruct':
            return 'single'
        elif player_name == 'gpt-3.5-turbo-instruct_SC_plan3':
            return 'SC_plan3'
        elif player_name == 'gpt-3.5-turbo-instruct_plan3':
            return 'plan3'
        elif player_name == 'gpt-3.5-turbo-instruct_SC':
            return 'SC'
        else:
            return 'single'  # Default fallback
    
    for filepath in json_files:
        filename = os.path.basename(filepath)
        if 'gpt-3.5-turbo-instruct' not in filename:
            continue
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            white = data.get('white_player', '')
            black = data.get('black_player', '')
            
            # Check if both are GPT-3.5 (base name or with config suffix)
            if not white.startswith('gpt-3.5-turbo-instruct') or not black.startswith('gpt-3.5-turbo-instruct'):
                continue
            
            # Extract configs from player names
            config1 = extract_config_from_player_name(white)
            config2 = extract_config_from_player_name(black)
            
            # Count this game for the matchup
            matchup = tuple(sorted([config1, config2]))
            if matchup in gpt_vs_gpt_matchups:
                gpt_vs_gpt_matchups[matchup] += 1
        except Exception as e:
            # Skip files that can't be read
            pass
    
    # Print results
    print("=" * 90)
    print("GPT-3.5 vs Other Models")
    print("=" * 90)
    
    total_expected = 0
    total_actual = 0
    missing_list = []
    
    for config in configs:
        for opp in expected_opponents:
            key = f'{config}_vs_{opp}'
            expected = 10
            actual = actual_games.get(key, 0)
            missing = expected - actual
            total_expected += expected
            total_actual += actual
            
            if missing > 0:
                status = f"❌ missing {missing}"
                missing_list.append((config, opp, missing))
            else:
                status = "✅"
            
            print(f"  {config:12} vs {opp:45} {actual:2}/10 {status}")
    
    print("=" * 90)
    print(f"Total GPT vs Others: {total_actual}/{total_expected} (missing {total_expected - total_actual})")
    print()
    
    # GPT vs GPT
    gpt_matchups_expected = 60  # 6 matchups × 10 games
    gpt_vs_gpt_actual = sum(gpt_vs_gpt_matchups.values())
    gpt_vs_gpt_missing = gpt_matchups_expected - gpt_vs_gpt_actual
    
    print("=" * 90)
    print("GPT-3.5 Configurations vs Each Other")
    print("=" * 90)
    # Display in a consistent order (single first, then alphabetical)
    display_order = [
        (('SC', 'single'), 'single', 'SC'),
        (('plan3', 'single'), 'single', 'plan3'),
        (('SC_plan3', 'single'), 'single', 'SC_plan3'),
        (('SC', 'plan3'), 'SC', 'plan3'),
        (('SC', 'SC_plan3'), 'SC', 'SC_plan3'),
        (('SC_plan3', 'plan3'), 'plan3', 'SC_plan3'),  # Note: key is ('SC_plan3', 'plan3') but display as plan3 vs SC_plan3
    ]
    for key, display1, display2 in display_order:
        count = gpt_vs_gpt_matchups.get(key, 0)
        missing = 10 - count
        if missing > 0:
            status = f"❌ missing {missing}"
        else:
            status = "✅"
        print(f"  {display1:12} vs {display2:12} {count:2}/10 {status}")
    print("=" * 90)
    print(f"Total GPT vs GPT: {gpt_vs_gpt_actual}/{gpt_matchups_expected} (missing {gpt_vs_gpt_missing})")
    print()
    
    # Grand total
    grand_total_expected = total_expected + gpt_matchups_expected
    grand_total_actual = total_actual + gpt_vs_gpt_actual
    grand_total_missing = grand_total_expected - grand_total_actual
    
    print("=" * 90)
    print("GRAND TOTAL")
    print("=" * 90)
    print(f"  Expected: {grand_total_expected} games")
    print(f"  Actual: {grand_total_actual} games")
    print(f"  Missing: {grand_total_missing} games ({grand_total_missing/grand_total_expected*100:.1f}%)")
    print("=" * 90)
    
    # Show missing games summary
    if missing_list:
        print()
        print("Missing Games Summary - GPT vs Other Models:")
        print("-" * 90)
        for config, opp, missing in sorted(missing_list):
            print(f"  {config:12} vs {opp:45} missing {missing:2} games")
    
    # Show missing GPT vs GPT games
    gpt_vs_gpt_missing_list = []
    for (config1, config2), count in gpt_vs_gpt_matchups.items():
        missing = 10 - count
        if missing > 0:
            gpt_vs_gpt_missing_list.append((config1, config2, missing))
    
    if gpt_vs_gpt_missing_list:
        print()
        print("Missing Games Summary - GPT vs GPT:")
        print("-" * 90)
        # Sort for display (single first, then alphabetical)
        display_missing = []
        for config1, config2, missing in gpt_vs_gpt_missing_list:
            # Convert key to display format
            if config1 == 'SC_plan3' and config2 == 'plan3':
                display_missing.append(('plan3', 'SC_plan3', missing))
            else:
                display_missing.append((config1, config2, missing))
        for config1, config2, missing in sorted(display_missing):
            print(f"  {config1:12} vs {config2:12} missing {missing:2} games")
    
    return grand_total_missing


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check remaining games needed for tournament completion"
    )
    parser.add_argument(
        "tournament_dir",
        nargs="?",
        default="data/tournaments/round_robin_single_10games",
        help="Tournament directory (default: data/tournaments/round_robin_single_10games)"
    )
    
    args = parser.parse_args()
    
    check_remaining_games(args.tournament_dir)


if __name__ == "__main__":
    main()

