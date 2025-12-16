#!/usr/bin/env python3
"""
Show the system and user prompts used for a specific puzzle from results CSV.
"""

import pandas as pd
import sys
from pathlib import Path

def show_prompts(csv_file: str, puzzle_id: str = None, puzzle_index: int = 0):
    """Show prompts for a specific puzzle."""
    df = pd.read_csv(csv_file)
    
    print("="*80)
    print(f"PROMPTS FROM: {Path(csv_file).name}")
    print("="*80)
    print()
    
    # Select puzzle
    if puzzle_id:
        puzzle_row = df[df["PuzzleId"] == puzzle_id]
        if len(puzzle_row) == 0:
            print(f"Error: Puzzle ID '{puzzle_id}' not found")
            return
        row = puzzle_row.iloc[0]
    else:
        if puzzle_index >= len(df):
            print(f"Error: Puzzle index {puzzle_index} out of range (max: {len(df)-1})")
            return
        row = df.iloc[puzzle_index]
        puzzle_id = row["PuzzleId"]
    
    print(f"Puzzle ID: {puzzle_id}")
    print(f"Puzzle Index: {puzzle_index}")
    print()
    
    # Show system prompt
    if "system_prompt" in df.columns:
        system_prompt = row["system_prompt"]
        print("="*80)
        print("SYSTEM PROMPT:")
        print("="*80)
        print(system_prompt)
        print()
    else:
        print("⚠️  System prompt not found in CSV")
        print()
    
    # Show user prompt
    if "user_prompt" in df.columns:
        user_prompt = row["user_prompt"]
        print("="*80)
        print("USER PROMPT (PGN):")
        print("="*80)
        print(user_prompt)
        print()
    else:
        print("⚠️  User prompt not found in CSV")
        print()
    
    # Show additional info
    print("="*80)
    print("ADDITIONAL INFO:")
    print("="*80)
    info_cols = ["PuzzleId", "correct_moves", "total_moves", "puzzle_solved", "single_model_move", "Moves", "Rating"]
    available_cols = [col for col in info_cols if col in df.columns]
    for col in available_cols:
        print(f"  {col}: {row[col]}")
    print()
    
    # Show model response if available
    if "single_model_response" in df.columns:
        response = row["single_model_response"]
        if pd.notna(response) and response:
            print("="*80)
            print("MODEL RESPONSE:")
            print("="*80)
            # Truncate if very long
            if len(str(response)) > 1000:
                print(str(response)[:1000])
                print(f"\n... (truncated, total length: {len(str(response))} characters)")
            else:
                print(response)
            print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 show_prompts.py <csv_file> [puzzle_id] [puzzle_index]")
        print("\nExamples:")
        print("  python3 show_prompts.py data/test_results/test_results_qwen_qwen3-235b-a22b-instruct-2507_single_50.csv")
        print("  python3 show_prompts.py data/test_results/test_results_qwen_qwen3-235b-a22b-instruct-2507_single_50.csv f9PQx")
        print("  python3 show_prompts.py data/test_results/test_results_qwen_qwen3-235b-a22b-instruct-2507_single_50.csv '' 0")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not Path(csv_file).exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    puzzle_id = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    puzzle_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    show_prompts(csv_file, puzzle_id, puzzle_index)

