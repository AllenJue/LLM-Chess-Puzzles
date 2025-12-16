#!/usr/bin/env python3
"""
Create sorted comparison graphs for accuracy and tokens across all models and paradigms.
Graphs are sorted by decreasing values and clearly show model + paradigm.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import academic style
from graph_style import set_academic_style, apply_academic_axes, create_academic_legend

# Set academic style
set_academic_style()

# Models to exclude (failed models or API issues)
EXCLUDED_MODELS = [
    "google/gemma-3-4b-it (free)",
    "google/gemma-3-12b-it (free)",
    "google/gemma-3-4b-it:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "meta-llama/llama-3.3-8b-instruct (free)",
]

def should_exclude_model(model_name: str) -> bool:
    """Check if a model should be excluded (handles variations)."""
    model_lower = model_name.lower()
    if model_name in EXCLUDED_MODELS:
        return True
    if 'gemma-3-4b-it' in model_lower and ('free' in model_lower or '(free)' in model_lower):
        return True
    if 'gemma-3-12b-it' in model_lower and ('free' in model_lower or '(free)' in model_lower):
        return True
    if 'llama-3.3-8b-instruct' in model_lower and ('free' in model_lower or '(free)' in model_lower):
        return True
    return False

def extract_model_name(filename: str, mode: str) -> str:
    """Extract a clean model name from filename."""
    name = filename.replace("test_results_", "").replace(f"_{mode}_50.csv", "")
    # Handle both _free and :free patterns BEFORE replacing other underscores
    name = name.replace("_free", ":free").replace(":free", " (free)")
    name = name.replace("_", "/")
    return name

def load_accuracy_data(csv_file: Path, num_puzzles: int = 50) -> Optional[Dict]:
    """Load and calculate accuracy metrics from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            return None
        
        unique_puzzles = df['PuzzleId'].drop_duplicates().head(num_puzzles).tolist()
        actual_num_puzzles = len(unique_puzzles)
        
        if actual_num_puzzles == 0:
            return None
        
        puzzle_solved_count = 0
        total_correct_moves = 0
        total_moves = 0
        
        for puzzle_id in unique_puzzles:
            puzzle_rows = df[df['PuzzleId'] == puzzle_id]
            if puzzle_rows['puzzle_solved'].any():
                puzzle_solved_count += 1
            
            puzzle_correct = puzzle_rows['correct_moves'].max() if 'correct_moves' in puzzle_rows.columns else 0
            puzzle_total = puzzle_rows['total_moves'].max() if 'total_moves' in puzzle_rows.columns else 0
            
            total_correct_moves += puzzle_correct
            total_moves += puzzle_total
        
        puzzle_accuracy = (puzzle_solved_count / actual_num_puzzles * 100) if actual_num_puzzles > 0 else 0
        move_accuracy = (total_correct_moves / total_moves * 100) if total_moves > 0 else 0
        
        return {
            'puzzle_accuracy': puzzle_accuracy,
            'move_accuracy': move_accuracy,
            'puzzles_solved': puzzle_solved_count,
            'total_puzzles': actual_num_puzzles,
            'correct_moves': total_correct_moves,
            'total_moves': total_moves
        }
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")
        return None

def load_token_data(csv_file: Path, num_puzzles: int = 50) -> Optional[Dict]:
    """Load and calculate token usage from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            return None
        
        unique_puzzles = df['PuzzleId'].drop_duplicates().head(num_puzzles).tolist()
        actual_num_puzzles = len(unique_puzzles)
        
        if actual_num_puzzles == 0:
            return None
        
        puzzle_rows = df[df['PuzzleId'].isin(unique_puzzles)]
        
        # Determine paradigm from file path or column names
        is_self_consistency = 'self_consistency' in str(csv_file) or 'self_consistency_total_prompt_tokens' in puzzle_rows.columns
        is_debate = 'debate' in str(csv_file) or 'debate_total_prompt_tokens' in puzzle_rows.columns
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        if is_self_consistency:
            if 'self_consistency_total_prompt_tokens' in puzzle_rows.columns:
                total_prompt_tokens = pd.to_numeric(puzzle_rows['self_consistency_total_prompt_tokens'], errors='coerce').fillna(0).sum()
            elif 'total_prompt_tokens' in puzzle_rows.columns:
                total_prompt_tokens = pd.to_numeric(puzzle_rows['total_prompt_tokens'], errors='coerce').fillna(0).sum()
            else:
                agent_prompt_cols = ['aggressive_prompt_tokens', 'positional_prompt_tokens', 'neutral_prompt_tokens']
                for col in agent_prompt_cols:
                    if col in puzzle_rows.columns:
                        total_prompt_tokens += pd.to_numeric(puzzle_rows[col], errors='coerce').fillna(0).sum()
            
            if 'self_consistency_total_completion_tokens' in puzzle_rows.columns:
                total_completion_tokens = pd.to_numeric(puzzle_rows['self_consistency_total_completion_tokens'], errors='coerce').fillna(0).sum()
            elif 'total_completion_tokens' in puzzle_rows.columns:
                total_completion_tokens = pd.to_numeric(puzzle_rows['total_completion_tokens'], errors='coerce').fillna(0).sum()
            else:
                agent_completion_cols = ['aggressive_completion_tokens', 'positional_completion_tokens', 'neutral_completion_tokens']
                for col in agent_completion_cols:
                    if col in puzzle_rows.columns:
                        total_completion_tokens += pd.to_numeric(puzzle_rows[col], errors='coerce').fillna(0).sum()
        
        elif is_debate:
            if 'debate_total_prompt_tokens' in puzzle_rows.columns:
                total_prompt_tokens = pd.to_numeric(puzzle_rows['debate_total_prompt_tokens'], errors='coerce').fillna(0).sum()
            elif 'total_prompt_tokens' in puzzle_rows.columns:
                total_prompt_tokens = pd.to_numeric(puzzle_rows['total_prompt_tokens'], errors='coerce').fillna(0).sum()
            
            if 'debate_total_completion_tokens' in puzzle_rows.columns:
                total_completion_tokens = pd.to_numeric(puzzle_rows['debate_total_completion_tokens'], errors='coerce').fillna(0).sum()
            elif 'total_completion_tokens' in puzzle_rows.columns:
                total_completion_tokens = pd.to_numeric(puzzle_rows['total_completion_tokens'], errors='coerce').fillna(0).sum()
        
        else:
            if 'single_model_prompt_tokens' in puzzle_rows.columns:
                total_prompt_tokens = pd.to_numeric(puzzle_rows['single_model_prompt_tokens'], errors='coerce').fillna(0).sum()
            if 'single_model_completion_tokens' in puzzle_rows.columns:
                total_completion_tokens = pd.to_numeric(puzzle_rows['single_model_completion_tokens'], errors='coerce').fillna(0).sum()
        
        total_tokens = total_prompt_tokens + total_completion_tokens
        
        # Calculate total moves
        total_moves = 0
        if 'total_moves' in puzzle_rows.columns:
            puzzle_moves = puzzle_rows.groupby('PuzzleId')['total_moves'].max()
            total_moves = puzzle_moves.sum()
        elif 'correct_moves' in puzzle_rows.columns:
            puzzle_moves = puzzle_rows.groupby('PuzzleId')['correct_moves'].max()
            total_moves = puzzle_moves.sum()
        
        # Calculate tokens per move
        tokens_per_move_prompt = (total_prompt_tokens / total_moves) if total_moves > 0 else 0
        tokens_per_move_completion = (total_completion_tokens / total_moves) if total_moves > 0 else 0
        tokens_per_move_total = (total_tokens / total_moves) if total_moves > 0 else 0
        
        return {
            'total_prompt_tokens': int(total_prompt_tokens),
            'total_completion_tokens': int(total_completion_tokens),
            'total_tokens': int(total_tokens),
            'total_moves': int(total_moves),
            'tokens_per_move_prompt': tokens_per_move_prompt,
            'tokens_per_move_completion': tokens_per_move_completion,
            'tokens_per_move_total': tokens_per_move_total,
        }
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")
        return None

def create_sorted_graphs(results_dir: str = "data/test_results", output_dir: str = "data/graphs"):
    """Create sorted comparison graphs for accuracy and tokens."""
    
    results_path = parent_dir / results_dir
    output_path = parent_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data from all three paradigms
    paradigms = {
        'single': results_path / "single_50",
        'self_consistency': results_path / "self_consistency_50",
        'debate': results_path / "debate_50",
    }
    
    paradigm_display_names = {
        'single': 'Single',
        'self_consistency': 'Self-Consistency',
        'debate': 'Debate',
    }
    
    paradigm_colors = {
        'single': '#3498db',      # Blue
        'self_consistency': '#2ecc71',  # Green
        'debate': '#e74c3c',      # Red
    }
    
    # Collect all data
    all_data = []  # List of dicts: {model, paradigm, puzzle_accuracy, move_accuracy, tokens_per_move}
    
    for paradigm, dir_path in paradigms.items():
        if not dir_path.exists():
            print(f"Warning: {dir_path} does not exist, skipping {paradigm}")
            continue
        
        csv_files = list(dir_path.glob("test_results_*_*.csv"))
        
        for csv_file in csv_files:
            model_name = extract_model_name(csv_file.name, paradigm)
            
            if should_exclude_model(model_name):
                continue
            
            # Load accuracy
            accuracy_data = load_accuracy_data(csv_file, num_puzzles=50)
            if not accuracy_data:
                continue
            
            # Load tokens
            token_data = load_token_data(csv_file, num_puzzles=50)
            if not token_data:
                continue
            
            all_data.append({
                'model': model_name,
                'paradigm': paradigm_display_names[paradigm],
                'puzzle_accuracy': accuracy_data['puzzle_accuracy'],
                'move_accuracy': accuracy_data['move_accuracy'],
                'tokens_per_move': token_data['tokens_per_move_total'],
            })
    
    if not all_data:
        print("No data found to graph.")
        return
    
    df = pd.DataFrame(all_data)
    
    # Create label: "Model (Paradigm)"
    df['label'] = df['model'] + ' (' + df['paradigm'] + ')'
    
    # Sort by puzzle accuracy (descending - highest first)
    df_sorted_accuracy = df.sort_values('puzzle_accuracy', ascending=False).reset_index(drop=True)
    
    # Sort by tokens per move (descending - highest first)
    df_sorted_tokens = df.sort_values('tokens_per_move', ascending=False).reset_index(drop=True)
    
    # Graph 1: Puzzle Accuracy (sorted by accuracy, highest at top)
    # Calculate figure height based on number of entries
    num_entries = len(df_sorted_accuracy)
    fig_height = max(10, num_entries * 0.4)
    fig1, ax1 = plt.subplots(figsize=(12, fig_height))
    
    # Map paradigm display names to color keys
    paradigm_to_key = {
        'Single': 'single',
        'Self-Consistency': 'self_consistency',
        'Debate': 'debate',
    }
    colors = [paradigm_colors.get(paradigm_to_key.get(p, ''), '#95a5a6') for p in df_sorted_accuracy['paradigm']]
    bars1 = ax1.barh(range(len(df_sorted_accuracy)), df_sorted_accuracy['puzzle_accuracy'], 
                     color=colors, edgecolor='black', linewidth=0.5, height=0.7)
    
    ax1.set_yticks(range(len(df_sorted_accuracy)))
    ax1.set_yticklabels(df_sorted_accuracy['label'], fontsize=10)
    ax1.invert_yaxis()  # Highest values at top
    ax1.set_xlabel('Puzzle Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Puzzle Accuracy by Model and Paradigm', fontsize=15, fontweight='bold', pad=20)
    max_acc = max(df_sorted_accuracy['puzzle_accuracy']) if max(df_sorted_accuracy['puzzle_accuracy']) > 0 else 10
    ax1.set_xlim(0, max_acc * 1.15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels with better positioning
    for i, (bar, acc) in enumerate(zip(bars1, df_sorted_accuracy['puzzle_accuracy'])):
        if acc > 0:
            label_x = acc + max_acc * 0.02
            ax1.text(label_x, i, f'{acc:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    # Add legend with better styling
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=paradigm_colors['single'], edgecolor='black', linewidth=0.5, label='Single Model'),
        Patch(facecolor=paradigm_colors['self_consistency'], edgecolor='black', linewidth=0.5, label='Self-Consistency'),
        Patch(facecolor=paradigm_colors['debate'], edgecolor='black', linewidth=0.5, label='Debate'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=True, 
               fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    output_file1 = output_path / "sorted_puzzle_accuracy.png"
    plt.savefig(output_file1, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig1)
    print(f"Graph saved to: {output_file1}")
    
    # Graph 2: Tokens per Move (sorted by tokens)
    num_entries2 = len(df_sorted_tokens)
    fig_height2 = max(10, num_entries2 * 0.4)
    fig2, ax2 = plt.subplots(figsize=(12, fig_height2))
    
    colors2 = [paradigm_colors.get(paradigm_to_key.get(p, ''), '#95a5a6') for p in df_sorted_tokens['paradigm']]
    bars2 = ax2.barh(range(len(df_sorted_tokens)), df_sorted_tokens['tokens_per_move'], 
                     color=colors2, edgecolor='black', linewidth=0.5, height=0.7)
    
    ax2.set_yticks(range(len(df_sorted_tokens)))
    ax2.set_yticklabels(df_sorted_tokens['label'], fontsize=10)
    ax2.invert_yaxis()  # Highest values at top
    ax2.set_xlabel('Tokens per Move', fontsize=13, fontweight='bold')
    ax2.set_title('Token Usage by Model and Paradigm', fontsize=15, fontweight='bold', pad=20)
    max_tokens = max(df_sorted_tokens['tokens_per_move']) if max(df_sorted_tokens['tokens_per_move']) > 0 else 100
    ax2.set_xlim(0, max_tokens * 1.15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add value labels with better positioning
    for i, (bar, tokens) in enumerate(zip(bars2, df_sorted_tokens['tokens_per_move'])):
        if tokens > 0:
            label_x = tokens + max_tokens * 0.02
            ax2.text(label_x, i, f'{tokens:.0f}', va='center', fontsize=9, fontweight='bold')
    
    # Add legend
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=True, 
               fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    output_file2 = output_path / "sorted_tokens_per_move.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig2)
    print(f"Graph saved to: {output_file2}")
    
    # Print summary
    print("\n" + "="*100)
    print("SORTED RESULTS SUMMARY")
    print("="*100)
    print(f"\nTop 10 by Puzzle Accuracy:")
    print(f"{'Model + Paradigm':<60} {'Puzzle Acc':<15} {'Move Acc':<15} {'Tokens/Move':<15}")
    print("-"*100)
    for idx, row in df_sorted_accuracy.head(10).iterrows():
        print(f"{row['label']:<60} {row['puzzle_accuracy']:>13.1f}% {row['move_accuracy']:>13.1f}% {row['tokens_per_move']:>13.0f}")
    
    print(f"\nTop 10 by Tokens per Move (most expensive):")
    print(f"{'Model + Paradigm':<60} {'Tokens/Move':<15} {'Puzzle Acc':<15} {'Move Acc':<15}")
    print("-"*100)
    for idx, row in df_sorted_tokens.head(10).iterrows():
        print(f"{row['label']:<60} {row['tokens_per_move']:>13.0f} {row['puzzle_accuracy']:>13.1f}% {row['move_accuracy']:>13.1f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create sorted comparison graphs")
    parser.add_argument("--results-dir", type=str, default="data/test_results",
                       help="Directory containing result CSV files")
    parser.add_argument("--output-dir", type=str, default="data/graphs",
                       help="Directory to save output graphs")
    
    args = parser.parse_args()
    create_sorted_graphs(args.results_dir, args.output_dir)

