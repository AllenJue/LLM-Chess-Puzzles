#!/usr/bin/env python3
"""
Graph comparison of all paradigms (single, self-consistency, debate) for all models.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

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
    # Check exact match
    if model_name in EXCLUDED_MODELS:
        return True
    # Check variations: gemma free models
    if 'gemma-3-4b-it' in model_lower and ('free' in model_lower or '(free)' in model_lower):
        return True
    if 'gemma-3-12b-it' in model_lower and ('free' in model_lower or '(free)' in model_lower):
        return True
    # Check llama 3.3 8b free
    if 'llama-3.3-8b-instruct' in model_lower and ('free' in model_lower or '(free)' in model_lower):
        return True
    return False

def extract_model_name(filename: str, mode: str) -> str:
    """Extract a clean model name from filename."""
    name = filename.replace("test_results_", "").replace(f"_{mode}_50.csv", "")
    # Clean up model name for display
    # Handle both _free and :free patterns
    name = name.replace("_free", ":free").replace(":free", " (free)")
    name = name.replace("_", "/")
    return name

def load_accuracy_data(csv_file: Path, num_puzzles: int = 50) -> Optional[Dict]:
    """Load and calculate accuracy metrics from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            return None
        
        # Get unique puzzles in order of first appearance
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
        is_single = 'single' in str(csv_file) or ('single_model_prompt_tokens' in puzzle_rows.columns and not is_self_consistency and not is_debate)
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        # Try to find token columns (prioritize paradigm-specific columns)
        if is_self_consistency:
            # For self-consistency, try multiple column options
            if 'self_consistency_total_prompt_tokens' in puzzle_rows.columns:
                total_prompt_tokens = pd.to_numeric(puzzle_rows['self_consistency_total_prompt_tokens'], errors='coerce').fillna(0).sum()
            if 'self_consistency_total_completion_tokens' in puzzle_rows.columns:
                total_completion_tokens = pd.to_numeric(puzzle_rows['self_consistency_total_completion_tokens'], errors='coerce').fillna(0).sum()
            
            # If totals are zero or missing, try generic total columns
            if total_prompt_tokens == 0 and 'total_prompt_tokens' in puzzle_rows.columns:
                total_prompt_tokens = pd.to_numeric(puzzle_rows['total_prompt_tokens'], errors='coerce').fillna(0).sum()
            if total_completion_tokens == 0 and 'total_completion_tokens' in puzzle_rows.columns:
                total_completion_tokens = pd.to_numeric(puzzle_rows['total_completion_tokens'], errors='coerce').fillna(0).sum()
            
            # If still zero, sum individual agent tokens
            if total_prompt_tokens == 0:
                agent_prompt_cols = ['aggressive_prompt_tokens', 'positional_prompt_tokens', 'neutral_prompt_tokens']
                for col in agent_prompt_cols:
                    if col in puzzle_rows.columns:
                        total_prompt_tokens += pd.to_numeric(puzzle_rows[col], errors='coerce').fillna(0).sum()
            
            if total_completion_tokens == 0:
                agent_completion_cols = ['aggressive_completion_tokens', 'positional_completion_tokens', 'neutral_completion_tokens']
                for col in agent_completion_cols:
                    if col in puzzle_rows.columns:
                        total_completion_tokens += pd.to_numeric(puzzle_rows[col], errors='coerce').fillna(0).sum()
        
        elif is_debate:
            # For debate, try debate-specific columns first
            if 'debate_total_prompt_tokens' in puzzle_rows.columns:
                total_prompt_tokens = pd.to_numeric(puzzle_rows['debate_total_prompt_tokens'], errors='coerce').fillna(0).sum()
            if 'debate_total_completion_tokens' in puzzle_rows.columns:
                total_completion_tokens = pd.to_numeric(puzzle_rows['debate_total_completion_tokens'], errors='coerce').fillna(0).sum()
            
            # If totals are zero or missing, try generic total columns
            if total_prompt_tokens == 0 and 'total_prompt_tokens' in puzzle_rows.columns:
                total_prompt_tokens = pd.to_numeric(puzzle_rows['total_prompt_tokens'], errors='coerce').fillna(0).sum()
            if total_completion_tokens == 0 and 'total_completion_tokens' in puzzle_rows.columns:
                total_completion_tokens = pd.to_numeric(puzzle_rows['total_completion_tokens'], errors='coerce').fillna(0).sum()
        
        else:
            # For single model
            if 'single_model_prompt_tokens' in puzzle_rows.columns:
                total_prompt_tokens = pd.to_numeric(puzzle_rows['single_model_prompt_tokens'], errors='coerce').fillna(0).sum()
            if 'single_model_completion_tokens' in puzzle_rows.columns:
                total_completion_tokens = pd.to_numeric(puzzle_rows['single_model_completion_tokens'], errors='coerce').fillna(0).sum()
        
        total_tokens = total_prompt_tokens + total_completion_tokens
        
        # Calculate total moves across all puzzles
        # For each puzzle, sum up the moves attempted (total_moves)
        # We need to aggregate by puzzle to avoid double counting
        total_moves = 0
        if 'total_moves' in puzzle_rows.columns:
            # Group by puzzle and take max (in case there are multiple rows per puzzle)
            puzzle_moves = puzzle_rows.groupby('PuzzleId')['total_moves'].max()
            total_moves = puzzle_moves.sum()
        elif 'correct_moves' in puzzle_rows.columns:
            # Fallback: use correct_moves if total_moves not available
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

def create_comparison_graphs(results_dir: str = "data/test_results", output_dir: str = "data/graphs"):
    """Create comparison graphs for all paradigms."""
    
    results_path = parent_dir / results_dir
    output_path = parent_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data from all three paradigms
    paradigms = {
        'single': results_path / "single_50",
        'self_consistency': results_path / "self_consistency_50",
        'debate': results_path / "debate_50",
    }
    
    # Collect all models and their data
    all_models = set()
    model_data = {}  # {model_name: {paradigm: {accuracy_data, token_data}}}
    
    for paradigm, dir_path in paradigms.items():
        if not dir_path.exists():
            print(f"Warning: {dir_path} does not exist, skipping {paradigm}")
            continue
        
        csv_files = list(dir_path.glob("test_results_*_*.csv"))
        
        for csv_file in csv_files:
            model_name = extract_model_name(csv_file.name, paradigm)
            
            # Skip excluded models
            if should_exclude_model(model_name):
                print(f"Skipping excluded model: {model_name} for {paradigm} paradigm")
                continue
            
            all_models.add(model_name)
            
            if model_name not in model_data:
                model_data[model_name] = {}
            
            # Load accuracy and token data
            accuracy_data = load_accuracy_data(csv_file, num_puzzles=50)
            token_data = load_token_data(csv_file, num_puzzles=50)
            
            if accuracy_data or token_data:
                model_data[model_name][paradigm] = {
                    'accuracy': accuracy_data,
                    'tokens': token_data
                }
    
    if len(all_models) == 0:
        print("No models found")
        return
    
    # Sort models alphabetically
    sorted_models = sorted(all_models)
    
    # Prepare data for plotting
    paradigms_list = ['single', 'self_consistency', 'debate']
    paradigm_labels = ['Single', 'Self-Consistency', 'Debate']
    paradigm_colors = {
        'single': '#3498db',      # Blue
        'self_consistency': '#2ecc71',  # Green
        'debate': '#e74c3c',      # Red
    }
    
    # Graph 1: Puzzle Accuracy Comparison
    fig1, ax1 = plt.subplots(figsize=(12, max(8, len(sorted_models) * 0.4)))
    
    x = np.arange(len(sorted_models))
    width = 0.25
    
    puzzle_accuracies = {p: [] for p in paradigms_list}
    for model in sorted_models:
        for paradigm in paradigms_list:
            if model in model_data and paradigm in model_data[model] and model_data[model][paradigm]['accuracy']:
                puzzle_accuracies[paradigm].append(model_data[model][paradigm]['accuracy']['puzzle_accuracy'])
            else:
                puzzle_accuracies[paradigm].append(0)
    
    bars1 = ax1.barh(x - width, puzzle_accuracies['single'], width, label='Single Model', 
                     color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax1.barh(x, puzzle_accuracies['self_consistency'], width, label='Self-Consistency', 
                     color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars3 = ax1.barh(x + width, puzzle_accuracies['debate'], width, label='Debate', 
                     color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(sorted_models, fontsize=10)
    ax1.set_xlabel('Puzzle Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Puzzle Accuracy by Paradigm', fontsize=15, fontweight='bold', pad=20)
    apply_academic_axes(ax1)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=paradigm_colors['single'], edgecolor='black', linewidth=0.5, label='Single Model'),
        Patch(facecolor=paradigm_colors['self_consistency'], edgecolor='black', linewidth=0.5, label='Self-Consistency'),
        Patch(facecolor=paradigm_colors['debate'], edgecolor='black', linewidth=0.5, label='Debate'),
    ]
    create_academic_legend(ax1, legend_elements)
    
    plt.tight_layout()
    output_file1 = output_path / "paradigm_comparison_puzzle_accuracy.png"
    plt.savefig(output_file1, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig1)
    print(f"✅ Graph saved to: {output_file1}")
    
    # Graph 2: Move Accuracy Comparison
    fig2, ax2 = plt.subplots(figsize=(12, max(8, len(sorted_models) * 0.4)))
    
    move_accuracies = {p: [] for p in paradigms_list}
    for model in sorted_models:
        for paradigm in paradigms_list:
            if model in model_data and paradigm in model_data[model] and model_data[model][paradigm]['accuracy']:
                move_accuracies[paradigm].append(model_data[model][paradigm]['accuracy']['move_accuracy'])
            else:
                move_accuracies[paradigm].append(0)
    
    bars1 = ax2.barh(x - width, move_accuracies['single'], width, label='Single Model', 
                     color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax2.barh(x, move_accuracies['self_consistency'], width, label='Self-Consistency', 
                     color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars3 = ax2.barh(x + width, move_accuracies['debate'], width, label='Debate', 
                     color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(sorted_models, fontsize=10)
    ax2.set_xlabel('Move Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Move Accuracy by Paradigm', fontsize=15, fontweight='bold', pad=20)
    apply_academic_axes(ax2)
    # Reuse legend elements for consistency
    from matplotlib.patches import Patch
    legend_elements_2 = [
        Patch(facecolor=paradigm_colors['single'], edgecolor='black', linewidth=0.5, label='Single Model'),
        Patch(facecolor=paradigm_colors['self_consistency'], edgecolor='black', linewidth=0.5, label='Self-Consistency'),
        Patch(facecolor=paradigm_colors['debate'], edgecolor='black', linewidth=0.5, label='Debate'),
    ]
    create_academic_legend(ax2, legend_elements_2)
    
    plt.tight_layout()
    output_file2 = output_path / "paradigm_comparison_move_accuracy.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig2)
    print(f"✅ Graph saved to: {output_file2}")
    
    # Graph 3: Token Usage Per Move Comparison (stacked)
    fig3, ax3 = plt.subplots(figsize=(12, max(8, len(sorted_models) * 0.4)))
    
    # Prepare token per move data
    prompt_tokens_per_move = {p: [] for p in paradigms_list}
    completion_tokens_per_move = {p: [] for p in paradigms_list}
    total_tokens_per_move = {p: [] for p in paradigms_list}
    
    for model in sorted_models:
        for paradigm in paradigms_list:
            if model in model_data and paradigm in model_data[model] and model_data[model][paradigm]['tokens']:
                token_data = model_data[model][paradigm]['tokens']
                prompt_tokens_per_move[paradigm].append(token_data.get('tokens_per_move_prompt', 0))
                completion_tokens_per_move[paradigm].append(token_data.get('tokens_per_move_completion', 0))
                total_tokens_per_move[paradigm].append(token_data.get('tokens_per_move_total', 0))
            else:
                prompt_tokens_per_move[paradigm].append(0)
                completion_tokens_per_move[paradigm].append(0)
                total_tokens_per_move[paradigm].append(0)
    
    # Create grouped stacked bars
    for i, paradigm in enumerate(paradigms_list):
        offset = (i - 1) * width
        ax3.barh(x + offset, prompt_tokens_per_move[paradigm], width, 
                label=f'{paradigm_labels[i]} (Prompt)', 
                color=['#3498db', '#2ecc71', '#e74c3c'][i], alpha=0.7, 
                edgecolor='black', linewidth=0.3)
        ax3.barh(x + offset, completion_tokens_per_move[paradigm], width, 
                left=prompt_tokens_per_move[paradigm],
                label=f'{paradigm_labels[i]} (Completion)', 
                color=['#2980b9', '#27ae60', '#c0392b'][i], alpha=0.9,
                edgecolor='black', linewidth=0.3)
    
    ax3.set_yticks(x)
    ax3.set_yticklabels(sorted_models, fontsize=10)
    ax3.set_xlabel('Tokens per Move', fontsize=13, fontweight='bold')
    ax3.set_title('Token Usage by Paradigm', fontsize=15, fontweight='bold', pad=20)
    apply_academic_axes(ax3)
    # Create custom legend
    from matplotlib.patches import Patch
    token_legend_elements = [
        Patch(facecolor='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5, label='Single (Prompt)'),
        Patch(facecolor='#2980b9', alpha=0.9, edgecolor='black', linewidth=0.5, label='Single (Completion)'),
        Patch(facecolor='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5, label='Self-Consistency (Prompt)'),
        Patch(facecolor='#27ae60', alpha=0.9, edgecolor='black', linewidth=0.5, label='Self-Consistency (Completion)'),
        Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5, label='Debate (Prompt)'),
        Patch(facecolor='#c0392b', alpha=0.9, edgecolor='black', linewidth=0.5, label='Debate (Completion)'),
    ]
    create_academic_legend(ax3, token_legend_elements, fontsize=10)
    
    # Add value labels on bars
    # Find max value across all paradigms for spacing
    max_tokens_per_move = max(
        max(total_tokens_per_move[p]) if total_tokens_per_move[p] else 0 
        for p in paradigms_list
    )
    
    for i, paradigm in enumerate(paradigms_list):
        offset = (i - 1) * width
        for j, (prompt, completion, total) in enumerate(zip(
            prompt_tokens_per_move[paradigm],
            completion_tokens_per_move[paradigm],
            total_tokens_per_move[paradigm]
        )):
            if total > 0:
                # Label on the right side showing total tokens per move
                ax3.text(total + max_tokens_per_move * 0.01, 
                        j + offset, f'{total:.0f}', 
                        va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    output_file3 = output_path / "paradigm_comparison_tokens.png"
    plt.savefig(output_file3, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig3)
    print(f"✅ Graph saved to: {output_file3}")
    
    # Print summary
    print("\n" + "="*80)
    print("PARADIGM COMPARISON SUMMARY (First 50 Puzzles)")
    print("="*80)
    print(f"{'Model':<50} {'Single Acc':<15} {'SC Acc':<15} {'Debate Acc':<15} {'Single Tok/Move':<18} {'SC Tok/Move':<18} {'Debate Tok/Move':<18}")
    print("-"*140)
    for model in sorted_models:
        single_acc = "N/A"
        sc_acc = "N/A"
        debate_acc = "N/A"
        single_tpm = "N/A"
        sc_tpm = "N/A"
        debate_tpm = "N/A"
        
        if model in model_data and 'single' in model_data[model] and model_data[model]['single']['accuracy']:
            single_acc = f"{model_data[model]['single']['accuracy']['puzzle_accuracy']:.1f}%"
        if model in model_data and 'self_consistency' in model_data[model] and model_data[model]['self_consistency']['accuracy']:
            sc_acc = f"{model_data[model]['self_consistency']['accuracy']['puzzle_accuracy']:.1f}%"
        if model in model_data and 'debate' in model_data[model] and model_data[model]['debate']['accuracy']:
            debate_acc = f"{model_data[model]['debate']['accuracy']['puzzle_accuracy']:.1f}%"
        
        if model in model_data and 'single' in model_data[model] and model_data[model]['single']['tokens']:
            tpm = model_data[model]['single']['tokens'].get('tokens_per_move_total', 0)
            single_tpm = f"{tpm:.0f}" if tpm > 0 else "N/A"
        if model in model_data and 'self_consistency' in model_data[model] and model_data[model]['self_consistency']['tokens']:
            tpm = model_data[model]['self_consistency']['tokens'].get('tokens_per_move_total', 0)
            sc_tpm = f"{tpm:.0f}" if tpm > 0 else "N/A"
        if model in model_data and 'debate' in model_data[model] and model_data[model]['debate']['tokens']:
            tpm = model_data[model]['debate']['tokens'].get('tokens_per_move_total', 0)
            debate_tpm = f"{tpm:.0f}" if tpm > 0 else "N/A"
        
        print(f"{model:<50} {single_acc:<15} {sc_acc:<15} {debate_acc:<15} {single_tpm:<18} {sc_tpm:<18} {debate_tpm:<18}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Graph comparison of all paradigms")
    parser.add_argument("--results-dir", type=str, default="data/test_results",
                       help="Directory containing result CSV files")
    parser.add_argument("--output-dir", type=str, default="data/graphs",
                       help="Directory to save output graphs")
    
    args = parser.parse_args()
    create_comparison_graphs(args.results_dir, args.output_dir)

