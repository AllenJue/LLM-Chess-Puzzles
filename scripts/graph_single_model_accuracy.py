#!/usr/bin/env python3
"""
Graph accuracy comparison for all single model results.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import academic style
from graph_style import set_academic_style, apply_academic_axes, create_academic_legend

# Set academic style
set_academic_style()

def extract_model_name(filename: str) -> str:
    """Extract a clean model name from filename."""
    name = filename.replace("test_results_", "").replace("_single_50.csv", "")
    # Clean up model name for display
    # Handle both _free and :free patterns
    name = name.replace("_free", ":free").replace(":free", " (free)")
    name = name.replace("_", "/")
    return name

def load_and_calculate_accuracy(csv_file: Path, num_puzzles: int = 50) -> Dict[str, float]:
    """Load a CSV and calculate accuracy metrics for the first N puzzles.
    
    Args:
        csv_file: Path to the CSV file
        num_puzzles: Number of puzzles to consider (default: 50)
    
    Returns:
        Dictionary with accuracy metrics
    """
    try:
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            return None
        
        # Get unique puzzles in order of first appearance
        # This preserves the order from the original puzzle list
        unique_puzzles = df['PuzzleId'].drop_duplicates().head(num_puzzles).tolist()
        actual_num_puzzles = len(unique_puzzles)
        
        if actual_num_puzzles == 0:
            return None
        
        # Calculate puzzle-level accuracy (puzzle_solved) and total moves
        puzzle_solved_count = 0
        total_correct_moves = 0
        total_moves = 0
        
        for puzzle_id in unique_puzzles:
            puzzle_rows = df[df['PuzzleId'] == puzzle_id]
            
            # Check if any row for this puzzle shows it was solved
            if puzzle_rows['puzzle_solved'].any():
                puzzle_solved_count += 1
            
            # Sum up correct_moves and total_moves for this puzzle
            # Take the max since they should be the same across rows for same puzzle
            puzzle_correct = puzzle_rows['correct_moves'].max() if 'correct_moves' in puzzle_rows.columns else 0
            puzzle_total = puzzle_rows['total_moves'].max() if 'total_moves' in puzzle_rows.columns else 0
            
            total_correct_moves += puzzle_correct
            total_moves += puzzle_total
        
        # Calculate accuracies
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
        import traceback
        traceback.print_exc()
        return None

def create_accuracy_graph(results_dir: str = "data/test_results", output_dir: str = "data/graphs"):
    """Create accuracy comparison graph for all single model results."""
    
    results_path = parent_dir / results_dir
    output_path = parent_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all single model CSV files (check both root and single_50 subdirectory)
    csv_files = list(results_path.glob("test_results_*_single_50.csv"))
    single_50_dir = results_path / "single_50"
    if single_50_dir.exists():
        csv_files.extend(list(single_50_dir.glob("test_results_*_single_50.csv")))
    
    if len(csv_files) == 0:
        print(f"No single model result files found in {results_path}")
        return
    
    print(f"Found {len(csv_files)} single model result files")
    
    # Models to exclude (failed models)
    excluded_models = [
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
        if model_name in excluded_models:
            return True
        if 'gemma-3-4b-it' in model_lower and ('free' in model_lower or '(free)' in model_lower):
            return True
        if 'gemma-3-12b-it' in model_lower and ('free' in model_lower or '(free)' in model_lower):
            return True
        if 'llama-3.3-8b-instruct' in model_lower and ('free' in model_lower or '(free)' in model_lower):
            return True
        return False
    
    # Load and calculate accuracy for each model (first 50 puzzles)
    model_results = {}
    for csv_file in csv_files:
        model_name = extract_model_name(csv_file.name)
        # Skip excluded models
        if should_exclude_model(model_name):
            print(f"Skipping excluded model: {model_name}")
            continue
        accuracy_data = load_and_calculate_accuracy(csv_file, num_puzzles=50)
        if accuracy_data:
            model_results[model_name] = accuracy_data
            print(f"{model_name}: {accuracy_data['puzzles_solved']}/{accuracy_data['total_puzzles']} puzzles solved ({accuracy_data['puzzle_accuracy']:.1f}%), {accuracy_data['correct_moves']}/{accuracy_data['total_moves']} moves correct ({accuracy_data['move_accuracy']:.1f}%)")
    
    if len(model_results) == 0:
        print("No valid results found")
        return
    
    # Sort models by puzzle accuracy (descending)
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['puzzle_accuracy'], reverse=True)
    
    # Prepare data for plotting
    model_names = [name for name, _ in sorted_models]
    puzzle_accuracies = [data['puzzle_accuracy'] for _, data in sorted_models]
    move_accuracies = [data['move_accuracy'] for _, data in sorted_models]
    
    # Color palette
    colors = sns.color_palette("husl", len(model_names))
    
    # Plot 1: Puzzle Accuracy (Puzzles Solved %) - Separate file
    fig1, ax1 = plt.subplots(figsize=(12, max(8, len(model_names) * 0.4)))
    bars1 = ax1.barh(range(len(model_names)), puzzle_accuracies, color=colors, 
                     edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels(model_names, fontsize=10)
    ax1.set_xlabel('Puzzle Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Puzzle Accuracy by Model', fontsize=15, fontweight='bold', pad=20)
    ax1.set_xlim(0, max(puzzle_accuracies) * 1.15 if max(puzzle_accuracies) > 0 else 10)
    apply_academic_axes(ax1)
    
    # Add value labels on bars
    max_acc = max(puzzle_accuracies) if max(puzzle_accuracies) > 0 else 10
    for i, (bar, acc) in enumerate(zip(bars1, puzzle_accuracies)):
        if acc > 0:
            ax1.text(acc + max_acc * 0.02, i, f'{acc:.1f}%', 
                    va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file1 = output_path / "single_model_puzzle_accuracy.png"
    plt.savefig(output_file1, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig1)
    print(f"✅ Graph saved to: {output_file1}")
    
    # Plot 2: Move Accuracy (Correct Moves %) - Separate file
    fig2, ax2 = plt.subplots(figsize=(12, max(8, len(model_names) * 0.4)))
    bars2 = ax2.barh(range(len(model_names)), move_accuracies, color=colors, 
                     edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(model_names)))
    ax2.set_yticklabels(model_names, fontsize=10)
    ax2.set_xlabel('Move Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Move Accuracy by Model', fontsize=15, fontweight='bold', pad=20)
    ax2.set_xlim(0, max(move_accuracies) * 1.15 if max(move_accuracies) > 0 else 10)
    apply_academic_axes(ax2)
    
    # Add value labels on bars
    max_move_acc = max(move_accuracies) if max(move_accuracies) > 0 else 10
    for i, (bar, acc) in enumerate(zip(bars2, move_accuracies)):
        if acc > 0:
            ax2.text(acc + max_move_acc * 0.02, i, f'{acc:.1f}%', 
                    va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file2 = output_path / "single_model_move_accuracy.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig2)
    print(f"✅ Graph saved to: {output_file2}")
    
    # Also create a summary table
    print("\n" + "="*80)
    print("ACCURACY SUMMARY (First 50 Puzzles)")
    print("="*80)
    print(f"{'Model':<50} {'Puzzles Solved':<25} {'Move Accuracy':<25}")
    print("-"*80)
    for name, data in sorted_models:
        puzzles_info = f"{data['puzzles_solved']}/{data['total_puzzles']} ({data['puzzle_accuracy']:.1f}%)"
        move_info = f"{data['correct_moves']}/{data['total_moves']} ({data['move_accuracy']:.1f}%)"
        print(f"{name:<50} {puzzles_info:<25} {move_info:<25}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Graph accuracy for single model results")
    parser.add_argument("--results-dir", type=str, default="data/test_results",
                       help="Directory containing result CSV files")
    parser.add_argument("--output-dir", type=str, default="data/graphs",
                       help="Directory to save output graphs")
    
    args = parser.parse_args()
    create_accuracy_graph(args.results_dir, args.output_dir)

