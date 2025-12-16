#!/usr/bin/env python3
"""
Create stacked bar chart comparing errors across paradigms (single, self-consistency, debate).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
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
    # Clean up model name for display
    name = name.replace("_free", ":free").replace(":free", " (free)")
    name = name.replace("_", "/")
    return name

def categorize_error(error_str: str) -> str:
    """Categorize an error string into error types.
    
    Note: "No legal move generated" includes cases where:
    - Model returned None (couldn't parse SAN)
    - Model generated an illegal move
    - Failed to extract a valid move from response
    """
    if pd.isna(error_str) or not error_str or not str(error_str).strip():
        return None
    
    error_lower = str(error_str).lower()
    
    if 'broken pipe' in error_lower or 'errno 32' in error_lower:
        return 'API connection error'
    elif 'failed to extract' in error_lower:
        return 'Failed to extract move'
    elif 'no legal move generated' in error_lower:
        return 'No legal move generated'
    elif 'mismatch' in error_lower:
        # Check if it's a "got None" case - couldn't parse SAN, so no legal move
        if 'got none' in error_lower or 'but got none' in error_lower:
            return 'No legal move generated'
        return 'Move mismatch'
    elif 'illegal' in error_lower or 'invalid' in error_lower:
        return 'No legal move generated'
    elif 'api' in error_lower or 'error calling' in error_lower:
        return 'API error'
    elif 'timeout' in error_lower:
        return 'Timeout'
    elif 'none' in error_lower and ('got' in error_lower or 'returned' in error_lower):
        return 'No legal move generated'
    else:
        return 'Other'

def analyze_errors_by_paradigm(csv_file: Path, paradigm: str, num_puzzles: int = 50) -> Dict:
    """Analyze errors from a CSV file for a specific paradigm."""
    try:
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            return None
        
        # Get unique puzzles in order of first appearance
        unique_puzzles = df['PuzzleId'].drop_duplicates().head(num_puzzles).tolist()
        actual_num_puzzles = len(unique_puzzles)
        
        if actual_num_puzzles == 0:
            return None
        
        # Collect errors by type
        error_type_counts = Counter()
        puzzles_with_errors = 0
        
        for puzzle_id in unique_puzzles:
            puzzle_rows = df[df['PuzzleId'] == puzzle_id]
            # Get the first row for this puzzle (should have the error if any)
            first_row = puzzle_rows.iloc[0]
            
            error = first_row.get('error', '')
            error_type = categorize_error(error)
            
            if error_type:
                error_type_counts[error_type] += 1
                puzzles_with_errors += 1
        
        return {
            'total_puzzles': actual_num_puzzles,
            'puzzles_with_errors': puzzles_with_errors,
            'error_type_counts': dict(error_type_counts),
        }
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_error_comparison_graph(results_dir: str = "data/test_results", output_dir: str = "data/graphs"):
    """Create horizontal stacked bar chart comparing errors per model across paradigms."""
    
    results_path = parent_dir / results_dir
    output_path = parent_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data from all three paradigms
    paradigms = {
        'single': results_path / "single_50",
        'self_consistency': results_path / "self_consistency_50",
        'debate': results_path / "debate_50",
    }
    
    # Collect all models and their error data
    all_models = set()
    model_error_data = {}  # {model_name: {paradigm: error_data}}
    
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
            
            if model_name not in model_error_data:
                model_error_data[model_name] = {}
            
            # Analyze errors
            error_data = analyze_errors_by_paradigm(csv_file, paradigm, num_puzzles=50)
            
            if error_data:
                model_error_data[model_name][paradigm] = error_data
    
    if len(all_models) == 0:
        print("No models found")
        return
    
    # Sort models alphabetically
    sorted_models = sorted(all_models)
    
    # Collect all error types across all models and paradigms
    all_error_types = set()
    for model_data in model_error_data.values():
        for paradigm_data in model_data.values():
            all_error_types.update(paradigm_data.get('error_type_counts', {}).keys())
    all_error_types = sorted(list(all_error_types))
    
    # Prepare data for stacked bar chart
    paradigms_list = ['single', 'self_consistency', 'debate']
    paradigm_labels = ['Single', 'Self-Consistency', 'Debate']
    
    # Color palette for error types
    error_colors = {
        'API connection error': '#8e44ad',         # Purple (broken pipe, connection issues)
        'Failed to extract move': '#e74c3c',      # Red
        'Move mismatch': '#f39c12',                # Orange
        'No legal move generated': '#c0392b',     # Dark red (includes None/parse failures)
        'API error': '#9b59b6',                    # Purple
        'Timeout': '#34495e',                      # Dark gray
        'Other': '#95a5a6',                        # Light gray
    }
    
    # Create color list for all error types
    colors_list = [error_colors.get(et, '#95a5a6') for et in all_error_types]
    
    # Create figure with subplots for each paradigm
    fig, axes = plt.subplots(1, 3, figsize=(18, max(8, len(sorted_models) * 0.5)))
    if len(paradigms_list) == 1:
        axes = [axes]
    
    # Create a stacked bar chart for each paradigm
    for paradigm_idx, paradigm in enumerate(paradigms_list):
        ax = axes[paradigm_idx]
        
        # Prepare data for this paradigm
        y = np.arange(len(sorted_models))
        height = 0.6
        
        # Build stacked bars for each model
        bottom = np.zeros(len(sorted_models))
        
        for i, error_type in enumerate(all_error_types):
            counts = []
            for model_name in sorted_models:
                if model_name in model_error_data and paradigm in model_error_data[model_name]:
                    count = model_error_data[model_name][paradigm].get('error_type_counts', {}).get(error_type, 0)
                else:
                    count = 0
                counts.append(count)
            
            if any(c > 0 for c in counts):  # Only plot if there are any errors of this type
                ax.barh(y, counts, height, left=bottom, 
                       label=error_type, color=colors_list[i],
                       edgecolor='black', linewidth=0.5)
                # Update bottom for next stack
                bottom = bottom + np.array(counts)
        
        # Customize axes
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_models, fontsize=9)
        ax.set_xlabel('Number of Errors', fontsize=11, fontweight='bold')
        ax.set_title(f'{paradigm_labels[paradigm_idx]}', fontsize=13, fontweight='bold', pad=10)
        apply_academic_axes(ax)
        
        # Add value labels on bars (total errors per model)
        max_width = max(bottom) if len(bottom) > 0 and max(bottom) > 0 else 1
        for i, model_name in enumerate(sorted_models):
            total_errors = bottom[i]
            if total_errors > 0:
                ax.text(total_errors + max_width * 0.01, i, f'{int(total_errors)}', 
                       va='center', fontsize=8, fontweight='bold')
    
    # Create shared legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=colors_list[i], edgecolor='black', linewidth=0.5, label=error_type) 
        for i, error_type in enumerate(all_error_types) 
        if any(
            (model_name in model_error_data and 
             paradigm in model_error_data[model_name] and
             model_error_data[model_name][paradigm].get('error_type_counts', {}).get(error_type, 0) > 0)
            for model_name in sorted_models
            for paradigm in paradigms_list
        )
    ]
    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=min(len(legend_patches), 4), fontsize=9, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.95)
    
    plt.suptitle('Error Comparison by Model Across Paradigms', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    output_file = output_path / "error_comparison_stacked.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✅ Graph saved to: {output_file}")
    
    # Print summary per model
    print("\n" + "="*80)
    print("ERROR COMPARISON SUMMARY BY MODEL (First 50 Puzzles)")
    print("="*80)
    for model_name in sorted_models:
        print(f"\n{model_name}:")
        print("-" * 80)
        for paradigm_idx, paradigm in enumerate(paradigms_list):
            if model_name in model_error_data and paradigm in model_error_data[model_name]:
                error_data = model_error_data[model_name][paradigm]
                total_errors = sum(error_data.get('error_type_counts', {}).values())
                if total_errors > 0:
                    error_breakdown = ", ".join([
                        f"{et}: {count}" 
                        for et, count in sorted(error_data.get('error_type_counts', {}).items(), 
                                               key=lambda x: x[1], reverse=True)
                    ])
                    print(f"  {paradigm_labels[paradigm_idx]:<20} {total_errors:>3} errors: {error_breakdown}")
                else:
                    print(f"  {paradigm_labels[paradigm_idx]:<20} {total_errors:>3} errors")
            else:
                print(f"  {paradigm_labels[paradigm_idx]:<20} No data")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create error comparison graph across paradigms")
    parser.add_argument("--results-dir", type=str, default="data/test_results",
                       help="Directory containing result CSV files")
    parser.add_argument("--output-dir", type=str, default="data/graphs",
                       help="Directory to save output graphs")
    
    args = parser.parse_args()
    create_error_comparison_graph(args.results_dir, args.output_dir)

