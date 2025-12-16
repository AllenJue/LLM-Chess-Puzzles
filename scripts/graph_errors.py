#!/usr/bin/env python3
"""
Graph error analysis for all single model results.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from typing import Dict, List

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
    # Handle both _free and :free patterns BEFORE replacing other underscores
    name = name.replace("_free", ":free").replace(":free", " (free)")
    name = name.replace("_", "/")
    return name

def analyze_errors(csv_file: Path, num_puzzles: int = 50) -> Dict:
    """Analyze errors from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            return None
        
        # Get unique puzzles in order of first appearance
        unique_puzzles = df['PuzzleId'].drop_duplicates().head(num_puzzles).tolist()
        actual_num_puzzles = len(unique_puzzles)
        
        if actual_num_puzzles == 0:
            return None
        
        # Collect errors
        errors = []
        error_types = []
        puzzles_with_errors = 0
        
        for puzzle_id in unique_puzzles:
            puzzle_rows = df[df['PuzzleId'] == puzzle_id]
            # Get the first row for this puzzle (should have the error if any)
            first_row = puzzle_rows.iloc[0]
            
            error = first_row.get('error', '')
            if pd.notna(error) and error and error.strip():
                errors.append({
                    'puzzle_id': puzzle_id,
                    'error': str(error).strip()
                })
                puzzles_with_errors += 1
                
                # Categorize error type
                error_str = str(error).lower()
                if 'failed to extract' in error_str:
                    error_types.append('Failed to extract move')
                elif 'mismatch' in error_str:
                    error_types.append('Move mismatch')
                elif 'illegal' in error_str or 'invalid' in error_str:
                    error_types.append('Illegal/invalid move')
                elif 'api' in error_str or 'error calling' in error_str:
                    error_types.append('API error')
                elif 'timeout' in error_str:
                    error_types.append('Timeout')
                else:
                    error_types.append('Other')
        
        error_counts = Counter(error_types)
        
        return {
            'total_puzzles': actual_num_puzzles,
            'puzzles_with_errors': puzzles_with_errors,
            'puzzles_without_errors': actual_num_puzzles - puzzles_with_errors,
            'error_rate': (puzzles_with_errors / actual_num_puzzles * 100) if actual_num_puzzles > 0 else 0,
            'errors': errors,
            'error_type_counts': dict(error_counts),
            'total_errors': len(errors)
        }
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_error_graphs(results_dir: str = "data/test_results", output_dir: str = "data/graphs"):
    """Create error analysis graphs for all single model results."""
    
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
    
    # Analyze errors for each model
    model_results = {}
    for csv_file in csv_files:
        model_name = extract_model_name(csv_file.name)
        # Skip excluded models
        if should_exclude_model(model_name):
            print(f"Skipping excluded model: {model_name}")
            continue
        error_data = analyze_errors(csv_file, num_puzzles=50)
        if error_data:
            model_results[model_name] = error_data
            print(f"{model_name}: {error_data['puzzles_with_errors']}/{error_data['total_puzzles']} puzzles with errors ({error_data['error_rate']:.1f}%)")
    
    if len(model_results) == 0:
        print("No valid results found")
        return
    
    # Sort models by error rate (highest first)
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['error_rate'], reverse=True)
    
    # Prepare data for plotting
    model_names = [name for name, _ in sorted_models]
    error_rates = [data['error_rate'] for _, data in sorted_models]
    error_counts = [data['puzzles_with_errors'] for _, data in sorted_models]
    total_puzzles = [data['total_puzzles'] for _, data in sorted_models]
    
    # Color palette
    colors = sns.color_palette("husl", len(model_names))
    
    # Plot 1: Error Rate (%) - Separate file
    fig1, ax1 = plt.subplots(figsize=(12, max(8, len(model_names) * 0.4)))
    bars1 = ax1.barh(range(len(model_names)), error_rates, color=colors, 
                     edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels(model_names, fontsize=10)
    ax1.set_xlabel('Error Rate (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Error Rate by Model', fontsize=15, fontweight='bold', pad=20)
    max_err_rate = max(error_rates) if max(error_rates) > 0 else 10
    ax1.set_xlim(0, max_err_rate * 1.15)
    apply_academic_axes(ax1)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars1, error_rates)):
        if rate > 0:
            ax1.text(rate + max_err_rate * 0.02, i, f'{rate:.1f}%', 
                    va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file1 = output_path / "single_model_error_rate.png"
    plt.savefig(output_file1, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig1)
    print(f"✅ Graph saved to: {output_file1}")
    
    # Plot 2: Error Count - Separate file
    fig2, ax2 = plt.subplots(figsize=(12, max(8, len(model_names) * 0.4)))
    bars2 = ax2.barh(range(len(model_names)), error_counts, color=colors, 
                     edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(model_names)))
    ax2.set_yticklabels(model_names, fontsize=10)
    ax2.set_xlabel('Number of Puzzles with Errors', fontsize=13, fontweight='bold')
    ax2.set_title('Error Count by Model', fontsize=15, fontweight='bold', pad=20)
    max_err_count = max(error_counts) if max(error_counts) > 0 else 5
    ax2.set_xlim(0, max_err_count * 1.15)
    apply_academic_axes(ax2)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars2, error_counts)):
        if count > 0:
            ax2.text(count + max_err_count * 0.02, i, f'{count}/{total_puzzles[i]}', 
                    va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file2 = output_path / "single_model_error_count.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig2)
    print(f"✅ Graph saved to: {output_file2}")
    
    # Plot 3: Error Type Distribution (stacked bar chart) - Separate file
    fig3, ax3 = plt.subplots(figsize=(14, max(8, len(model_names) * 0.4)))
    
    # Collect all error types
    all_error_types = set()
    for data in model_results.values():
        all_error_types.update(data['error_type_counts'].keys())
    all_error_types = sorted(list(all_error_types))
    
    # Prepare data for stacked bar chart
    error_type_data = {error_type: [] for error_type in all_error_types}
    for model_name, _ in sorted_models:
        data = model_results[model_name]
        for error_type in all_error_types:
            count = data['error_type_counts'].get(error_type, 0)
            error_type_data[error_type].append(count)
    
    # Create stacked bar chart
    bottom = [0] * len(model_names)
    colors_stacked = sns.color_palette("Set2", len(all_error_types))
    
    for i, error_type in enumerate(all_error_types):
        counts = error_type_data[error_type]
        if any(c > 0 for c in counts):  # Only plot if there are any errors of this type
            ax3.barh(range(len(model_names)), counts, left=bottom, 
                    label=error_type, color=colors_stacked[i])
            # Update bottom for next stack
            bottom = [b + c for b, c in zip(bottom, counts)]
    
    ax3.set_yticks(range(len(model_names)))
    ax3.set_yticklabels(model_names, fontsize=10)
    ax3.set_xlabel('Number of Errors', fontsize=13, fontweight='bold')
    ax3.set_title('Error Type Distribution by Model', fontsize=15, fontweight='bold', pad=20)
    apply_academic_axes(ax3)
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors_stacked[i], edgecolor='black', linewidth=0.5, label=error_type) 
                     for i, error_type in enumerate(all_error_types) if any(error_type_data[error_type])]
    create_academic_legend(ax3, legend_patches, fontsize=10)
    
    plt.tight_layout()
    output_file3 = output_path / "single_model_error_types.png"
    plt.savefig(output_file3, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig3)
    print(f"✅ Graph saved to: {output_file3}")
    
    # Print summary
    print("\n" + "="*80)
    print("ERROR SUMMARY (First 50 Puzzles)")
    print("="*80)
    print(f"{'Model':<50} {'Errors':<15} {'Error Rate':<15}")
    print("-"*80)
    for name, data in sorted_models:
        error_info = f"{data['puzzles_with_errors']}/{data['total_puzzles']}"
        error_rate = f"{data['error_rate']:.1f}%"
        print(f"{name:<50} {error_info:<15} {error_rate:<15}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Graph errors for single model results")
    parser.add_argument("--results-dir", type=str, default="data/test_results",
                       help="Directory containing result CSV files")
    parser.add_argument("--output-dir", type=str, default="data/graphs",
                       help="Directory to save output graphs")
    
    args = parser.parse_args()
    create_error_graphs(args.results_dir, args.output_dir)

