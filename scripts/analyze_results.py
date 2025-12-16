#!/usr/bin/env python3
"""
Comprehensive analysis of chess puzzle evaluation results.
Analyzes relationships between model size, performance, error types, puzzle ratings, etc.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import re
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Models to exclude
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
    """Check if a model should be excluded."""
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
    name = name.replace("_free", ":free").replace(":free", " (free)")
    name = name.replace("_", "/")
    return name

def get_model_size(model_name: str) -> Optional[float]:
    """Extract model size in billions of parameters from model name.
    Returns None if size cannot be determined."""
    model_lower = model_name.lower()
    
    # Pattern matching for common model naming conventions
    patterns = [
        (r'(\d+(?:\.\d+)?)b(?:-|$)', 1),  # e.g., "24b", "3.5b", "235b"
        (r'(\d+(?:\.\d+)?)b-instruct', 1),  # e.g., "24b-instruct"
        (r'(\d+(?:\.\d+)?)b-it', 1),  # e.g., "2b-it"
        (r'(\d+(?:\.\d+)?)b-', 1),  # e.g., "8b-"
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, model_lower)
        if match:
            size = float(match.group(1))
            return size * multiplier
    
    # Special cases
    if 'gpt-3.5' in model_lower or 'gpt-3.5-turbo' in model_lower:
        return 175.0  # GPT-3.5 is ~175B
    if 'gpt-4' in model_lower:
        return 1000.0  # GPT-4 is estimated ~1T
    if 'deepseek-v3' in model_lower:
        return 67.0  # DeepSeek V3 is 67B
    if 'gemma-3n-e4b' in model_lower:
        return 4.0  # Gemma 3N E4B is 4B
    
    return None

def categorize_error(error_str: str) -> str:
    """Categorize an error string into error types."""
    if pd.isna(error_str) or not error_str or not str(error_str).strip():
        return None
    
    error_lower = str(error_str).lower()
    
    if 'failed to extract' in error_lower:
        return 'Failed to extract move'
    elif 'no legal move generated' in error_lower:
        return 'No legal move generated'
    elif 'mismatch' in error_lower:
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

def load_model_results(results_dir: str = "data/test_results", num_puzzles: int = 50) -> Dict:
    """Load all model results from CSV files."""
    results_path = parent_dir / results_dir
    paradigms = {
        'single': results_path / "single_50",
        'self_consistency': results_path / "self_consistency_50",
        'debate': results_path / "debate_50",
    }
    
    all_data = []
    
    for paradigm, dir_path in paradigms.items():
        if not dir_path.exists():
            continue
        
        csv_files = list(dir_path.glob("test_results_*_*.csv"))
        
        for csv_file in csv_files:
            model_name = extract_model_name(csv_file.name, paradigm)
            
            if should_exclude_model(model_name):
                continue
            
            try:
                df = pd.read_csv(csv_file)
                if len(df) == 0:
                    continue
                
                # Get unique puzzles
                unique_puzzles = df['PuzzleId'].drop_duplicates().head(num_puzzles).tolist()
                
                # Calculate metrics per puzzle
                for puzzle_id in unique_puzzles:
                    puzzle_rows = df[df['PuzzleId'] == puzzle_id]
                    first_row = puzzle_rows.iloc[0]
                    
                    # Extract puzzle rating
                    puzzle_rating = first_row.get('Rating', None)
                    if pd.notna(puzzle_rating):
                        puzzle_rating = float(puzzle_rating)
                    
                    # Puzzle solved
                    puzzle_solved = first_row.get('puzzle_solved', False)
                    if pd.notna(puzzle_solved):
                        puzzle_solved = bool(puzzle_solved)
                    
                    # Move accuracy
                    correct_moves = first_row.get('correct_moves', 0)
                    total_moves = first_row.get('total_moves', 0)
                    move_accuracy = (correct_moves / total_moves * 100) if total_moves > 0 else 0
                    
                    # Error information
                    error = first_row.get('error', '')
                    error_type = categorize_error(error)
                    has_error = bool(pd.notna(error) and error and str(error).strip())
                    
                    # Token usage
                    total_tokens = first_row.get('total_tokens', 0)
                    if pd.isna(total_tokens) or total_tokens == 0:
                        # Try paradigm-specific columns
                        if paradigm == 'single':
                            total_tokens = first_row.get('single_model_total_tokens', 0)
                        elif paradigm == 'self_consistency':
                            total_tokens = first_row.get('self_consistency_total_tokens', 0)
                        elif paradigm == 'debate':
                            total_tokens = first_row.get('debate_total_tokens', 0)
                    
                    # Model size
                    model_size = get_model_size(model_name)
                    
                    all_data.append({
                        'model_name': model_name,
                        'paradigm': paradigm,
                        'puzzle_id': puzzle_id,
                        'puzzle_rating': puzzle_rating,
                        'puzzle_solved': puzzle_solved,
                        'move_accuracy': move_accuracy,
                        'correct_moves': correct_moves,
                        'total_moves': total_moves,
                        'error_type': error_type,
                        'has_error': has_error,
                        'total_tokens': total_tokens if pd.notna(total_tokens) else 0,
                        'model_size': model_size,
                    })
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")
                continue
    
    return pd.DataFrame(all_data)

def analyze_relationships(df: pd.DataFrame) -> Dict:
    """Analyze relationships between various metrics."""
    results = {}
    
    # 1. Model size vs Performance
    # Convert boolean columns to numeric for aggregation
    df_agg = df.copy()
    df_agg['puzzle_solved'] = df_agg['puzzle_solved'].astype(float)
    df_agg['has_error'] = df_agg['has_error'].astype(int).astype(float)
    
    model_metrics = df_agg.groupby(['model_name', 'paradigm', 'model_size']).agg({
        'puzzle_solved': 'mean',
        'move_accuracy': 'mean',
        'has_error': 'mean',
    }).reset_index()
    
    # Filter out models without size information
    model_metrics_with_size = model_metrics[model_metrics['model_size'].notna()]
    
    if len(model_metrics_with_size) > 0:
        # Correlation: Model size vs Puzzle accuracy
        corr_puzzle = pearsonr(model_metrics_with_size['model_size'], 
                               model_metrics_with_size['puzzle_solved'])
        corr_move = pearsonr(model_metrics_with_size['model_size'], 
                            model_metrics_with_size['move_accuracy'])
        corr_error = pearsonr(model_metrics_with_size['model_size'], 
                             model_metrics_with_size['has_error'])
        
        results['model_size_vs_performance'] = {
            'puzzle_accuracy_correlation': corr_puzzle[0],
            'puzzle_accuracy_pvalue': corr_puzzle[1],
            'move_accuracy_correlation': corr_move[0],
            'move_accuracy_pvalue': corr_move[1],
            'error_rate_correlation': corr_error[0],
            'error_rate_pvalue': corr_error[1],
        }
    
    # 2. Puzzle rating vs Performance
    puzzle_metrics = df_agg[df_agg['puzzle_rating'].notna()].groupby(['puzzle_id', 'puzzle_rating']).agg({
        'puzzle_solved': 'mean',
        'move_accuracy': 'mean',
        'has_error': 'mean',
    }).reset_index()
    
    if len(puzzle_metrics) > 0:
        corr_rating_puzzle = pearsonr(puzzle_metrics['puzzle_rating'], 
                                     puzzle_metrics['puzzle_solved'])
        corr_rating_move = pearsonr(puzzle_metrics['puzzle_rating'], 
                                   puzzle_metrics['move_accuracy'])
        corr_rating_error = pearsonr(puzzle_metrics['puzzle_rating'], 
                                    puzzle_metrics['has_error'])
        
        results['puzzle_rating_vs_performance'] = {
            'puzzle_accuracy_correlation': corr_rating_puzzle[0],
            'puzzle_accuracy_pvalue': corr_rating_puzzle[1],
            'move_accuracy_correlation': corr_rating_move[0],
            'move_accuracy_pvalue': corr_rating_move[1],
            'error_rate_correlation': corr_rating_error[0],
            'error_rate_pvalue': corr_rating_error[1],
        }
    
    # 3. Error types vs Performance
    error_analysis = df[df['error_type'].notna()].groupby('error_type').agg({
        'puzzle_solved': 'mean',
        'move_accuracy': 'mean',
        'puzzle_id': 'count',
    }).reset_index()
    error_analysis.columns = ['error_type', 'avg_puzzle_accuracy', 'avg_move_accuracy', 'count']
    
    results['error_types_analysis'] = error_analysis.to_dict('records')
    
    # 4. Paradigm comparison
    paradigm_metrics = df_agg.groupby('paradigm').agg({
        'puzzle_solved': ['mean', 'sum', 'count'],
        'move_accuracy': 'mean',
        'has_error': 'mean',
        'total_tokens': 'mean',
    }).reset_index()
    paradigm_metrics.columns = ['paradigm', 'puzzle_solved_mean', 'puzzle_solved_sum', 'puzzle_count', 
                                'move_accuracy', 'has_error', 'total_tokens']
    # Calculate puzzles solved out of 50 (assuming 50 puzzles per model-paradigm)
    paradigm_metrics['puzzles_solved_out_of_50'] = (paradigm_metrics['puzzle_solved_sum'] / 
                                                    (paradigm_metrics['puzzle_count'] / 50)).round().astype(int)
    results['paradigm_comparison'] = paradigm_metrics.to_dict('records')
    
    # 5. Model performance summary
    model_summary = df_agg.groupby(['model_name', 'paradigm']).agg({
        'puzzle_solved': ['mean', 'sum'],
        'move_accuracy': 'mean',
        'has_error': 'mean',
        'model_size': 'first',
    }).reset_index()
    model_summary.columns = ['model_name', 'paradigm', 'puzzle_solved_mean', 'puzzle_solved_sum',
                            'move_accuracy', 'has_error', 'model_size']
    # Calculate puzzles solved out of 50
    model_summary['puzzles_solved_out_of_50'] = model_summary['puzzle_solved_sum'].round().astype(int)
    results['model_summary'] = model_summary.to_dict('records')
    
    # 6. Error type distribution by model size
    error_by_size = df[df['error_type'].notna() & df['model_size'].notna()].groupby(
        ['model_size', 'error_type']
    ).size().reset_index(name='count')
    
    results['error_by_size'] = error_by_size.to_dict('records')
    
    return results

def print_analysis_report(df: pd.DataFrame, analysis: Dict):
    """Print a comprehensive analysis report."""
    print("=" * 80)
    print("CHESS PUZZLE EVALUATION - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print()
    
    # Overall statistics
    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total puzzles evaluated: {df['puzzle_id'].nunique()}")
    print(f"Total model-paradigm combinations: {df.groupby(['model_name', 'paradigm']).ngroups}")
    print(f"Total data points: {len(df)}")
    print(f"Models with size information: {df[df['model_size'].notna()]['model_name'].nunique()}")
    print()
    
    # Model size vs Performance
    if 'model_size_vs_performance' in analysis:
        print("=" * 80)
        print("MODEL SIZE vs PERFORMANCE")
        print("-" * 80)
        size_perf = analysis['model_size_vs_performance']
        print(f"Model Size vs Puzzle Accuracy:")
        print(f"  Correlation: {size_perf['puzzle_accuracy_correlation']:.3f}")
        print(f"  P-value: {size_perf['puzzle_accuracy_pvalue']:.4f}")
        print(f"  {'Significant' if size_perf['puzzle_accuracy_pvalue'] < 0.05 else 'Not significant'} (p < 0.05)")
        print()
        print(f"Model Size vs Move Accuracy:")
        print(f"  Correlation: {size_perf['move_accuracy_correlation']:.3f}")
        print(f"  P-value: {size_perf['move_accuracy_pvalue']:.4f}")
        print(f"  {'Significant' if size_perf['move_accuracy_pvalue'] < 0.05 else 'Not significant'} (p < 0.05)")
        print()
        print(f"Model Size vs Error Rate:")
        print(f"  Correlation: {size_perf['error_rate_correlation']:.3f}")
        print(f"  P-value: {size_perf['error_rate_pvalue']:.4f}")
        print(f"  {'Significant' if size_perf['error_rate_pvalue'] < 0.05 else 'Not significant'} (p < 0.05)")
        print()
    
    # Puzzle rating vs Performance
    if 'puzzle_rating_vs_performance' in analysis:
        print("=" * 80)
        print("PUZZLE RATING vs PERFORMANCE")
        print("-" * 80)
        rating_perf = analysis['puzzle_rating_vs_performance']
        print(f"Puzzle Rating vs Puzzle Accuracy:")
        print(f"  Correlation: {rating_perf['puzzle_accuracy_correlation']:.3f}")
        print(f"  P-value: {rating_perf['puzzle_accuracy_pvalue']:.4f}")
        print(f"  {'Significant' if rating_perf['puzzle_accuracy_pvalue'] < 0.05 else 'Not significant'} (p < 0.05)")
        print()
        print(f"Puzzle Rating vs Move Accuracy:")
        print(f"  Correlation: {rating_perf['move_accuracy_correlation']:.3f}")
        print(f"  P-value: {rating_perf['move_accuracy_pvalue']:.4f}")
        print(f"  {'Significant' if rating_perf['move_accuracy_pvalue'] < 0.05 else 'Not significant'} (p < 0.05)")
        print()
        print(f"Puzzle Rating vs Error Rate:")
        print(f"  Correlation: {rating_perf['error_rate_correlation']:.3f}")
        print(f"  P-value: {rating_perf['error_rate_pvalue']:.4f}")
        print(f"  {'Significant' if rating_perf['error_rate_pvalue'] < 0.05 else 'Not significant'} (p < 0.05)")
        print()
    
    # Error types analysis
    if 'error_types_analysis' in analysis:
        print("=" * 80)
        print("ERROR TYPES vs PERFORMANCE")
        print("-" * 80)
        print(f"{'Error Type':<30} {'Count':<10} {'Avg Puzzle Acc':<15} {'Avg Move Acc':<15}")
        print("-" * 80)
        for err in sorted(analysis['error_types_analysis'], key=lambda x: x['count'], reverse=True):
            print(f"{err['error_type']:<30} {err['count']:<10} {err['avg_puzzle_accuracy']*100:>13.1f}% {err['avg_move_accuracy']:>13.1f}%")
        print()
    
    # Paradigm comparison
    if 'paradigm_comparison' in analysis:
        print("=" * 80)
        print("PARADIGM COMPARISON")
        print("-" * 80)
        print(f"{'Paradigm':<20} {'Puzzle Acc':<15} {'Move Acc':<15} {'Error Rate':<15} {'Avg Tokens':<15}")
        print("-" * 80)
        for p in analysis['paradigm_comparison']:
            puzzles_solved = p.get('puzzles_solved_out_of_50', int(p['puzzle_solved_mean'] * 50))
            print(f"{p['paradigm']:<20} {puzzles_solved:>4}/50 ({p['puzzle_solved_mean']*100:>5.1f}%) {p['move_accuracy']:>13.1f}% {p['has_error']*100:>13.1f}% {p['total_tokens']:>13.0f}")
        print()
    
    # Top performing models
    if 'model_summary' in analysis:
        print("=" * 80)
        print("TOP PERFORMING MODELS (by Puzzle Accuracy)")
        print("-" * 80)
        model_summary_df = pd.DataFrame(analysis['model_summary'])
        top_models = model_summary_df.nlargest(10, 'puzzle_solved_mean')
        print(f"{'Model':<40} {'Paradigm':<20} {'Puzzle Acc':<15} {'Move Acc':<15} {'Size (B)':<12}")
        print("-" * 80)
        for _, row in top_models.iterrows():
            size_str = f"{row['model_size']:.1f}" if pd.notna(row['model_size']) else "Unknown"
            puzzles_solved = row.get('puzzles_solved_out_of_50', int(row['puzzle_solved_mean'] * 50))
            print(f"{row['model_name']:<40} {row['paradigm']:<20} {puzzles_solved:>4}/50 ({row['puzzle_solved_mean']*100:>5.1f}%) {row['move_accuracy']:>13.1f}% {size_str:>12}")
        print()
    
    # Error distribution by model size
    if 'error_by_size' in analysis and len(analysis['error_by_size']) > 0:
        print("=" * 80)
        print("ERROR DISTRIBUTION BY MODEL SIZE")
        print("-" * 80)
        error_by_size_df = pd.DataFrame(analysis['error_by_size'])
        size_bins = [0, 10, 50, 100, 200, float('inf')]
        size_labels = ['<10B', '10-50B', '50-100B', '100-200B', '200B+']
        error_by_size_df['size_bin'] = pd.cut(error_by_size_df['model_size'], bins=size_bins, labels=size_labels)
        
        for bin_label in size_labels:
            bin_data = error_by_size_df[error_by_size_df['size_bin'] == bin_label]
            if len(bin_data) > 0:
                # Aggregate by error type within each bin
                bin_summary = bin_data.groupby('error_type')['count'].sum().reset_index()
                total = bin_summary['count'].sum()
                if total > 0:
                    print(f"\n{bin_label} Models (Total errors: {total}):")
                    for _, row in bin_summary.iterrows():
                        pct = (row['count'] / total * 100) if total > 0 else 0
                        print(f"  {row['error_type']:<30} {row['count']:>5} ({pct:>5.1f}%)")
        print()
        
        # Additional insight: Error type ratio by size
        print("ERROR TYPE RATIO BY MODEL SIZE (Move Mismatch / No Legal Move)")
        print("-" * 80)
        for bin_label in size_labels:
            bin_data = error_by_size_df[error_by_size_df['size_bin'] == bin_label]
            if len(bin_data) > 0:
                bin_summary = bin_data.groupby('error_type')['count'].sum().to_dict()
                mismatch = bin_summary.get('Move mismatch', 0)
                no_legal = bin_summary.get('No legal move generated', 0)
                if no_legal > 0:
                    ratio = mismatch / no_legal
                    print(f"{bin_label:<15} Ratio: {ratio:.2f} (Mismatch: {mismatch}, No Legal: {no_legal})")
                elif mismatch > 0:
                    print(f"{bin_label:<15} Ratio: ∞ (Only Move Mismatch: {mismatch})")
        print()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze chess puzzle evaluation results")
    parser.add_argument("--results-dir", type=str, default="data/test_results",
                       help="Directory containing result CSV files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for text report (default: print to stdout)")
    parser.add_argument("--num-puzzles", type=int, default=50,
                       help="Number of puzzles to analyze")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading results...")
    df = load_model_results(args.results_dir, args.num_puzzles)
    
    if len(df) == 0:
        print("No data found!")
        return
    
    # Analyze relationships
    print("Analyzing relationships...")
    analysis = analyze_relationships(df)
    
    # Print report
    if args.output:
        with open(args.output, 'w') as f:
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            print_analysis_report(df, analysis)
            sys.stdout = old_stdout
        print(f"✅ Analysis report saved to: {args.output}")
    else:
        print_analysis_report(df, analysis)

if __name__ == "__main__":
    main()

