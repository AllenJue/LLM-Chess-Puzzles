#!/usr/bin/env python3
"""
Test script to run models on chess puzzles.
Supports testing any model via Anannas or OpenAI API.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional, Set

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from main import load_environment, evaluate_puzzles, read_chess_puzzles_csv
from model_interface import ChessModelInterface
import pandas as pd

# # Working free models from smoke test
# WORKING_FREE_MODELS = [
#     "google/gemma-3-12b-it:free",
#     "google/gemma-3-27b-it:free",
#     "google/gemma-3-4b-it:free",
#     "meta-llama/llama-3.3-8b-instruct:free",
#     "mistralai/mistral-small-3.2-24b-instruct:free",
#     "qwen/qwen-2.5-coder-32b-instruct:free",
#     "qwen/qwen2.5-vl-32b-instruct:free",
#     "qwen/qwen3-4b:free",  # Reasoning model
# ]

# Budget OpenAI models (via OpenAI API)
# Running debate and self-consistency for these models
BUDGET_OPENAI_MODELS = [
    "gpt-4o-mini",           # $0.15/$0.60 per 1M tokens - Run SC and debate
    "gpt-4.1-mini",          # $0.40/$1.60 per 1M tokens - Run SC and debate
    # "gpt-5-mini",          # $0.25/$2.00 per 1M tokens - Reasoning model (not tested)
    # "gpt-5-nano",            # $0.05/$0.40 per 1M tokens - ❌ Empty responses
]

# Anannas models - EXCLUDED for this run
ANANNAS_MODELS = [
    # "openai/gpt-oss-120b-turbo",     # via deepinfra - ✅ Working (not running)
    # "deepseek-ai/deepseek-v3",              # via deepinfra, deepseek-ai - ✅ Working (EXCLUDED per user request)
]

# Combined list for testing
WORKING_FREE_MODELS = BUDGET_OPENAI_MODELS + ANANNAS_MODELS


CONFIG_MODES = ["single", "self_consistency", "debate"]


def _select_output_file(model_name: str, mode: str, num_puzzles: int) -> str:
    """Helper to build consistent output filenames."""
    safe_model = model_name.replace("/", "_").replace(":", "_")
    return os.path.join(
        parent_dir,
        "data",
        "test_results",
        f"test_results_{safe_model}_{mode}_{num_puzzles}.csv",
    )


def _create_model_interface(model_name: str, base_url: Optional[str], api_key: str, use_openai: bool) -> ChessModelInterface:
    """Factory to produce a ChessModelInterface with correct provider prioritization."""
    # Prioritize provider based on flag
    provider_base_url = base_url
    if use_openai:
        provider_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY") or api_key

    # Reasoning models excluded for now, so we don't need special token handling
    max_tokens = 640

    return ChessModelInterface(
        api_key=api_key,
        model_name=model_name,
        base_url=provider_base_url,
        max_completion_tokens=max_tokens,
        default_temperature=0.1,
        retry_attempts=2,
    )


def test_model(
    model_name: str,
    *,
    num_puzzles: int,
    csv_file: str,
    base_url: Optional[str],
    api_delay: float,
    use_openai: bool,
    modes: Optional[List[str]] = None,
    single_model_only: bool = False,
):
    """Test a single model on puzzles."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}\n")
    
    # Load environment
    load_environment()
    
    # Get API key and base URL - prioritize based on selected provider
    api_key = os.getenv("OPENAI_API_KEY") if use_openai else os.getenv("ANANNAS_API_KEY")
    if not api_key:
        provider_label = "OPENAI_API_KEY" if use_openai else "ANANNAS_API_KEY"
        print(f"Error: {provider_label} not found in .env file")
        print("Please ensure the appropriate API key is set in chess_puzzles/.env")
        return []

    if base_url is None and not use_openai:
        base_url = os.getenv("ANANNAS_API_URL", "https://api.anannas.ai/v1")
    if use_openai:
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

    print(f"<debug> : API key loaded: {bool(api_key)} (provider={'openai' if use_openai else 'anannas'})")
    print(f"<debug> : Base URL: {base_url or 'OpenAI default'}")
    
    # Default CSV file
    if csv_file is None:
        csv_file = os.path.join(parent_dir, "data", "input", "lichess_puzzles_with_pgn_1000.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return None
    
    # Load puzzles
    try:
        df = read_chess_puzzles_csv(csv_file)
        print(f"Loaded {len(df)} puzzles from {csv_file}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    modes = modes or CONFIG_MODES
    results: List[pd.DataFrame] = []

    for mode in modes:
        if single_model_only and mode != "single":
            continue

        print(f"\n--- Running mode: {mode} ---")
        output_file = _select_output_file(model_name, mode, num_puzzles)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            if mode == "single":
                model_interface = _create_model_interface(model_name, base_url, api_key, use_openai)
                result_df = evaluate_puzzles(
                    df=df,
                    model_interface=model_interface,
                    max_puzzles=num_puzzles,
                    start_puzzle=0,
                    planning_plies=0,
                    api_delay=api_delay,
                )
            elif mode == "self_consistency":
                from main import ChessSelfConsistency  # delayed import to avoid circular

                sc_player = ChessSelfConsistency(
                    model_name=model_name,
                    temperature=0.1,
                    openai_api_key=api_key,
                    base_url=base_url,
                    max_rounds=2,
                    sleep_time=api_delay,
                    plan_plies=0,
                )
                result_df = evaluate_puzzles(
                    df=df,
                    debate=sc_player,
                    max_puzzles=num_puzzles,
                    start_puzzle=0,
                    planning_plies=0,
                    api_delay=api_delay,
                )
            else:  # debate
                from chess_debate import ChessDebateV2

                debate_player = ChessDebateV2(
                    model_name=model_name,
                    temperature=0.1,
                    openai_api_key=api_key,
                    base_url=base_url,
                    max_rounds=3,
                    sleep_time=api_delay,
                    plan_plies=0,
                )
                result_df = evaluate_puzzles(
                    df=df,
                    debate_v2=debate_player,
                    max_puzzles=num_puzzles,
                    start_puzzle=0,
                    planning_plies=0,
                    api_delay=api_delay,
                )

            trimmed_df = trim_results_dataframe(result_df, mode)
            trimmed_df.to_csv(output_file, index=False)
            print(f"✅ Saved {mode} results to: {output_file}")

            solved = trimmed_df["puzzle_solved"].sum() if "puzzle_solved" in trimmed_df.columns else 0
            total = len(trimmed_df)
            print(f"   Solved: {solved}/{total} ({100 * solved / total:.1f}%)")

            if "total_prompt_tokens" in trimmed_df.columns:
                total_prompt = trimmed_df["total_prompt_tokens"].sum()
                total_completion = trimmed_df["total_completion_tokens"].sum()
                print(f"   Tokens -> prompt {total_prompt}, completion {total_completion}")

            results.append(trimmed_df)
        except Exception as e:
            print(f"❌ Error running mode {mode} for {model_name}: {e}")
            continue

    return results


def trim_results_dataframe(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Reduce dataframe to essential columns for downstream analysis."""
    common_keep: Set[str] = {
        "PuzzleId",
        "FEN",
        "Moves",
        "Rating",
        "Themes",
        "correct_moves",
        "puzzle_solved",
        "total_moves",
        "accuracy_percentage",
        "error",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "prompt_token_log",
        "board_fen",
        "played_plies",
        "current_turn",
        "is_white_to_move",
        "user_prompt",
        "system_prompt",
    }
    single_keep: Set[str] = {
        "single_model_move",
        "single_model_response",
        "single_model_prompt_tokens",
        "single_model_completion_tokens",
        "single_model_total_tokens",
        "single_model_total_prompt_tokens",
        "single_model_total_completion_tokens",
        "single_model_model",
        "single_model_finish_reason",
        "single_model_prompt_log",
        "single_model_fallback",
    }
    sc_keep: Set[str] = {
        "aggressive_move",
        "aggressive_response",
        "aggressive_prompt_tokens",
        "aggressive_completion_tokens",
        "aggressive_total_tokens",
        "aggressive_model",
        "aggressive_finish_reason",
        "positional_move",
        "positional_response",
        "positional_prompt_tokens",
        "positional_completion_tokens",
        "positional_total_tokens",
        "positional_model",
        "positional_finish_reason",
        "neutral_move",
        "neutral_response",
        "neutral_prompt_tokens",
        "neutral_completion_tokens",
        "neutral_total_tokens",
        "neutral_model",
        "neutral_finish_reason",
        "final_consensus_move",
        "self_consistency_total_prompt_tokens",
        "self_consistency_total_completion_tokens",
        "self_consistency_total_tokens",
        "self_consistency_prompt_log",
    }
    debate_keep: Set[str] = {
        "debate_history",
        "debate_v2_history",
        "moderator_decision",
        "moderator_prompt_tokens",
        "moderator_completion_tokens",
        "moderator_total_tokens",
        "moderator_model",
        "moderator_finish_reason",
        "moderator_response",
        "judge_decision",
        "judge_prompt_tokens",
        "judge_completion_tokens",
        "judge_total_tokens",
        "judge_model",
        "judge_finish_reason",
        "judge_response",
        "final_reason",
        "supported_side",
        "debate_total_prompt_tokens",
        "debate_total_completion_tokens",
        "debate_total_tokens",
        "debate_prompt_log",
        "debate_success",
        "debate_rounds",
    }

    keep_columns = set(col for col in df.columns if col in common_keep)
    if mode == "single":
        keep_columns |= single_keep
    elif mode == "self_consistency":
        keep_columns |= sc_keep | single_keep  # retain single columns for comparison
    elif mode == "debate":
        keep_columns |= debate_keep | single_keep

    present_keep = [col for col in df.columns if col in keep_columns]
    trimmed_df = df.loc[:, present_keep].copy()
    return trimmed_df


def main():
    parser = argparse.ArgumentParser(
        description="Test free models on chess puzzles"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model to test (default: all). Use 'all' to test all models, or specify a model name."
    )
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=50,
        help="Number of puzzles to test per model (default: 50)"
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="single",
        help="Comma-separated list of modes to run (single,self_consistency,debate). Default: single"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay between models in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--api-delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Path to puzzle CSV file (default: data/lichess_puzzles_with_pgn_1000.csv)"
    )
    parser.add_argument(
        "--anannas-base-url",
        type=str,
        default=None,
        help="Anannas API base URL (defaults to https://api.anannas.ai/v1)"
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI API instead of Anannas"
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Only run single-model mode"
    )
    
    args = parser.parse_args()
    
    # Determine which models to test
    if args.model.lower() == "all":
        models_to_test = WORKING_FREE_MODELS
    else:
        if args.model not in WORKING_FREE_MODELS:
            print(f"⚠️  Warning: {args.model} not in known working models list")
            print(f"   Known models: {', '.join(WORKING_FREE_MODELS)}")
        models_to_test = [args.model]
    
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip() in CONFIG_MODES]
    if not modes:
        print("No valid modes selected; defaulting to single.")
        modes = ["single"]

    print(f"\nTesting {len(models_to_test)} model(s) on {args.num_puzzles} puzzle(s) each")
    print(f"Modes: {', '.join(modes)}")
    print(f"Delay between models: {args.delay}s, Delay between API calls: {args.api_delay}s\n")

    results = {}
    api_issues = []
    
    for i, model in enumerate(models_to_test):
        # Add delay between models (except for the first one)
        if i > 0:
            print(f"\nWaiting {args.delay} seconds before next model...")
            time.sleep(args.delay)
        
        # Auto-detect API provider based on model name
        # OpenAI models: gpt-4o-mini, gpt-5-mini, gpt-5-nano, gpt-4.1-mini
        # Anannas models: everything else (openai/gpt-oss-*, deepseek-ai/*)
        use_openai_for_model = model in BUDGET_OPENAI_MODELS
        if args.openai:
            use_openai_for_model = True  # Override with flag
        
        provider_label = "OpenAI" if use_openai_for_model else "Anannas"
        print(f"\n[{i+1}/{len(models_to_test)}] Using {provider_label} API for {model}")
        
        result = test_model(
            model,
            num_puzzles=args.num_puzzles,
            csv_file=args.csv_file or os.path.join(parent_dir, "data", "input", "lichess_puzzles_with_pgn_1000.csv"),
            base_url=args.anannas_base_url,
            api_delay=args.api_delay,
            use_openai=use_openai_for_model,
            modes=modes,
            single_model_only=args.single_only,
        )
        results[model] = result
        
        # Track API issues
        if not result:
            api_issues.append({
                "model": model,
                "issue": "Failed to initialize or run",
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = []
    failed = []
    
    for model, model_results in results.items():
        if model_results:
            # Aggregate summary per model across modes
            combined_df = pd.concat(model_results, ignore_index=True)
            solved = combined_df["puzzle_solved"].sum() if "puzzle_solved" in combined_df.columns else 0
            total = len(combined_df)
            status = "✅" if solved > 0 else "⚠️"
            print(f"{status} {model}: {solved}/{total} solved ({100 * solved / total:.1f}%) across modes {', '.join(modes)}")
            successful.append(model)
        elif model in models_to_test:
            print(f"❌ {model}: Failed")
            failed.append(model)
            if model not in [issue["model"] for issue in api_issues]:
                api_issues.append({
                    "model": model,
                    "issue": "Evaluation failed",
                })
    
    # Print API issues summary
    if api_issues:
        print(f"\n{'='*60}")
        print("API ISSUES DETECTED")
        print(f"{'='*60}")
        for issue in api_issues:
            print(f"  ❌ {issue['model']}: {issue.get('issue', 'Unknown error')}")
    
    print(f"\nCompleted: {len(successful)} successful, {len(failed)} failed out of {len(results)} total")


if __name__ == "__main__":
    main()

