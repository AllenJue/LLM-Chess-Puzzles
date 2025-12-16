#!/usr/bin/env python3
"""Show full prompts for qwen model and suggest other models to try."""

import csv
import sys

# Read the CSV
csv_file = 'chess_puzzles/data/test_results/test_results_qwen_qwen3-235b-a22b-instruct-2507_single_1.csv'

print("=" * 80)
print("QWEN3-235B-A22B-INSTRUCT-2507 - Puzzle 0 Analysis")
print("=" * 80)

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    row = next(reader)  # First puzzle
    
    print(f"\nPuzzle ID: {row.get('PuzzleId', 'N/A')}")
    print(f"Expected moves (UCI): {row.get('Moves', 'N/A')}")
    print(f"Predicted move (SAN): {row.get('single_model_move', 'N/A')}")
    print(f"Correct moves: {row.get('correct_moves', '0')}")
    
    print("\n" + "=" * 80)
    print("MODEL RESPONSE (Full):")
    print("=" * 80)
    print(row.get('single_model_response', ''))
    
    # Reconstruct what the prompts should be
    print("\n" + "=" * 80)
    print("EXPECTED SYSTEM PROMPT (reconstructed):")
    print("=" * 80)
    system_prompt = """You are a chess grandmaster.
You will be given a partially completed game.
Complete the algebraic notation by repeating the ENTIRE GAME and then giving the next 3 moves.
After repeating the game, immediately continue by listing those moves in order on a single line, separated by spaces, starting with the side to move now.
Use standard algebraic notation, e.g. 'e4' or 'Rdf8'.
ALWAYS repeat the entire representation of the game so far.
NO other explanations. Just complete the algebraic notation."""
    print(system_prompt)
    
    print("\n" + "=" * 80)
    print("NOTE: User prompt (PGN) not saved in CSV")
    print("The user_prompt would contain the PGN up to move 27... Rxf1+")
    print("=" * 80)

print("\n" + "=" * 80)
print("SUGGESTED MODELS TO TRY (High Performance):")
print("=" * 80)

models_to_try = [
    # Top reasoning models
    ("openai/o3-pro", "Reasoning model, very strong"),
    ("openai/o3-mini", "Reasoning model, cheaper than o3-pro"),
    ("openai/o1-pro", "Strong reasoning model"),
    ("openai/o1-mini", "Cheaper reasoning model"),
    
    # Top flagship models
    ("openai/gpt-5-pro", "Latest flagship"),
    ("openai/gpt-5-codex", "Code-focused but strong"),
    ("anthropic/claude-3.7-sonnet", "Very strong general model"),
    ("anthropic/claude-opus-4.1", "Top-tier Claude"),
    ("qwen/qwen3-max", "Qwen's best model"),
    ("qwen/qwen3-235b-a22b-instruct-2507", "Already tested - got 1 right"),
    
    # Strong 70B+ models
    ("meta-llama/llama-3.1-70b-instruct", "Strong open-source"),
    ("meta-llama/llama-3.3-70b-instruct", "Newer version"),
    ("nvidia/llama-3.1-nemotron-70b-instruct", "NVIDIA's version"),
    ("qwen/qwen2.5-72b-instruct", "Large Qwen model"),
    
    # Fast/light but capable
    ("openai/gpt-4o", "Fast and strong"),
    ("openai/gpt-4-turbo", "Strong OpenAI model"),
    ("anthropic/claude-3.5-sonnet", "Fast Claude"),
    ("deepseek/deepseek-v3", "Strong DeepSeek model"),
    
    # Free/cheap options
    ("google/gemma-3-27b-it:free", "Free large model"),
    ("meta-llama/llama-3.3-8b-instruct:free", "Free smaller model"),
]

print("\nCommands to test puzzle 0 (single model, 1 puzzle):\n")
for model, desc in models_to_try:
    model_safe = model.replace("/", "_").replace(":", "_")
    print(f"# {desc}")
    print(f"python chess_puzzles/scripts/test_free_models.py --model \"{model}\" --num-puzzles 1 --single-only --delay 0 --api-delay 0.5")
    print()

