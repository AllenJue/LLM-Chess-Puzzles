import pandas as pd
import sys

csv_file = 'chess_puzzles/data/test_results/test_results_qwen_qwen3-235b-a22b-instruct-2507_single_1.csv'
df = pd.read_csv(csv_file)

print("=" * 80)
print("QWEN3-235B-A22B-INSTRUCT-2507 RESULTS")
print("=" * 80)
print(f"\nTotal puzzles: {len(df)}")
print(f"Puzzles solved: {df['puzzle_solved'].sum()}")
print(f"Correct moves: {df['correct_moves'].tolist()}")

# Find solved puzzle
solved_rows = df[df['puzzle_solved'] == True]
if len(solved_rows) > 0:
    idx = solved_rows.index[0]
    row = df.loc[idx]
    
    print("\n" + "=" * 80)
    print(f"SOLVED PUZZLE (Index: {idx})")
    print("=" * 80)
    
    print("\n--- SYSTEM PROMPT ---")
    print(row['system_prompt'])
    
    print("\n--- USER PROMPT (Full PGN) ---")
    print(row['user_prompt'])
    
    print("\n--- MODEL RESPONSE ---")
    print(row['single_model_response'])
    
    print("\n--- PREDICTED MOVE (SAN) ---")
    print(row['single_model_move'])
    
    print("\n--- EXPECTED MOVES (UCI) ---")
    print(row['Moves'])
    
    print("\n--- CORRECT MOVES COUNT ---")
    print(row['correct_moves'])
else:
    print("\nNo puzzles were fully solved.")
    # Show the one with most correct moves
    best_idx = df['correct_moves'].idxmax()
    row = df.loc[best_idx]
    print(f"\nBest result: Puzzle {best_idx} with {row['correct_moves']} correct moves")
    print("\n--- SYSTEM PROMPT ---")
    print(row['system_prompt'])
    print("\n--- USER PROMPT ---")
    print(row['user_prompt'])
    print("\n--- MODEL RESPONSE ---")
    print(row['single_model_response'])

