# Advanced Usage Guide

This document provides detailed command examples and advanced usage patterns for the chess puzzle evaluator.

## Puzzle Evaluation

### Basic Evaluation

```bash
# Show help
python main.py --help

# Evaluate 1 puzzle (minimal test) - Single Model
python main.py --evaluate --max-puzzles 1

# Evaluate 10 puzzles with GPT-3.5 - Single Model
python main.py --evaluate --max-puzzles 10

# Evaluate with self-consistency (3 agents, majority vote)
python main.py --evaluate --max-puzzles 10 --self-consistency

# Evaluate with debate (2 agents + moderator)
python main.py --evaluate --max-puzzles 10 --debate

# Evaluate with planning (request 3 future plies)
python main.py --evaluate --max-puzzles 10 --plan-plies 3

# Evaluate with open-source model via Anannas API
python main.py --evaluate --max-puzzles 10 --model deepseek-ai/deepseek-v3 --use-anannas

# Evaluate with GPT-4
python main.py --evaluate --max-puzzles 5 --model gpt-4-turbo

# Calculate Glicko-2 rating (after evaluation)
python main.py --rating
```

### Command Line Options

- `--evaluate`: Run puzzle evaluation
- `--max-puzzles N`: Number of puzzles to evaluate
- `--self-consistency`: Use self-consistency paradigm (3 agents)
- `--debate`: Use multi-agent debate paradigm
- `--plan-plies N`: Enable planning with N future plies
- `--model MODEL_NAME`: Specify model to use
- `--use-anannas`: Use Anannas API for open-source models
- `--rating`: Calculate Glicko-2 ratings from results

## Batch Testing

### Using test_models.py

```bash
cd scripts

# Test specific model on puzzles
python test_models.py --model gpt-4o-mini --num-puzzles 50 --modes single,self_consistency,debate

# Test all models
python test_models.py --model all --num-puzzles 50 --modes single,self_consistency,debate

# Test with custom CSV file
python test_models.py --model gpt-4o-mini --num-puzzles 50 --csv-file data/input/custom_puzzles.csv

# Test with API delay (rate limiting)
python test_models.py --model gpt-4o-mini --num-puzzles 50 --api-delay 1.0

# Test only single model mode
python test_models.py --model gpt-4o-mini --num-puzzles 50 --single-only
```

### test_models.py Options

- `--model MODEL`: Model to test (or "all" for all models)
- `--num-puzzles N`: Number of puzzles per model
- `--modes MODES`: Comma-separated list (single,self_consistency,debate)
- `--csv-file PATH`: Custom puzzle CSV file
- `--delay SECONDS`: Delay between models
- `--api-delay SECONDS`: Delay between API calls
- `--openai`: Force use of OpenAI API
- `--single-only`: Only run single-model mode

## Full Game Play

### Single Games

```bash
cd chess_game_engine

# Basic game (LLM as white, Stockfish skill 5)
python chess_game.py --save-pgn

# With self-consistency
python chess_game.py --self-consistency --save-pgn

# With debate
python chess_game.py --debate --save-pgn

# With planning (3 plies ahead)
python chess_game.py --plan-plies 3 --save-pgn

# Play as black
python chess_game.py --model-color black --save-pgn

# Custom Stockfish skill level (0-20)
python chess_game.py --skill 10 --save-pgn

# Custom Stockfish thinking time
python chess_game.py --time 2.0 --save-pgn

# Fixed Stockfish depth
python chess_game.py --stockfish-depth 7 --save-pgn

# Use open-source model
python chess_game.py --model deepseek-ai/deepseek-v3 --use-anannas --save-pgn

# Play against random moves instead of Stockfish
python chess_game.py --random-opponent --save-pgn

# Save as JSON instead of PGN
python chess_game.py --save-json

# Save both formats
python chess_game.py --save-json --save-pgn
```

### chess_game.py Options

- `--model MODEL`: Model to use (default: gpt-3.5-turbo-instruct)
- `--stockfish PATH`: Path to Stockfish executable
- `--time SECONDS`: Stockfish thinking time (default: 1.0)
- `--skill N`: Stockfish skill level 0-20 (default: 5)
- `--stockfish-depth N`: Fixed search depth for Stockfish
- `--self-consistency`: Use self-consistency approach
- `--debate`: Use multi-agent debate approach
- `--plan-plies N`: Number of future plies to plan
- `--random-opponent`: Use random legal moves instead of Stockfish
- `--model-color COLOR`: Color for model (white/black, default: white)
- `--save-json`: Save game as JSON file
- `--save-pgn`: Save game as PGN file
- `--no-save`: Don't save any files
- `--use-anannas`: Use Anannas API for open-source models

## Tournaments

### Basic Tournament

```bash
cd chess_game_engine

# Single matchup (2 models, 1 game each)
python run_tournament.py --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 --games-per-matchup 1

# Multiple games per matchup
python run_tournament.py --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 --games-per-matchup 5

# With self-consistency
python run_tournament.py --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 --games-per-matchup 1 --use-self-consistency

# With planning
python run_tournament.py --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 --games-per-matchup 1 --plan-plies 3

# Include Stockfish as baseline
python run_tournament.py --pair gpt-3.5-turbo-instruct Stockfish --games-per-matchup 1 --include-stockfish

# Custom Stockfish skill
python run_tournament.py --pair gpt-3.5-turbo-instruct Stockfish --games-per-matchup 1 --include-stockfish --stockfish-skill 10

# Use opening variations
python run_tournament.py --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 --games-per-matchup 1 --use-openings

# Specific opening
python run_tournament.py --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 --games-per-matchup 1 --use-openings --opening e4_e5
```

### Human vs LLM

```bash
cd chess_game_engine

# Play as white against LLM
python run_tournament.py --human-vs-llm --model gpt-3.5-turbo-instruct

# Play as black
python run_tournament.py --human-vs-llm --model gpt-3.5-turbo-instruct --human-black

# With opening
python run_tournament.py --human-vs-llm --model gpt-3.5-turbo-instruct --use-openings --opening e4_e5
```

### run_tournament.py Options

- `--pair MODEL1 MODEL2`: Run matchup between two models
- `--games-per-matchup N`: Number of games per matchup (default: 2)
- `--use-self-consistency`: Use self-consistency for all LLM players
- `--plan-plies N`: Number of future plies to plan
- `--include-stockfish`: Include Stockfish as baseline player
- `--stockfish-skill N`: Stockfish skill level 0-20 (default: 5)
- `--stockfish-path PATH`: Path to Stockfish executable
- `--human-vs-llm`: Play interactively against an LLM
- `--human-black`: Human plays black (for --human-vs-llm)
- `--model MODEL`: Model to use (for --human-vs-llm)
- `--output-dir PATH`: Output directory (default: data/tournaments)
- `--use-openings`: Use opening variations
- `--opening KEY`: Specific opening key (e.g., 'e4_e5')

## Graph Generation

### Main Graphs

```bash
cd scripts

# Generate paradigm comparison graphs
python graph_all_paradigms.py

# Generate error comparison graphs
python graph_error_comparison.py

# Generate single model error graphs
python graph_errors.py

# Generate single model accuracy graphs
python graph_single_model_accuracy.py

# Generate sorted comparison graphs
python graph_sorted_comparison.py

# Generate token usage graphs
python graph_tokens.py
```

### Graph Script Options

Most graph scripts support:
- `--results-dir PATH`: Directory containing results (default: data/test_results)
- `--output-dir PATH`: Directory to save graphs (default: data/graphs)

Example:
```bash
python graph_all_paradigms.py --results-dir data/test_results --output-dir data/graphs
```

## Tournament Utilities

```bash
cd chess_game_engine

# Organize game files by matchup
python organize_games.py data/tournaments/round_robin_single_10games

# Check remaining games in tournament
python check_remaining_games.py data/tournaments/round_robin_single_10games

# Reconstruct ratings from saved games
python reconstruct_ratings.py data/tournaments/round_robin_single_10games

# Fix disqualification scores (bug fix utility)
python fix_disqualification_scores.py data/tournaments/tournament_name
```

## Programmatic Usage

```python
from csv_reader import read_chess_puzzles_csv, sample_puzzles
from model_interface import ChessModelInterface
from chess_utils import build_chess_prompts
from glicko_rating import update_agent_rating_from_puzzles

# Load data
df = read_chess_puzzles_csv("data/input/lichess_puzzles_with_pgn_1000.csv")
df_sample = sample_puzzles(df, n=100)

# Initialize model
model = ChessModelInterface(
    api_key="your-api-key",
    model_name="gpt-3.5-turbo-instruct",
    max_completion_tokens=640,
    default_temperature=0.1
)

# Evaluate puzzles programmatically
from main import evaluate_puzzles

results = evaluate_puzzles(
    df=df_sample,
    model_interface=model,
    max_puzzles=10,
    start_puzzle=0,
    planning_plies=0,
    api_delay=0.5
)

# Calculate ratings
from glicko_rating import update_agent_rating_from_puzzles
ratings = update_agent_rating_from_puzzles(results)
```

## Environment Variables

Create a `.env` file in the `chess_puzzles` directory:

```env
# OpenAI API (for GPT models)
OPENAI_API_KEY=your-openai-key-here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional, for custom endpoints

# Anannas API (for open-source models)
ANANNAS_API_KEY=your-anannas-key-here
ANANNAS_API_URL=https://api.anannas.ai/v1
```

The system automatically selects the correct API based on model name:
- Models with "gpt" in name → OpenAI API
- Other models → Anannas API (unless `--openai` flag is used)

## Stockfish Installation

### macOS
```bash
brew install stockfish
```

### Ubuntu/Debian
```bash
sudo apt-get install stockfish
```

### Windows
Download from [stockfishchess.org/download/](https://stockfishchess.org/download/)

### Verify Installation
```bash
cd chess_game_engine
python test_stockfish.py
```

## Troubleshooting

### Common Issues

1. **Module Import Errors**: Run scripts from the correct directory or use absolute imports
2. **API Key Errors**: Verify `.env` file is in `chess_puzzles/` directory with correct key names
3. **Stockfish Not Found**: Install Stockfish and ensure it's in PATH, or use `--stockfish /path/to/stockfish`
4. **Graph Generation Errors**: Install dependencies: `pip install matplotlib seaborn`
5. **Rate Limiting**: Use `--api-delay` flag to add delays between API calls

### Getting Help

- Check script help: `python script_name.py --help`
- Review README files in subdirectories
- Check console output for detailed error messages


