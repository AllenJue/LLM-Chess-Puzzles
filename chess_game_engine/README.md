# Chess Game Engine

Play full chess games against Stockfish using LLM models with single model or self-consistency approaches.

## Features

- **Single Model**: Use a single LLM call to generate moves
- **Self-Consistency**: Use 3 independent LLM queries with majority voting
- **Stockfish Integration**: Play against the Stockfish chess engine
- **Fallback System**: Automatically uses random legal moves when AI fails
- **Game Recording**: Save games as JSON and PGN formats
- **Flexible Configuration**: Choose model, time limits, and colors
- **Move Tracking**: Records when fallback moves were used and why

## Requirements

- Python 3.7+
- `python-chess` library
- `stockfish` chess engine installed
- OpenAI API key

## Installation

```bash
# Install required packages
pip install python-chess openai

# Install Stockfish (varies by system)
# Ubuntu/Debian:
sudo apt-get install stockfish

# macOS:
brew install stockfish

# Windows: Download from https://stockfishchess.org/download/
```

## Usage

### Quick Tests

```bash
# Test 1 puzzle and 1 game (from parent directory)
cd ..
python test_minimal.py
cd chess_game_engine
```

### Single Game Tests

```bash
# Play with single model as white (Stockfish skill level 5/20)
python chess_game.py --save-pgn

# Play with self-consistency as black (easier Stockfish)
python chess_game.py --self-consistency --model-color black --skill 3 --save-pgn

# Use GPT-4 with custom Stockfish time and skill level
python chess_game.py --model gpt-4-turbo --time 2.0 --skill 8 --save-pgn

# Play with planning (3 plies ahead)
python chess_game.py --plan-plies 3 --save-pgn

# Play with open-source model via Anannas API
python chess_game.py --model deepseek-ai/deepseek-v3 --use-anannas --save-pgn

# Play against random legal moves instead of Stockfish
python chess_game.py --random-opponent --save-pgn

# Self-consistency vs random opponent
python chess_game.py --self-consistency --random-opponent --save-pgn
```

### Tournament Tests

```bash
# Run a single matchup between two models (1 game)
python run_tournament.py --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 --games-per-matchup 1

# Run a matchup with self-consistency
python run_tournament.py --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 --games-per-matchup 1 --use-self-consistency

# Run a matchup with planning (3 plies)
python run_tournament.py --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 --games-per-matchup 1 --plan-plies 3

# Run a matchup with self-consistency + planning
python run_tournament.py --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 --games-per-matchup 1 --use-self-consistency --plan-plies 3

# Include Stockfish in a matchup
python run_tournament.py --pair gpt-3.5-turbo-instruct Stockfish --games-per-matchup 1 --include-stockfish

# Human vs LLM (interactive)
python run_tournament.py --human-vs-llm --model gpt-3.5-turbo-instruct
```

### Command Line Options for `chess_game.py`

- `--model`: Model to use (default: gpt-3.5-turbo-instruct)
- `--stockfish`: Path to Stockfish executable (default: stockfish, auto-detected)
- `--time`: Time limit for Stockfish moves in seconds (default: 1.0)
- `--skill`: Stockfish skill level 0-20 (default: 5, where 20 is maximum strength)
- `--stockfish-depth`: Optional fixed search depth for Stockfish (overrides time)
- `--self-consistency`: Use self-consistency approach (3 agents, majority vote)
- `--debate`: Use multi-agent debate approach (2 agents + moderator)
- `--plan-plies`: Number of future plies to plan (default: 0, disabled)
- `--random-opponent`: Use random legal moves instead of Stockfish
- `--model-color`: Color for the model (white/black, default: white)
- `--save-json`: Save game result as JSON file
- `--save-pgn`: Save game as PGN file
- `--no-save`: Don't save any files
- `--use-anannas`: Use Anannas API for open-source models

### Command Line Options for `run_tournament.py`

- `--pair MODEL1 MODEL2`: Run a single matchup between two models
- `--games-per-matchup`: Number of games per matchup (default: 2)
- `--use-self-consistency`: Use self-consistency for all LLM players
- `--plan-plies`: Number of future plies to plan (default: 0)
- `--include-stockfish`: Include Stockfish as baseline
- `--stockfish-skill`: Stockfish skill level 0-20 (default: 5)
- `--stockfish-path`: Path to Stockfish executable
- `--human-vs-llm`: Play interactively against an LLM
- `--model`: Model to use (for --human-vs-llm)
- `--output-dir`: Output directory for results (default: data/tournaments)
- `--use-openings`: Use opening variations
- `--opening`: Specific opening key (e.g., 'e4_e5')

### Stockfish Skill Levels

- **0-3**: Beginner level (good for testing, makes obvious mistakes)
- **4-7**: Intermediate level (recommended for LLMs, balanced play)
- **8-12**: Advanced level (strong tactical play)
- **13-17**: Expert level (very strong, few mistakes)
- **18-20**: Maximum strength (grandmaster level, near-perfect play)

**Default**: Skill level 5 (intermediate) - provides a good challenge for LLMs without being overwhelming.

### More Examples

```bash
# Quick game with single model
python chess_game.py --save-pgn

# Self-consistency game with longer Stockfish thinking time
python chess_game.py --self-consistency --time 3.0 --save-json --save-pgn

# Model plays black with GPT-4
python chess_game.py --model gpt-4-turbo --model-color black --self-consistency --save-pgn

# Tournament: Run 10 games between GPT-3.5 and DeepSeek
python run_tournament.py --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 --games-per-matchup 10 --output-dir data/tournaments/test_tournament

# Tournament: GPT-3.5 with self-consistency vs Stockfish
python run_tournament.py --pair gpt-3.5-turbo-instruct Stockfish --games-per-matchup 5 --use-self-consistency --include-stockfish

# Tournament: GPT-3.5 with planning vs all other models
python run_tournament.py --pair gpt-3.5-turbo-instruct deepseek-ai/deepseek-v3 --games-per-matchup 5 --plan-plies 3
python run_tournament.py --pair gpt-3.5-turbo-instruct mistralai/mistral-small-24b-instruct-2501 --games-per-matchup 5 --plan-plies 3
python run_tournament.py --pair gpt-3.5-turbo-instruct meta-llama/llama-3.3-70b-instruct --games-per-matchup 5 --plan-plies 3
```

## Game Output

The engine provides real-time feedback:
- Current board position
- Move analysis
- Game progress
- Final result

## File Outputs

### JSON Format
Contains detailed game information:
- Move history with timestamps
- Player information
- Board states
- Game metadata
- **Fallback move tracking**: Records when and why fallback moves were used
- **Move quality indicators**: Shows which moves were AI-generated vs fallback

### PGN Format
Standard chess notation format that can be imported into chess software.

## Environment Setup

Create a `.env` file in the `chess_puzzles` directory with your API keys:

```env
# For OpenAI models (GPT-3.5, GPT-4, etc.)
OPENAI_API_KEY=your-openai-key-here

# For open-source models via Anannas API
ANANNAS_API_KEY=your-anannas-key-here
ANANNAS_API_URL=https://api.anannas.ai/v1
```

The system automatically selects the correct API key based on the model name:
- Models with "gpt" in the name → OpenAI API
- Other models → Anannas API

## Troubleshooting

### Stockfish Not Found
- Ensure Stockfish is installed and in your PATH
- Use `--stockfish /path/to/stockfish` to specify custom path

### API Errors
- Check your OpenAI API key is valid
- Ensure you have sufficient API credits
- Check rate limits

### Move Generation Issues
- The model may generate invalid moves occasionally
- Self-consistency approach is more robust but uses more tokens
- **Disqualification**: Models that fail to produce a legal move within 3 retries are disqualified (no random fallback)
- Temperature increases on retries to encourage different moves
- Check the console output for error messages and retry attempts

## Performance Notes

- **Single Model**: Faster, uses fewer tokens, but may be less accurate
- **Self-Consistency**: More robust, uses 3x tokens, better move quality (3 agents vote)
- **Planning**: Requests future moves, can improve strategic play
- **Stockfish Time**: Longer time = stronger play but slower games
- **Model Choice**: GPT-4 is stronger but more expensive than GPT-3.5

## Tournament Management

After running tournaments, you can:

```bash
# Organize game files by matchup
python organize_games.py data/tournaments/round_robin_single_10games

# Check remaining games
python check_remaining_games.py data/tournaments/round_robin_single_10games

# Reconstruct ratings from saved games
python reconstruct_ratings.py data/tournaments/round_robin_single_10games
```

See `TOURNAMENT_README.md` for detailed tournament documentation.

## Game Analysis

After playing games, you can:
- Import PGN files into chess analysis software
- Analyze JSON files for move patterns
- Compare single model vs self-consistency performance
- Study the model's chess understanding
