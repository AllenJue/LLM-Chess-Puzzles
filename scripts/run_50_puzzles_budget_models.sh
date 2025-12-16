#!/bin/bash
# Script to run 50 puzzles on budget OpenAI models and Anannas models (oss120b + deepseek-ai)
# Results saved in single model mode only

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "Running 50 Puzzles - Budget Models"
echo "=========================================="
echo ""
echo "Models to test:"
echo "  Budget OpenAI: gpt-4o-mini, gpt-5-mini, gpt-5-nano, gpt-4.1-mini"
echo "  Anannas: openai/gpt-oss-120b-turbo, deepseek-ai/* (8 models)"
echo ""
echo "Mode: single model only"
echo "Puzzles: 50 per model"
echo ""

# Run all models (auto-detects API provider)
python scripts/test_models.py \
    --model all \
    --num-puzzles 50 \
    --modes single \
    --single-only \
    --api-delay 0.5 \
    --delay 3.0

echo ""
echo "=========================================="
echo "All tests completed!"
echo "Results saved in: data/test_results/"
echo "=========================================="




