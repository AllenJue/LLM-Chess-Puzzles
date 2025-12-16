#!/usr/bin/env python3
"""
Opening Variations for Chess Tournaments

Provides common two-ply opening positions similar to Kaggle benchmark
"""

import chess
from typing import List, Dict, Optional


# Common two-ply opening positions (after 2 moves)
OPENING_VARIATIONS = {
    "e4_e5": {
        "name": "King's Pawn Game (e4 e5)",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "description": "Classical king's pawn opening"
    },
    "e4_c5": {
        "name": "Sicilian Defense (e4 c5)",
        "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "description": "Sicilian Defense"
    },
    "e4_e6": {
        "name": "French Defense (e4 e6)",
        "fen": "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "description": "French Defense"
    },
    "d4_d5": {
        "name": "Queen's Gambit Declined (d4 d5)",
        "fen": "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
        "description": "Queen's pawn opening"
    },
    "d4_Nf6": {
        "name": "Indian Defense (d4 Nf6)",
        "fen": "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
        "description": "Indian Defense"
    },
    "Nf3_d5": {
        "name": "Reti Opening (Nf3 d5)",
        "fen": "rnbqkbnr/ppp1pppp/8/3p4/8/5N2/PPPPPPPP/RNBQKB1R w KQkq d6 0 2",
        "description": "Reti Opening"
    },
    "c4_e5": {
        "name": "English Opening (c4 e5)",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq e6 0 2",
        "description": "English Opening"
    },
    "e4_c6": {
        "name": "Caro-Kann Defense (e4 c6)",
        "fen": "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "description": "Caro-Kann Defense"
    },
}


def get_opening_variation(opening_key: str) -> Optional[Dict[str, str]]:
    """
    Get a specific opening variation
    
    Args:
        opening_key: Key for the opening (e.g., "e4_e5")
        
    Returns:
        Dictionary with opening info or None if not found
    """
    return OPENING_VARIATIONS.get(opening_key)


def get_all_openings() -> List[Dict[str, str]]:
    """
    Get all available opening variations
    
    Returns:
        List of opening dictionaries
    """
    return [
        {**opening, "key": key}
        for key, opening in OPENING_VARIATIONS.items()
    ]


def get_random_opening() -> Dict[str, str]:
    """
    Get a random opening variation
    
    Returns:
        Dictionary with opening info including 'key'
    """
    import random
    key = random.choice(list(OPENING_VARIATIONS.keys()))
    return {**OPENING_VARIATIONS[key], "key": key}


def validate_opening_fen(fen: str) -> bool:
    """
    Validate that a FEN string represents a valid chess position
    
    Args:
        fen: FEN string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        board = chess.Board(fen)
        return True
    except:
        return False


if __name__ == "__main__":
    # Test opening variations
    print("Available Opening Variations:")
    print("=" * 60)
    for opening in get_all_openings():
        print(f"\n{opening['key']}: {opening['name']}")
        print(f"  Description: {opening['description']}")
        print(f"  FEN: {opening['fen']}")
        print(f"  Valid: {validate_opening_fen(opening['fen'])}")

