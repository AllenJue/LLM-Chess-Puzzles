#!/usr/bin/env python3
"""
Tournament Manager

Orchestrates matches between players (LLMs, Stockfish, Humans) in tournaments
"""

import sys
import os
import json
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Callable
from itertools import combinations, product

# Add parent directory to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Import from same directory (works both as module and script)
try:
    from .model_vs_model_game import Player, PlayerType, ModelVsModelGame
    from .tournament_ratings import TournamentRatings
    from .opening_variations import get_all_openings, get_opening_variation, get_random_opening
except ImportError:
    # Fallback for direct script execution
    from model_vs_model_game import Player, PlayerType, ModelVsModelGame
    from tournament_ratings import TournamentRatings
    from opening_variations import get_all_openings, get_opening_variation, get_random_opening


class TournamentManager:
    """Manages tournament execution and match pairing"""
    
    def __init__(
        self,
        players: List[Player],
        ratings: Optional[TournamentRatings] = None,
        games_per_matchup: int = 2,
        use_openings: bool = False,
        opening_keys: Optional[List[str]] = None,
        output_dir: str = "data/tournaments",
        verbose: bool = True,
    ):
        """
        Initialize tournament manager
        
        Args:
            players: List of Player objects to compete
            ratings: TournamentRatings instance (creates new if None)
            games_per_matchup: Number of games per player pair (each plays white/black)
            use_openings: Whether to use opening variations
            opening_keys: List of opening keys to use (None = all openings)
            output_dir: Directory to save game results
            verbose: Whether to print progress
        """
        self.players = players
        self.ratings = ratings or TournamentRatings()
        self.games_per_matchup = games_per_matchup
        self.use_openings = use_openings
        self.opening_keys = opening_keys
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Tournament state
        self.completed_games: List[Dict] = []
        self.match_results: List[Dict] = []
        
        # Setup openings
        if use_openings:
            if opening_keys:
                self.openings = [get_opening_variation(key) for key in opening_keys if get_opening_variation(key)]
            else:
                self.openings = get_all_openings()
        else:
            self.openings = [None]  # Standard starting position
    
    def create_player_pairs(self) -> List[Tuple[Player, Player]]:
        """
        Create all possible player pairs for round-robin tournament
        
        Returns:
            List of (player1, player2) tuples
        """
        pairs = []
        for p1, p2 in combinations(self.players, 2):
            pairs.append((p1, p2))
        return pairs
    
    def run_tournament(self) -> Dict:
        """
        Run complete tournament
        
        Returns:
            Dictionary with tournament results
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🏆 Starting Tournament")
            print(f"{'='*80}")
            print(f"Players: {len(self.players)}")
            print(f"Games per matchup: {self.games_per_matchup}")
            print(f"Using openings: {self.use_openings}")
            if self.use_openings:
                print(f"Openings: {len(self.openings)}")
            print(f"{'='*80}\n")
        
        # Create all pairs
        pairs = self.create_player_pairs()
        total_games = len(pairs) * self.games_per_matchup * len(self.openings)
        game_count = 0
        
        # Run matches
        for p1, p2 in pairs:
            for game_num in range(self.games_per_matchup):
                # Alternate colors
                white_player = p1 if game_num % 2 == 0 else p2
                black_player = p2 if game_num % 2 == 0 else p1
                
                # Try each opening (or standard start)
                for opening in self.openings:
                    game_count += 1
                    
                    if self.verbose:
                        print(f"\n[{game_count}/{total_games}] {white_player.name} vs {black_player.name}")
                        if opening:
                            print(f"Opening: {opening['name']}")
                    
                    # Create game
                    initial_fen = opening['fen'] if opening else None
                    game = ModelVsModelGame(
                        white_player=white_player,
                        black_player=black_player,
                        initial_fen=initial_fen,
                        verbose=self.verbose,
                    )
                    
                    # Play game
                    game_result = game.play_game()
                    
                    # Record result (sanitize names for game_id)
                    white_safe = white_player.name.replace('/', '_').replace('\\', '_')
                    black_safe = black_player.name.replace('/', '_').replace('\\', '_')
                    game_id = f"{white_safe}_vs_{black_safe}_{game_count}"
                    game_result['game_id'] = game_id
                    game_result['opening'] = opening['key'] if opening else 'standard'
                    
                    # Update ratings
                    self.ratings.record_game(
                        white_player=white_player.name,
                        black_player=black_player.name,
                        white_score=game_result['white_score'],
                        black_score=game_result['black_score'],
                        game_id=game_id,
                    )
                    
                    # Save game
                    self._save_game(game_result, game)
                    
                    # Track results
                    self.completed_games.append(game_result)
                    
                    # Update match results
                    self._update_match_results(white_player.name, black_player.name, game_result)
        
        # Generate tournament summary
        summary = self._generate_summary()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🏆 Tournament Complete!")
            print(f"{'='*80}")
            print(f"Total games: {len(self.completed_games)}")
            print(f"Total players: {len(self.players)}")
            print(f"{'='*80}\n")
            
            # Print leaderboard
            self.ratings.print_leaderboard()
        
        return summary
    
    def _save_game(self, game_result: Dict, game: ModelVsModelGame):
        """Save game to file"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Sanitize game_id for filename (replace slashes and other invalid chars)
        safe_game_id = game_result['game_id'].replace('/', '_').replace('\\', '_')
        
        # Save JSON
        json_path = os.path.join(self.output_dir, f"{safe_game_id}.json")
        with open(json_path, 'w') as f:
            json.dump(game_result, f, indent=2)
        
        # Save PGN
        pgn_path = os.path.join(self.output_dir, f"{safe_game_id}.pgn")
        game.save_pgn(game_result, pgn_path)
    
    def _update_match_results(self, white_name: str, black_name: str, game_result: Dict):
        """Update match-level results"""
        match_key = tuple(sorted([white_name, black_name]))
        
        # Find or create match record
        match_record = None
        for match in self.match_results:
            if tuple(sorted([match['player1'], match['player2']])) == match_key:
                match_record = match
                break
        
        if not match_record:
            match_record = {
                'player1': white_name,
                'player2': black_name,
                'games': [],
                'player1_wins': 0,
                'player2_wins': 0,
                'draws': 0,
            }
            self.match_results.append(match_record)
        
        # Update match record
        match_record['games'].append(game_result['game_id'])
        
        if game_result['white_score'] == 1.0:
            if white_name == match_record['player1']:
                match_record['player1_wins'] += 1
            else:
                match_record['player2_wins'] += 1
        elif game_result['black_score'] == 1.0:
            if black_name == match_record['player1']:
                match_record['player1_wins'] += 1
            else:
                match_record['player2_wins'] += 1
        else:
            match_record['draws'] += 1
    
    def _generate_summary(self) -> Dict:
        """Generate tournament summary"""
        # Save ratings
        ratings_path = os.path.join(self.output_dir, "ratings.json")
        self.ratings.save_ratings(ratings_path)
        
        # Generate summary
        summary = {
            "tournament_info": {
                "timestamp": datetime.now().isoformat(),
                "total_players": len(self.players),
                "total_games": len(self.completed_games),
                "games_per_matchup": self.games_per_matchup,
                "use_openings": self.use_openings,
                "openings_used": len(self.openings) if self.use_openings else 0,
            },
            "leaderboard": [
                {
                    "rank": rank,
                    "player": name,
                    "rating": rating,
                    "rd": rd,
                    "games": games,
                }
                for rank, (name, rating, rd, games) in enumerate(self.ratings.get_leaderboard(), 1)
            ],
            "match_results": self.match_results,
            "ratings_file": ratings_path,
        }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, "tournament_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def add_stockfish_player(
        self,
        name: str = "Stockfish",
        skill_level: int = 5,
        stockfish_path: str = "stockfish",
        stockfish_time: float = 1.0,
    ) -> Player:
        """
        Add Stockfish as a baseline player
        
        Args:
            name: Name for Stockfish player
            skill_level: Stockfish skill level (0-20)
            stockfish_path: Path to Stockfish executable
            stockfish_time: Time limit for Stockfish moves
            
        Returns:
            Created Player object
        """
        stockfish_player = Player(
            name=name,
            player_type=PlayerType.STOCKFISH,
            stockfish_path=stockfish_path,
            stockfish_time=stockfish_time,
            stockfish_skill=skill_level,
        )
        self.players.append(stockfish_player)
        return stockfish_player
    
    def add_llm_player(
        self,
        name: str,
        model_name: str,
        use_self_consistency: bool = False,
        use_debate: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Player:
        """
        Add an LLM player
        
        Args:
            name: Name for the player
            model_name: Model identifier
            use_self_consistency: Use self-consistency approach
            use_debate: Use debate approach
            api_key: API key (uses env if None)
            base_url: Base URL (uses env if None)
            
        Returns:
            Created Player object
        """
        llm_player = Player(
            name=name,
            player_type=PlayerType.LLM,
            model_name=model_name,
            use_self_consistency=use_self_consistency,
            use_debate=use_debate,
            api_key=api_key,
            base_url=base_url,
        )
        self.players.append(llm_player)
        return llm_player


def create_default_players(
    include_stockfish: bool = True,
    stockfish_skill: int = 5,
) -> List[Player]:
    """
    Create default set of players for tournament
    
    Args:
        include_stockfish: Whether to include Stockfish baseline
        stockfish_skill: Stockfish skill level
        
    Returns:
        List of Player objects
    """
    players = []
    
    # Get API credentials
    api_key = os.getenv('ANANNAS_API_KEY') or os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('ANANNAS_API_URL') or os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_BASE')
    
    # Default models
    models = [
        ("GPT-3.5-turbo-instruct", "gpt-3.5-turbo-instruct"),
        ("DeepSeek-V3", "deepseek-ai/deepseek-v3"),
        ("Mistral-Small-24B", "mistralai/mistral-small-24b-instruct-2501"),
        ("Llama-3.3-70B", "meta-llama/llama-3.3-70b-instruct"),
    ]
    
    for display_name, model_name in models:
        player = Player(
            name=display_name,
            player_type=PlayerType.LLM,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )
        players.append(player)
    
    # Add Stockfish if requested
    if include_stockfish:
        stockfish = Player(
            name="Stockfish",
            player_type=PlayerType.STOCKFISH,
            stockfish_skill=stockfish_skill,
        )
        players.append(stockfish)
    
    return players


if __name__ == "__main__":
    # Test tournament manager
    players = create_default_players(include_stockfish=True)
    
    manager = TournamentManager(
        players=players,
        games_per_matchup=1,  # Quick test
        use_openings=False,
        verbose=True,
    )
    
    # Run small test tournament
    summary = manager.run_tournament()
    print("\nTournament Summary:")
    print(json.dumps(summary, indent=2))

