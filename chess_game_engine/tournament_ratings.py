#!/usr/bin/env python3
"""
Tournament Rating System using Glicko-2

Manages ratings for players in tournaments and generates leaderboards
"""

import sys
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Add parent directory to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from glicko_rating import Glicko2, Rating, WIN, DRAW, LOSS


class TournamentRatings:
    """Manages ratings for tournament players using Glicko-2"""
    
    def __init__(self, initial_rating: float = 1500.0, initial_rd: float = 350.0):
        """
        Initialize tournament ratings system
        
        Args:
            initial_rating: Starting rating for new players
            initial_rd: Starting rating deviation for new players
        """
        self.glicko = Glicko2()
        self.ratings: Dict[str, Rating] = {}
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.game_history: List[Dict] = []
    
    def get_or_create_rating(self, player_name: str) -> Rating:
        """
        Get rating for a player, creating if it doesn't exist
        
        Args:
            player_name: Name of the player
            
        Returns:
            Rating object for the player
        """
        if player_name not in self.ratings:
            self.ratings[player_name] = self.glicko.create_rating(
                mu=self.initial_rating,
                phi=self.initial_rd,
                sigma=0.06
            )
        return self.ratings[player_name]
    
    def record_game(
        self,
        white_player: str,
        black_player: str,
        white_score: float,
        black_score: float,
        game_id: Optional[str] = None,
    ):
        """
        Record a game result and update ratings
        
        Args:
            white_player: Name of white player
            black_player: Name of black player
            white_score: Score for white (1.0 = win, 0.5 = draw, 0.0 = loss)
            black_score: Score for black (1.0 = win, 0.5 = draw, 0.0 = loss)
            game_id: Optional game identifier
        """
        # Get current ratings
        white_rating = self.get_or_create_rating(white_player)
        black_rating = self.get_or_create_rating(black_player)
        
        # Update ratings
        if white_score == 1.0:
            # White wins
            new_white_rating, new_black_rating = self.glicko.rate_1vs1(
                white_rating, black_rating, drawn=False
            )
        elif black_score == 1.0:
            # Black wins (swap order)
            new_black_rating, new_white_rating = self.glicko.rate_1vs1(
                black_rating, white_rating, drawn=False
            )
        else:
            # Draw
            new_white_rating, new_black_rating = self.glicko.rate_1vs1(
                white_rating, black_rating, drawn=True
            )
        
        # Update stored ratings
        self.ratings[white_player] = new_white_rating
        self.ratings[black_player] = new_black_rating
        
        # Record game history
        self.game_history.append({
            "game_id": game_id,
            "white_player": white_player,
            "black_player": black_player,
            "white_score": white_score,
            "black_score": black_score,
            "white_rating_before": white_rating.mu,
            "black_rating_before": black_rating.mu,
            "white_rating_after": new_white_rating.mu,
            "black_rating_after": new_black_rating.mu,
            "white_rd_before": white_rating.phi,
            "black_rd_before": black_rating.phi,
            "white_rd_after": new_white_rating.phi,
            "black_rd_after": new_black_rating.phi,
        })
    
    def get_rating(self, player_name: str) -> Optional[Rating]:
        """Get current rating for a player"""
        return self.ratings.get(player_name)
    
    def get_elo_estimate(self, player_name: str) -> Optional[float]:
        """
        Get Elo estimate from Glicko-2 rating
        
        Glicko-2 mu is approximately equivalent to Elo rating
        """
        rating = self.get_rating(player_name)
        if rating:
            return rating.mu
        return None
    
    def get_leaderboard(self, min_games: int = 0) -> List[Tuple[str, float, float, int]]:
        """
        Get sorted leaderboard
        
        Args:
            min_games: Minimum number of games to appear on leaderboard
            
        Returns:
            List of (player_name, rating, rd, games_played) tuples, sorted by rating
        """
        # Count games per player
        games_count: Dict[str, int] = defaultdict(int)
        for game in self.game_history:
            games_count[game["white_player"]] += 1
            games_count[game["black_player"]] += 1
        
        # Build leaderboard
        leaderboard = []
        for player_name, rating in self.ratings.items():
            games = games_count[player_name]
            if games >= min_games:
                leaderboard.append((player_name, rating.mu, rating.phi, games))
        
        # Sort by rating (descending)
        leaderboard.sort(key=lambda x: x[1], reverse=True)
        return leaderboard
    
    def print_leaderboard(self, min_games: int = 0, title: str = "Tournament Leaderboard"):
        """Print formatted leaderboard"""
        leaderboard = self.get_leaderboard(min_games)
        
        print(f"\n{'='*80}")
        print(f"{title}")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Player':<40} {'Rating':<12} {'RD':<12} {'Games':<8}")
        print(f"{'-'*80}")
        
        for rank, (player_name, rating, rd, games) in enumerate(leaderboard, 1):
            print(f"{rank:<6} {player_name:<40} {rating:>10.1f}  {rd:>10.1f}  {games:>6}")
        
        print(f"{'='*80}\n")
    
    def get_player_stats(self, player_name: str) -> Dict:
        """Get detailed statistics for a player"""
        rating = self.get_rating(player_name)
        if not rating:
            return None
        
        # Count games and results
        wins = 0
        losses = 0
        draws = 0
        games_as_white = 0
        games_as_black = 0
        
        for game in self.game_history:
            if game["white_player"] == player_name:
                games_as_white += 1
                if game["white_score"] == 1.0:
                    wins += 1
                elif game["white_score"] == 0.5:
                    draws += 1
                else:
                    losses += 1
            elif game["black_player"] == player_name:
                games_as_black += 1
                if game["black_score"] == 1.0:
                    wins += 1
                elif game["black_score"] == 0.5:
                    draws += 1
                else:
                    losses += 1
        
        total_games = wins + losses + draws
        
        return {
            "player_name": player_name,
            "rating": rating.mu,
            "rd": rating.phi,
            "sigma": rating.sigma,
            "total_games": total_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / total_games if total_games > 0 else 0.0,
            "games_as_white": games_as_white,
            "games_as_black": games_as_black,
        }
    
    def save_ratings(self, filepath: str):
        """Save ratings to JSON file"""
        import json
        from datetime import datetime
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "ratings": {
                name: {
                    "mu": rating.mu,
                    "phi": rating.phi,
                    "sigma": rating.sigma,
                }
                for name, rating in self.ratings.items()
            },
            "game_history": self.game_history,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_ratings(self, filepath: str):
        """Load ratings from JSON file"""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Restore ratings
        for name, rating_data in data.get("ratings", {}).items():
            self.ratings[name] = self.glicko.create_rating(
                mu=rating_data["mu"],
                phi=rating_data["phi"],
                sigma=rating_data.get("sigma", 0.06)
            )
        
        # Restore game history
        self.game_history = data.get("game_history", [])


if __name__ == "__main__":
    # Test the rating system
    ratings = TournamentRatings()
    
    # Simulate some games
    ratings.record_game("Player A", "Player B", 1.0, 0.0)
    ratings.record_game("Player B", "Player C", 1.0, 0.0)
    ratings.record_game("Player A", "Player C", 0.5, 0.5)
    
    ratings.print_leaderboard()

