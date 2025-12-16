#!/usr/bin/env python3
"""
Bradley-Terry Model for Rating Estimation

Implements the Bradley-Terry model for estimating player ratings from game results.
This model is more stable than Elo because it doesn't weigh recent games more heavily.

Reference: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class BradleyTerry:
    """
    Bradley-Terry model for rating estimation
    
    Models the probability of player i beating player j as:
    P(i > j) = R_i / (R_i + R_j)
    
    Where R_i = exp(β_i) are the ratings.
    """
    
    def __init__(self):
        """Initialize Bradley-Terry model"""
        self.ratings: Dict[str, float] = {}
        self.beta: Dict[str, float] = {}  # Log-ratings (β_i)
        self.player_index: Dict[str, int] = {}
        self.index_player: Dict[int, str] = {}
    
    def fit(
        self,
        games: List[Tuple[str, str, float, float]],
        max_iter: int = 1000,
        tol: float = 1e-6,
        initial_rating: float = 1500.0,
    ) -> Dict[str, float]:
        """
        Fit Bradley-Terry model to game results
        
        Args:
            games: List of (player1, player2, score1, score2) tuples
                  score1 is 1.0 if player1 wins, 0.0 if loses, 0.5 if draw
            max_iter: Maximum iterations for optimization
            tol: Convergence tolerance
            initial_rating: Initial rating for all players (will be adjusted)
            
        Returns:
            Dictionary mapping player names to ratings
        """
        # Collect all unique players
        players = set()
        for p1, p2, _, _ in games:
            players.add(p1)
            players.add(p2)
        
        players = sorted(list(players))
        n_players = len(players)
        
        # Create player index mapping
        self.player_index = {player: i for i, player in enumerate(players)}
        self.index_player = {i: player for i, player in enumerate(players)}
        
        # Initialize beta (log-ratings) - convert initial_rating to log scale
        # Using log scale for numerical stability
        initial_beta = np.log(initial_rating / 400.0)  # Scale to reasonable range
        beta = np.full(n_players, initial_beta)
        
        # Count wins and total games for each player
        wins = defaultdict(int)
        total_games = defaultdict(int)
        
        for p1, p2, s1, s2 in games:
            total_games[p1] += 1
            total_games[p2] += 1
            if s1 > s2:
                wins[p1] += 1
            elif s2 > s1:
                wins[p2] += 1
            # Draws don't count as wins
        
        # Maximum likelihood estimation using iterative method
        # Standard Bradley-Terry algorithm: β_i = log(wins_i / sum_over_j(prob(i beats j)))
        for iteration in range(max_iter):
            beta_old = beta.copy()
            
            # Update each player's beta
            for i, player in enumerate(players):
                if total_games[player] == 0:
                    continue
                
                # Sum over all opponents: sum of probabilities that player beats each opponent
                denominator = 0.0
                for p1, p2, s1, s2 in games:
                    if p1 == player:
                        j = self.player_index[p2]
                        # Probability that player beats opponent
                        exp_beta_i = np.exp(beta[i])
                        exp_beta_j = np.exp(beta[j])
                        prob = exp_beta_i / (exp_beta_i + exp_beta_j)
                        denominator += prob
                    elif p2 == player:
                        j = self.player_index[p1]
                        # Probability that opponent beats player (we want inverse)
                        exp_beta_i = np.exp(beta[i])
                        exp_beta_j = np.exp(beta[j])
                        prob = exp_beta_i / (exp_beta_i + exp_beta_j)
                        denominator += prob
                
                if denominator > 0 and wins[player] > 0:
                    # Update beta: log(wins / denominator)
                    beta[i] = np.log(wins[player] / denominator)
                elif wins[player] == 0:
                    # Player has no wins, set low rating
                    beta[i] = beta[i] - 0.5
            
            # Normalize: set one player's beta to 0 (for identifiability)
            # This doesn't affect relative ratings
            beta = beta - beta[0]
            
            # Check convergence
            if np.max(np.abs(beta - beta_old)) < tol:
                if iteration > 0:  # Don't print on first iteration
                    print(f"  Converged after {iteration + 1} iterations")
                break
        
        # Convert beta to ratings: R_i = exp(β_i)
        # Scale to Elo-like range (centered around 1500)
        # Since we normalized by setting beta[0] = 0, we need to scale the differences
        # Use a scale factor to convert log-ratings to Elo-like ratings
        # Standard Elo scale: 400 points = 10:1 odds ratio
        # So if beta difference is 1, that's exp(1) ≈ 2.7:1 odds, which is ~400 Elo points
        
        # Calculate range of betas for scaling
        beta_min = np.min(beta)
        beta_max = np.max(beta)
        beta_range = beta_max - beta_min if beta_max > beta_min else 1.0
        
        # Scale factor: map beta range to ~800 Elo points (400 above and below center)
        # This gives a reasonable spread
        if beta_range > 0:
            scale_factor = 800.0 / beta_range
        else:
            scale_factor = 400.0  # Default if all betas are the same
        
        # Center around initial_rating
        beta_center = np.mean(beta)
        
        # Store ratings
        self.beta = {player: float(beta[i]) for i, player in enumerate(players)}
        self.ratings = {
            player: float(initial_rating + scale_factor * (beta[i] - beta_center))
            for i, player in enumerate(players)
        }
        
        return self.ratings
    
    def get_rating(self, player: str) -> Optional[float]:
        """Get rating for a player"""
        return self.ratings.get(player)
    
    def get_probability(self, player1: str, player2: str) -> Optional[float]:
        """
        Get probability that player1 beats player2
        
        Args:
            player1: First player name
            player2: Second player name
            
        Returns:
            Probability that player1 wins (0.0 to 1.0)
        """
        if player1 not in self.beta or player2 not in self.beta:
            return None
        
        beta1 = self.beta[player1]
        beta2 = self.beta[player2]
        
        # P(i > j) = exp(β_i) / (exp(β_i) + exp(β_j))
        exp_beta1 = np.exp(beta1)
        exp_beta2 = np.exp(beta2)
        
        return exp_beta1 / (exp_beta1 + exp_beta2)
    
    def get_all_ratings(self) -> Dict[str, float]:
        """Get all player ratings"""
        return self.ratings.copy()


if __name__ == "__main__":
    # Test with simple example
    bt = BradleyTerry()
    
    # Simulate some games
    games = [
        ("Player A", "Player B", 1.0, 0.0),  # A beats B
        ("Player B", "Player C", 1.0, 0.0),  # B beats C
        ("Player A", "Player C", 1.0, 0.0),  # A beats C
        ("Player A", "Player B", 0.5, 0.5),  # Draw
    ]
    
    ratings = bt.fit(games)
    
    print("Bradley-Terry Ratings:")
    for player, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"  {player}: {rating:.1f}")
    
    print(f"\nProbability A beats B: {bt.get_probability('Player A', 'Player B'):.3f}")
    print(f"Probability A beats C: {bt.get_probability('Player A', 'Player C'):.3f}")
    print(f"Probability B beats C: {bt.get_probability('Player B', 'Player C'):.3f}")

