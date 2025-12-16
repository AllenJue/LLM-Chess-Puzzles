"""
Glicko-2 Rating System Implementation

This module implements the Glicko-2 rating system for chess puzzle evaluation.
The implementation is based on the work by Heungsub Lee.

Original implementation: https://github.com/heungsub/glicko2
Copyright (c) 2012 by Heungsub Lee
License: BSD

This is a Python port of the Glicko-2 rating system as described in:
Glickman, Mark E. "The Glicko system." Boston University (1995).
"""

import math
from typing import List, Tuple, Optional


# Constants for Glicko-2 system
MU = 1500
PHI = 350
SIGMA = 0.06
TAU = 1.0
EPSILON = 0.000001

# Game outcomes
WIN = 1.0
DRAW = 0.5
LOSS = 0.0


class Rating:
    """
    Represents a player's rating in the Glicko-2 system.
    
    Attributes:
        mu (float): Player's rating (μ)
        phi (float): Rating deviation (φ)
        sigma (float): Volatility (σ)
    """
    
    def __init__(self, mu: float = MU, phi: float = PHI, sigma: float = SIGMA):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma

    def __repr__(self):
        return f"Rating(mu={self.mu:.3f}, phi={self.phi:.3f}, sigma={self.sigma:.3f})"


class Glicko2:
    """
    Glicko-2 rating system implementation.
    
    This class provides methods to update player ratings based on game results.
    """
    
    def __init__(self, mu: float = MU, phi: float = PHI, sigma: float = SIGMA, 
                 tau: float = TAU, epsilon: float = EPSILON):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.tau = tau
        self.epsilon = epsilon

    def create_rating(self, mu: Optional[float] = None, phi: Optional[float] = None, 
                     sigma: Optional[float] = None) -> Rating:
        """
        Create a new rating object.
        
        Args:
            mu: Player's rating (default: system default)
            phi: Rating deviation (default: system default)
            sigma: Volatility (default: system default)
            
        Returns:
            Rating: New rating object
        """
        if mu is None:
            mu = self.mu
        if phi is None:
            phi = self.phi
        if sigma is None:
            sigma = self.sigma
        return Rating(mu, phi, sigma)

    def scale_down(self, rating: Rating, ratio: float = 173.7178) -> Rating:
        """Scale rating down to Glicko-2 scale."""
        mu = (rating.mu - self.mu) / ratio
        phi = rating.phi / ratio
        return self.create_rating(mu, phi, rating.sigma)

    def scale_up(self, rating: Rating, ratio: float = 173.7178) -> Rating:
        """Scale rating up from Glicko-2 scale."""
        mu = rating.mu * ratio + self.mu
        phi = rating.phi * ratio
        return self.create_rating(mu, phi, rating.sigma)

    def reduce_impact(self, rating: Rating) -> float:
        """
        Reduce the impact of games as a function of opponent's RD.
        Original form: g(RD)
        """
        return 1.0 / math.sqrt(1 + (3 * rating.phi ** 2) / (math.pi ** 2))

    def expect_score(self, rating: Rating, other_rating: Rating, impact: float) -> float:
        """Calculate expected score against an opponent."""
        return 1.0 / (1 + math.exp(-impact * (rating.mu - other_rating.mu)))

    def determine_sigma(self, rating: Rating, difference: float, variance: float) -> float:
        """
        Determine new sigma using iterative algorithm.
        
        This is the most complex part of the Glicko-2 system.
        """
        phi = rating.phi
        difference_squared = difference ** 2
        alpha = math.log(rating.sigma ** 2)

        def f(x):
            """Twice the conditional log-posterior density of phi."""
            tmp = phi ** 2 + variance + math.exp(x)
            a = math.exp(x) * (difference_squared - tmp) / (2 * tmp ** 2)
            b = (x - alpha) / (self.tau ** 2)
            return a - b

        # Set initial values
        a = alpha
        if difference_squared > phi ** 2 + variance:
            b = math.log(difference_squared - phi ** 2 - variance)
        else:
            k = 1
            while f(alpha - k * math.sqrt(self.tau ** 2)) < 0:
                k += 1
            b = alpha - k * math.sqrt(self.tau ** 2)

        # Iterative algorithm
        f_a, f_b = f(a), f(b)
        while abs(b - a) > self.epsilon:
            c = a + (a - b) * f_a / (f_b - f_a)
            f_c = f(c)
            if f_c * f_b < 0:
                a, f_a = b, f_b
            else:
                f_a /= 2
            b, f_b = c, f_c

        return math.exp(1) ** (a / 2)

    def rate(self, rating: Rating, series: List[Tuple[float, Rating]]) -> Rating:
        """
        Update rating based on a series of game results.
        
        Args:
            rating: Current player rating
            series: List of (score, opponent_rating) tuples
            
        Returns:
            Rating: Updated rating
        """
        # Convert to Glicko-2 scale
        rating = self.scale_down(rating)
        
        if not series:
            # No games played, only update volatility
            phi_star = math.sqrt(rating.phi ** 2 + rating.sigma ** 2)
            return self.scale_up(self.create_rating(rating.mu, phi_star, rating.sigma))

        # Calculate variance and difference
        variance_inv = 0
        difference = 0
        
        for actual_score, other_rating in series:
            other_rating = self.scale_down(other_rating)
            impact = self.reduce_impact(other_rating)
            expected_score = self.expect_score(rating, other_rating, impact)
            variance_inv += impact ** 2 * expected_score * (1 - expected_score)
            difference += impact * (actual_score - expected_score)

        difference /= variance_inv
        variance = 1.0 / variance_inv

        # Determine new sigma
        sigma = self.determine_sigma(rating, difference, variance)

        # Update rating deviation
        phi_star = math.sqrt(rating.phi ** 2 + sigma ** 2)

        # Update rating and RD
        phi = 1.0 / math.sqrt(1 / phi_star ** 2 + 1 / variance)
        mu = rating.mu + phi ** 2 * (difference / variance)

        # Convert back to original scale
        return self.scale_up(self.create_rating(mu, phi, sigma))

    def rate_1vs1(self, rating1: Rating, rating2: Rating, drawn: bool = False) -> Tuple[Rating, Rating]:
        """
        Update ratings for a 1v1 game.
        
        Args:
            rating1: First player's rating
            rating2: Second player's rating
            drawn: Whether the game was a draw
            
        Returns:
            Tuple of updated ratings
        """
        if drawn:
            return (self.rate(rating1, [(DRAW, rating2)]),
                    self.rate(rating2, [(DRAW, rating1)]))
        else:
            return (self.rate(rating1, [(WIN, rating2)]),
                    self.rate(rating2, [(LOSS, rating1)]))

    def quality_1vs1(self, rating1: Rating, rating2: Rating) -> float:
        """
        Calculate the quality of a 1v1 matchup.
        
        Returns a value between 0 and 1, where 1 means perfectly balanced.
        """
        expected_score1 = self.expect_score(rating1, rating2, self.reduce_impact(rating1))
        expected_score2 = self.expect_score(rating2, rating1, self.reduce_impact(rating2))
        expected_score = (expected_score1 + expected_score2) / 2
        return 2 * (0.5 - abs(0.5 - expected_score))


def update_agent_rating_from_puzzles(df, initial_rating: Optional[Rating] = None) -> Rating:
    """
    Update agent's Glicko2 rating based on puzzle results.
    
    Args:
        df: DataFrame containing puzzle results with columns:
            - 'Rating': Puzzle rating
            - 'RatingDeviation': Puzzle rating deviation
            - 'puzzle_solved': Boolean indicating if puzzle was solved
        initial_rating: Starting rating for the agent
        
    Returns:
        Rating: Updated agent rating
    """
    if initial_rating is None:
        initial_rating = Rating()  # Default 1500, 350, 0.06

    glicko_system = Glicko2()
    rating = initial_rating

    series = []
    for idx, row in df.iterrows():
        puzzle_mu = row['Rating']
        puzzle_phi = row.get('RatingDeviation', 350)  # Use default if missing
        puzzle_sigma = 0.06  # Fixed sigma for puzzles

        puzzle_rating = Rating(mu=puzzle_mu, phi=puzzle_phi, sigma=puzzle_sigma)
        outcome = 1 if row['puzzle_solved'] else 0  # Win=1, Loss=0

        series.append((outcome, puzzle_rating))

    if series:
        rating = glicko_system.rate(rating, series)
        print("\nAgent rating updated.")
    else:
        print("\nNo puzzles to update from.")

    print(f"Final Agent Rating: {rating.mu:.2f}")
    print(f"Final Agent RD:     {rating.phi:.2f}")
    print(f"Final Agent Sigma:  {rating.sigma:.4f}")

    return rating


if __name__ == "__main__":
    # Example usage
    print("Glicko-2 Rating System Example")
    print("=" * 40)
    
    # Create a new player
    new_player = Rating()
    print(f"New player's initial rating: {new_player}")

    # Example of updating a player's rating after a series of matches
    player_rating = Rating(mu=1500, phi=350, sigma=0.06)
    opponent_rating = Rating(mu=1400, phi=30, sigma=0.06)
    series = [(WIN, opponent_rating), (LOSS, opponent_rating), (DRAW, opponent_rating)]
    
    glicko2_system = Glicko2()
    updated_rating = glicko2_system.rate(player_rating, series)
    print(f"Updated player's rating after series: {updated_rating}")
    
    # Test 1v1 rating update
    rating1, rating2 = glicko2_system.rate_1vs1(player_rating, opponent_rating, drawn=False)
    print(f"After 1v1 (player wins): {rating1}")
    print(f"After 1v1 (opponent loses): {rating2}")
    
    # Calculate matchup quality
    quality = glicko2_system.quality_1vs1(player_rating, opponent_rating)
    print(f"Matchup quality: {quality:.3f}")


