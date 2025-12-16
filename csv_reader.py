"""
CSV Reader Module for Chess Puzzles

This module handles reading and processing CSV files containing chess puzzle data.
"""

import pandas as pd
import os
from typing import Optional, Tuple


def read_chess_puzzles_csv(file_path: str) -> pd.DataFrame:
    """
    Read chess puzzles from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing puzzle data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: For other reading errors
    """
    try:
        df = pd.read_csv(file_path)
        print(f"CSV file loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        raise
    except Exception as e:
        print(f"An error occurred while reading CSV: {e}")
        raise


def sample_puzzles(df: pd.DataFrame, n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Sample random puzzles from the DataFrame.
    
    Args:
        df (pd.DataFrame): Source DataFrame
        n (int): Number of puzzles to sample
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Sampled DataFrame
    """
    if len(df) < n:
        print(f"Warning: Requested {n} samples but only {len(df)} available")
        return df.copy()
    
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def save_puzzles_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Save puzzles DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        file_path (str): Output file path
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
        raise


def get_puzzle_by_index(df: pd.DataFrame, index: int) -> pd.Series:
    """
    Get a specific puzzle by index.
    
    Args:
        df (pd.DataFrame): Source DataFrame
        index (int): Index of the puzzle
        
    Returns:
        pd.Series: Puzzle data
    """
    if index >= len(df):
        raise IndexError(f"Index {index} out of range for DataFrame with {len(df)} rows")
    
    return df.iloc[index]


def get_puzzle_stats(df: pd.DataFrame) -> dict:
    """
    Get basic statistics about the puzzles.
    
    Args:
        df (pd.DataFrame): Puzzle DataFrame
        
    Returns:
        dict: Dictionary containing statistics
    """
    stats = {
        'total_puzzles': len(df),
        'rating_stats': df['Rating'].describe() if 'Rating' in df.columns else None,
        'popularity_stats': df['Popularity'].describe() if 'Popularity' in df.columns else None,
        'nb_plays_stats': df['NbPlays'].describe() if 'NbPlays' in df.columns else None,
    }
    
    return stats


def filter_puzzles_by_rating(df: pd.DataFrame, min_rating: Optional[int] = None, 
                           max_rating: Optional[int] = None) -> pd.DataFrame:
    """
    Filter puzzles by rating range.
    
    Args:
        df (pd.DataFrame): Source DataFrame
        min_rating (int, optional): Minimum rating
        max_rating (int, optional): Maximum rating
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if min_rating is not None:
        filtered_df = filtered_df[filtered_df['Rating'] >= min_rating]
    
    if max_rating is not None:
        filtered_df = filtered_df[filtered_df['Rating'] <= max_rating]
    
    return filtered_df


def filter_puzzles_by_theme(df: pd.DataFrame, themes: list) -> pd.DataFrame:
    """
    Filter puzzles by themes.
    
    Args:
        df (pd.DataFrame): Source DataFrame
        themes (list): List of themes to filter by
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if 'Themes' not in df.columns:
        print("Warning: 'Themes' column not found")
        return df.copy()
    
    def has_theme(themes_str, target_themes):
        if pd.isna(themes_str):
            return False
        puzzle_themes = themes_str.split()
        return any(theme in puzzle_themes for theme in target_themes)
    
    mask = df['Themes'].apply(lambda x: has_theme(x, themes))
    return df[mask].copy()


if __name__ == "__main__":
    # Example usage
    csv_file = "lichess_puzzles_with_pgn_1000.csv"
    
    if os.path.exists(csv_file):
        df = read_chess_puzzles_csv(csv_file)
        print(f"Loaded {len(df)} puzzles")
        
        # Sample some puzzles
        sample_df = sample_puzzles(df, n=10)
        print(f"Sampled {len(sample_df)} puzzles")
        
        # Get stats
        stats = get_puzzle_stats(df)
        print("Puzzle statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print(f"CSV file {csv_file} not found")


