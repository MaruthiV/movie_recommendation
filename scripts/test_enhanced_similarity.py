#!/usr/bin/env python3
"""
Test script for enhanced movie similarity with explanations.
"""

import argparse
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.append('.')

from src.models.enhanced_similarity import EnhancedMovieSimilarity


def load_movies_data(data_dir: str) -> pd.DataFrame:
    """Load movies data from MovieLens dataset."""
    movies_df = pd.read_csv(f"{data_dir}/movies.csv")
    
    # Extract year from title if not already present
    if 'year' not in movies_df.columns:
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
    
    # For demo purposes, add some mock director and actor data
    # In a real implementation, you'd fetch this from TMDB API
    movies_df['director'] = 'Unknown'  # Placeholder
    movies_df['actors'] = ''  # Placeholder
    movies_df['rating'] = 3.5  # Placeholder average rating
    
    return movies_df


def get_user_preferences_interactive() -> Dict[str, Any]:
    """Get user preferences interactively."""
    print("\n🎬 Let's personalize your recommendations!")
    print("=" * 50)
    
    preferences = {}
    
    # Genre preferences
    print("\nWhat genres do you enjoy most? (comma-separated)")
    print("Options: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western")
    genres_input = input("Your preferred genres: ").strip()
    if genres_input:
        preferences['preferred_genres'] = [g.strip() for g in genres_input.split(',')]
    
    # Era preferences
    print("\nWhat movie era do you prefer?")
    print("1. 80s and earlier")
    print("2. 90s")
    print("3. 2000s")
    print("4. Recent (2010+)")
    print("5. No preference")
    era_choice = input("Your choice (1-5): ").strip()
    era_mapping = {
        '1': '80s_and_earlier',
        '2': '90s',
        '3': '2000s',
        '4': 'recent',
        '5': 'no_preference'
    }
    if era_choice in era_mapping:
        preferences['preferred_era'] = era_mapping[era_choice]
    
    # Mood preferences
    print("\nWhat mood are you in for movies?")
    print("1. Feel-good and uplifting")
    print("2. Exciting and action-packed")
    print("3. Thought-provoking and deep")
    print("4. Scary and thrilling")
    print("5. Relaxing and easy-going")
    print("6. No preference")
    mood_choice = input("Your choice (1-6): ").strip()
    mood_mapping = {
        '1': 'feel_good',
        '2': 'exciting',
        '3': 'thought_provoking',
        '4': 'scary',
        '5': 'relaxing',
        '6': 'no_preference'
    }
    if mood_choice in mood_mapping:
        preferences['preferred_mood'] = mood_mapping[mood_choice]
    
    return preferences


def find_movie_by_title(movies_df: pd.DataFrame, title_query: str) -> pd.DataFrame:
    """Find movies by title query."""
    return movies_df[movies_df['title'].str.contains(title_query, case=False, na=False)]


def display_movie_options(movies_df: pd.DataFrame, title_query: str):
    """Display movie options for user selection."""
    matching_movies = find_movie_by_title(movies_df, title_query)
    
    if matching_movies.empty:
        print(f"No movies found matching '{title_query}'")
        return None
    
    print(f"\nFound {len(matching_movies)} movies matching '{title_query}':")
    print("-" * 60)
    
    for i, (_, movie) in enumerate(matching_movies.iterrows(), 1):
        year = int(movie['year']) if not pd.isna(movie['year']) else 'Unknown'
        print(f"{i:2d}. {movie['title']:<40} | {movie['genres']:<20} | {year}")
    
    return matching_movies


def main():
    parser = argparse.ArgumentParser(description="Test enhanced movie similarity")
    parser.add_argument("--data-dir", type=str, default="data/ml-100k", help="Data directory")
    parser.add_argument("--movie-title", type=str, help="Movie title to find similar movies for")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load movies data
    logger.info("Loading movies data...")
    movies_df = load_movies_data(args.data_dir)
    
    # Initialize enhanced similarity calculator
    similarity_calculator = EnhancedMovieSimilarity(movies_df)
    
    if args.interactive:
        # Interactive mode
        print("🎬 Enhanced Movie Recommendation System")
        print("=" * 50)
        
        # Get user preferences
        user_preferences = get_user_preferences_interactive()
        
        # Get movie to find similar movies for
        print(f"\nWhat movie would you like to find similar movies for?")
        movie_query = input("Enter movie title (or part of title): ").strip()
        
        # Find matching movies
        matching_movies = display_movie_options(movies_df, movie_query)
        if matching_movies is None or matching_movies.empty:
            return
        
        # Let user select a movie
        if len(matching_movies) > 1:
            choice = input(f"\nSelect a movie (1-{len(matching_movies)}): ").strip()
            try:
                choice_idx = int(choice) - 1
                selected_movie = matching_movies.iloc[choice_idx]
            except (ValueError, IndexError):
                print("Invalid choice. Using first movie.")
                selected_movie = matching_movies.iloc[0]
        else:
            selected_movie = matching_movies.iloc[0]
        
        movie_id = selected_movie['movieId']
        movie_title = selected_movie['title']
        
        # Get feature weights based on user preferences
        feature_weights = similarity_calculator.get_user_preference_weights(user_preferences)
        
        print(f"\n🎭 Finding movies similar to '{movie_title}' based on your preferences...")
        print("=" * 80)
        
        # Find similar movies
        similar_movies = similarity_calculator.find_similar_movies(
            movie_id, 
            top_k=args.top_k,
            feature_weights=feature_weights
        )
        
        # Display results
        for i, movie in enumerate(similar_movies, 1):
            year = int(movie['year']) if not pd.isna(movie['year']) else 'Unknown'
            print(f"{i:2d}. {movie['title']:<50} | {movie['genres']:<25} | Similarity: {movie['similarity']:.3f}")
            print(f"    📝 {movie['explanation']}")
            print()
    
    elif args.movie_title:
        # Command line mode
        logger.info(f"Finding movies similar to '{args.movie_title}'")
        
        # Find the movie
        matching_movies = find_movie_by_title(movies_df, args.movie_title)
        if matching_movies.empty:
            logger.error(f"No movies found matching '{args.movie_title}'")
            return
        
        movie_id = matching_movies.iloc[0]['movieId']
        movie_title = matching_movies.iloc[0]['title']
        
        # Find similar movies
        similar_movies = similarity_calculator.find_similar_movies(movie_id, top_k=args.top_k)
        
        # Display results
        print(f"\n🎭 Movies Similar to '{movie_title}':")
        print("=" * 80)
        
        for i, movie in enumerate(similar_movies, 1):
            year = int(movie['year']) if not pd.isna(movie['year']) else 'Unknown'
            print(f"{i:2d}. {movie['title']:<50} | {movie['genres']:<25} | Similarity: {movie['similarity']:.3f}")
            print(f"    📝 {movie['explanation']}")
            print()
    
    else:
        # Demo mode - show examples
        print("🎬 Enhanced Movie Similarity Demo")
        print("=" * 50)
        
        # Example movies to test
        example_movies = [
            "Grumpier Old Men",
            "Star Wars",
            "Titanic",
            "The Matrix"
        ]
        
        for movie_title in example_movies:
            print(f"\n🎭 Movies Similar to '{movie_title}':")
            print("-" * 60)
            
            matching_movies = find_movie_by_title(movies_df, movie_title)
            if matching_movies.empty:
                print(f"No movies found matching '{movie_title}'")
                continue
            
            movie_id = matching_movies.iloc[0]['movieId']
            similar_movies = similarity_calculator.find_similar_movies(movie_id, top_k=5)
            
            for i, movie in enumerate(similar_movies, 1):
                year = int(movie['year']) if not pd.isna(movie['year']) else 'Unknown'
                print(f"{i}. {movie['title']:<40} | Similarity: {movie['similarity']:.3f}")
                print(f"   📝 {movie['explanation']}")


if __name__ == "__main__":
    main() 