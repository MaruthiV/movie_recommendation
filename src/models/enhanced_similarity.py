"""
Enhanced movie similarity calculation using multiple features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from collections import Counter
import re


class EnhancedMovieSimilarity:
    """Enhanced movie similarity calculator using multiple features."""
    
    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize with movies dataframe.
        
        Args:
            movies_df: DataFrame with columns: movieId, title, genres, year, director, actors, etc.
        """
        self.movies_df = movies_df.copy()
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess movie data for similarity calculations."""
        # Extract year from title if not present
        if 'year' not in self.movies_df.columns:
            self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
        
        # Parse genres into sets
        self.movies_df['genre_set'] = self.movies_df['genres'].str.split('|').apply(set)
        
        # Parse actors into sets (if available)
        if 'actors' in self.movies_df.columns:
            self.movies_df['actor_set'] = self.movies_df['actors'].str.split('|').apply(set)
        else:
            self.movies_df['actor_set'] = self.movies_df.apply(lambda x: set(), axis=1)
        
        # Create mood categories based on genres
        self.movies_df['mood'] = self.movies_df['genre_set'].apply(self._categorize_mood)
        
        # Create budget categories (if available)
        if 'budget' in self.movies_df.columns:
            self.movies_df['budget_category'] = self.movies_df['budget'].apply(self._categorize_budget)
        else:
            self.movies_df['budget_category'] = 'unknown'
    
    def _categorize_mood(self, genres: Set[str]) -> str:
        """Categorize movie mood based on genres."""
        mood_mapping = {
            'feel_good': {'Comedy', 'Romance', 'Musical', 'Family'},
            'exciting': {'Action', 'Adventure', 'Thriller', 'War'},
            'thought_provoking': {'Drama', 'Documentary', 'Biography'},
            'scary': {'Horror', 'Mystery'},
            'relaxing': {'Animation', 'Family', 'Comedy'},
            'intense': {'Thriller', 'Crime', 'Mystery'}
        }
        
        for mood, mood_genres in mood_mapping.items():
            if genres & mood_genres:
                return mood
        
        return 'neutral'
    
    def _categorize_budget(self, budget: float) -> str:
        """Categorize movie budget."""
        if pd.isna(budget):
            return 'unknown'
        elif budget < 1000000:
            return 'low'
        elif budget < 50000000:
            return 'medium'
        else:
            return 'high'
    
    def calculate_similarity(self, movie1_id: int, movie2_id: int, 
                           feature_weights: Dict[str, float] = None) -> float:
        """
        Calculate similarity between two movies using multiple features.
        
        Args:
            movie1_id: ID of first movie
            movie2_id: ID of second movie
            feature_weights: Weights for different features (default: equal weights)
        
        Returns:
            Similarity score between 0 and 1
        """
        movie1 = self.movies_df[self.movies_df['movieId'] == movie1_id].iloc[0]
        movie2 = self.movies_df[self.movies_df['movieId'] == movie2_id].iloc[0]
        
        # Default weights
        if feature_weights is None:
            feature_weights = {
                'genre': 0.3,
                'year': 0.15,
                'director': 0.2,
                'actors': 0.15,
                'mood': 0.1,
                'budget': 0.05,
                'rating': 0.05
            }
        
        # Calculate individual similarities
        similarities = {
            'genre': self._genre_similarity(movie1, movie2),
            'year': self._year_similarity(movie1, movie2),
            'director': self._director_similarity(movie1, movie2),
            'actors': self._actor_similarity(movie1, movie2),
            'mood': self._mood_similarity(movie1, movie2),
            'budget': self._budget_similarity(movie1, movie2),
            'rating': self._rating_similarity(movie1, movie2)
        }
        
        # Calculate weighted similarity
        weighted_similarity = sum(
            similarities[feature] * feature_weights[feature]
            for feature in feature_weights
        )
        
        return weighted_similarity
    
    def _genre_similarity(self, movie1: pd.Series, movie2: pd.Series) -> float:
        """Calculate genre similarity using Jaccard index."""
        genres1 = movie1['genre_set']
        genres2 = movie2['genre_set']
        
        if not genres1 or not genres2:
            return 0.0
        
        intersection = len(genres1 & genres2)
        union = len(genres1 | genres2)
        
        return intersection / union if union > 0 else 0.0
    
    def _year_similarity(self, movie1: pd.Series, movie2: pd.Series) -> float:
        """Calculate year similarity using exponential decay."""
        year1 = movie1['year']
        year2 = movie2['year']
        
        if pd.isna(year1) or pd.isna(year2):
            return 0.5  # Neutral similarity for unknown years
        
        year_diff = abs(year1 - year2)
        return np.exp(-year_diff / 10)  # Decay over 10 years
    
    def _director_similarity(self, movie1: pd.Series, movie2: pd.Series) -> float:
        """Calculate director similarity."""
        director1 = movie1.get('director', '')
        director2 = movie2.get('director', '')
        
        if not director1 or not director2:
            return 0.0
        
        return 1.0 if director1 == director2 else 0.0
    
    def _actor_similarity(self, movie1: pd.Series, movie2: pd.Series) -> float:
        """Calculate actor similarity using Jaccard index."""
        actors1 = movie1['actor_set']
        actors2 = movie2['actor_set']
        
        if not actors1 or not actors2:
            return 0.0
        
        intersection = len(actors1 & actors2)
        union = len(actors1 | actors2)
        
        return intersection / union if union > 0 else 0.0
    
    def _mood_similarity(self, movie1: pd.Series, movie2: pd.Series) -> float:
        """Calculate mood similarity."""
        mood1 = movie1['mood']
        mood2 = movie2['mood']
        
        return 1.0 if mood1 == mood2 else 0.0
    
    def _budget_similarity(self, movie1: pd.Series, movie2: pd.Series) -> float:
        """Calculate budget category similarity."""
        budget1 = movie1['budget_category']
        budget2 = movie2['budget_category']
        
        return 1.0 if budget1 == budget2 else 0.0
    
    def _rating_similarity(self, movie1: pd.Series, movie2: pd.Series) -> float:
        """Calculate rating similarity."""
        rating1 = movie1.get('rating', 0)
        rating2 = movie2.get('rating', 0)
        
        if rating1 == 0 or rating2 == 0:
            return 0.5  # Neutral similarity for unknown ratings
        
        rating_diff = abs(rating1 - rating2)
        return 1 / (1 + rating_diff / 2)  # Decay over 2 rating points
    
    def find_similar_movies(self, movie_id: int, top_k: int = 10,
                          feature_weights: Dict[str, float] = None,
                          exclude_self: bool = True) -> List[Dict[str, Any]]:
        """
        Find movies similar to a given movie.
        
        Args:
            movie_id: ID of target movie
            top_k: Number of similar movies to return
            feature_weights: Weights for different features
            exclude_self: Whether to exclude the target movie from results
        
        Returns:
            List of similar movies with similarity scores and explanations
        """
        if movie_id not in self.movies_df['movieId'].values:
            raise ValueError(f"Movie ID {movie_id} not found in dataset")
        
        similarities = []
        
        for _, movie in self.movies_df.iterrows():
            if exclude_self and movie['movieId'] == movie_id:
                continue
            
            similarity = self.calculate_similarity(movie_id, movie['movieId'], feature_weights)
            explanation = self._generate_explanation(movie_id, movie['movieId'])
            
            similarities.append({
                'movie_id': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'year': movie['year'],
                'similarity': similarity,
                'explanation': explanation
            })
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def _generate_explanation(self, movie1_id: int, movie2_id: int) -> str:
        """Generate explanation for why two movies are similar."""
        movie1 = self.movies_df[self.movies_df['movieId'] == movie1_id].iloc[0]
        movie2 = self.movies_df[self.movies_df['movieId'] == movie2_id].iloc[0]
        
        reasons = []
        
        # Genre similarity
        shared_genres = movie1['genre_set'] & movie2['genre_set']
        if shared_genres:
            genre_list = ', '.join(sorted(shared_genres))
            reasons.append(f"Both are {genre_list} movies")
        
        # Director similarity
        director1 = movie1.get('director', '')
        director2 = movie2.get('director', '')
        if director1 and director2 and director1 == director2:
            reasons.append(f"Both directed by {director1}")
        
        # Actor similarity
        shared_actors = movie1['actor_set'] & movie2['actor_set']
        if shared_actors:
            actor_list = ', '.join(sorted(list(shared_actors))[:2])  # Top 2 actors
            reasons.append(f"Both feature {actor_list}")
        
        # Year similarity
        year1 = movie1['year']
        year2 = movie2['year']
        if not pd.isna(year1) and not pd.isna(year2) and abs(year1 - year2) <= 5:
            reasons.append(f"Both from the same era ({int(year1)} and {int(year2)})")
        
        # Mood similarity
        if movie1['mood'] == movie2['mood'] and movie1['mood'] != 'neutral':
            mood_display = movie1['mood'].replace('_', ' ').title()
            reasons.append(f"Both have a {mood_display} feel")
        
        if reasons:
            return " and ".join(reasons)
        else:
            return "Similar in style and tone"
    
    def get_user_preference_weights(self, user_preferences: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert user preferences to feature weights.
        
        Args:
            user_preferences: Dictionary of user preferences
        
        Returns:
            Dictionary of feature weights
        """
        # Default weights
        weights = {
            'genre': 0.3,
            'year': 0.15,
            'director': 0.2,
            'actors': 0.15,
            'mood': 0.1,
            'budget': 0.05,
            'rating': 0.05
        }
        
        # Adjust weights based on user preferences
        if 'preferred_genres' in user_preferences:
            weights['genre'] += 0.1
            weights['mood'] += 0.05
        
        if 'preferred_era' in user_preferences:
            weights['year'] += 0.1
        
        if 'preferred_directors' in user_preferences:
            weights['director'] += 0.1
        
        if 'preferred_actors' in user_preferences:
            weights['actors'] += 0.1
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights 