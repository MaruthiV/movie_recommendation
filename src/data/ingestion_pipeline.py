import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import requests
import zipfile
import os
import time
import json
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta

from src.data.tmdb_client import TMDBClient, TMDBMovie
from src.data.movie_database import MovieDatabase, Movie, Person
from src.data.graph_database import GraphDatabase
from src.data.vector_database import VectorDatabaseManager as VectorDatabase
try:
    from src.data.feature_store_manager import FeatureStoreManager
except ImportError:
    FeatureStoreManager = None
from src.config.api_config import APIConfig
from src.config.database_config import DatabaseConfig

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Comprehensive data ingestion pipeline for MovieLens-20M and TMDB data."""
    
    def __init__(self, 
                 tmdb_api_key: Optional[str] = None,
                 data_dir: str = "data",
                 sample_size: Optional[int] = None):
        """
        Initialize the data ingestion pipeline.
        
        Args:
            tmdb_api_key: TMDB API key for fetching movie metadata
            data_dir: Directory to store downloaded data
            sample_size: Number of movies to process (None for all)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.sample_size = sample_size
        
        # Initialize clients and databases
        self.tmdb_client = TMDBClient(tmdb_api_key or APIConfig.TMDB_API_KEY)
        self.movie_db = MovieDatabase()
        self.graph_db = GraphDatabase()
        self.vector_db = VectorDatabase()
        if FeatureStoreManager is not None:
            self.feature_store = FeatureStoreManager()
        else:
            self.feature_store = None
        
        # Data storage
        self.movielens_data = {}
        self.tmdb_data = {}
        self.processed_data = {}
        
        logger.info("Data ingestion pipeline initialized")
    
    def download_movielens_data(self) -> Dict[str, pd.DataFrame]:
        """Download and extract MovieLens-20M dataset."""
        logger.info("Downloading MovieLens-20M dataset...")
        
        movielens_url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
        zip_path = self.data_dir / "ml-20m.zip"
        extract_path = self.data_dir / "ml-20m"
        
        # Download if not exists
        if not zip_path.exists():
            logger.info("Downloading MovieLens-20M zip file...")
            response = requests.get(movielens_url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract if not exists
        if not extract_path.exists():
            logger.info("Extracting MovieLens-20M data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
        
        # Load data files
        logger.info("Loading MovieLens data files...")
        self.movielens_data = {
            'ratings': pd.read_csv(extract_path / "ratings.csv"),
            'movies': pd.read_csv(extract_path / "movies.csv"),
            'tags': pd.read_csv(extract_path / "tags.csv") if (extract_path / "tags.csv").exists() else pd.DataFrame(),
            'genome_scores': pd.read_csv(extract_path / "genome-scores.csv") if (extract_path / "genome-scores.csv").exists() else pd.DataFrame(),
            'genome_tags': pd.read_csv(extract_path / "genome-tags.csv") if (extract_path / "genome-tags.csv").exists() else pd.DataFrame()
        }
        
        logger.info(f"Loaded MovieLens data: {len(self.movielens_data['ratings'])} ratings, {len(self.movielens_data['movies'])} movies")
        return self.movielens_data
    
    def fetch_tmdb_metadata(self, movie_ids: List[int]) -> Dict[int, Dict]:
        """Fetch movie metadata from TMDB API."""
        logger.info(f"Fetching TMDB metadata for {len(movie_ids)} movies...")
        
        tmdb_data = {}
        failed_movies = []
        
        for movie_id in tqdm(movie_ids, desc="Fetching TMDB data"):
            try:
                # Get movie details
                movie_details = self.tmdb_client.get_movie_details(movie_id)
                if movie_details:
                    tmdb_data[movie_id] = movie_details
                
                # Rate limiting
                time.sleep(0.1)  # 10 requests per second
                
            except Exception as e:
                logger.warning(f"Failed to fetch TMDB data for movie {movie_id}: {e}")
                failed_movies.append(movie_id)
        
        logger.info(f"Successfully fetched TMDB data for {len(tmdb_data)} movies")
        if failed_movies:
            logger.warning(f"Failed to fetch data for {len(failed_movies)} movies")
        
        self.tmdb_data = tmdb_data
        return tmdb_data
    
    def transform_movielens_data(self) -> Dict[str, pd.DataFrame]:
        """Transform MovieLens data into our schema."""
        logger.info("Transforming MovieLens data...")
        
        # Transform movies data
        movies_df = self.movielens_data['movies'].copy()
        
        # Extract year from title
        movies_df['release_year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
        movies_df['title_clean'] = movies_df['title'].str.replace(r'\(\d{4}\)', '').str.strip()
        
        # Split genres
        movies_df['genres_list'] = movies_df['genres'].str.split('|')
        
        # Transform ratings data
        ratings_df = self.movielens_data['ratings'].copy()
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        
        # Calculate movie statistics
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count', 'std'],
            'timestamp': ['min', 'max']
        }).round(2)
        
        movie_stats.columns = ['avg_rating', 'total_ratings', 'rating_std', 'first_rating', 'last_rating']
        movie_stats = movie_stats.reset_index()
        
        # Merge with movies data
        movies_transformed = movies_df.merge(movie_stats, left_on='movieId', right_on='movieId', how='left')
        
        # Transform user statistics
        user_stats = ratings_df.groupby('userId').agg({
            'rating': ['mean', 'count', 'std'],
            'timestamp': ['min', 'max']
        }).round(2)
        
        user_stats.columns = ['avg_rating', 'total_ratings', 'rating_std', 'first_rating', 'last_rating']
        user_stats = user_stats.reset_index()
        
        # Calculate user activity level
        user_stats['activity_level'] = pd.cut(
            user_stats['total_ratings'],
            bins=[0, 10, 50, 100, 500, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Get user preferred genres
        user_genres = ratings_df.merge(movies_df[['movieId', 'genres_list']], on='movieId')
        user_genres = user_genres.explode('genres_list')
        user_genre_counts = user_genres.groupby(['userId', 'genres_list']).size().reset_index(name='count')
        user_preferred_genres = user_genre_counts.loc[
            user_genre_counts.groupby('userId')['count'].idxmax()
        ][['userId', 'genres_list']].rename(columns={'genres_list': 'preferred_genre'})
        
        user_stats = user_stats.merge(user_preferred_genres, on='userId', how='left')
        
        self.processed_data = {
            'movies': movies_transformed,
            'ratings': ratings_df,
            'users': user_stats,
            'tags': self.movielens_data.get('tags', pd.DataFrame()),
            'genome_scores': self.movielens_data.get('genome_scores', pd.DataFrame()),
            'genome_tags': self.movielens_data.get('genome_tags', pd.DataFrame())
        }
        
        logger.info(f"Transformed data: {len(movies_transformed)} movies, {len(ratings_df)} ratings, {len(user_stats)} users")
        return self.processed_data
    
    def merge_tmdb_data(self) -> pd.DataFrame:
        """Merge TMDB metadata with MovieLens data."""
        logger.info("Merging TMDB metadata with MovieLens data...")
        
        movies_df = self.processed_data['movies'].copy()
        
        # Create TMDB data DataFrame
        tmdb_list = []
        for movie_id, tmdb_info in self.tmdb_data.items():
            tmdb_list.append({
                'movieId': movie_id,
                'tmdb_id': tmdb_info.get('id'),
                'title': tmdb_info.get('title'),
                'overview': tmdb_info.get('overview'),
                'poster_path': tmdb_info.get('poster_path'),
                'backdrop_path': tmdb_info.get('backdrop_path'),
                'release_date': tmdb_info.get('release_date'),
                'runtime': tmdb_info.get('runtime'),
                'budget': tmdb_info.get('budget'),
                'revenue': tmdb_info.get('revenue'),
                'popularity': tmdb_info.get('popularity'),
                'vote_average': tmdb_info.get('vote_average'),
                'vote_count': tmdb_info.get('vote_count'),
                'status': tmdb_info.get('status'),
                'original_language': tmdb_info.get('original_language'),
                'production_companies': json.dumps(tmdb_info.get('production_companies', [])),
                'genres': json.dumps(tmdb_info.get('genres', [])),
                'spoken_languages': json.dumps(tmdb_info.get('spoken_languages', [])),
                'cast': json.dumps(tmdb_info.get('credits', {}).get('cast', [])[:10]),  # Top 10 cast
                'crew': json.dumps(tmdb_info.get('credits', {}).get('crew', [])[:10]),  # Top 10 crew
                'keywords': json.dumps(tmdb_info.get('keywords', {}).get('keywords', [])),
                'videos': json.dumps(tmdb_info.get('videos', {}).get('results', []))
            })
        
        tmdb_df = pd.DataFrame(tmdb_list)
        
        # Merge with MovieLens data
        movies_merged = movies_df.merge(tmdb_df, on='movieId', how='left')
        
        # Update release year from TMDB if available
        movies_merged['release_year'] = movies_merged['release_year'].fillna(
            pd.to_datetime(movies_merged['release_date']).dt.year
        )
        
        self.processed_data['movies'] = movies_merged
        
        logger.info(f"Merged TMDB data: {len(tmdb_df)} movies with TMDB metadata")
        return movies_merged
    
    def validate_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Validate the processed data for quality issues using the enhanced quality system."""
        logger.info("Validating data quality using enhanced validation system...")
        
        try:
            # Import quality components
            from .quality import DataValidator, QualityMetrics, QualityMonitor, QualityReporter
            
            # Initialize quality components
            data_validator = DataValidator()
            quality_metrics = QualityMetrics()
            quality_monitor = QualityMonitor(data_validator, quality_metrics)
            quality_reporter = QualityReporter(data_validator, quality_metrics, quality_monitor)
            
            # Prepare data for validation
            data = {
                'movies': self.processed_data['movies'],
                'ratings': self.processed_data['ratings'],
                'users': self.processed_data['users']
            }
            
            # Perform comprehensive validation
            validation_results = data_validator.validate_data(data, self.tmdb_data)
            
            # Calculate quality score
            data_stats = {table: len(df) for table, df in data.items()}
            quality_score = quality_metrics.calculate_quality_score(validation_results, data_stats, self.tmdb_data)
            
            # Monitor quality and generate alerts
            quality_monitor.monitor_data_quality(data, self.tmdb_data, trigger_alerts=True)
            
            # Generate quality report
            quality_report = quality_reporter.generate_comprehensive_report(
                data, self.tmdb_data, include_trends=True, include_recommendations=True
            )
            
            # Log quality summary
            logger.info(f"Quality Score: {quality_score.score:.2f} ({quality_score.level.value})")
            logger.info(f"Total Issues: {quality_score.issues_count}")
            logger.info(f"Critical Issues: {quality_score.critical_issues}")
            logger.info(f"Warning Issues: {quality_score.warning_issues}")
            
            # Log insights and recommendations
            insights = quality_metrics.generate_quality_insights(quality_score, validation_results)
            recommendations = quality_metrics.get_quality_recommendations(quality_score, validation_results)
            
            for insight in insights:
                logger.info(f"Quality Insight: {insight}")
            
            for recommendation in recommendations[:3]:  # Log top 3 recommendations
                logger.info(f"Quality Recommendation: {recommendation}")
            
            # Store quality components for later use
            self.quality_validator = data_validator
            self.quality_metrics = quality_metrics
            self.quality_monitor = quality_monitor
            self.quality_reporter = quality_reporter
            self.quality_score = quality_score
            self.quality_report = quality_report
            
            return validation_results
            
        except ImportError as e:
            logger.warning(f"Quality system not available: {e}. Falling back to basic validation.")
            return self._basic_validate_data()
        except Exception as e:
            logger.error(f"Error in enhanced validation: {e}. Falling back to basic validation.")
            return self._basic_validate_data()
    
    def _basic_validate_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Basic validation fallback when quality system is not available."""
        logger.info("Performing basic data validation...")
        
        issues = {
            'movies': [],
            'ratings': [],
            'users': [],
            'tmdb': []
        }
        
        movies_df = self.processed_data['movies']
        ratings_df = self.processed_data['ratings']
        users_df = self.processed_data['users']
        
        # Movie validation
        if movies_df['movieId'].duplicated().any():
            issues['movies'].append({
                'rule': 'movie_id_unique',
                'severity': 'critical',
                'message': 'Duplicate movie IDs found',
                'row_data': {}
            })
        
        if movies_df['avg_rating'].isna().sum() > 0:
            issues['movies'].append({
                'rule': 'movie_rating_missing',
                'severity': 'warning',
                'message': f"{movies_df['avg_rating'].isna().sum()} movies have no ratings",
                'row_data': {}
            })
        
        if (movies_df['avg_rating'] < 0).any() or (movies_df['avg_rating'] > 5).any():
            issues['movies'].append({
                'rule': 'movie_rating_range',
                'severity': 'warning',
                'message': 'Invalid rating values found',
                'row_data': {}
            })
        
        # Rating validation
        if ratings_df['rating'].isna().sum() > 0:
            issues['ratings'].append({
                'rule': 'rating_missing',
                'severity': 'critical',
                'message': f"{ratings_df['rating'].isna().sum()} ratings are missing",
                'row_data': {}
            })
        
        if (ratings_df['rating'] < 0).any() or (ratings_df['rating'] > 5).any():
            issues['ratings'].append({
                'rule': 'rating_range',
                'severity': 'critical',
                'message': 'Invalid rating values found',
                'row_data': {}
            })
        
        # User validation
        if users_df['userId'].duplicated().any():
            issues['users'].append({
                'rule': 'user_id_unique',
                'severity': 'critical',
                'message': 'Duplicate user IDs found',
                'row_data': {}
            })
        
        # TMDB validation
        tmdb_movies = movies_df[movies_df['tmdb_id'].notna()]
        if len(tmdb_movies) < len(movies_df) * 0.5:
            issues['tmdb'].append({
                'rule': 'tmdb_coverage',
                'severity': 'info',
                'message': f"Only {len(tmdb_movies)}/{len(movies_df)} movies have TMDB metadata",
                'row_data': {}
            })
        
        # Log issues
        for category, category_issues in issues.items():
            if category_issues:
                logger.warning(f"{category.upper()} issues: {len(category_issues)} found")
            else:
                logger.info(f"{category.upper()} data validation passed")
        
        return issues
    
    def load_to_postgresql(self) -> bool:
        """Load data to PostgreSQL database."""
        logger.info("Loading data to PostgreSQL...")
        
        try:
            movies_df = self.processed_data['movies']
            ratings_df = self.processed_data['ratings']
            users_df = self.processed_data['users']
            
            # Load movies
            logger.info("Loading movies to PostgreSQL...")
            for _, movie in movies_df.iterrows():
                self.movie_db.add_movie(
                    movie_id=int(movie['movieId']),
                    title=movie['title_clean'],
                    genres=movie['genres_list'],
                    release_year=int(movie['release_year']) if pd.notna(movie['release_year']) else None,
                    avg_rating=float(movie['avg_rating']) if pd.notna(movie['avg_rating']) else None,
                    total_ratings=int(movie['total_ratings']) if pd.notna(movie['total_ratings']) else 0,
                    tmdb_id=int(movie['tmdb_id']) if pd.notna(movie['tmdb_id']) else None,
                    overview=movie['overview'],
                    poster_path=movie['poster_path'],
                    budget=int(movie['budget']) if pd.notna(movie['budget']) else None,
                    revenue=int(movie['revenue']) if pd.notna(movie['revenue']) else None,
                    popularity=float(movie['popularity']) if pd.notna(movie['popularity']) else None
                )
            
            # Load ratings (sample for performance)
            logger.info("Loading ratings to PostgreSQL...")
            sample_ratings = ratings_df.sample(min(10000, len(ratings_df)))  # Sample for demo
            for _, rating in sample_ratings.iterrows():
                self.movie_db.add_rating(
                    user_id=int(rating['userId']),
                    movie_id=int(rating['movieId']),
                    rating=float(rating['rating']),
                    timestamp=rating['timestamp']
                )
            
            logger.info("PostgreSQL data loading completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data to PostgreSQL: {e}")
            return False
    
    def load_to_neo4j(self) -> bool:
        """Load data to Neo4j graph database."""
        logger.info("Loading data to Neo4j...")
        
        try:
            movies_df = self.processed_data['movies']
            ratings_df = self.processed_data['ratings']
            
            # Create movie nodes
            logger.info("Creating movie nodes in Neo4j...")
            for _, movie in movies_df.iterrows():
                self.graph_db.create_movie_node(
                    movie_id=int(movie['movieId']),
                    title=movie['title_clean'],
                    genres=movie['genres_list'],
                    release_year=int(movie['release_year']) if pd.notna(movie['release_year']) else None
                )
            
            # Create rating relationships (sample for performance)
            logger.info("Creating rating relationships in Neo4j...")
            sample_ratings = ratings_df.sample(min(5000, len(ratings_df)))  # Sample for demo
            for _, rating in sample_ratings.iterrows():
                self.graph_db.create_rating_relationship(
                    user_id=int(rating['userId']),
                    movie_id=int(rating['movieId']),
                    rating=float(rating['rating']),
                    timestamp=rating['timestamp']
                )
            
            logger.info("Neo4j data loading completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data to Neo4j: {e}")
            return False
    
    def load_to_feature_store(self) -> bool:
        """Load data to Feast feature store."""
        if self.feature_store is None:
            logger.warning("Feature store is not available (Feast not installed). Skipping feature store loading.")
            return False
        logger.info("Loading data to Feast feature store...")
        
        try:
            movies_df = self.processed_data['movies']
            users_df = self.processed_data['users']
            ratings_df = self.processed_data['ratings']
            
            # Prepare user features
            user_features = users_df.copy()
            user_features['event_timestamp'] = datetime.now()
            user_features = user_features.rename(columns={
                'userId': 'user_id',
                'avg_rating': 'user_avg_rating',
                'total_ratings': 'user_total_ratings',
                'preferred_genre': 'user_preferred_genres',
                'activity_level': 'user_activity_level'
            })
            
            # Prepare movie features
            movie_features = movies_df.copy()
            movie_features['event_timestamp'] = datetime.now()
            movie_features = movie_features.rename(columns={
                'movieId': 'movie_id',
                'avg_rating': 'movie_avg_rating',
                'total_ratings': 'movie_total_ratings',
                'popularity': 'movie_popularity',
                'genres_list': 'movie_genres',
                'release_year': 'movie_release_year',
                'budget': 'movie_budget',
                'revenue': 'movie_revenue'
            })
            
            # Save to parquet files
            user_features.to_parquet(self.data_dir / "user_stats.parquet", index=False)
            movie_features.to_parquet(self.data_dir / "movie_stats.parquet", index=False)
            
            logger.info("Feast feature store data loading completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data to Feast feature store: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete data ingestion pipeline."""
        logger.info("Starting full data ingestion pipeline...")
        
        try:
            # Step 1: Download MovieLens data
            self.download_movielens_data()
            
            # Step 2: Transform MovieLens data
            self.transform_movielens_data()
            
            # Step 3: Sample data if specified
            if self.sample_size:
                logger.info(f"Sampling {self.sample_size} movies...")
                sample_movies = self.processed_data['movies'].sample(self.sample_size)
                movie_ids = sample_movies['movieId'].tolist()
            else:
                movie_ids = self.processed_data['movies']['movieId'].tolist()
            
            # Step 4: Fetch TMDB metadata
            self.fetch_tmdb_metadata(movie_ids)
            
            # Step 5: Merge TMDB data
            self.merge_tmdb_data()
            
            # Step 6: Validate data
            issues = self.validate_data()
            
            # Step 7: Load to storage systems
            postgres_success = self.load_to_postgresql()
            neo4j_success = self.load_to_neo4j()
            feature_store_success = self.load_to_feature_store()
            
            # Summary
            logger.info("Data ingestion pipeline completed!")
            logger.info(f"Processed {len(self.processed_data['movies'])} movies")
            logger.info(f"Processed {len(self.processed_data['ratings'])} ratings")
            logger.info(f"Processed {len(self.processed_data['users'])} users")
            logger.info(f"TMDB metadata: {len(self.tmdb_data)} movies")
            logger.info(f"PostgreSQL: {'✓' if postgres_success else '✗'}")
            logger.info(f"Neo4j: {'✓' if neo4j_success else '✗'}")
            logger.info(f"Feature Store: {'✓' if feature_store_success else '✗'}")
            
            return all([postgres_success, neo4j_success, feature_store_success])
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
    
    def get_pipeline_stats(self) -> Dict:
        """Get statistics about the ingested data."""
        if not self.processed_data:
            return {}
        
        return {
            'movies': {
                'total': len(self.processed_data['movies']),
                'with_tmdb': len(self.processed_data['movies'][self.processed_data['movies']['tmdb_id'].notna()]),
                'with_ratings': len(self.processed_data['movies'][self.processed_data['movies']['avg_rating'].notna()]),
                'avg_rating': self.processed_data['movies']['avg_rating'].mean(),
                'avg_ratings_count': self.processed_data['movies']['total_ratings'].mean()
            },
            'ratings': {
                'total': len(self.processed_data['ratings']),
                'avg_rating': self.processed_data['ratings']['rating'].mean(),
                'date_range': {
                    'start': self.processed_data['ratings']['timestamp'].min(),
                    'end': self.processed_data['ratings']['timestamp'].max()
                }
            },
            'users': {
                'total': len(self.processed_data['users']),
                'avg_ratings_per_user': self.processed_data['users']['total_ratings'].mean(),
                'activity_distribution': self.processed_data['users']['activity_level'].value_counts().to_dict()
            },
            'tmdb': {
                'total_fetched': len(self.tmdb_data),
                'success_rate': len(self.tmdb_data) / len(self.processed_data['movies']) * 100
            }
        }
    
    def get_quality_report(self) -> Optional[Dict[str, Any]]:
        """Get the comprehensive quality report if available."""
        if hasattr(self, 'quality_report'):
            return self.quality_report
        return None
    
    def get_quality_summary(self) -> Optional[Dict[str, Any]]:
        """Get a summary of quality metrics if available."""
        if hasattr(self, 'quality_score'):
            return {
                'score': self.quality_score.score,
                'level': self.quality_score.level.value,
                'total_records': self.quality_score.total_records,
                'issues_count': self.quality_score.issues_count,
                'critical_issues': self.quality_score.critical_issues,
                'warning_issues': self.quality_score.warning_issues,
                'info_issues': self.quality_score.info_issues,
                'timestamp': self.quality_score.timestamp.isoformat(),
                'metadata': self.quality_score.metadata
            }
        return None
    
    def get_quality_insights(self) -> Optional[List[str]]:
        """Get quality insights if available."""
        if hasattr(self, 'quality_metrics') and hasattr(self, 'quality_score'):
            from .quality import DataValidator
            data_validator = DataValidator()
            validation_results = data_validator.validate_data(self.processed_data, self.tmdb_data)
            return self.quality_metrics.generate_quality_insights(self.quality_score, validation_results)
        return None
    
    def get_quality_recommendations(self) -> Optional[List[str]]:
        """Get quality recommendations if available."""
        if hasattr(self, 'quality_metrics') and hasattr(self, 'quality_score'):
            from .quality import DataValidator
            data_validator = DataValidator()
            validation_results = data_validator.validate_data(self.processed_data, self.tmdb_data)
            return self.quality_metrics.get_quality_recommendations(self.quality_score, validation_results)
        return None