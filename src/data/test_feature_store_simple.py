import pytest
import pandas as pd
from src.config.feature_store_config import FeatureStoreConfig

class TestFeatureStoreConfig:
    """Test cases for FeatureStoreConfig."""
    
    def test_config_values(self):
        """Test feature store configuration values."""
        assert FeatureStoreConfig.FEAST_ONLINE_STORE_HOST == "localhost"
        assert FeatureStoreConfig.FEAST_ONLINE_STORE_PORT == 6379
        assert FeatureStoreConfig.FEAST_OFFLINE_STORE_HOST == "localhost"
        assert FeatureStoreConfig.FEAST_OFFLINE_STORE_PORT == 5433
        assert FeatureStoreConfig.FEAST_REGISTRY_HOST == "localhost"
        assert FeatureStoreConfig.FEAST_REGISTRY_PORT == 5434
        
        # Test feature view definitions
        assert "user_features" in FeatureStoreConfig.FEATURE_VIEWS
        assert "movie_features" in FeatureStoreConfig.FEATURE_VIEWS
        assert "user_movie_interaction_features" in FeatureStoreConfig.FEATURE_VIEWS
        
        # Test user features
        user_features = FeatureStoreConfig.FEATURE_VIEWS["user_features"]
        assert "user_avg_rating" in user_features["features"]
        assert "user_total_ratings" in user_features["features"]
        assert "user_preferred_genres" in user_features["features"]
        assert "user_activity_level" in user_features["features"]
        assert "user_avg_watch_duration" in user_features["features"]
        assert "user_id" in user_features["entities"]
        
        # Test movie features
        movie_features = FeatureStoreConfig.FEATURE_VIEWS["movie_features"]
        assert "movie_avg_rating" in movie_features["features"]
        assert "movie_popularity" in movie_features["features"]
        assert "movie_genres" in movie_features["features"]
        assert "movie_release_year" in movie_features["features"]
        assert "movie_budget" in movie_features["features"]
        assert "movie_revenue" in movie_features["features"]
        assert "movie_id" in movie_features["entities"]
        
        # Test user-movie interaction features
        interaction_features = FeatureStoreConfig.FEATURE_VIEWS["user_movie_interaction_features"]
        assert "user_movie_rating" in interaction_features["features"]
        assert "user_movie_watch_duration" in interaction_features["features"]
        assert "user_movie_completion_rate" in interaction_features["features"]
        assert "user_movie_liked" in interaction_features["features"]
        assert "user_movie_would_recommend" in interaction_features["features"]
        assert "user_id" in interaction_features["entities"]
        assert "movie_id" in interaction_features["entities"]

class TestSampleDataCreation:
    """Test sample data creation for feature store."""
    
    def test_user_sample_data(self):
        """Test creating sample user data."""
        # Create sample user data similar to what the feature store manager would create
        user_data = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "user_avg_rating": [4.2, 3.8, 4.5, 3.9, 4.1],
            "user_total_ratings": [50, 25, 100, 30, 75],
            "user_preferred_genres": ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller"],
            "user_activity_level": ["High", "Medium", "High", "Low", "Medium"],
            "user_avg_watch_duration": [120.5, 95.2, 150.8, 80.1, 110.3],
            "event_timestamp": pd.Timestamp.now()
        })
        
        # Check user data
        assert len(user_data) == 5
        assert "user_id" in user_data.columns
        assert "user_avg_rating" in user_data.columns
        assert "user_total_ratings" in user_data.columns
        assert "user_preferred_genres" in user_data.columns
        assert "user_activity_level" in user_data.columns
        assert "user_avg_watch_duration" in user_data.columns
        assert "event_timestamp" in user_data.columns
        
        # Check data types
        assert user_data["user_id"].dtype == "int64"
        assert user_data["user_avg_rating"].dtype == "float64"
        assert user_data["user_total_ratings"].dtype == "int64"
        assert user_data["user_preferred_genres"].dtype == "object"
        assert user_data["user_activity_level"].dtype == "object"
        assert user_data["user_avg_watch_duration"].dtype == "float64"
        
        # Check value ranges
        assert user_data["user_avg_rating"].min() >= 0
        assert user_data["user_avg_rating"].max() <= 5
        assert user_data["user_total_ratings"].min() >= 0
        assert user_data["user_avg_watch_duration"].min() >= 0
    
    def test_movie_sample_data(self):
        """Test creating sample movie data."""
        # Create sample movie data similar to what the feature store manager would create
        movie_data = pd.DataFrame({
            "movie_id": [101, 102, 103, 104, 105],
            "movie_avg_rating": [4.3, 3.9, 4.1, 4.5, 3.7],
            "movie_total_ratings": [1000, 500, 750, 1200, 300],
            "movie_popularity": [8.5, 6.2, 7.8, 9.1, 5.4],
            "movie_genres": ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller"],
            "movie_release_year": [2020, 2019, 2021, 2018, 2022],
            "movie_budget": [50000000, 30000000, 40000000, 60000000, 25000000],
            "movie_revenue": [150000000, 80000000, 120000000, 200000000, 60000000],
            "event_timestamp": pd.Timestamp.now()
        })
        
        # Check movie data
        assert len(movie_data) == 5
        assert "movie_id" in movie_data.columns
        assert "movie_avg_rating" in movie_data.columns
        assert "movie_total_ratings" in movie_data.columns
        assert "movie_popularity" in movie_data.columns
        assert "movie_genres" in movie_data.columns
        assert "movie_release_year" in movie_data.columns
        assert "movie_budget" in movie_data.columns
        assert "movie_revenue" in movie_data.columns
        assert "event_timestamp" in movie_data.columns
        
        # Check data types
        assert movie_data["movie_id"].dtype == "int64"
        assert movie_data["movie_avg_rating"].dtype == "float64"
        assert movie_data["movie_total_ratings"].dtype == "int64"
        assert movie_data["movie_popularity"].dtype == "float64"
        assert movie_data["movie_genres"].dtype == "object"
        assert movie_data["movie_release_year"].dtype == "int64"
        assert movie_data["movie_budget"].dtype == "int64"
        assert movie_data["movie_revenue"].dtype == "int64"
        
        # Check value ranges
        assert movie_data["movie_avg_rating"].min() >= 0
        assert movie_data["movie_avg_rating"].max() <= 5
        assert movie_data["movie_total_ratings"].min() >= 0
        assert movie_data["movie_popularity"].min() >= 0
        assert movie_data["movie_popularity"].max() <= 10
        assert movie_data["movie_release_year"].min() >= 1900
        assert movie_data["movie_release_year"].max() <= 2030
        assert movie_data["movie_budget"].min() >= 0
        assert movie_data["movie_revenue"].min() >= 0
    
    def test_feature_view_consistency(self):
        """Test that feature view definitions are consistent with sample data."""
        # Get feature view definitions
        user_features = FeatureStoreConfig.FEATURE_VIEWS["user_features"]
        movie_features = FeatureStoreConfig.FEATURE_VIEWS["movie_features"]
        
        # Create sample data
        user_data = pd.DataFrame({
            "user_id": [1],
            "user_avg_rating": [4.2],
            "user_total_ratings": [50],
            "user_preferred_genres": ["Action"],
            "user_activity_level": ["High"],
            "user_avg_watch_duration": [120.5],
            "event_timestamp": pd.Timestamp.now()
        })
        
        movie_data = pd.DataFrame({
            "movie_id": [101],
            "movie_avg_rating": [4.3],
            "movie_total_ratings": [1000],
            "movie_popularity": [8.5],
            "movie_genres": ["Action"],
            "movie_release_year": [2020],
            "movie_budget": [50000000],
            "movie_revenue": [150000000],
            "event_timestamp": pd.Timestamp.now()
        })
        
        # Check that all features defined in config exist in sample data
        for feature in user_features["features"]:
            if feature != "event_timestamp":  # Skip timestamp as it's added by the system
                assert feature in user_data.columns, f"Feature {feature} not found in user data"
        
        for feature in movie_features["features"]:
            if feature != "event_timestamp":  # Skip timestamp as it's added by the system
                assert feature in movie_data.columns, f"Feature {feature} not found in movie data"
        
        # Check that all entities defined in config exist in sample data
        for entity in user_features["entities"]:
            assert entity in user_data.columns, f"Entity {entity} not found in user data"
        
        for entity in movie_features["entities"]:
            assert entity in movie_data.columns, f"Entity {entity} not found in movie data" 