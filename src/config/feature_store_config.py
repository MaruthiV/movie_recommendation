import os

class FeatureStoreConfig:
    """Configuration for Feast feature store."""
    
    # Feast Registry
    FEAST_REGISTRY_HOST = os.getenv("FEAST_REGISTRY_HOST", "localhost")
    FEAST_REGISTRY_PORT = int(os.getenv("FEAST_REGISTRY_PORT", "5434"))
    FEAST_REGISTRY_DB = os.getenv("FEAST_REGISTRY_DB", "feast_registry")
    FEAST_REGISTRY_USER = os.getenv("FEAST_REGISTRY_USER", "feast_user")
    FEAST_REGISTRY_PASSWORD = os.getenv("FEAST_REGISTRY_PASSWORD", "feast_password")
    
    # Feast Online Store (Redis)
    FEAST_ONLINE_STORE_HOST = os.getenv("FEAST_ONLINE_STORE_HOST", "localhost")
    FEAST_ONLINE_STORE_PORT = int(os.getenv("FEAST_ONLINE_STORE_PORT", "6379"))
    
    # Feast Offline Store (PostgreSQL)
    FEAST_OFFLINE_STORE_HOST = os.getenv("FEAST_OFFLINE_STORE_HOST", "localhost")
    FEAST_OFFLINE_STORE_PORT = int(os.getenv("FEAST_OFFLINE_STORE_PORT", "5433"))
    FEAST_OFFLINE_STORE_DB = os.getenv("FEAST_OFFLINE_STORE_DB", "feast")
    FEAST_OFFLINE_STORE_USER = os.getenv("FEAST_OFFLINE_STORE_USER", "feast_user")
    FEAST_OFFLINE_STORE_PASSWORD = os.getenv("FEAST_OFFLINE_STORE_PASSWORD", "feast_password")
    
    # Feature Definitions
    FEATURE_VIEWS = {
        "user_features": {
            "entities": ["user_id"],
            "features": [
                "user_avg_rating",
                "user_total_ratings",
                "user_preferred_genres",
                "user_activity_level",
                "user_avg_watch_duration"
            ]
        },
        "movie_features": {
            "entities": ["movie_id"],
            "features": [
                "movie_avg_rating",
                "movie_total_ratings",
                "movie_popularity",
                "movie_genres",
                "movie_release_year",
                "movie_budget",
                "movie_revenue"
            ]
        },
        "user_movie_interaction_features": {
            "entities": ["user_id", "movie_id"],
            "features": [
                "user_movie_rating",
                "user_movie_watch_duration",
                "user_movie_completion_rate",
                "user_movie_liked",
                "user_movie_would_recommend"
            ]
        }
    } 