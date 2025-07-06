import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from feast import FeatureStore
from feast.repo_config import RepoConfig
from src.config.feature_store_config import FeatureStoreConfig

logger = logging.getLogger(__name__)

class FeatureStoreManager:
    """Feast feature store manager for feature serving and management."""
    
    def __init__(self, repo_path: str = "feature_store"):
        self.repo_path = repo_path
        self.store = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the Feast feature store."""
        try:
            repo_config = RepoConfig(
                registry=f"{self.repo_path}/data/registry.db",
                project="movie_recommendation",
                provider="local",
                online_store={
                    "type": "redis",
                    "connection_string": f"redis://{FeatureStoreConfig.FEAST_ONLINE_STORE_HOST}:{FeatureStoreConfig.FEAST_ONLINE_STORE_PORT}"
                },
                offline_store={
                    "type": "postgres",
                    "host": FeatureStoreConfig.FEAST_OFFLINE_STORE_HOST,
                    "port": FeatureStoreConfig.FEAST_OFFLINE_STORE_PORT,
                    "database": FeatureStoreConfig.FEAST_OFFLINE_STORE_DB,
                    "user": FeatureStoreConfig.FEAST_OFFLINE_STORE_USER,
                    "password": FeatureStoreConfig.FEAST_OFFLINE_STORE_PASSWORD
                }
            )
            
            self.store = FeatureStore(config=repo_config)
            logger.info("Feast feature store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Feast feature store: {e}")
            raise
    
    def apply_feature_definitions(self):
        """Apply feature definitions to the feature store."""
        try:
            self.store.apply([self.store.get_repo()])
            logger.info("Feature definitions applied successfully")
        except Exception as e:
            logger.error(f"Failed to apply feature definitions: {e}")
            raise
    
    def get_online_features(
        self,
        entity_rows: List[Dict[str, Any]],
        features: List[str],
        feature_service_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Get features from the online store for real-time serving."""
        try:
            if feature_service_name:
                feature_service = self.store.get_feature_service(feature_service_name)
                online_features = self.store.get_online_features(
                    features=feature_service,
                    entity_rows=entity_rows
                )
            else:
                online_features = self.store.get_online_features(
                    features=features,
                    entity_rows=entity_rows
                )
            
            logger.debug(f"Retrieved {len(online_features)} online features")
            return online_features.to_df()
        except Exception as e:
            logger.error(f"Failed to get online features: {e}")
            raise
    
    def get_offline_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
        feature_service_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Get features from the offline store for training."""
        try:
            if feature_service_name:
                feature_service = self.store.get_feature_service(feature_service_name)
                offline_features = self.store.get_historical_features(
                    features=feature_service,
                    entity_df=entity_df
                )
            else:
                offline_features = self.store.get_historical_features(
                    features=features,
                    entity_df=entity_df
                )
            
            logger.debug(f"Retrieved {len(offline_features)} offline features")
            return offline_features.to_df()
        except Exception as e:
            logger.error(f"Failed to get offline features: {e}")
            raise
    
    def materialize_features(
        self,
        start_date: str,
        end_date: str,
        feature_views: Optional[List[str]] = None
    ):
        """Materialize features from offline to online store."""
        try:
            if feature_views:
                self.store.materialize(
                    start_date=start_date,
                    end_date=end_date,
                    feature_views=feature_views
                )
            else:
                self.store.materialize(
                    start_date=start_date,
                    end_date=end_date
                )
            logger.info(f"Materialized features from {start_date} to {end_date}")
        except Exception as e:
            logger.error(f"Failed to materialize features: {e}")
            raise
    
    def push_features(
        self,
        feature_view_name: str,
        df: pd.DataFrame
    ):
        """Push features to the online store."""
        try:
            feature_view = self.store.get_feature_view(feature_view_name)
            self.store.push(
                feature_view=feature_view,
                df=df
            )
            logger.info(f"Pushed features to {feature_view_name}")
        except Exception as e:
            logger.error(f"Failed to push features: {e}")
            raise
    
    def list_feature_views(self) -> List[str]:
        """List all available feature views."""
        try:
            feature_views = self.store.list_feature_views()
            return [fv.name for fv in feature_views]
        except Exception as e:
            logger.error(f"Failed to list feature views: {e}")
            return []
    
    def list_feature_services(self) -> List[str]:
        """List all available feature services."""
        try:
            feature_services = self.store.list_feature_services()
            return [fs.name for fs in feature_services]
        except Exception as e:
            logger.error(f"Failed to list feature services: {e}")
            return []
    
    def get_feature_view_info(self, feature_view_name: str) -> Dict[str, Any]:
        """Get information about a specific feature view."""
        try:
            feature_view = self.store.get_feature_view(feature_view_name)
            return {
                "name": feature_view.name,
                "entities": [entity.name for entity in feature_view.entities],
                "features": [field.name for field in feature_view.schema],
                "ttl": feature_view.ttl,
                "online": feature_view.online
            }
        except Exception as e:
            logger.error(f"Failed to get feature view info: {e}")
            return {}
    
    def create_sample_data(self):
        """Create sample data for testing the feature store."""
        try:
            # Sample user features
            user_data = pd.DataFrame({
                "user_id": [1, 2, 3, 4, 5],
                "user_avg_rating": [4.2, 3.8, 4.5, 3.9, 4.1],
                "user_total_ratings": [50, 25, 100, 30, 75],
                "user_preferred_genres": ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller"],
                "user_activity_level": ["High", "Medium", "High", "Low", "Medium"],
                "user_avg_watch_duration": [120.5, 95.2, 150.8, 80.1, 110.3],
                "event_timestamp": pd.Timestamp.now()
            })
            
            # Sample movie features
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
            
            # Save sample data
            user_data.to_parquet("feature_store/data/user_stats.parquet", index=False)
            movie_data.to_parquet("feature_store/data/movie_stats.parquet", index=False)
            
            logger.info("Sample data created successfully")
            return user_data, movie_data
        except Exception as e:
            logger.error(f"Failed to create sample data: {e}")
            raise 