#!/usr/bin/env python3
"""
Setup script for Feast feature store.
Initializes the feature store and tests connections.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.feature_store_manager import FeatureStoreManager
from src.config.feature_store_config import FeatureStoreConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_services():
    """Wait for Feast services to be ready."""
    logger.info("Waiting for Feast services to be ready...")
    
    # Wait for Redis (online store)
    import redis
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            r = redis.Redis(
                host=FeatureStoreConfig.FEAST_ONLINE_STORE_HOST,
                port=FeatureStoreConfig.FEAST_ONLINE_STORE_PORT,
                decode_responses=True
            )
            r.ping()
            logger.info("✅ Redis (online store) is ready")
            break
        except Exception as e:
            retry_count += 1
            logger.info(f"Waiting for Redis... ({retry_count}/{max_retries})")
            time.sleep(2)
    
    if retry_count >= max_retries:
        raise Exception("Redis service failed to start")
    
    # Wait for PostgreSQL (offline store)
    import psycopg2
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            conn = psycopg2.connect(
                host=FeatureStoreConfig.FEAST_OFFLINE_STORE_HOST,
                port=FeatureStoreConfig.FEAST_OFFLINE_STORE_PORT,
                database=FeatureStoreConfig.FEAST_OFFLINE_STORE_DB,
                user=FeatureStoreConfig.FEAST_OFFLINE_STORE_USER,
                password=FeatureStoreConfig.FEAST_OFFLINE_STORE_PASSWORD
            )
            conn.close()
            logger.info("✅ PostgreSQL (offline store) is ready")
            break
        except Exception as e:
            retry_count += 1
            logger.info(f"Waiting for PostgreSQL... ({retry_count}/{max_retries})")
            time.sleep(2)
    
    if retry_count >= max_retries:
        raise Exception("PostgreSQL service failed to start")
    
    # Wait for PostgreSQL (registry)
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            conn = psycopg2.connect(
                host=FeatureStoreConfig.FEAST_REGISTRY_HOST,
                port=FeatureStoreConfig.FEAST_REGISTRY_PORT,
                database=FeatureStoreConfig.FEAST_REGISTRY_DB,
                user=FeatureStoreConfig.FEAST_REGISTRY_USER,
                password=FeatureStoreConfig.FEAST_REGISTRY_PASSWORD
            )
            conn.close()
            logger.info("✅ PostgreSQL (registry) is ready")
            break
        except Exception as e:
            retry_count += 1
            logger.info(f"Waiting for PostgreSQL registry... ({retry_count}/{max_retries})")
            time.sleep(2)
    
    if retry_count >= max_retries:
        raise Exception("PostgreSQL registry service failed to start")

def setup_feature_store():
    """Set up the Feast feature store."""
    try:
        logger.info("🚀 Setting up Feast feature store...")
        
        # Create necessary directories
        os.makedirs("feature_store/data", exist_ok=True)
        
        # Wait for services to be ready
        wait_for_services()
        
        # Initialize feature store manager
        feature_store = FeatureStoreManager()
        
        # Create sample data
        logger.info("Creating sample data...")
        user_data, movie_data = feature_store.create_sample_data()
        logger.info(f"Created sample data: {len(user_data)} users, {len(movie_data)} movies")
        
        # Apply feature definitions
        logger.info("Applying feature definitions...")
        feature_store.apply_feature_definitions()
        
        # Test feature store operations
        logger.info("Testing feature store operations...")
        
        # Test listing feature views
        feature_views = feature_store.list_feature_views()
        logger.info(f"Available feature views: {feature_views}")
        
        # Test listing feature services
        feature_services = feature_store.list_feature_services()
        logger.info(f"Available feature services: {feature_services}")
        
        # Test getting feature view info
        if feature_views:
            feature_info = feature_store.get_feature_view_info(feature_views[0])
            logger.info(f"Feature view info: {feature_info}")
        
        # Test online feature retrieval
        entity_rows = [{"user_id": 1}, {"user_id": 2}]
        try:
            online_features = feature_store.get_online_features(
                entity_rows=entity_rows,
                features=["user_features:user_avg_rating", "user_features:user_total_ratings"]
            )
            logger.info(f"Retrieved online features: {len(online_features)} rows")
        except Exception as e:
            logger.warning(f"Online feature retrieval failed (expected for new setup): {e}")
        
        logger.info("✅ Feast feature store setup completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to set up Feast feature store: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("🎬 Movie Recommendation System - Feast Feature Store Setup")
    logger.info("=" * 60)
    
    success = setup_feature_store()
    
    if success:
        logger.info("\n🎉 Setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start the services: docker-compose up -d")
        logger.info("2. Run the setup script: python scripts/setup_feature_store.py")
        logger.info("3. Test the feature store with sample data")
    else:
        logger.error("\n❌ Setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 