#!/usr/bin/env python3
"""
Data ingestion pipeline runner script.
Downloads MovieLens-20M data, fetches TMDB metadata, and loads to all storage systems.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.ingestion_pipeline import DataIngestionPipeline
from src.config.api_config import APIConfig

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ingestion_pipeline.log')
        ]
    )

def main():
    """Main function to run the data ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Movie Recommendation System Data Ingestion Pipeline")
    
    parser.add_argument(
        "--mode",
        choices=["full", "movielens-only", "tmdb-only", "sample"],
        default="sample",
        help="Pipeline mode: full (all data), movielens-only, tmdb-only, or sample (default)"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of movies to process in sample mode (default: 100)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store downloaded data (default: data)"
    )
    
    parser.add_argument(
        "--tmdb-api-key",
        type=str,
        help="TMDB API key (if not set, will use environment variable)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--skip-postgres",
        action="store_true",
        help="Skip loading data to PostgreSQL"
    )
    
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip loading data to Neo4j"
    )
    
    parser.add_argument(
        "--skip-feature-store",
        action="store_true",
        help="Skip loading data to feature store"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🎬 Movie Recommendation System - Data Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Data directory: {args.data_dir}")
    
    # Check TMDB API key
    tmdb_api_key = args.tmdb_api_key or APIConfig.TMDB_API_KEY
    if not tmdb_api_key and args.mode in ["full", "tmdb-only"]:
        logger.error("TMDB API key is required for this mode. Set TMDB_API_KEY environment variable or use --tmdb-api-key")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = DataIngestionPipeline(
            tmdb_api_key=tmdb_api_key,
            data_dir=args.data_dir,
            sample_size=args.sample_size if args.mode == "sample" else None
        )
        
        # Run pipeline based on mode
        if args.mode == "full":
            logger.info("Running full pipeline...")
            success = pipeline.run_full_pipeline()
            
        elif args.mode == "movielens-only":
            logger.info("Running MovieLens-only pipeline...")
            pipeline.download_movielens_data()
            pipeline.transform_movielens_data()
            
            # Load to storage systems
            postgres_success = not args.skip_postgres and pipeline.load_to_postgresql()
            neo4j_success = not args.skip_neo4j and pipeline.load_to_neo4j()
            feature_store_success = not args.skip_feature_store and pipeline.load_to_feature_store()
            
            success = all([postgres_success, neo4j_success, feature_store_success])
            
        elif args.mode == "tmdb-only":
            logger.info("Running TMDB-only pipeline...")
            # This would require existing MovieLens data
            logger.warning("TMDB-only mode requires existing MovieLens data")
            success = False
            
        elif args.mode == "sample":
            logger.info(f"Running sample pipeline with {args.sample_size} movies...")
            success = pipeline.run_full_pipeline()
        
        # Display results
        if success:
            logger.info("✅ Pipeline completed successfully!")
            
            # Show statistics
            stats = pipeline.get_pipeline_stats()
            if stats:
                logger.info("\n📊 Pipeline Statistics:")
                logger.info(f"Movies: {stats['movies']['total']} total, {stats['movies']['with_tmdb']} with TMDB metadata")
                logger.info(f"Ratings: {stats['ratings']['total']:,} total")
                logger.info(f"Users: {stats['users']['total']:,} total")
                logger.info(f"TMDB success rate: {stats['tmdb']['success_rate']:.1f}%")
                
                if stats['movies']['avg_rating']:
                    logger.info(f"Average movie rating: {stats['movies']['avg_rating']:.2f}")
                if stats['ratings']['avg_rating']:
                    logger.info(f"Average user rating: {stats['ratings']['avg_rating']:.2f}")
            
            logger.info("\n🎉 Data ingestion completed! You can now:")
            logger.info("1. Start the recommendation API: uvicorn src.api.main:app --reload")
            logger.info("2. Run tests: python -m pytest")
            logger.info("3. Explore the data in your databases")
            
        else:
            logger.error("❌ Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 