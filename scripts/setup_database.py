#!/usr/bin/env python3
"""
Database setup script for Movie Recommendation System.
Initializes PostgreSQL database and creates all necessary tables.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.movie_database import movie_db, Base
from src.config.database_config import db_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def wait_for_database(max_retries=30, retry_interval=2):
    """Wait for database to be ready."""
    logger.info("Waiting for database to be ready...")
    
    for attempt in range(max_retries):
        try:
            if movie_db.health_check():
                logger.info("Database is ready!")
                return True
        except Exception as e:
            logger.warning(f"Database not ready (attempt {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_interval)
    
    logger.error("Database failed to become ready within the expected time")
    return False


def create_database_tables():
    """Create all database tables."""
    try:
        logger.info("Creating database tables...")
        movie_db.create_tables()
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        return False


def verify_database_setup():
    """Verify that the database setup is correct."""
    try:
        logger.info("Verifying database setup...")
        
        # Test basic operations
        with movie_db.get_session() as session:
            # Check if tables exist
            tables = session.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """).fetchall()
            
            expected_tables = {
                'movies', 'persons', 'cast_members', 'crew_members',
                'users', 'user_ratings', 'user_watches', 'search_history'
            }
            
            actual_tables = {table[0] for table in tables}
            missing_tables = expected_tables - actual_tables
            
            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                return False
            
            logger.info(f"Found {len(actual_tables)} tables: {sorted(actual_tables)}")
            
            # Check if indexes exist
            indexes = session.execute("""
                SELECT indexname, tablename 
                FROM pg_indexes 
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """).fetchall()
            
            logger.info(f"Found {len(indexes)} indexes")
            
            # Check if views exist
            views = session.execute("""
                SELECT viewname 
                FROM pg_views 
                WHERE schemaname = 'public'
                ORDER BY viewname
            """).fetchall()
            
            logger.info(f"Found {len(views)} views: {[view[0] for view in views]}")
        
        logger.info("Database setup verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database setup verification failed: {e}")
        return False


def create_sample_data():
    """Create sample data for testing."""
    try:
        logger.info("Creating sample data...")
        
        from src.data.movie_database import Movie, Person, CastMember, User
        
        with movie_db.get_session() as session:
            # Check if sample data already exists
            existing_movies = session.query(Movie).count()
            if existing_movies > 0:
                logger.info(f"Sample data already exists ({existing_movies} movies)")
                return True
            
            # Create sample movies
            sample_movies = [
                Movie(
                    title="The Matrix",
                    original_title="The Matrix",
                    release_date="1999-03-31",
                    runtime=136,
                    overview="A computer programmer discovers that reality as he knows it is a simulation created by machines, and joins a rebellion to break free.",
                    genres=["Action", "Sci-Fi"],
                    vote_average=8.7,
                    vote_count=1800000,
                    popularity=85.0,
                    tmdb_id=603,
                    imdb_id="tt0133093"
                ),
                Movie(
                    title="Inception",
                    original_title="Inception",
                    release_date="2010-07-16",
                    runtime=148,
                    overview="A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
                    genres=["Action", "Sci-Fi", "Thriller"],
                    vote_average=8.8,
                    vote_count=2200000,
                    popularity=90.0,
                    tmdb_id=27205,
                    imdb_id="tt1375666"
                ),
                Movie(
                    title="The Dark Knight",
                    original_title="The Dark Knight",
                    release_date="2008-07-18",
                    runtime=152,
                    overview="When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
                    genres=["Action", "Crime", "Drama"],
                    vote_average=9.0,
                    vote_count=2500000,
                    popularity=95.0,
                    tmdb_id=155,
                    imdb_id="tt0468569"
                )
            ]
            
            for movie in sample_movies:
                session.add(movie)
            
            # Create sample persons
            sample_persons = [
                Person(
                    name="Keanu Reeves",
                    tmdb_id=6384,
                    imdb_id="nm0000206"
                ),
                Person(
                    name="Leonardo DiCaprio",
                    tmdb_id=6193,
                    imdb_id="nm0000138"
                ),
                Person(
                    name="Christian Bale",
                    tmdb_id=3894,
                    imdb_id="nm0000288"
                )
            ]
            
            for person in sample_persons:
                session.add(person)
            
            session.commit()
            
            # Create sample cast members
            matrix_id = session.query(Movie).filter(Movie.title == "The Matrix").first().id
            keanu_id = session.query(Person).filter(Person.name == "Keanu Reeves").first().id
            
            cast_member = CastMember(
                movie_id=matrix_id,
                person_id=keanu_id,
                character_name="Neo",
                order=1
            )
            session.add(cast_member)
            
            session.commit()
            
            logger.info("Sample data created successfully")
            return True
            
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("Starting database setup for Movie Recommendation System")
    
    # Display configuration
    logger.info(f"Database host: {db_config.postgres_host}")
    logger.info(f"Database port: {db_config.postgres_port}")
    logger.info(f"Database name: {db_config.postgres_database}")
    logger.info(f"Database user: {db_config.postgres_user}")
    
    # Wait for database to be ready
    if not wait_for_database():
        logger.error("Database setup failed: database not ready")
        sys.exit(1)
    
    # Create tables
    if not create_database_tables():
        logger.error("Database setup failed: could not create tables")
        sys.exit(1)
    
    # Verify setup
    if not verify_database_setup():
        logger.error("Database setup failed: verification failed")
        sys.exit(1)
    
    # Create sample data (optional)
    if os.getenv("CREATE_SAMPLE_DATA", "false").lower() == "true":
        if not create_sample_data():
            logger.warning("Failed to create sample data, but setup continues")
    
    logger.info("Database setup completed successfully!")
    
    # Display connection information
    logger.info("\n" + "="*50)
    logger.info("DATABASE SETUP COMPLETE")
    logger.info("="*50)
    logger.info(f"Database URL: postgresql://{db_config.postgres_user}:***@{db_config.postgres_host}:{db_config.postgres_port}/{db_config.postgres_database}")
    logger.info(f"pgAdmin URL: http://localhost:5050")
    logger.info(f"pgAdmin Email: admin@movie-recommendation.com")
    logger.info(f"pgAdmin Password: admin_password")
    logger.info("="*50)


if __name__ == "__main__":
    main() 