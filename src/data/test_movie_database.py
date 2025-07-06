"""
Unit tests for movie database functionality.
Tests database operations, models, and data integrity.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.data.movie_database import (
    Base, Movie, Person, CastMember, CrewMember, User, 
    UserRating, UserWatch, SearchHistory, MovieDatabase
)
from src.config.database_config import DatabaseConfig


class TestMovieDatabase(unittest.TestCase):
    """Test cases for MovieDatabase class."""
    
    def setUp(self):
        """Set up test database."""
        # Create in-memory SQLite database for testing
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Mock the database configuration
        self.mock_config = DatabaseConfig(
            postgres_host="localhost",
            postgres_port=5432,
            postgres_database="test_db",
            postgres_user="test_user",
            postgres_password="test_password"
        )
        
        # Create test database instance
        with patch('src.data.movie_database.db_config', self.mock_config):
            with patch('src.data.movie_database.create_engine') as mock_create_engine:
                mock_create_engine.return_value = self.engine
                self.db = MovieDatabase()
                self.db.engine = self.engine
                self.db.SessionLocal = self.SessionLocal
    
    def tearDown(self):
        """Clean up test database."""
        Base.metadata.drop_all(bind=self.engine)
    
    def test_create_tables(self):
        """Test database table creation."""
        # Tables should already be created in setUp
        with self.db.get_session() as session:
            # Check if tables exist by querying them
            movies = session.query(Movie).all()
            users = session.query(User).all()
            self.assertIsInstance(movies, list)
            self.assertIsInstance(users, list)
    
    def test_health_check(self):
        """Test database health check."""
        self.assertTrue(self.db.health_check())
    
    def test_add_and_get_movie(self):
        """Test adding and retrieving a movie."""
        # Create test movie
        test_movie = Movie(
            title="Test Movie",
            original_title="Test Movie Original",
            release_date=datetime(2023, 1, 1),
            runtime=120,
            overview="A test movie for testing",
            genres=["Action", "Adventure"],
            vote_average=8.5,
            vote_count=1000,
            popularity=75.5,
            tmdb_id=12345,
            imdb_id="tt1234567"
        )
        
        # Add movie to database
        with self.db.get_session() as session:
            session.add(test_movie)
            session.commit()
            movie_id = test_movie.id
        
        # Retrieve movie
        retrieved_movie = self.db.get_movie_by_id(movie_id)
        self.assertIsNotNone(retrieved_movie)
        self.assertEqual(retrieved_movie.title, "Test Movie")
        self.assertEqual(retrieved_movie.tmdb_id, 12345)
        self.assertEqual(retrieved_movie.genres, ["Action", "Adventure"])
    
    def test_get_movie_by_tmdb_id(self):
        """Test retrieving movie by TMDB ID."""
        # Create test movie
        test_movie = Movie(
            title="TMDB Test Movie",
            tmdb_id=54321,
            imdb_id="tt7654321"
        )
        
        # Add movie to database
        with self.db.get_session() as session:
            session.add(test_movie)
            session.commit()
        
        # Retrieve movie by TMDB ID
        retrieved_movie = self.db.get_movie_by_tmdb_id(54321)
        self.assertIsNotNone(retrieved_movie)
        self.assertEqual(retrieved_movie.title, "TMDB Test Movie")
        self.assertEqual(retrieved_movie.tmdb_id, 54321)
    
    def test_search_movies(self):
        """Test movie search functionality."""
        # Create test movies
        movies = [
            Movie(title="The Matrix", tmdb_id=1),
            Movie(title="Matrix Reloaded", tmdb_id=2),
            Movie(title="Star Wars", tmdb_id=3),
            Movie(title="The Matrix Revolutions", tmdb_id=4)
        ]
        
        # Add movies to database
        with self.db.get_session() as session:
            for movie in movies:
                session.add(movie)
            session.commit()
        
        # Search for Matrix movies
        matrix_movies = self.db.search_movies("Matrix")
        self.assertEqual(len(matrix_movies), 3)
        
        # Search for Star Wars
        star_wars_movies = self.db.search_movies("Star Wars")
        self.assertEqual(len(star_wars_movies), 1)
        self.assertEqual(star_wars_movies[0].title, "Star Wars")
    
    def test_get_movies_by_genre(self):
        """Test retrieving movies by genre."""
        # Create test movies with different genres
        movies = [
            Movie(title="Action Movie 1", genres=["Action"], popularity=80.0),
            Movie(title="Action Movie 2", genres=["Action"], popularity=90.0),
            Movie(title="Comedy Movie", genres=["Comedy"], popularity=70.0),
            Movie(title="Action Comedy", genres=["Action", "Comedy"], popularity=85.0)
        ]
        
        # Add movies to database
        with self.db.get_session() as session:
            for movie in movies:
                session.add(movie)
            session.commit()
        
        # Get action movies
        action_movies = self.db.get_movies_by_genre("Action")
        self.assertEqual(len(action_movies), 3)  # Should include "Action Comedy" too
        
        # Get comedy movies
        comedy_movies = self.db.get_movies_by_genre("Comedy")
        self.assertEqual(len(comedy_movies), 2)  # Should include "Action Comedy" too
    
    def test_user_rating_operations(self):
        """Test user rating operations."""
        # Create test user and movie
        test_user = User(username="testuser", email="test@example.com", password_hash="hash")
        test_movie = Movie(title="Rating Test Movie", tmdb_id=999)
        
        with self.db.get_session() as session:
            session.add(test_user)
            session.add(test_movie)
            session.commit()
            user_id = test_user.id
            movie_id = test_movie.id
        
        # Add rating
        success = self.db.add_user_rating(user_id, movie_id, 4.5, "Great movie!")
        self.assertTrue(success)
        
        # Get user ratings
        ratings = self.db.get_user_ratings(user_id)
        self.assertEqual(len(ratings), 1)
        self.assertEqual(ratings[0].rating, 4.5)
        self.assertEqual(ratings[0].review_text, "Great movie!")
        
        # Update rating
        success = self.db.add_user_rating(user_id, movie_id, 5.0, "Excellent movie!")
        self.assertTrue(success)
        
        # Check updated rating
        ratings = self.db.get_user_ratings(user_id)
        self.assertEqual(len(ratings), 1)  # Should still be one rating
        self.assertEqual(ratings[0].rating, 5.0)
        self.assertEqual(ratings[0].review_text, "Excellent movie!")
    
    def test_movie_relationships(self):
        """Test movie relationships with cast and crew."""
        # Create test movie and person
        test_movie = Movie(title="Relationship Test Movie", tmdb_id=111)
        test_person = Person(name="Test Actor", tmdb_id=222)
        
        with self.db.get_session() as session:
            session.add(test_movie)
            session.add(test_person)
            session.commit()
            movie_id = test_movie.id
            person_id = test_person.id
        
        # Add cast member
        cast_member = CastMember(
            movie_id=movie_id,
            person_id=person_id,
            character_name="Test Character",
            order=1
        )
        
        # Add crew member
        crew_member = CrewMember(
            movie_id=movie_id,
            person_id=person_id,
            job="Director",
            department="Production"
        )
        
        with self.db.get_session() as session:
            session.add(cast_member)
            session.add(crew_member)
            session.commit()
        
        # Test relationships
        with self.db.get_session() as session:
            movie = session.query(Movie).filter(Movie.id == movie_id).first()
            person = session.query(Person).filter(Person.id == person_id).first()
            
            self.assertEqual(len(movie.cast_members), 1)
            self.assertEqual(len(movie.crew_members), 1)
            self.assertEqual(len(person.cast_roles), 1)
            self.assertEqual(len(person.crew_roles), 1)
            
            self.assertEqual(movie.cast_members[0].character_name, "Test Character")
            self.assertEqual(movie.crew_members[0].job, "Director")
    
    def test_user_watch_history(self):
        """Test user watch history functionality."""
        # Create test user and movie
        test_user = User(username="watchuser", email="watch@example.com", password_hash="hash")
        test_movie = Movie(title="Watch Test Movie", tmdb_id=333)
        
        with self.db.get_session() as session:
            session.add(test_user)
            session.add(test_movie)
            session.commit()
            user_id = test_user.id
            movie_id = test_movie.id
        
        # Add watch history
        watch_record = UserWatch(
            user_id=user_id,
            movie_id=movie_id,
            watch_date=datetime.utcnow(),
            watch_duration=7200,  # 2 hours in seconds
            completion_rate=1.0,  # 100% completion
            device_type="desktop",
            platform="web",
            liked=True,
            would_recommend=True
        )
        
        with self.db.get_session() as session:
            session.add(watch_record)
            session.commit()
        
        # Verify watch history
        with self.db.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            self.assertEqual(len(user.watches), 1)
            self.assertEqual(user.watches[0].completion_rate, 1.0)
            self.assertTrue(user.watches[0].liked)
    
    def test_search_history(self):
        """Test search history functionality."""
        # Create test user and movie
        test_user = User(username="searchuser", email="search@example.com", password_hash="hash")
        test_movie = Movie(title="Search Test Movie", tmdb_id=444)
        
        with self.db.get_session() as session:
            session.add(test_user)
            session.add(test_movie)
            session.commit()
            user_id = test_user.id
            movie_id = test_movie.id
        
        # Add search history
        search_record = SearchHistory(
            user_id=user_id,
            query="test movie",
            query_type="natural_language",
            results_count=5,
            clicked_movie_id=movie_id
        )
        
        with self.db.get_session() as session:
            session.add(search_record)
            session.commit()
        
        # Verify search history
        with self.db.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            self.assertEqual(len(user.search_history), 1)
            self.assertEqual(user.search_history[0].query, "test movie")
            self.assertEqual(user.search_history[0].query_type, "natural_language")
    
    def test_database_connection_error_handling(self):
        """Test error handling for database connection issues."""
        # Test with invalid connection
        with patch('src.data.movie_database.create_engine') as mock_create_engine:
            mock_create_engine.side_effect = Exception("Connection failed")
            
            with self.assertRaises(Exception):
                MovieDatabase()
    
    def test_session_management(self):
        """Test proper session management."""
        session = self.db.get_session()
        self.assertIsNotNone(session)
        
        # Test session closure
        self.db.close_session(session)
        # Session should be closed (though we can't easily test this without accessing internal state)


if __name__ == "__main__":
    unittest.main() 