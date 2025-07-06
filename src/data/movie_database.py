"""
Core movie database management for the Movie Recommendation System.
Contains SQLAlchemy models and database operations for movies, users, and interactions.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Index, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import TypeDecorator, JSON
import logging

from src.config.database_config import db_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create SQLAlchemy base class
Base = declarative_base()


class PGArrayOrJSON(TypeDecorator):
    """
    Use ARRAY for PostgreSQL, JSON for SQLite (for test compatibility).
    """
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            from sqlalchemy.dialects.postgresql import ARRAY
            return dialect.type_descriptor(ARRAY(String))
        else:
            return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        return value

    def process_result_value(self, value, dialect):
        return value


class PGJSONBOrJSON(TypeDecorator):
    """
    Use JSONB for PostgreSQL, JSON for SQLite (for test compatibility).
    """
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            from sqlalchemy.dialects.postgresql import JSONB
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        return value

    def process_result_value(self, value, dialect):
        return value


class Movie(Base):
    """Movie entity with metadata and content information."""
    
    __tablename__ = "movies"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Basic movie information
    title = Column(String(500), nullable=False, index=True)
    original_title = Column(String(500))
    release_date = Column(DateTime)
    runtime = Column(Integer)  # in minutes
    overview = Column(Text)
    tagline = Column(String(500))
    
    # Metadata
    genres = Column(PGArrayOrJSON)  # Array of genre names
    keywords = Column(PGArrayOrJSON)  # Array of keywords/tags
    language = Column(String(10))
    country = Column(String(100))
    budget = Column(Integer)
    revenue = Column(Integer)
    
    # Ratings and popularity
    vote_average = Column(Float)
    vote_count = Column(Integer)
    popularity = Column(Float)
    
    # Content embeddings and metadata
    poster_path = Column(String(500))
    backdrop_path = Column(String(500))
    trailer_url = Column(String(500))
    content_embeddings = Column(PGJSONBOrJSON)  # Store CLIP/VideoCLIP embeddings
    
    # External IDs
    tmdb_id = Column(Integer, unique=True, index=True)
    imdb_id = Column(String(20), unique=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    cast_members = relationship("CastMember", back_populates="movie")
    crew_members = relationship("CrewMember", back_populates="movie")
    user_ratings = relationship("UserRating", back_populates="movie")
    user_watches = relationship("UserWatch", back_populates="movie")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_movies_genres', 'genres', postgresql_using='gin'),
        Index('idx_movies_keywords', 'keywords', postgresql_using='gin'),
        Index('idx_movies_release_date', 'release_date'),
        Index('idx_movies_popularity', 'popularity'),
        Index('idx_movies_vote_average', 'vote_average'),
    )


class Person(Base):
    """Person entity for actors, directors, and crew members."""
    
    __tablename__ = "persons"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    biography = Column(Text)
    birth_date = Column(DateTime)
    death_date = Column(DateTime)
    place_of_birth = Column(String(200))
    gender = Column(Integer)  # 1=female, 2=male, 0=unknown
    profile_path = Column(String(500))
    tmdb_id = Column(Integer, unique=True, index=True)
    imdb_id = Column(String(20), unique=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    cast_roles = relationship("CastMember", back_populates="person")
    crew_roles = relationship("CrewMember", back_populates="person")


class CastMember(Base):
    """Cast member relationship between movies and persons."""
    
    __tablename__ = "cast_members"
    
    id = Column(Integer, primary_key=True, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    character_name = Column(String(200))
    order = Column(Integer)  # Order of appearance in credits
    cast_id = Column(Integer)  # TMDB cast ID
    
    # Relationships
    movie = relationship("Movie", back_populates="cast_members")
    person = relationship("Person", back_populates="cast_roles")
    
    __table_args__ = (
        Index('idx_cast_members_movie_person', 'movie_id', 'person_id'),
    )


class CrewMember(Base):
    """Crew member relationship between movies and persons."""
    
    __tablename__ = "crew_members"
    
    id = Column(Integer, primary_key=True, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    job = Column(String(100))  # e.g., "Director", "Producer", "Screenplay"
    department = Column(String(100))  # e.g., "Production", "Sound", "Camera"
    credit_id = Column(String(100))  # TMDB credit ID
    
    # Relationships
    movie = relationship("Movie", back_populates="crew_members")
    person = relationship("Person", back_populates="crew_roles")
    
    __table_args__ = (
        Index('idx_crew_members_movie_person', 'movie_id', 'person_id'),
        Index('idx_crew_members_job', 'job'),
        Index('idx_crew_members_department', 'department'),
    )


class User(Base):
    """User entity for recommendation system."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    
    # User preferences
    preferred_genres = Column(PGArrayOrJSON)
    preferred_languages = Column(PGArrayOrJSON)
    age_group = Column(String(20))  # e.g., "18-25", "26-35", etc.
    watch_preferences = Column(PGJSONBOrJSON)  # Store user preferences as JSON
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime)
    
    # Relationships
    ratings = relationship("UserRating", back_populates="user")
    watches = relationship("UserWatch", back_populates="user")
    search_history = relationship("SearchHistory", back_populates="user")


class UserRating(Base):
    """User movie ratings."""
    
    __tablename__ = "user_ratings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    rating = Column(Float, nullable=False)  # Rating value (e.g., 1-5 stars)
    review_text = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="user_ratings")
    
    __table_args__ = (
        Index('idx_user_ratings_user_movie', 'user_id', 'movie_id', unique=True),
        Index('idx_user_ratings_rating', 'rating'),
    )


class UserWatch(Base):
    """User movie watching history."""
    
    __tablename__ = "user_watches"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    
    # Watch behavior
    watch_date = Column(DateTime, nullable=False)
    watch_duration = Column(Integer)  # Duration watched in seconds
    completion_rate = Column(Float)  # Percentage of movie watched (0-1)
    device_type = Column(String(50))  # e.g., "mobile", "desktop", "tv"
    platform = Column(String(50))  # e.g., "web", "ios", "android"
    
    # User feedback
    liked = Column(Boolean)  # User liked/disliked
    would_recommend = Column(Boolean)  # Would recommend to others
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user = relationship("User", back_populates="watches")
    movie = relationship("Movie", back_populates="user_watches")
    
    __table_args__ = (
        Index('idx_user_watches_user_movie', 'user_id', 'movie_id'),
        Index('idx_user_watches_date', 'watch_date'),
        Index('idx_user_watches_completion', 'completion_rate'),
    )


class SearchHistory(Base):
    """User search history for recommendations."""
    
    __tablename__ = "search_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    query = Column(String(500), nullable=False)
    query_type = Column(String(50))  # e.g., "natural_language", "movie_title", "genre"
    results_count = Column(Integer)
    clicked_movie_id = Column(Integer, ForeignKey("movies.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user = relationship("User", back_populates="search_history")
    clicked_movie = relationship("Movie")
    
    __table_args__ = (
        Index('idx_search_history_user_date', 'user_id', 'created_at'),
        Index('idx_search_history_query', 'query'),
    )


class MovieDatabase:
    """Database manager for movie recommendation system."""
    
    def __init__(self):
        """Initialize database connection."""
        self.engine = None
        self.SessionLocal = None
        self._setup_connection()
    
    def _setup_connection(self):
        """Set up database connection and session factory."""
        try:
            connection_string = db_config.get_connection_string()
            self.engine = create_engine(
                connection_string,
                pool_size=db_config.postgres_pool_size,
                max_overflow=db_config.postgres_max_overflow,
                pool_pre_ping=True,
                echo=False  # Set to True for SQL debugging
            )
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to establish database connection: {e}")
            raise
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def close_session(self, session: Session):
        """Close database session."""
        session.close()
    
    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_movie_by_id(self, movie_id: int) -> Optional[Movie]:
        """Get movie by ID."""
        try:
            with self.get_session() as session:
                return session.query(Movie).filter(Movie.id == movie_id).first()
        except Exception as e:
            logger.error(f"Failed to get movie by ID {movie_id}: {e}")
            return None
    
    def get_movie_by_tmdb_id(self, tmdb_id: int) -> Optional[Movie]:
        """Get movie by TMDB ID."""
        try:
            with self.get_session() as session:
                return session.query(Movie).filter(Movie.tmdb_id == tmdb_id).first()
        except Exception as e:
            logger.error(f"Failed to get movie by TMDB ID {tmdb_id}: {e}")
            return None
    
    def search_movies(self, query: str, limit: int = 20) -> List[Movie]:
        """Search movies by title."""
        try:
            with self.get_session() as session:
                return session.query(Movie).filter(
                    Movie.title.ilike(f"%{query}%")
                ).limit(limit).all()
        except Exception as e:
            logger.error(f"Failed to search movies with query '{query}': {e}")
            return []
    
    def get_movies_by_genre(self, genre: str, limit: int = 50) -> List[Movie]:
        """Get movies by genre."""
        try:
            with self.get_session() as session:
                if session.bind.dialect.name == 'postgresql':
                    return session.query(Movie).filter(
                        Movie.genres.contains([genre])
                    ).order_by(Movie.popularity.desc()).limit(limit).all()
                else:
                    # SQLite/JSON fallback: fetch all and filter in Python
                    all_movies = session.query(Movie).all()
                    filtered = [m for m in all_movies if m.genres and genre in m.genres]
                    filtered.sort(key=lambda m: m.popularity or 0, reverse=True)
                    return filtered[:limit]
        except Exception as e:
            logger.error(f"Failed to get movies by genre '{genre}': {e}")
            return []
    
    def get_user_ratings(self, user_id: int) -> List[UserRating]:
        """Get all ratings for a user."""
        try:
            with self.get_session() as session:
                return session.query(UserRating).filter(
                    UserRating.user_id == user_id
                ).all()
        except Exception as e:
            logger.error(f"Failed to get ratings for user {user_id}: {e}")
            return []
    
    def add_user_rating(self, user_id: int, movie_id: int, rating: float, review_text: str = None) -> bool:
        """Add or update user rating."""
        try:
            with self.get_session() as session:
                existing_rating = session.query(UserRating).filter(
                    UserRating.user_id == user_id,
                    UserRating.movie_id == movie_id
                ).first()
                
                if existing_rating:
                    existing_rating.rating = rating
                    existing_rating.review_text = review_text
                    existing_rating.updated_at = datetime.now(timezone.utc)
                else:
                    new_rating = UserRating(
                        user_id=user_id,
                        movie_id=movie_id,
                        rating=rating,
                        review_text=review_text
                    )
                    session.add(new_rating)
                
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to add rating for user {user_id}, movie {movie_id}: {e}")
            return False


# Global database instance
movie_db = MovieDatabase() 