import os
from typing import Optional


class APIConfig:
    """Configuration for external APIs."""
    
    # TMDB API Configuration
    TMDB_API_KEY: Optional[str] = os.getenv('TMDB_API_KEY')
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
    TMDB_IMAGE_BASE_URL: str = "https://image.tmdb.org/t/p"
    
    # API Rate Limiting
    TMDB_RATE_LIMIT_PER_SECOND: int = 10
    TMDB_RATE_LIMIT_PER_MINUTE: int = 40
    
    # Data Ingestion Settings
    MAX_MOVIES_PER_REQUEST: int = 20  # TMDB API limit
    DEFAULT_LANGUAGE: str = "en-US"
    INCLUDE_ADULT: bool = False
    
    @classmethod
    def validate_tmdb_config(cls) -> bool:
        """Validate that TMDB API key is configured."""
        if not cls.TMDB_API_KEY:
            raise ValueError(
                "TMDB_API_KEY environment variable is required. "
                "Please set it with your TMDB API key."
            )
        return True
    
    @classmethod
    def get_tmdb_headers(cls) -> dict:
        """Get headers for TMDB API requests."""
        cls.validate_tmdb_config()
        return {
            "Authorization": f"Bearer {cls.TMDB_API_KEY}",
            "Content-Type": "application/json"
        } 