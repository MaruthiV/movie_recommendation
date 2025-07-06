import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from src.config.api_config import APIConfig

logger = logging.getLogger(__name__)


@dataclass
class TMDBMovie:
    """Data class for TMDB movie data."""
    id: int
    title: str
    original_title: str
    overview: str
    release_date: str
    poster_path: Optional[str]
    backdrop_path: Optional[str]
    popularity: float
    vote_average: float
    vote_count: int
    genre_ids: List[int]
    adult: bool
    video: bool
    original_language: str


class TMDBClient:
    """Client for interacting with TMDB API with rate limiting."""
    
    def __init__(self):
        self.base_url = APIConfig.TMDB_BASE_URL
        self.headers = APIConfig.get_tmdb_headers()
        self.rate_limit_per_second = APIConfig.TMDB_RATE_LIMIT_PER_SECOND
        self.rate_limit_per_minute = APIConfig.TMDB_RATE_LIMIT_PER_MINUTE
        self.last_request_time = 0
        self.requests_this_minute = 0
        self.minute_start_time = time.time()
    
    async def _rate_limit(self):
        """Implement rate limiting for API requests."""
        current_time = time.time()
        
        # Reset minute counter if a minute has passed
        if current_time - self.minute_start_time >= 60:
            self.requests_this_minute = 0
            self.minute_start_time = current_time
        
        # Check minute rate limit
        if self.requests_this_minute >= self.rate_limit_per_minute:
            sleep_time = 60 - (current_time - self.minute_start_time)
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
                self.requests_this_minute = 0
                self.minute_start_time = time.time()
        
        # Check second rate limit
        time_since_last = current_time - self.last_request_time
        if time_since_last < 1.0 / self.rate_limit_per_second:
            sleep_time = (1.0 / self.rate_limit_per_second) - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.requests_this_minute += 1
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a rate-limited request to TMDB API."""
        await self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        default_params = {
            "language": APIConfig.DEFAULT_LANGUAGE,
            "include_adult": APIConfig.INCLUDE_ADULT
        }
        
        if params:
            default_params.update(params)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=default_params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 401:
                        raise ValueError("Invalid TMDB API key")
                    elif response.status == 429:
                        logger.warning("Rate limit exceeded, waiting before retry")
                        await asyncio.sleep(60)
                        return await self._make_request(endpoint, params)
                    else:
                        response.raise_for_status()
        except aiohttp.ClientError as e:
            logger.error(f"Network error during TMDB API request: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during TMDB API request: {e}")
            raise
    
    async def get_popular_movies(self, page: int = 1) -> Dict[str, Any]:
        """Get popular movies from TMDB."""
        return await self._make_request("movie/popular", {"page": page})
    
    async def get_top_rated_movies(self, page: int = 1) -> Dict[str, Any]:
        """Get top rated movies from TMDB."""
        return await self._make_request("movie/top_rated", {"page": page})
    
    async def get_movie_details(self, movie_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific movie."""
        return await self._make_request(f"movie/{movie_id}")
    
    async def get_movie_credits(self, movie_id: int) -> Dict[str, Any]:
        """Get cast and crew information for a movie."""
        return await self._make_request(f"movie/{movie_id}/credits")
    
    async def get_genres(self) -> Dict[str, Any]:
        """Get list of movie genres."""
        return await self._make_request("genre/movie/list")
    
    async def search_movies(self, query: str, page: int = 1) -> Dict[str, Any]:
        """Search for movies by title."""
        return await self._make_request("search/movie", {
            "query": query,
            "page": page
        })
    
    async def get_movies_by_genre(self, genre_id: int, page: int = 1) -> Dict[str, Any]:
        """Get movies by genre ID."""
        return await self._make_request("discover/movie", {
            "with_genres": genre_id,
            "page": page
        })
    
    def parse_movie_data(self, movie_data: Dict[str, Any]) -> TMDBMovie:
        """Parse raw movie data into TMDBMovie object."""
        return TMDBMovie(
            id=movie_data.get("id"),
            title=movie_data.get("title", ""),
            original_title=movie_data.get("original_title", ""),
            overview=movie_data.get("overview", ""),
            release_date=movie_data.get("release_date", ""),
            poster_path=movie_data.get("poster_path"),
            backdrop_path=movie_data.get("backdrop_path"),
            popularity=movie_data.get("popularity", 0.0),
            vote_average=movie_data.get("vote_average", 0.0),
            vote_count=movie_data.get("vote_count", 0),
            genre_ids=movie_data.get("genre_ids", []),
            adult=movie_data.get("adult", False),
            video=movie_data.get("video", False),
            original_language=movie_data.get("original_language", "")
        ) 