#!/usr/bin/env python3
"""
Movie data enrichment script for RAG system.
Fetches metadata from TMDB, OMDB, and other sources.
"""

import os
import json
import time
import logging
import pandas as pd
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieDataEnricher:
    """Enriches movie data with metadata from various APIs."""
    
    def __init__(self):
        """Initialize the enricher with API keys."""
        self.tmdb_api_key = os.getenv('TMDB_API_KEY')
        self.omdb_api_key = os.getenv('OMDB_API_KEY')
        
        if not self.tmdb_api_key:
            logger.warning("TMDB_API_KEY not found. TMDB enrichment will be skipped.")
        if not self.omdb_api_key:
            logger.warning("OMDB_API_KEY not found. OMDB enrichment will be skipped.")
        
        # API endpoints
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        self.omdb_base_url = "http://www.omdbapi.com/"
        
        # Rate limiting
        self.request_delay = 0.1  # 100ms between requests
        
        # Initialize tokenizer for text processing
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make a rate-limited API request."""
        try:
            time.sleep(self.request_delay)  # Rate limiting
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def search_tmdb_movie(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Search for a movie on TMDB."""
        if not self.tmdb_api_key:
            return None
        
        params = {
            'api_key': self.tmdb_api_key,
            'query': title,
            'language': 'en-US',
            'page': 1,
            'include_adult': False
        }
        
        if year:
            params['year'] = year
        
        url = f"{self.tmdb_base_url}/search/movie"
        data = self._make_request(url, params)
        
        if data and data.get('results'):
            return data['results'][0]  # Return first match
        return None
    
    def get_tmdb_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """Get detailed movie information from TMDB."""
        if not self.tmdb_api_key:
            return None
        
        params = {
            'api_key': self.tmdb_api_key,
            'language': 'en-US',
            'append_to_response': 'credits,keywords,reviews'
        }
        
        url = f"{self.tmdb_base_url}/movie/{tmdb_id}"
        return self._make_request(url, params)
    
    def get_omdb_movie(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Get movie information from OMDB."""
        if not self.omdb_api_key:
            return None
        
        params = {
            'apikey': self.omdb_api_key,
            't': title,
            'type': 'movie'
        }
        
        if year:
            params['y'] = year
        
        return self._make_request(self.omdb_base_url, params)
    
    def extract_year_from_title(self, title: str) -> Optional[int]:
        """Extract year from movie title like 'Movie Name (1995)'."""
        import re
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else None
    
    def clean_title(self, title: str) -> str:
        """Remove year from title for better search."""
        import re
        return re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
    
    def create_movie_text_for_embedding(self, movie_data: Dict) -> str:
        """Create a comprehensive text representation of a movie for embedding."""
        parts = []
        
        # Basic info
        if movie_data.get('title'):
            parts.append(f"Title: {movie_data['title']}")
        
        if movie_data.get('genres'):
            genres = ', '.join(movie_data['genres'])
            parts.append(f"Genres: {genres}")
        
        if movie_data.get('year'):
            parts.append(f"Year: {movie_data['year']}")
        
        # Plot/overview
        if movie_data.get('overview'):
            parts.append(f"Plot: {movie_data['overview']}")
        
        if movie_data.get('plot'):
            parts.append(f"Plot: {movie_data['plot']}")
        
        # Cast and crew
        if movie_data.get('cast'):
            cast = ', '.join(movie_data['cast'][:5])  # Top 5 actors
            parts.append(f"Cast: {cast}")
        
        if movie_data.get('director'):
            parts.append(f"Director: {movie_data['director']}")
        
        # Keywords/tags
        if movie_data.get('keywords'):
            keywords = ', '.join(movie_data['keywords'][:10])  # Top 10 keywords
            parts.append(f"Keywords: {keywords}")
        
        # Awards and ratings
        if movie_data.get('awards'):
            parts.append(f"Awards: {movie_data['awards']}")
        
        if movie_data.get('rating'):
            parts.append(f"Rating: {movie_data['rating']}")
        
        return ' | '.join(parts)
    
    def enrich_movie(self, movie_row: pd.Series) -> Dict[str, Any]:
        """Enrich a single movie with metadata from multiple sources."""
        movie_id = movie_row['movieId']
        title = movie_row['title']
        genres = movie_row['genres']
        
        # Extract year from title
        year = self.extract_year_from_title(title)
        clean_title = self.clean_title(title)
        
        enriched_data = {
            'movieId': movie_id,
            'title': title,
            'genres': genres,
            'year': year,
            'clean_title': clean_title,
            'enriched': False
        }
        
        # Try TMDB first
        tmdb_movie = self.search_tmdb_movie(clean_title, year)
        if tmdb_movie:
            tmdb_id = tmdb_movie['id']
            tmdb_details = self.get_tmdb_movie_details(tmdb_id)
            
            if tmdb_details:
                # Extract TMDB data
                enriched_data.update({
                    'tmdb_id': tmdb_id,
                    'overview': tmdb_details.get('overview', ''),
                    'tagline': tmdb_details.get('tagline', ''),
                    'runtime': tmdb_details.get('runtime'),
                    'budget': tmdb_details.get('budget'),
                    'revenue': tmdb_details.get('revenue'),
                    'vote_average': tmdb_details.get('vote_average'),
                    'vote_count': tmdb_details.get('vote_count'),
                    'popularity': tmdb_details.get('popularity'),
                    'release_date': tmdb_details.get('release_date'),
                    'status': tmdb_details.get('status'),
                    'original_language': tmdb_details.get('original_language'),
                    'production_companies': [comp['name'] for comp in tmdb_details.get('production_companies', [])],
                    'production_countries': [country['name'] for country in tmdb_details.get('production_countries', [])],
                    'spoken_languages': [lang['name'] for lang in tmdb_details.get('spoken_languages', [])],
                    'enriched': True
                })
                
                # Extract cast and crew
                if 'credits' in tmdb_details:
                    credits = tmdb_details['credits']
                    
                    # Cast (top 10 actors)
                    cast = []
                    for person in credits.get('cast', [])[:10]:
                        cast.append(person['name'])
                    enriched_data['cast'] = cast
                    
                    # Director
                    directors = []
                    for person in credits.get('crew', []):
                        if person.get('job') == 'Director':
                            directors.append(person['name'])
                    enriched_data['director'] = ', '.join(directors) if directors else ''
                
                # Extract keywords
                if 'keywords' in tmdb_details:
                    keywords = [kw['name'] for kw in tmdb_details['keywords'].get('keywords', [])]
                    enriched_data['keywords'] = keywords
                
                # Extract reviews
                if 'reviews' in tmdb_details:
                    reviews = tmdb_details['reviews'].get('results', [])
                    enriched_data['review_count'] = len(reviews)
                    if reviews:
                        enriched_data['sample_review'] = reviews[0].get('content', '')[:500]  # First 500 chars
        
        # Try OMDB as backup or for additional data
        omdb_movie = self.get_omdb_movie(clean_title, year)
        if omdb_movie and omdb_movie.get('Response') == 'True':
            enriched_data.update({
                'omdb_plot': omdb_movie.get('Plot', ''),
                'omdb_director': omdb_movie.get('Director', ''),
                'omdb_writer': omdb_movie.get('Writer', ''),
                'omdb_actors': omdb_movie.get('Actors', ''),
                'omdb_awards': omdb_movie.get('Awards', ''),
                'omdb_metascore': omdb_movie.get('Metascore', ''),
                'omdb_imdb_rating': omdb_movie.get('imdbRating', ''),
                'omdb_imdb_votes': omdb_movie.get('imdbVotes', ''),
                'omdb_box_office': omdb_movie.get('BoxOffice', ''),
                'omdb_production': omdb_movie.get('Production', ''),
                'omdb_website': omdb_movie.get('Website', ''),
                'enriched': True
            })
        
        # Create text representation for embeddings
        enriched_data['text_for_embedding'] = self.create_movie_text_for_embedding(enriched_data)
        
        # Calculate text length (useful for chunking)
        enriched_data['text_length'] = len(enriched_data['text_for_embedding'])
        enriched_data['token_count'] = len(self.tokenizer.encode(enriched_data['text_for_embedding']))
        
        return enriched_data
    
    def enrich_movies_batch(self, movies_df: pd.DataFrame, max_movies: int = None) -> List[Dict]:
        """Enrich a batch of movies."""
        if max_movies:
            movies_df = movies_df.head(max_movies)
        
        enriched_movies = []
        
        logger.info(f"Starting enrichment of {len(movies_df)} movies...")
        
        for idx, movie_row in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Enriching movies"):
            try:
                enriched_movie = self.enrich_movie(movie_row)
                enriched_movies.append(enriched_movie)
                
                # Log progress every 50 movies
                if (idx + 1) % 50 == 0:
                    enriched_count = sum(1 for m in enriched_movies if m['enriched'])
                    logger.info(f"Processed {idx + 1}/{len(movies_df)} movies. Enriched: {enriched_count}")
                    
            except Exception as e:
                logger.error(f"Error enriching movie {movie_row['title']}: {e}")
                # Add basic data even if enrichment fails
                enriched_movies.append({
                    'movieId': movie_row['movieId'],
                    'title': movie_row['title'],
                    'genres': movie_row['genres'],
                    'enriched': False,
                    'error': str(e)
                })
        
        return enriched_movies


def main():
    """Main function to run the enrichment pipeline."""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Load existing movie data
    movies_file = data_dir / "movies.csv"
    if not movies_file.exists():
        logger.error(f"Movies file not found: {movies_file}")
        logger.info("Please ensure you have movies.csv in the data/ directory")
        return
    
    # Load movies data
    movies_df = pd.read_csv(movies_file)
    logger.info(f"Loaded {len(movies_df)} movies from {movies_file}")
    
    # Initialize enricher
    enricher = MovieDataEnricher()
    
    # Enrich movies (limit to first 100 for testing)
    enriched_movies = enricher.enrich_movies_batch(movies_df, max_movies=100)
    
    # Save enriched data
    output_file = data_dir / "enriched_movies.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_movies, f, indent=2, ensure_ascii=False)
    
    # Print summary
    enriched_count = sum(1 for m in enriched_movies if m['enriched'])
    logger.info(f"Enrichment complete!")
    logger.info(f"Total movies processed: {len(enriched_movies)}")
    logger.info(f"Successfully enriched: {enriched_count}")
    logger.info(f"Enrichment rate: {enriched_count/len(enriched_movies)*100:.1f}%")
    logger.info(f"Output saved to: {output_file}")
    
    # Show sample enriched movie
    if enriched_movies:
        sample = next((m for m in enriched_movies if m['enriched']), enriched_movies[0])
        logger.info(f"\nSample enriched movie:")
        logger.info(f"Title: {sample['title']}")
        logger.info(f"Enriched: {sample['enriched']}")
        if sample.get('overview'):
            logger.info(f"Overview: {sample['overview'][:100]}...")
        if sample.get('text_length'):
            logger.info(f"Text length: {sample['text_length']} chars, {sample['token_count']} tokens")


if __name__ == "__main__":
    main()
