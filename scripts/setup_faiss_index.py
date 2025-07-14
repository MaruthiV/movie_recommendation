#!/usr/bin/env python3
"""
FAISS vector database setup for movie similarity search.
Creates embeddings and builds FAISS index for fast retrieval.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSMovieIndex:
    """FAISS-based movie similarity search index."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the FAISS index with a sentence transformer model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.movie_ids = []
        self.movie_metadata = []
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Initialized FAISS index with model: {model_name}")
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def create_embeddings(self, enriched_movies: List[Dict]) -> np.ndarray:
        """Create embeddings for all movies."""
        texts = []
        valid_movies = []
        
        for movie in enriched_movies:
            if movie.get('enriched') and movie.get('text_for_embedding'):
                texts.append(movie['text_for_embedding'])
                valid_movies.append(movie)
            else:
                logger.warning(f"Skipping movie {movie.get('title', 'Unknown')} - no enriched text")
        
        if not texts:
            raise ValueError("No valid texts found for embedding generation")
        
        logger.info(f"Generating embeddings for {len(texts)} movies...")
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings, valid_movies
    
    def build_index(self, embeddings: np.ndarray, movies: List[Dict]) -> None:
        """Build FAISS index from embeddings."""
        logger.info("Building FAISS index...")
        
        # Create index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Add vectors to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.movie_ids = [movie['movieId'] for movie in movies]
        self.movie_metadata = movies
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        logger.info(f"Index type: {type(self.index)}")
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search for similar movies using text query."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.movie_metadata):
                movie = self.movie_metadata[idx].copy()
                movie['similarity_score'] = float(score)
                movie['rank'] = i + 1
                results.append(movie)
        
        return results
    
    def search_by_movie_id(self, movie_id: int, k: int = 10) -> List[Dict]:
        """Search for movies similar to a given movie ID."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Find the movie in our metadata
        movie_idx = None
        for i, movie in enumerate(self.movie_metadata):
            if movie['movieId'] == movie_id:
                movie_idx = i
                break
        
        if movie_idx is None:
            raise ValueError(f"Movie ID {movie_id} not found in index")
        
        # Get the movie's embedding
        movie_embedding = self.index.reconstruct(movie_idx).reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(movie_embedding.astype('float32'), k + 1)  # +1 to exclude self
        
        # Return results (excluding the query movie)
        results = []
        for i, (score, idx) in enumerate(zip(scores[0][1:], indices[0][1:])):  # Skip first result (self)
            if idx < len(self.movie_metadata):
                movie = self.movie_metadata[idx].copy()
                movie['similarity_score'] = float(score)
                movie['rank'] = i + 1
                results.append(movie)
        
        return results
    
    def save_index(self, output_dir: Path) -> None:
        """Save the FAISS index and metadata."""
        output_dir.mkdir(exist_ok=True)
        
        # Save FAISS index
        index_path = output_dir / "movie_index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = output_dir / "movie_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'movie_ids': self.movie_ids,
                'movie_metadata': self.movie_metadata,
                'model_name': self.model_name,
                'dimension': self.dimension
            }, f)
        
        logger.info(f"Saved FAISS index to: {index_path}")
        logger.info(f"Saved metadata to: {metadata_path}")
    
    def load_index(self, index_dir: Path) -> None:
        """Load the FAISS index and metadata."""
        index_path = index_dir / "movie_index.faiss"
        metadata_path = index_dir / "movie_metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index files not found in {index_dir}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.movie_ids = metadata['movie_ids']
        self.movie_metadata = metadata['movie_metadata']
        self.model_name = metadata['model_name']
        self.dimension = metadata['dimension']
        
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")


def test_faiss_index(faiss_index: FAISSMovieIndex) -> None:
    """Test the FAISS index with sample queries."""
    logger.info("\n" + "="*50)
    logger.info("TESTING FAISS INDEX")
    logger.info("="*50)
    
    # Test text search
    test_queries = [
        "sci-fi space adventure",
        "romantic comedy",
        "action thriller",
        "animated family movie",
        "drama about war"
    ]
    
    for query in test_queries:
        logger.info(f"\nSearching for: '{query}'")
        results = faiss_index.search(query, k=5)
        
        for result in results:
            logger.info(f"  {result['rank']}. {result['title']} (Score: {result['similarity_score']:.3f})")
    
    # Test movie similarity
    if faiss_index.movie_metadata:
        sample_movie = faiss_index.movie_metadata[0]
        logger.info(f"\nFinding movies similar to: {sample_movie['title']}")
        similar_movies = faiss_index.search_by_movie_id(sample_movie['movieId'], k=5)
        
        for movie in similar_movies:
            logger.info(f"  {movie['rank']}. {movie['title']} (Score: {movie['similarity_score']:.3f})")


def main():
    """Main function to build and test FAISS index."""
    # Load enriched movie data
    enriched_file = Path("data/enriched_movies.json")
    if not enriched_file.exists():
        logger.error(f"Enriched movies file not found: {enriched_file}")
        logger.info("Please run the enrichment script first: python scripts/enrich_movie_data.py")
        return
    
    with open(enriched_file, 'r', encoding='utf-8') as f:
        enriched_movies = json.load(f)
    
    logger.info(f"Loaded {len(enriched_movies)} enriched movies")
    
    # Initialize FAISS index
    faiss_index = FAISSMovieIndex()
    
    # Create embeddings and build index
    embeddings, valid_movies = faiss_index.create_embeddings(enriched_movies)
    faiss_index.build_index(embeddings, valid_movies)
    
    # Test the index
    test_faiss_index(faiss_index)
    
    # Save the index
    output_dir = Path("data/faiss_index")
    faiss_index.save_index(output_dir)
    
    logger.info(f"\nFAISS index setup complete!")
    logger.info(f"Index contains {faiss_index.index.ntotal} movies")
    logger.info(f"Index saved to: {output_dir}")


if __name__ == "__main__":
    main()
