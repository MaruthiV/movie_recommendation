#!/usr/bin/env python3
"""
Test script to verify RAG explanation integration with movie similarity calculator.
"""

import sys
from pathlib import Path
import torch
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.lightgcn_model import LightGCN
from models.movie_similarity import MovieSimilarityCalculator
from data.movie_database import MovieDatabase

def test_rag_integration():
    """Test the integration of RAG explanations with movie similarity."""
    print("Testing RAG Integration with Movie Similarity Calculator")
    print("=" * 60)
    
    # Load movie database
    movie_db = MovieDatabase()
    movies_df = movie_db.get_movies()
    
    # Create a simple test model (you can replace this with your trained model)
    num_users = 1000
    num_items = len(movies_df)
    embedding_dim = 64
    
    model = LightGCN(num_users, num_items, embedding_dim)
    
    # Create dummy edge index for testing
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long)
    
    # Initialize similarity calculator with RAG system
    rag_index_dir = Path("data/faiss_index")
    similarity_calc = MovieSimilarityCalculator(
        model, 
        device='cpu', 
        rag_index_dir=rag_index_dir
    )
    
    # Test with a few movie IDs
    test_movie_ids = [1, 2, 3, 4, 5]  # First 5 movies
    
    for movie_id in test_movie_ids:
        print(f"\nTesting movie ID: {movie_id}")
        print(f"Movie title: {movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]}")
        
        try:
            # Get similar movies with explanations
            similar_movies = similarity_calc.get_similar_movies_with_explanations(
                movie_id, edge_index, top_k=3
            )
            
            print("Similar movies with explanations:")
            for i, result in enumerate(similar_movies, 1):
                rec_movie_id = result['movie_id']
                rec_title = movies_df[movies_df['movieId'] == rec_movie_id]['title'].iloc[0]
                score = result['similarity_score']
                explanation = result['explanation']
                
                print(f"  {i}. {rec_title} (ID: {rec_movie_id}, Score: {score:.3f})")
                if explanation:
                    print(f"     Explanation: {explanation}")
                else:
                    print(f"     Explanation: Not available")
                    
        except Exception as e:
            print(f"Error testing movie {movie_id}: {e}")
    
    print("\n" + "=" * 60)
    print("RAG Integration Test Complete!")

if __name__ == "__main__":
    test_rag_integration() 