#!/usr/bin/env python3
"""
Simple test script to verify RAG explanation system works independently.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.rag_system import RAGExplanationSystem

def test_rag_system():
    """Test the RAG explanation system directly."""
    print("Testing RAG Explanation System")
    print("=" * 50)
    
    # Initialize RAG system
    rag_index_dir = Path("data/faiss_index")
    rag = RAGExplanationSystem(rag_index_dir)
    
    print(f"Loaded RAG system with {len(rag.movie_metadata)} movies")
    
    # Test explanations for first few movies
    test_pairs = [
        (1, 2),  # Toy Story -> Jumanji
        (1, 3),  # Toy Story -> Grumpier Old Men
        (2, 4),  # Jumanji -> Waiting to Exhale
        (3, 5),  # Grumpier Old Men -> Father of the Bride Part II
    ]
    
    for source_id, rec_id in test_pairs:
        print(f"\nTesting: Movie {source_id} -> Movie {rec_id}")
        
        # Get movie titles
        source_movie = rag.get_movie_by_id(source_id)
        rec_movie = rag.get_movie_by_id(rec_id)
        
        if source_movie and rec_movie:
            print(f"Source: {source_movie['title']}")
            print(f"Recommendation: {rec_movie['title']}")
            
            # Get explanation
            explanation = rag.explain_recommendation(source_id, rec_id)
            print(f"Explanation: {explanation}")
        else:
            print("One or both movies not found")
    
    # Test similar movies
    print(f"\n" + "=" * 50)
    print("Testing Similar Movies")
    print("=" * 50)
    
    for movie_id in [1, 2, 3]:
        movie = rag.get_movie_by_id(movie_id)
        if movie:
            print(f"\nSimilar movies to: {movie['title']}")
            similar = rag.get_similar_movies(movie_id, k=3)
            for i, sim_movie in enumerate(similar, 1):
                print(f"  {i}. {sim_movie['title']} (Score: {sim_movie['similarity_score']:.3f})")
    
    print(f"\n" + "=" * 50)
    print("RAG System Test Complete!")

if __name__ == "__main__":
    test_rag_system() 