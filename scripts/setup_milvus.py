#!/usr/bin/env python3
"""
Script to initialize Milvus vector database and test connection.
"""
import logging
import time
from src.data.vector_database import VectorDatabaseManager

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("Setting up Milvus vector database...")
    
    # Wait for Milvus to be ready
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            vector_db = VectorDatabaseManager()
            print("Connected to Milvus successfully!")
            break
        except Exception as e:
            retry_count += 1
            print(f"Attempt {retry_count}/{max_retries}: Milvus not ready yet. Waiting...")
            time.sleep(2)
    
    if retry_count >= max_retries:
        print("Failed to connect to Milvus after maximum retries.")
        return
    
    try:
        # Create collections
        print("Creating movie embeddings collection...")
        vector_db.create_movie_embeddings_collection()
        
        print("Creating user embeddings collection...")
        vector_db.create_user_embeddings_collection()
        
        # Test with sample data
        print("Testing with sample data...")
        sample_movie_data = [
            {
                "movie_id": 1,
                "title": "Sample Movie 1",
                "embedding": [0.1] * 512  # 512-dimensional vector
            },
            {
                "movie_id": 2,
                "title": "Sample Movie 2", 
                "embedding": [0.2] * 512
            }
        ]
        
        vector_db.insert_movie_embeddings(sample_movie_data)
        
        # Test search
        query_embedding = [0.15] * 512
        results = vector_db.search_similar_movies(query_embedding, top_k=2)
        
        print(f"Search test successful. Found {len(results)} similar movies:")
        for result in results:
            print(f"  - {result['title']} (score: {result['score']:.4f})")
        
        # Get collection stats
        movie_stats = vector_db.get_collection_stats("movie_embeddings")
        print(f"Movie embeddings collection: {movie_stats['num_entities']} entities")
        
        print("Milvus setup completed successfully!")
        
    except Exception as e:
        print(f"Error during Milvus setup: {e}")
    finally:
        vector_db.disconnect()

if __name__ == "__main__":
    main() 