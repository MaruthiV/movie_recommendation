#!/usr/bin/env python3
"""
Unified end-to-end test for the movie recommendation system:
- Model inference
- Similarity search
- RAG explanation
- Explanation pipeline
"""

import sys
from pathlib import Path
import logging
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.enhanced_rag_system import create_enhanced_rag_system
from rag.explanation_pipeline import create_explanation_pipeline

# Try to import the real model, else use a mock
try:
    from models.lightgcn_model import LightGCN
    from models.movie_similarity import MovieSimilarityCalculator
    from data.movie_database import MovieDatabase
    REAL_MODEL = True
except ImportError:
    REAL_MODEL = False
    print("Warning: Real model not found, using mock model for test.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("\n=== Unified End-to-End System Test ===\n")
    # 1. Load model and data
    if REAL_MODEL:
        movie_db = MovieDatabase()
        movies_df = movie_db.get_movies()
        num_users = 1000
        num_items = len(movies_df)
        embedding_dim = 64
        model = LightGCN(num_users, num_items, embedding_dim)
        # Dummy edge index for test
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long)
        similarity_calc = MovieSimilarityCalculator(model, device='cpu', rag_index_dir=Path("data/faiss_index"))
    else:
        # Mock data
        class MockModel:
            def to(self, device): pass
            def eval(self): pass
            def forward(self, edge_index):
                # Return dummy user/item embeddings
                return torch.randn(5, 64), torch.randn(100, 64)
        model = MockModel()
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long)
        from models.movie_similarity import MovieSimilarityCalculator
        similarity_calc = MovieSimilarityCalculator(model, device='cpu', rag_index_dir=Path("data/faiss_index"))
        movies_df = None

    # 2. Generate recommendations for a sample user (user 0)
    print("--- Model Inference: Recommendations for User 0 ---")
    # For demo, just use similar movies to movieId 1 (Toy Story)
    movie_id = 1
    similar_movies = similarity_calc.get_similar_movies_with_explanations(movie_id, edge_index, top_k=5)
    for i, rec in enumerate(similar_movies, 1):
        print(f"{i}. MovieId: {rec['movie_id']} | Score: {rec['similarity_score']:.3f}")
        print(f"   Explanation: {rec['explanation']}")
    print()

    # 3. FAISS similarity search
    print("--- FAISS Similarity Search for 'Toy Story' ---")
    enhanced_rag = create_enhanced_rag_system()
    faiss_results = enhanced_rag.base_rag.get_similar_movies(movie_id, k=5)
    for i, m in enumerate(faiss_results, 1):
        print(f"{i}. MovieId: {m['movieId']} | Score: {m['similarity_score']:.3f}")
    print()

    # 4. RAG explanation
    print("--- RAG Explanation for Toy Story -> Jumanji ---")
    explanation = enhanced_rag.explain_recommendation(1, 2)
    print(f"Explanation: {explanation}")
    print()

    # 5. Explanation pipeline (full report)
    print("--- Explanation Pipeline Report for Toy Story -> Jumanji ---")
    pipeline = create_explanation_pipeline(enhanced_rag, enhanced_rag.knowledge_graph)
    report = pipeline.generate_explanation_report(1, 2)
    print(f"Explanation: {report['explanation']}")
    print(f"Confidence Score: {report['confidence_score']:.3f}")
    print(f"Explanation Type: {report['explanation_type']}")
    print(f"Similarity Score: {report['similarity_score']}")
    print(f"Supporting Facts: {len(report['supporting_facts'])}")
    for i, fact in enumerate(report['supporting_facts'], 1):
        print(f"  {i}. {fact['fact']} (Confidence: {fact['confidence']:.2f})")
    print(f"Fact Summary: {report['fact_summary']}")
    print("\n=== End of Unified System Test ===\n")

if __name__ == "__main__":
    main() 