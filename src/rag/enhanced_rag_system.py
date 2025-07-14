#!/usr/bin/env python3
"""
Enhanced RAG system with knowledge graph integration for movie explanations.
"""

import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .rag_system import RAGExplanationSystem
    from .knowledge_graph import MovieKnowledgeGraph
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False
    logger.warning("Knowledge graph not available")


class EnhancedRAGSystem:
    """Enhanced RAG system with knowledge graph integration."""
    
    def __init__(self, index_dir: Path, knowledge_graph: Optional[MovieKnowledgeGraph] = None):
        """Initialize the enhanced RAG system."""
        # Initialize base RAG system
        self.base_rag = RAGExplanationSystem(index_dir)
        
        # Initialize knowledge graph
        self.knowledge_graph = knowledge_graph
        if knowledge_graph:
            logger.info("Knowledge graph integration enabled")
        else:
            logger.info("Knowledge graph integration disabled")
    
    def explain_recommendation(self, source_movie_id: int, rec_movie_id: int) -> str:
        """Generate enhanced explanation using both RAG and knowledge graph."""
        # Get base explanation
        base_explanation = self.base_rag.explain_recommendation(source_movie_id, rec_movie_id)
        
        # Enhance with knowledge graph facts if available
        if self.knowledge_graph:
            try:
                # Get contextual facts from knowledge graph
                contextual_facts = self.knowledge_graph.get_contextual_facts(source_movie_id, rec_movie_id)
                
                # Get shared connections
                connections = self.knowledge_graph.find_shared_connections(source_movie_id, rec_movie_id)
                
                # Enhance explanation with additional facts
                enhanced_facts = []
                
                # Add franchise information
                sequels = self.knowledge_graph.get_sequel_relationships(source_movie_id)
                for sequel in sequels:
                    if sequel['movieId'] == rec_movie_id:
                        enhanced_facts.append("are part of the same franchise")
                        break
                
                # Add award information if available
                awards = self.knowledge_graph.get_award_connections(source_movie_id)
                if awards:
                    enhanced_facts.append("both have received critical acclaim")
                
                # Add production company information
                if connections.get('shared_companies'):
                    companies = connections['shared_companies'][:1]
                    enhanced_facts.append(f"both produced by {', '.join(companies)}")
                
                # Combine facts
                if enhanced_facts:
                    # Replace the fallback in base explanation with enhanced facts
                    if "have similar plot or style" in base_explanation:
                        base_explanation = base_explanation.replace(
                            "have similar plot or style", 
                            f"{', and '.join(enhanced_facts)}"
                        )
                    else:
                        # Add enhanced facts to the end
                        base_explanation += f", and {', and '.join(enhanced_facts)}"
                
                logger.info(f"Enhanced explanation with {len(enhanced_facts)} additional facts")
                
            except Exception as e:
                logger.warning(f"Failed to enhance explanation with knowledge graph: {e}")
        
        return base_explanation
    
    def explain_for_user(self, user_liked_movie_ids: List[int], rec_movie_id: int) -> str:
        """Generate explanation for user with multiple liked movies."""
        # Find the most similar liked movie
        best_explanation = None
        best_score = -1
        
        for liked_id in user_liked_movie_ids:
            sim_movies = self.base_rag.get_similar_movies(liked_id, k=10)
            for m in sim_movies:
                if int(m['movieId']) == int(rec_movie_id) and m['similarity_score'] > best_score:
                    best_score = m['similarity_score']
                    best_explanation = self.explain_recommendation(liked_id, rec_movie_id)
        
        if best_explanation:
            return best_explanation
        
        # Fallback: use the first liked movie
        if user_liked_movie_ids:
            return self.explain_recommendation(user_liked_movie_ids[0], rec_movie_id)
        
        return "Explanation unavailable."
    
    def get_detailed_explanation(self, source_movie_id: int, rec_movie_id: int) -> Dict[str, Any]:
        """Get detailed explanation with all supporting facts."""
        explanation = self.explain_recommendation(source_movie_id, rec_movie_id)
        
        details = {
            'explanation': explanation,
            'source_movie': self.base_rag.get_movie_by_id(source_movie_id),
            'recommended_movie': self.base_rag.get_movie_by_id(rec_movie_id),
            'similarity_score': None,
            'shared_connections': {},
            'contextual_facts': []
        }
        
        # Get similarity score
        sim_movies = self.base_rag.get_similar_movies(source_movie_id, k=20)
        for m in sim_movies:
            if int(m['movieId']) == int(rec_movie_id):
                details['similarity_score'] = m['similarity_score']
                break
        
        # Get knowledge graph details if available
        if self.knowledge_graph:
            try:
                details['shared_connections'] = self.knowledge_graph.find_shared_connections(
                    source_movie_id, rec_movie_id
                )
                details['contextual_facts'] = self.knowledge_graph.get_contextual_facts(
                    source_movie_id, rec_movie_id
                )
            except Exception as e:
                logger.warning(f"Failed to get knowledge graph details: {e}")
        
        return details


def create_enhanced_rag_system(index_dir: Path = None, 
                              neo4j_uri: str = "bolt://localhost:7687",
                              neo4j_user: str = "neo4j", 
                              neo4j_password: str = "test_password") -> EnhancedRAGSystem:
    """Create an enhanced RAG system with knowledge graph integration."""
    if index_dir is None:
        index_dir = Path("data/faiss_index")
    
    # Initialize knowledge graph
    knowledge_graph = None
    if KNOWLEDGE_GRAPH_AVAILABLE:
        try:
            knowledge_graph = MovieKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
            logger.info("Knowledge graph connected successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to knowledge graph: {e}")
            knowledge_graph = None
    
    # Create enhanced RAG system
    return EnhancedRAGSystem(index_dir, knowledge_graph)


if __name__ == "__main__":
    # Test the enhanced RAG system
    enhanced_rag = create_enhanced_rag_system()
    
    # Test explanations
    test_pairs = [
        (1, 2),  # Toy Story -> Jumanji
        (1, 3),  # Toy Story -> Grumpier Old Men
        (2, 4),  # Jumanji -> Waiting to Exhale
    ]
    
    print("Enhanced RAG System Test")
    print("=" * 50)
    
    for source_id, rec_id in test_pairs:
        print(f"\nTesting: Movie {source_id} -> Movie {rec_id}")
        
        # Get basic explanation
        explanation = enhanced_rag.explain_recommendation(source_id, rec_id)
        print(f"Explanation: {explanation}")
        
        # Get detailed explanation
        details = enhanced_rag.get_detailed_explanation(source_id, rec_id)
        print(f"Similarity Score: {details['similarity_score']}")
        print(f"Shared Connections: {details['shared_connections']}")
        print(f"Contextual Facts: {details['contextual_facts']}")
    
    print("\nEnhanced RAG System Test Complete!") 