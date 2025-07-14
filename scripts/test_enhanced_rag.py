#!/usr/bin/env python3
"""
Test script for enhanced RAG system with knowledge graph integration.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.enhanced_rag_system import create_enhanced_rag_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_rag():
    """Test the enhanced RAG system."""
    logger.info("Testing Enhanced RAG System with Knowledge Graph")
    logger.info("=" * 60)
    
    # Create enhanced RAG system
    enhanced_rag = create_enhanced_rag_system()
    
    # Test movie pairs
    test_pairs = [
        (1, 2),   # Toy Story -> Jumanji
        (1, 3),   # Toy Story -> Grumpier Old Men  
        (2, 4),   # Jumanji -> Waiting to Exhale
        (3, 5),   # Grumpier Old Men -> Father of the Bride Part II
        (10, 15), # GoldenEye -> Casino
    ]
    
    for source_id, rec_id in test_pairs:
        logger.info(f"\n{'='*40}")
        logger.info(f"Testing: Movie {source_id} -> Movie {rec_id}")
        
        # Get source and recommended movie info
        source_movie = enhanced_rag.base_rag.get_movie_by_id(source_id)
        rec_movie = enhanced_rag.base_rag.get_movie_by_id(rec_id)
        
        if source_movie and rec_movie:
            logger.info(f"Source: {source_movie['title']}")
            logger.info(f"Recommendation: {rec_movie['title']}")
            
            # Get enhanced explanation
            explanation = enhanced_rag.explain_recommendation(source_id, rec_id)
            logger.info(f"Enhanced Explanation: {explanation}")
            
            # Get detailed explanation
            details = enhanced_rag.get_detailed_explanation(source_id, rec_id)
            if details['similarity_score'] is not None:
                logger.info(f"Similarity Score: {details['similarity_score']:.3f}")
            else:
                logger.info("Similarity Score: Not available")
            
            if details['shared_connections']:
                logger.info("Shared Connections:")
                for key, value in details['shared_connections'].items():
                    if value:
                        logger.info(f"  {key}: {value}")
            
            if details['contextual_facts']:
                logger.info(f"Contextual Facts: {details['contextual_facts']}")
        else:
            logger.warning(f"One or both movies not found: {source_id}, {rec_id}")
    
    logger.info(f"\n{'='*60}")
    logger.info("Enhanced RAG System Test Complete!")

if __name__ == "__main__":
    test_enhanced_rag() 