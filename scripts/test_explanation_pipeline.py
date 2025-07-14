#!/usr/bin/env python3
"""
Test script for explanation pipeline with supporting facts.
"""

import sys
from pathlib import Path
import logging
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.enhanced_rag_system import create_enhanced_rag_system
from rag.explanation_pipeline import create_explanation_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_explanation_pipeline():
    """Test the explanation pipeline."""
    logger.info("Testing Explanation Pipeline with Supporting Facts")
    logger.info("=" * 70)
    
    # Create enhanced RAG system
    enhanced_rag = create_enhanced_rag_system()
    
    # Create explanation pipeline
    pipeline = create_explanation_pipeline(enhanced_rag, enhanced_rag.knowledge_graph)
    
    # Test movie pairs
    test_pairs = [
        (1, 2),   # Toy Story -> Jumanji
        (1, 3),   # Toy Story -> Grumpier Old Men
        (2, 4),   # Jumanji -> Waiting to Exhale
        (3, 5),   # Grumpier Old Men -> Father of the Bride Part II
        (10, 15), # GoldenEye -> Cutthroat Island
    ]
    
    for source_id, rec_id in test_pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: Movie {source_id} -> Movie {rec_id}")
        
        # Get movie titles
        source_movie = enhanced_rag.base_rag.get_movie_by_id(source_id)
        rec_movie = enhanced_rag.base_rag.get_movie_by_id(rec_id)
        
        if source_movie and rec_movie:
            logger.info(f"Source: {source_movie['title']}")
            logger.info(f"Recommendation: {rec_movie['title']}")
            
            # Generate explanation
            explanation = pipeline.generate_explanation(source_id, rec_id)
            
            logger.info(f"Explanation: {explanation.explanation_text}")
            logger.info(f"Confidence Score: {explanation.confidence_score:.3f}")
            logger.info(f"Explanation Type: {explanation.explanation_type.value}")
            
            if explanation.similarity_score:
                logger.info(f"Similarity Score: {explanation.similarity_score:.3f}")
            else:
                logger.info("Similarity Score: Not available")
            
            logger.info(f"Supporting Facts ({len(explanation.supporting_facts)}):")
            for i, fact in enumerate(explanation.supporting_facts, 1):
                logger.info(f"  {i}. {fact.fact}")
                logger.info(f"     Evidence: {fact.evidence}")
                logger.info(f"     Confidence: {fact.confidence:.2f} | Source: {fact.source} | Type: {fact.fact_type}")
            
            # Generate detailed report
            report = pipeline.generate_explanation_report(source_id, rec_id)
            logger.info(f"Fact Summary: {report['fact_summary']}")
            
        else:
            logger.warning(f"One or both movies not found: {source_id}, {rec_id}")
    
    logger.info(f"\n{'='*70}")
    logger.info("Explanation Pipeline Test Complete!")

def test_explanation_types():
    """Test different explanation types."""
    logger.info("\nTesting Explanation Types")
    logger.info("=" * 40)
    
    # Create pipeline
    enhanced_rag = create_enhanced_rag_system()
    pipeline = create_explanation_pipeline(enhanced_rag, enhanced_rag.knowledge_graph)
    
    # Test different types of movie pairs
    test_cases = [
        (1, 2, "Similarity-based (high similarity)"),
        (1, 3, "Content-based (shared genres)"),
        (3, 5, "Contextual (shared keywords)"),
    ]
    
    for source_id, rec_id, description in test_cases:
        logger.info(f"\n{description}:")
        explanation = pipeline.generate_explanation(source_id, rec_id)
        logger.info(f"  Type: {explanation.explanation_type.value}")
        logger.info(f"  Confidence: {explanation.confidence_score:.3f}")
        logger.info(f"  Facts: {len(explanation.supporting_facts)}")

if __name__ == "__main__":
    test_explanation_pipeline()
    test_explanation_types() 