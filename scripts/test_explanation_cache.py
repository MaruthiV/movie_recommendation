#!/usr/bin/env python3
"""
Test script for explanation pipeline caching.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.enhanced_rag_system import create_enhanced_rag_system
from rag.explanation_pipeline import create_explanation_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_explanation_cache():
    logger.info("Testing Explanation Pipeline Caching")
    logger.info("=" * 60)
    
    enhanced_rag = create_enhanced_rag_system()
    pipeline = create_explanation_pipeline(enhanced_rag, enhanced_rag.knowledge_graph)
    
    test_pair = (1, 2)  # Toy Story -> Jumanji
    n_repeats = 5
    
    # First run (should be a miss)
    logger.info("First run (should be cache miss):")
    explanation = pipeline.generate_explanation(*test_pair)
    logger.info(f"Explanation: {explanation.explanation_text}")
    logger.info(f"Cache stats: {pipeline.cache_stats()}")
    
    # Repeat runs (should be cache hits)
    for i in range(1, n_repeats):
        logger.info(f"\nRepeat run {i+1} (should be cache hit):")
        explanation = pipeline.generate_explanation(*test_pair)
        logger.info(f"Explanation: {explanation.explanation_text}")
        logger.info(f"Cache stats: {pipeline.cache_stats()}")
    
    # Test cache clear
    logger.info("\nClearing cache...")
    pipeline.clear_cache()
    logger.info(f"Cache stats after clear: {pipeline.cache_stats()}")
    
    # Run again (should be a miss)
    logger.info("\nAfter cache clear (should be cache miss):")
    explanation = pipeline.generate_explanation(*test_pair)
    logger.info(f"Explanation: {explanation.explanation_text}")
    logger.info(f"Cache stats: {pipeline.cache_stats()}")
    
    logger.info("\nExplanation Pipeline Cache Test Complete!")

if __name__ == "__main__":
    test_explanation_cache() 