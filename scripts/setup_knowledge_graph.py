#!/usr/bin/env python3
"""
Setup script for Neo4j knowledge graph with enriched movie data.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.knowledge_graph import load_knowledge_graph_from_enriched_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to set up the knowledge graph."""
    logger.info("Setting up Neo4j knowledge graph...")
    
    # Load knowledge graph
    kg = load_knowledge_graph_from_enriched_data()
    
    if kg:
        logger.info("Knowledge graph setup complete!")
        
        # Test the knowledge graph
        logger.info("Testing knowledge graph functionality...")
        
        # Test shared connections between Toy Story and Jumanji
        connections = kg.find_shared_connections(1, 2)
        logger.info(f"Shared connections between Toy Story and Jumanji: {connections}")
        
        # Test contextual facts
        facts = kg.get_contextual_facts(1, 2)
        logger.info(f"Contextual facts: {facts}")
        
        # Test actor filmography
        if connections.get('shared_actors'):
            actor = connections['shared_actors'][0]
            filmography = kg.get_actor_filmography(actor, limit=5)
            logger.info(f"Filmography for {actor}: {[f['title'] for f in filmography]}")
        
        kg.close()
        logger.info("Knowledge graph test complete!")
    else:
        logger.error("Failed to set up knowledge graph")

if __name__ == "__main__":
    main() 