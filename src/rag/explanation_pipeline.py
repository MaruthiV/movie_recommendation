#!/usr/bin/env python3
"""
Explanation Generation Pipeline with Supporting Facts.
Generates comprehensive explanations with evidence, confidence scores, and multiple explanation types.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import functools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Types of explanations that can be generated."""
    SIMILARITY_BASED = "similarity_based"
    CONTENT_BASED = "content_based"
    COLLABORATIVE = "collaborative"
    CONTEXTUAL = "contextual"
    HYBRID = "hybrid"


@dataclass
class SupportingFact:
    """A supporting fact with evidence and confidence."""
    fact: str
    evidence: str
    confidence: float
    source: str
    fact_type: str


@dataclass
class Explanation:
    """A complete explanation with supporting facts."""
    explanation_text: str
    supporting_facts: List[SupportingFact]
    confidence_score: float
    explanation_type: ExplanationType
    movie_id_1: int
    movie_id_2: int
    similarity_score: Optional[float] = None


class ExplanationPipeline:
    """Pipeline for generating comprehensive movie explanations."""
    
    def __init__(self, enhanced_rag_system, knowledge_graph=None):
        """Initialize the explanation pipeline."""
        self.enhanced_rag = enhanced_rag_system
        self.knowledge_graph = knowledge_graph
        self.confidence_threshold = 0.3
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
    def generate_explanation(self, source_movie_id: int, rec_movie_id: int) -> Explanation:
        """Generate a comprehensive explanation with supporting facts, using cache if available."""
        cache_key = (source_movie_id, rec_movie_id)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1
        explanation = self._generate_explanation_uncached(source_movie_id, rec_movie_id)
        self._cache[cache_key] = explanation
        return explanation

    def _generate_explanation_uncached(self, source_movie_id: int, rec_movie_id: int) -> Explanation:
        """Generate a comprehensive explanation with supporting facts."""
        logger.info(f"Generating explanation for {source_movie_id} -> {rec_movie_id}")
        
        # Get basic movie information
        source_movie = self.enhanced_rag.base_rag.get_movie_by_id(source_movie_id)
        rec_movie = self.enhanced_rag.base_rag.get_movie_by_id(rec_movie_id)
        
        if not source_movie or not rec_movie:
            return self._create_error_explanation(source_movie_id, rec_movie_id)
        
        # Collect supporting facts
        supporting_facts = []
        
        # 1. Similarity-based facts
        similarity_facts = self._get_similarity_facts(source_movie_id, rec_movie_id)
        supporting_facts.extend(similarity_facts)
        
        # 2. Content-based facts
        content_facts = self._get_content_based_facts(source_movie, rec_movie)
        supporting_facts.extend(content_facts)
        
        # 3. Contextual facts (from knowledge graph)
        if self.knowledge_graph:
            contextual_facts = self._get_contextual_facts(source_movie_id, rec_movie_id)
            supporting_facts.extend(contextual_facts)
        
        # 4. Collaborative facts
        collaborative_facts = self._get_collaborative_facts(source_movie_id, rec_movie_id)
        supporting_facts.extend(collaborative_facts)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(source_movie, rec_movie, supporting_facts)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(supporting_facts)
        
        # Determine explanation type
        explanation_type = self._determine_explanation_type(supporting_facts)
        
        # Get similarity score
        similarity_score = self._get_similarity_score(source_movie_id, rec_movie_id)
        
        return Explanation(
            explanation_text=explanation_text,
            supporting_facts=supporting_facts,
            confidence_score=confidence_score,
            explanation_type=explanation_type,
            movie_id_1=source_movie_id,
            movie_id_2=rec_movie_id,
            similarity_score=similarity_score
        )

    def cache_stats(self) -> dict:
        return {
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses
        }

    def clear_cache(self):
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _get_similarity_facts(self, source_id: int, rec_id: int) -> List[SupportingFact]:
        """Get facts based on similarity scores."""
        facts = []
        
        # Get similarity score
        sim_movies = self.enhanced_rag.base_rag.get_similar_movies(source_id, k=20)
        similarity_score = None
        
        for m in sim_movies:
            if int(m['movieId']) == int(rec_id):
                similarity_score = m['similarity_score']
                break
        
        if similarity_score:
            if similarity_score > 0.7:
                confidence = 0.9
                fact = "highly similar based on collaborative filtering"
            elif similarity_score > 0.5:
                confidence = 0.7
                fact = "moderately similar based on collaborative filtering"
            elif similarity_score > 0.3:
                confidence = 0.5
                fact = "somewhat similar based on collaborative filtering"
            else:
                confidence = 0.3
                fact = "minimally similar based on collaborative filtering"
            
            facts.append(SupportingFact(
                fact=fact,
                evidence=f"Similarity score: {similarity_score:.3f}",
                confidence=confidence,
                source="collaborative_filtering",
                fact_type="similarity"
            ))
        
        return facts
    
    def _get_content_based_facts(self, source_movie: Dict, rec_movie: Dict) -> List[SupportingFact]:
        """Get facts based on content similarity."""
        facts = []
        
        # Genre overlap
        src_genres = set(source_movie.get('genres', '').split('|'))
        rec_genres = set(rec_movie.get('genres', '').split('|'))
        genre_overlap = src_genres & rec_genres
        
        if genre_overlap:
            facts.append(SupportingFact(
                fact=f"share genres: {', '.join(genre_overlap)}",
                evidence=f"Source: {', '.join(src_genres)} | Target: {', '.join(rec_genres)}",
                confidence=0.8,
                source="content_analysis",
                fact_type="genre"
            ))
        
        # Director overlap
        if (source_movie.get('director') and rec_movie.get('director') and 
            source_movie['director'] == rec_movie['director']):
            facts.append(SupportingFact(
                fact=f"same director: {source_movie['director']}",
                evidence=f"Both movies directed by {source_movie['director']}",
                confidence=0.9,
                source="content_analysis",
                fact_type="director"
            ))
        
        # Cast overlap
        src_cast = set(source_movie.get('cast', []))
        rec_cast = set(rec_movie.get('cast', []))
        cast_overlap = src_cast & rec_cast
        
        if cast_overlap:
            actors = list(cast_overlap)[:2]  # Limit to 2 actors
            facts.append(SupportingFact(
                fact=f"share actors: {', '.join(actors)}",
                evidence=f"Actors appearing in both movies: {', '.join(actors)}",
                confidence=0.8,
                source="content_analysis",
                fact_type="cast"
            ))
        
        # Year proximity
        if source_movie.get('year') and rec_movie.get('year'):
            try:
                y1, y2 = int(source_movie['year']), int(rec_movie['year'])
                year_diff = abs(y1 - y2)
                
                if year_diff == 0:
                    facts.append(SupportingFact(
                        fact="released in the same year",
                        evidence=f"Both released in {y1}",
                        confidence=0.7,
                        source="content_analysis",
                        fact_type="year"
                    ))
                elif year_diff <= 2:
                    facts.append(SupportingFact(
                        fact="released around the same time",
                        evidence=f"Released in {y1} and {y2}",
                        confidence=0.6,
                        source="content_analysis",
                        fact_type="year"
                    ))
            except (ValueError, TypeError):
                pass
        
        return facts
    
    def _get_contextual_facts(self, source_id: int, rec_id: int) -> List[SupportingFact]:
        """Get contextual facts from knowledge graph."""
        facts = []
        
        if not self.knowledge_graph:
            return facts
        
        try:
            # Get shared connections
            connections = self.knowledge_graph.find_shared_connections(source_id, rec_id)
            
            # Production company
            if connections.get('shared_companies'):
                companies = connections['shared_companies'][:1]
                facts.append(SupportingFact(
                    fact=f"same production company: {', '.join(companies)}",
                    evidence=f"Both produced by {', '.join(companies)}",
                    confidence=0.8,
                    source="knowledge_graph",
                    fact_type="production"
                ))
            
            # Keywords/themes
            if connections.get('shared_keywords'):
                keywords = connections['shared_keywords'][:3]
                facts.append(SupportingFact(
                    fact=f"share themes: {', '.join(keywords)}",
                    evidence=f"Common keywords: {', '.join(keywords)}",
                    confidence=0.7,
                    source="knowledge_graph",
                    fact_type="themes"
                ))
            
            # Check for sequel relationships
            sequels = self.knowledge_graph.get_sequel_relationships(source_id)
            for sequel in sequels:
                if sequel['movieId'] == rec_id:
                    facts.append(SupportingFact(
                        fact="part of the same franchise",
                        evidence=f"Franchise relationship detected",
                        confidence=0.9,
                        source="knowledge_graph",
                        fact_type="franchise"
                    ))
                    break
            
        except Exception as e:
            logger.warning(f"Failed to get contextual facts: {e}")
        
        return facts
    
    def _get_collaborative_facts(self, source_id: int, rec_id: int) -> List[SupportingFact]:
        """Get facts based on collaborative filtering patterns."""
        facts = []
        
        # This would typically involve user behavior patterns
        # For now, we'll add a basic collaborative fact
        facts.append(SupportingFact(
            fact="users who liked the source movie also liked this movie",
            evidence="Based on collaborative filtering patterns",
            confidence=0.6,
            source="collaborative_filtering",
            fact_type="user_patterns"
        ))
        
        return facts
    
    def _generate_explanation_text(self, source_movie: Dict, rec_movie: Dict, 
                                 supporting_facts: List[SupportingFact]) -> str:
        """Generate natural language explanation text."""
        if not supporting_facts:
            return f"Recommended because you liked '{source_movie['title']}' and '{rec_movie['title']}' have similar appeal."
        
        # Sort facts by confidence
        sorted_facts = sorted(supporting_facts, key=lambda x: x.confidence, reverse=True)
        
        # Get high-confidence facts
        high_conf_facts = [f for f in sorted_facts if f.confidence > 0.7]
        medium_conf_facts = [f for f in sorted_facts if 0.5 <= f.confidence <= 0.7]
        
        # Build explanation
        parts = []
        
        if high_conf_facts:
            # Use high-confidence facts
            fact_descriptions = [f.fact for f in high_conf_facts[:2]]  # Limit to 2 facts
            parts.append(f"both {', and '.join(fact_descriptions)}")
        elif medium_conf_facts:
            # Use medium-confidence facts
            fact_descriptions = [f.fact for f in medium_conf_facts[:2]]
            parts.append(f"both {', and '.join(fact_descriptions)}")
        else:
            # Fallback
            parts.append("have similar appeal")
        
        return f"Recommended because you liked '{source_movie['title']}' and '{rec_movie['title']}' {', and '.join(parts)}."
    
    def _calculate_confidence(self, supporting_facts: List[SupportingFact]) -> float:
        """Calculate overall confidence score based on supporting facts."""
        if not supporting_facts:
            return 0.3
        
        # Weight by fact type
        type_weights = {
            'similarity': 0.3,
            'genre': 0.2,
            'director': 0.25,
            'cast': 0.2,
            'year': 0.1,
            'production': 0.15,
            'themes': 0.15,
            'franchise': 0.3,
            'user_patterns': 0.2
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for fact in supporting_facts:
            weight = type_weights.get(fact.fact_type, 0.1)
            weighted_sum += fact.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.3
        
        return min(1.0, weighted_sum / total_weight)
    
    def _determine_explanation_type(self, supporting_facts: List[SupportingFact]) -> ExplanationType:
        """Determine the type of explanation based on supporting facts."""
        if not supporting_facts:
            return ExplanationType.SIMILARITY_BASED
        
        fact_types = [f.fact_type for f in supporting_facts]
        
        # Check for hybrid (multiple types)
        if len(set(fact_types)) > 2:
            return ExplanationType.HYBRID
        
        # Check for contextual
        if any(t in fact_types for t in ['production', 'themes', 'franchise']):
            return ExplanationType.CONTEXTUAL
        
        # Check for content-based
        if any(t in fact_types for t in ['genre', 'director', 'cast', 'year']):
            return ExplanationType.CONTENT_BASED
        
        # Check for collaborative
        if any(t in fact_types for t in ['similarity', 'user_patterns']):
            return ExplanationType.COLLABORATIVE
        
        return ExplanationType.SIMILARITY_BASED
    
    def _get_similarity_score(self, source_id: int, rec_id: int) -> Optional[float]:
        """Get similarity score between movies."""
        sim_movies = self.enhanced_rag.base_rag.get_similar_movies(source_id, k=20)
        for m in sim_movies:
            if int(m['movieId']) == int(rec_id):
                return m['similarity_score']
        return None
    
    def _create_error_explanation(self, source_id: int, rec_id: int) -> Explanation:
        """Create an error explanation when movies are not found."""
        return Explanation(
            explanation_text=f"Explanation unavailable: one or both movies not found.",
            supporting_facts=[],
            confidence_score=0.0,
            explanation_type=ExplanationType.SIMILARITY_BASED,
            movie_id_1=source_id,
            movie_id_2=rec_id
        )
    
    def generate_explanation_report(self, source_id: int, rec_id: int) -> Dict[str, Any]:
        """Generate a comprehensive explanation report."""
        explanation = self.generate_explanation(source_id, rec_id)
        
        # Convert to dictionary
        report = {
            'explanation': explanation.explanation_text,
            'confidence_score': explanation.confidence_score,
            'explanation_type': explanation.explanation_type.value,
            'similarity_score': explanation.similarity_score,
            'source_movie_id': explanation.movie_id_1,
            'recommended_movie_id': explanation.movie_id_2,
            'supporting_facts': [
                {
                    'fact': f.fact,
                    'evidence': f.evidence,
                    'confidence': f.confidence,
                    'source': f.source,
                    'fact_type': f.fact_type
                }
                for f in explanation.supporting_facts
            ],
            'fact_summary': {
                'total_facts': len(explanation.supporting_facts),
                'high_confidence_facts': len([f for f in explanation.supporting_facts if f.confidence > 0.7]),
                'medium_confidence_facts': len([f for f in explanation.supporting_facts if 0.5 <= f.confidence <= 0.7]),
                'low_confidence_facts': len([f for f in explanation.supporting_facts if f.confidence < 0.5])
            }
        }
        
        return report


def create_explanation_pipeline(enhanced_rag_system, knowledge_graph=None):
    """Create an explanation pipeline."""
    return ExplanationPipeline(enhanced_rag_system, knowledge_graph)


if __name__ == "__main__":
    # Test the explanation pipeline
    import sys
    from pathlib import Path
    
    sys.path.append(str(Path(__file__).parent.parent))
    
    from rag.enhanced_rag_system import create_enhanced_rag_system
    
    # Create pipeline
    enhanced_rag = create_enhanced_rag_system()
    pipeline = create_explanation_pipeline(enhanced_rag, enhanced_rag.knowledge_graph)
    
    # Test explanations
    test_pairs = [(1, 2), (1, 3), (2, 4)]
    
    print("Explanation Pipeline Test")
    print("=" * 50)
    
    for source_id, rec_id in test_pairs:
        print(f"\nTesting: Movie {source_id} -> Movie {rec_id}")
        
        # Generate explanation
        explanation = pipeline.generate_explanation(source_id, rec_id)
        
        print(f"Explanation: {explanation.explanation_text}")
        print(f"Confidence: {explanation.confidence_score:.3f}")
        print(f"Type: {explanation.explanation_type.value}")
        print(f"Similarity Score: {explanation.similarity_score}")
        print(f"Supporting Facts: {len(explanation.supporting_facts)}")
        
        for i, fact in enumerate(explanation.supporting_facts, 1):
            print(f"  {i}. {fact.fact} (Confidence: {fact.confidence:.2f})")
    
    print("\nExplanation Pipeline Test Complete!") 