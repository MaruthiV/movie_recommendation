import os
import json
import logging
import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import knowledge graph
try:
    from .knowledge_graph import MovieKnowledgeGraph
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False

class RAGExplanationSystem:
    """Enriches movie data with metadata from various APIs."""
    
    def __init__(self, index_dir: Path, knowledge_graph: Optional[MovieKnowledgeGraph] = None):
        """Initialize the enricher with API keys."""
        self.tmdb_api_key = os.getenv('TMDB_API_KEY')
        self.omdb_api_key = os.getenv('OMDB_API_KEY')
        
        if not self.tmdb_api_key:
            logger.warning("TMDB_API_KEY not found. TMDB enrichment will be skipped.")
        if not self.omdb_api_key:
            logger.warning("OMDB_API_KEY not found. OMDB enrichment will be skipped.")
        
        # API endpoints
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        self.omdb_base_url = "http://www.omdbapi.com/"
        
        # Rate limiting
        self.request_delay = 0.1  # 100ms between requests
        
        # Initialize tokenizer for text processing
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize knowledge graph
        self.knowledge_graph = knowledge_graph
        if knowledge_graph:
            print(f"Knowledge graph integration enabled")
        else:
            print(f"Knowledge graph integration disabled")

        # Load index and metadata
        self.index_dir = index_dir
        self._load_index()

    def _load_index(self):
        index_path = self.index_dir / "movie_index.faiss"
        metadata_path = self.index_dir / "movie_metadata.pkl"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index or metadata not found in {self.index_dir}")
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, 'rb') as f:
            meta = pickle.load(f)
        self.movie_metadata = meta['movie_metadata']
        self.movie_ids = meta['movie_ids']

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict]:
        for m in self.movie_metadata:
            if int(m['movieId']) == int(movie_id):
                return m
        return None

    def get_similar_movies(self, movie_id: int, k: int = 5) -> List[Dict]:
        idx = None
        for i, m in enumerate(self.movie_metadata):
            if int(m['movieId']) == int(movie_id):
                idx = i
                break
        if idx is None:
            return []
        emb = self.index.reconstruct(idx).reshape(1, -1)
        scores, indices = self.index.search(emb.astype('float32'), k+1)
        results = []
        for score, i2 in zip(scores[0][1:], indices[0][1:]):
            if i2 < len(self.movie_metadata):
                m2 = self.movie_metadata[i2].copy()
                m2['similarity_score'] = float(score)
                results.append(m2)
        return results

    def explain_recommendation(self, source_movie_id: int, rec_movie_id: int) -> str:
        src = self.get_movie_by_id(source_movie_id)
        rec = self.get_movie_by_id(rec_movie_id)
        if not src or not rec:
            return "Explanation unavailable: movie not found."
        facts = []
        # Genre overlap
        src_genres = set(src.get('genres', '').split('|'))
        rec_genres = set(rec.get('genres', '').split('|'))
        genre_overlap = src_genres & rec_genres
        if genre_overlap:
            facts.append(f"both are {', '.join(genre_overlap)} movies")
        # Director
        if src.get('director') and rec.get('director') and src['director'] == rec['director']:
            facts.append(f"both directed by {src['director']}")
        # Cast overlap
        src_cast = set(src.get('cast', []))
        rec_cast = set(rec.get('cast', []))
        cast_overlap = src_cast & rec_cast
        if cast_overlap:
            facts.append(f"both feature {', '.join(list(cast_overlap)[:2])}")
        # Keywords
        src_kw = set(src.get('keywords', []))
        rec_kw = set(rec.get('keywords', []))
        kw_overlap = src_kw & rec_kw
        if kw_overlap:
            facts.append(f"share themes like {', '.join(list(kw_overlap)[:2])}")
        # Year proximity
        if src.get('year') and rec.get('year'):
            try:
                y1, y2 = int(src['year']), int(rec['year'])
                if abs(y1 - y2) <= 2:
                    facts.append(f"released around the same time ({y1} & {y2})")
            except Exception:
                pass
        # Fallback
        if not facts:
            facts.append("have similar plot or style")
        return f"Recommended because you liked '{src['title']}' and '{rec['title']}' {', and '.join(facts)}."

    def explain_for_user(self, user_liked_movie_ids: List[int], rec_movie_id: int) -> str:
        # Find the most similar liked movie
        best_explanation = None
        best_score = -1
        for liked_id in user_liked_movie_ids:
            sim_movies = self.get_similar_movies(liked_id, k=10)
            for m in sim_movies:
                if int(m['movieId']) == int(rec_movie_id) and m['similarity_score'] > best_score:
                    best_score = m['similarity_score']
                    best_explanation = self.explain_recommendation(liked_id, rec_movie_id)
        if best_explanation:
            return best_explanation
        # Fallback: just use the first liked movie
        if user_liked_movie_ids:
            return self.explain_recommendation(user_liked_movie_ids[0], rec_movie_id)
        return "Explanation unavailable."

if __name__ == "__main__":
    # Example CLI usage
    import argparse
    parser = argparse.ArgumentParser(description="RAG Explanation System Test")
    parser.add_argument('--index_dir', type=str, default="data/faiss_index", help="Path to FAISS index dir")
    parser.add_argument('--source_movie_id', type=int, help="Source (liked) movieId")
    parser.add_argument('--rec_movie_id', type=int, help="Recommended movieId")
    args = parser.parse_args()
    rag = RAGExplanationSystem(Path(args.index_dir))
    if args.source_movie_id and args.rec_movie_id:
        print(rag.explain_recommendation(args.source_movie_id, args.rec_movie_id))
    else:
        # Demo: show explanations for first 5 movies
        for i in range(5):
            src = rag.movie_metadata[i]['movieId']
            sim = rag.get_similar_movies(src, k=1)
            if sim:
                print(rag.explain_recommendation(src, sim[0]['movieId'])) 