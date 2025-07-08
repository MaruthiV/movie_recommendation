import numpy as np
import torch
from typing import List, Dict, Optional, Tuple

class DiversityConstraint:
    """
    Methods to rerank or filter recommendations to maximize diversity.
    Supports MMR, genre coverage, and embedding-based diversity.
    """
    def __init__(self, movie_metadata: Optional[Dict[int, Dict]] = None):
        """
        Args:
            movie_metadata: Optional dict mapping movie_id to metadata (e.g., genres)
        """
        self.movie_metadata = movie_metadata or {}

    def mmr_rerank(
        self,
        candidate_ids: List[int],
        relevance_scores: List[float],
        embeddings: Optional[np.ndarray] = None,
        lambda_diversity: float = 0.3,
        top_k: int = 20
    ) -> List[int]:
        """
        Maximal Marginal Relevance (MMR) reranking.
        Args:
            candidate_ids: List of candidate movie IDs
            relevance_scores: List of relevance scores (same order as candidate_ids)
            embeddings: Optional [num_candidates, emb_dim] array for diversity
            lambda_diversity: Weight for diversity (0=only relevance, 1=only diversity)
            top_k: Number of items to select
        Returns:
            Reranked list of movie IDs
        """
        if len(candidate_ids) == 0:
            return []
        if embeddings is not None:
            emb_matrix = embeddings
        else:
            emb_matrix = None
        selected = []
        candidate_set = set(candidate_ids)
        scores = np.array(relevance_scores, dtype=float)
        idx_map = {mid: i for i, mid in enumerate(candidate_ids)}
        while len(selected) < min(top_k, len(candidate_ids)):
            if len(selected) == 0 or emb_matrix is None:
                # Pick highest relevance
                idx = np.argmax(scores)
                selected.append(candidate_ids[idx])
                candidate_set.remove(candidate_ids[idx])
                scores[idx] = -np.inf
            else:
                mmr_scores = []
                for mid in candidate_set:
                    i = idx_map[mid]
                    rel = scores[i]
                    # Diversity: min distance to already selected
                    selected_indices = [idx_map[sid] for sid in selected]
                    diversity = np.min(
                        np.linalg.norm(emb_matrix[i] - emb_matrix[selected_indices], axis=1)
                    )
                    mmr = (1 - lambda_diversity) * rel + lambda_diversity * diversity
                    mmr_scores.append((mid, mmr))
                # Pick max MMR
                mmr_scores.sort(key=lambda x: x[1], reverse=True)
                best_mid = mmr_scores[0][0]
                selected.append(best_mid)
                candidate_set.remove(best_mid)
        return selected

    def genre_coverage(
        self,
        candidate_ids: List[int],
        top_k: int = 20
    ) -> List[int]:
        """
        Select a diverse set of movies to maximize genre coverage.
        Args:
            candidate_ids: List of candidate movie IDs
            top_k: Number of items to select
        Returns:
            List of movie IDs maximizing genre coverage
        """
        if not self.movie_metadata:
            return candidate_ids[:top_k]
        selected = []
        covered_genres = set()
        for mid in candidate_ids:
            genres = set(self.movie_metadata.get(mid, {}).get('genres', []))
            if not genres.issubset(covered_genres):
                selected.append(mid)
                covered_genres.update(genres)
            if len(selected) >= top_k:
                break
        # If not enough, pad with remaining
        if len(selected) < top_k:
            for mid in candidate_ids:
                if mid not in selected:
                    selected.append(mid)
                if len(selected) >= top_k:
                    break
        return selected

    def embedding_diversity(
        self,
        candidate_ids: List[int],
        embeddings: np.ndarray,
        top_k: int = 20
    ) -> List[int]:
        """
        Select a set of movies maximizing average pairwise embedding distance.
        Args:
            candidate_ids: List of candidate movie IDs
            embeddings: [num_candidates, emb_dim] array
            top_k: Number of items to select
        Returns:
            List of movie IDs maximizing embedding diversity
        """
        if len(candidate_ids) == 0:
            return []
        if len(candidate_ids) <= top_k:
            return candidate_ids
        selected = [candidate_ids[0]]
        idx_map = {mid: i for i, mid in enumerate(candidate_ids)}
        while len(selected) < top_k:
            best_mid = None
            best_div = -np.inf
            for mid in candidate_ids:
                if mid in selected:
                    continue
                indices = [idx_map[sid] for sid in selected]
                dists = np.linalg.norm(embeddings[idx_map[mid]] - embeddings[indices], axis=1)
                avg_dist = np.mean(dists)
                if avg_dist > best_div:
                    best_div = avg_dist
                    best_mid = mid
            selected.append(best_mid)
        return selected 