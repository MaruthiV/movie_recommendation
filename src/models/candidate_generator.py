import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from lightgcn_model import LightGCN
from movie_similarity import MovieSimilarityCalculator


class CandidateGenerator:
    """
    Generate candidate movies for users using various strategies (LightGCN, popularity, hybrid).
    """
    def __init__(
        self,
        model: LightGCN,
        edge_index: torch.Tensor,
        movie_popularity: Optional[np.ndarray] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            model: Trained LightGCN model
            edge_index: User-item interaction edge index
            movie_popularity: Optional array of movie popularity scores (for popularity-based)
            device: Device to run calculations on
        """
        self.model = model
        self.edge_index = edge_index
        self.device = device
        self.sim_calc = MovieSimilarityCalculator(model, device)
        self.num_items = model.num_items
        self.movie_popularity = movie_popularity

    def generate_for_user_lightgcn(
        self,
        user_id: int,
        top_k: int = 1000,
        exclude_watched: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generate candidates for a user using LightGCN (embedding dot product).
        Args:
            user_id: User ID
            top_k: Number of candidates to return
            exclude_watched: List of item IDs to exclude (e.g., already watched)
        Returns:
            List of candidate movie IDs
        """
        self.model.eval()
        with torch.no_grad():
            user_emb, item_emb = self.model.forward(self.edge_index)
            user_vec = user_emb[user_id]
            scores = torch.matmul(item_emb, user_vec)
            scores = scores.cpu().numpy()
            candidate_ids = np.argsort(-scores)  # Descending order
            if exclude_watched is not None:
                mask = np.isin(candidate_ids, exclude_watched, invert=True)
                candidate_ids = candidate_ids[mask]
            return candidate_ids[:top_k].tolist()

    def generate_for_user_popularity(
        self,
        top_k: int = 1000,
        exclude_watched: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generate candidates using popularity (most popular movies).
        Args:
            top_k: Number of candidates
            exclude_watched: List of item IDs to exclude
        Returns:
            List of candidate movie IDs
        """
        if self.movie_popularity is None:
            # Default: uniform popularity
            popularity = np.arange(self.num_items)
        else:
            popularity = np.argsort(-self.movie_popularity)  # Descending
        if exclude_watched is not None:
            mask = np.isin(popularity, exclude_watched, invert=True)
            popularity = popularity[mask]
        return popularity[:top_k].tolist()

    def generate_for_user_hybrid(
        self,
        user_id: int,
        top_k: int = 1000,
        exclude_watched: Optional[List[int]] = None,
        alpha: float = 0.7
    ) -> List[int]:
        """
        Hybrid candidate generation: weighted sum of LightGCN and popularity.
        Args:
            user_id: User ID
            top_k: Number of candidates
            exclude_watched: List of item IDs to exclude
            alpha: Weight for LightGCN (0=only popularity, 1=only LightGCN)
        Returns:
            List of candidate movie IDs
        """
        lgnc_scores = np.zeros(self.num_items)
        pop_scores = np.zeros(self.num_items)
        # LightGCN scores
        self.model.eval()
        with torch.no_grad():
            user_emb, item_emb = self.model.forward(self.edge_index)
            user_vec = user_emb[user_id]
            lgnc_scores = torch.matmul(item_emb, user_vec).cpu().numpy()
        # Popularity scores
        if self.movie_popularity is not None:
            pop_scores = self.movie_popularity
        # Normalize
        lgnc_scores = (lgnc_scores - lgnc_scores.min()) / (lgnc_scores.ptp() + 1e-8)
        if self.movie_popularity is not None:
            pop_scores = (pop_scores - pop_scores.min()) / (pop_scores.ptp() + 1e-8)
        # Weighted sum
        scores = alpha * lgnc_scores + (1 - alpha) * pop_scores
        candidate_ids = np.argsort(-scores)
        if exclude_watched is not None:
            mask = np.isin(candidate_ids, exclude_watched, invert=True)
            candidate_ids = candidate_ids[mask]
        return candidate_ids[:top_k].tolist()

    def generate_for_users_batch(
        self,
        user_ids: List[int],
        strategy: str = 'lightgcn',
        top_k: int = 1000,
        exclude_watched: Optional[Dict[int, List[int]]] = None,
        alpha: float = 0.7
    ) -> Dict[int, List[int]]:
        """
        Batch candidate generation for multiple users.
        Args:
            user_ids: List of user IDs
            strategy: 'lightgcn', 'popularity', or 'hybrid'
            top_k: Number of candidates per user
            exclude_watched: Dict mapping user_id to list of watched item IDs
            alpha: Hybrid weight
        Returns:
            Dict mapping user_id to list of candidate movie IDs
        """
        results = {}
        for user_id in user_ids:
            exclude = exclude_watched[user_id] if exclude_watched and user_id in exclude_watched else None
            if strategy == 'lightgcn':
                results[user_id] = self.generate_for_user_lightgcn(user_id, top_k, exclude)
            elif strategy == 'popularity':
                results[user_id] = self.generate_for_user_popularity(top_k, exclude)
            elif strategy == 'hybrid':
                results[user_id] = self.generate_for_user_hybrid(user_id, top_k, exclude, alpha)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        return results 