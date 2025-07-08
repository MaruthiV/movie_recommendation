import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from lightgcn_model import LightGCN
from candidate_generator import CandidateGenerator


class ColdStartHandler:
    """
    Handle cold-start scenarios for new users and new content.
    """
    def __init__(
        self,
        model: LightGCN,
        edge_index: torch.Tensor,
        movie_features: Optional[Dict] = None,
        movie_popularity: Optional[np.ndarray] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            model: Trained LightGCN model
            edge_index: User-item interaction edge index
            movie_features: Dict mapping movie_id to features (genres, actors, etc.)
            movie_popularity: Array of movie popularity scores
            device: Device to run calculations on
        """
        self.model = model
        self.edge_index = edge_index
        self.device = device
        self.candidate_generator = CandidateGenerator(model, edge_index, movie_popularity, device)
        self.movie_features = movie_features or {}
        self.movie_popularity = movie_popularity
        self.num_items = model.num_items

    def recommend_for_new_user(
        self,
        user_demographics: Optional[Dict] = None,
        initial_preferences: Optional[List[int]] = None,
        strategy: str = 'hybrid',
        top_k: int = 20
    ) -> List[int]:
        """
        Generate recommendations for a new user.
        
        Args:
            user_demographics: Dict with user demographics (age, gender, location, etc.)
            initial_preferences: List of movie IDs the user has initially rated/liked
            strategy: 'popularity', 'content_based', 'hybrid'
            top_k: Number of recommendations to return
            
        Returns:
            List of recommended movie IDs
        """
        if strategy == 'popularity':
            return self._popularity_based_new_user(top_k)
        elif strategy == 'content_based':
            return self._content_based_new_user(initial_preferences, top_k)
        elif strategy == 'hybrid':
            return self._hybrid_new_user(user_demographics, initial_preferences, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def recommend_for_new_content(
        self,
        movie_id: int,
        movie_features: Optional[np.ndarray] = None,
        strategy: str = 'content_based',
        top_k: int = 20
    ) -> List[int]:
        """
        Find similar movies for new content.
        
        Args:
            movie_id: ID of the new movie
            movie_features: Features of the new movie (if not in self.movie_features)
            strategy: 'content_based', 'popularity', 'hybrid'
            top_k: Number of similar movies to return
            
        Returns:
            List of similar movie IDs
        """
        if strategy == 'content_based':
            return self._content_based_new_content(movie_id, movie_features, top_k)
        elif strategy == 'popularity':
            return self._popularity_based_new_content(top_k)
        elif strategy == 'hybrid':
            return self._hybrid_new_content(movie_id, movie_features, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _popularity_based_new_user(self, top_k: int) -> List[int]:
        """Recommend popular movies for new users."""
        if self.movie_popularity is not None:
            popular_movies = np.argsort(-self.movie_popularity)
            return popular_movies[:top_k].tolist()
        else:
            # Fallback: random selection
            popular_movies = np.random.choice(self.num_items, top_k, replace=False)
            return sorted(popular_movies.tolist())

    def _content_based_new_user(
        self,
        initial_preferences: Optional[List[int]],
        top_k: int
    ) -> List[int]:
        """Use content-based filtering for new users with initial preferences."""
        if not initial_preferences or not self.movie_features:
            return self._popularity_based_new_user(top_k)
        
        # Get features of preferred movies
        preferred_features = []
        for movie_id in initial_preferences:
            if movie_id in self.movie_features:
                preferred_features.append(self.movie_features[movie_id])
        
        if not preferred_features:
            return self._popularity_based_new_user(top_k)
        
        # Calculate average feature vector
        avg_features = np.mean(preferred_features, axis=0)
        
        # Find movies with similar features
        similarities = []
        for movie_id in range(self.num_items):
            if movie_id in self.movie_features:
                movie_feat = self.movie_features[movie_id]
                similarity = np.dot(avg_features, movie_feat) / (
                    np.linalg.norm(avg_features) * np.linalg.norm(movie_feat) + 1e-8
                )
                similarities.append((movie_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in similarities[:top_k]]

    def _hybrid_new_user(
        self,
        user_demographics: Optional[Dict],
        initial_preferences: Optional[List[int]],
        top_k: int
    ) -> List[int]:
        """Hybrid approach combining multiple strategies."""
        # Get recommendations from different strategies
        pop_recs = self._popularity_based_new_user(top_k)
        content_recs = self._content_based_new_user(initial_preferences, top_k)
        
        # Combine and deduplicate
        combined = list(set(pop_recs + content_recs))
        
        # If we have demographics, we could add demographic-based filtering here
        if user_demographics:
            # Placeholder for demographic-based filtering
            pass
        
        # Ensure we return exactly top_k items, padding with more popular items if needed
        if len(combined) < top_k:
            additional = [i for i in pop_recs if i not in combined]
            combined.extend(additional[:top_k - len(combined)])
        
        return combined[:top_k]

    def _content_based_new_content(
        self,
        movie_id: int,
        movie_features: Optional[np.ndarray],
        top_k: int
    ) -> List[int]:
        """Find similar movies based on content features."""
        target_features = movie_features if movie_features is not None else self.movie_features.get(movie_id)
        
        if target_features is None or not self.movie_features:
            return self._popularity_based_new_user(top_k)
        
        # Calculate similarities with all other movies
        similarities = []
        for other_id in range(self.num_items):
            if other_id != movie_id and other_id in self.movie_features:
                other_features = self.movie_features[other_id]
                similarity = np.dot(target_features, other_features) / (
                    np.linalg.norm(target_features) * np.linalg.norm(other_features) + 1e-8
                )
                similarities.append((other_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in similarities[:top_k]]

    def _popularity_based_new_content(self, top_k: int) -> List[int]:
        """Recommend popular movies for new content (fallback)."""
        return self._popularity_based_new_user(top_k)

    def _hybrid_new_content(
        self,
        movie_id: int,
        movie_features: Optional[np.ndarray],
        top_k: int
    ) -> List[int]:
        """Hybrid approach for new content."""
        content_recs = self._content_based_new_content(movie_id, movie_features, top_k)
        pop_recs = self._popularity_based_new_user(top_k)
        
        # Combine and deduplicate
        combined = list(dict.fromkeys(content_recs + pop_recs))
        # Pad with more popular items if needed
        if len(combined) < top_k:
            additional = [i for i in pop_recs if i not in combined]
            combined.extend(additional[:top_k - len(combined)])
        return combined[:top_k]

    def get_user_embedding_for_new_user(
        self,
        initial_preferences: Optional[List[int]] = None,
        user_demographics: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Generate a pseudo-embedding for a new user.
        This can be used to integrate with the existing LightGCN model.
        
        Args:
            initial_preferences: List of movie IDs the user has rated
            user_demographics: User demographic information
            
        Returns:
            Pseudo user embedding tensor
        """
        # Get average item embedding of preferred movies
        if initial_preferences:
            self.model.eval()
            with torch.no_grad():
                _, item_emb = self.model.forward(self.edge_index)
                preferred_embeddings = []
                for movie_id in initial_preferences:
                    if movie_id < self.num_items:
                        preferred_embeddings.append(item_emb[movie_id])
                
                if preferred_embeddings:
                    avg_embedding = torch.stack(preferred_embeddings).mean(dim=0)
                    return avg_embedding
        
        # Fallback: return zero embedding or random embedding
        return torch.zeros(self.model.embedding_dim, device=self.device)

    def recommend_with_pseudo_embedding(
        self,
        pseudo_embedding: torch.Tensor,
        top_k: int = 20,
        exclude_watched: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generate recommendations using a pseudo user embedding.
        
        Args:
            pseudo_embedding: Pseudo user embedding
            top_k: Number of recommendations
            exclude_watched: List of movies to exclude
            
        Returns:
            List of recommended movie IDs
        """
        self.model.eval()
        with torch.no_grad():
            _, item_emb = self.model.forward(self.edge_index)
            scores = torch.matmul(item_emb, pseudo_embedding)
            scores = scores.cpu().numpy()
            
            candidate_ids = np.argsort(-scores)
            
            if exclude_watched:
                mask = np.isin(candidate_ids, exclude_watched, invert=True)
                candidate_ids = candidate_ids[mask]
            
            return candidate_ids[:top_k].tolist() 