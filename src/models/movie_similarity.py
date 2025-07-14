import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .lightgcn_model import LightGCN
from rag.rag_system import RAGExplanationSystem


class MovieSimilarityCalculator:
    """
    Calculate movie similarities using embeddings from recommendation models.
    
    This class provides various similarity metrics and methods to find similar movies
    based on learned embeddings from models like LightGCN.
    """
    
    def __init__(self, model: LightGCN, device: str = 'cpu', rag_index_dir: Optional[Path] = None):
        """
        Initialize the similarity calculator.
        
        Args:
            model: Trained LightGCN model
            device: Device to run calculations on ('cpu' or 'cuda')
            rag_index_dir: Path to FAISS index directory for explanations (optional)
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Cache for embeddings to avoid recomputation
        self._cached_embeddings = None
        self._cached_edge_index = None
        
        # Initialize RAG explanation system if index directory provided
        self.rag_system = None
        if rag_index_dir and rag_index_dir.exists():
            try:
                self.rag_system = RAGExplanationSystem(rag_index_dir)
                print(f"RAG explanation system loaded from {rag_index_dir}")
            except Exception as e:
                print(f"Warning: Could not load RAG system: {e}")
                self.rag_system = None
    
    def get_movie_embeddings(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get movie embeddings from the model.
        
        Args:
            edge_index: Tensor of shape (2, num_edges) containing user-item interactions
            
        Returns:
            Movie embeddings tensor of shape (num_items, embedding_dim)
        """
        # Check if we can use cached embeddings
        if (self._cached_embeddings is not None and 
            self._cached_edge_index is not None and
            torch.equal(self._cached_edge_index, edge_index)):
            return self._cached_embeddings
        
        # Get embeddings from model
        with torch.no_grad():
            _, item_emb = self.model.forward(edge_index)
        
        # Cache embeddings
        self._cached_embeddings = item_emb
        self._cached_edge_index = edge_index.clone()
        
        return item_emb
    
    def cosine_similarity(self, movie_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate cosine similarity between all pairs of movies.
        
        Args:
            movie_embeddings: Movie embeddings tensor of shape (num_items, embedding_dim)
            
        Returns:
            Similarity matrix of shape (num_items, num_items)
        """
        # Normalize embeddings
        normalized_emb = F.normalize(movie_embeddings, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity_matrix = torch.mm(normalized_emb, normalized_emb.t())
        
        return similarity_matrix
    
    def dot_product_similarity(self, movie_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate dot product similarity between all pairs of movies.
        
        Args:
            movie_embeddings: Movie embeddings tensor of shape (num_items, embedding_dim)
            
        Returns:
            Similarity matrix of shape (num_items, num_items)
        """
        return torch.mm(movie_embeddings, movie_embeddings.t())
    
    def euclidean_distance(self, movie_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate Euclidean distance between all pairs of movies.
        
        Args:
            movie_embeddings: Movie embeddings tensor of shape (num_items, embedding_dim)
            
        Returns:
            Distance matrix of shape (num_items, num_items)
        """
        # Calculate pairwise Euclidean distances
        dist_matrix = torch.cdist(movie_embeddings, movie_embeddings, p=2)
        return dist_matrix
    
    def manhattan_distance(self, movie_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate Manhattan distance between all pairs of movies.
        
        Args:
            movie_embeddings: Movie embeddings tensor of shape (num_items, embedding_dim)
            
        Returns:
            Distance matrix of shape (num_items, num_items)
        """
        # Calculate pairwise Manhattan distances
        dist_matrix = torch.cdist(movie_embeddings, movie_embeddings, p=1)
        return dist_matrix
    
    def get_similar_movies(
        self,
        movie_id: int,
        edge_index: torch.Tensor,
        similarity_metric: str = 'cosine',
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Find the most similar movies to a given movie.
        
        Args:
            movie_id: ID of the target movie
            edge_index: Tensor of shape (2, num_edges) containing user-item interactions
            similarity_metric: Similarity metric to use ('cosine', 'dot_product', 'euclidean', 'manhattan')
            top_k: Number of similar movies to return
            exclude_self: Whether to exclude the movie itself from results
            
        Returns:
            List of tuples (movie_id, similarity_score) sorted by similarity
        """
        movie_embeddings = self.get_movie_embeddings(edge_index)
        
        # Calculate similarity matrix based on metric
        if similarity_metric == 'cosine':
            similarity_matrix = self.cosine_similarity(movie_embeddings)
        elif similarity_metric == 'dot_product':
            similarity_matrix = self.dot_product_similarity(movie_embeddings)
        elif similarity_metric == 'euclidean':
            # Convert distance to similarity (1 / (1 + distance))
            distance_matrix = self.euclidean_distance(movie_embeddings)
            similarity_matrix = 1.0 / (1.0 + distance_matrix)
        elif similarity_metric == 'manhattan':
            # Convert distance to similarity (1 / (1 + distance))
            distance_matrix = self.manhattan_distance(movie_embeddings)
            similarity_matrix = 1.0 / (1.0 + distance_matrix)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
        # Get similarities for the target movie
        movie_similarities = similarity_matrix[movie_id]
        
        # Create list of (movie_id, similarity) pairs
        similarities = [(i, movie_similarities[i].item()) for i in range(len(movie_similarities))]
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Exclude self if requested
        if exclude_self:
            similarities = [(mid, score) for mid, score in similarities if mid != movie_id]
        
        # Return top_k results
        return similarities[:top_k]
    
    def get_similar_movies_with_explanations(
        self,
        movie_id: int,
        edge_index: torch.Tensor,
        similarity_metric: str = 'cosine',
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Dict]:
        """
        Find similar movies with explanations for why they are recommended.
        
        Args:
            movie_id: ID of the target movie
            edge_index: Tensor of shape (2, num_edges) containing user-item interactions
            similarity_metric: Similarity metric to use
            top_k: Number of similar movies to return
            exclude_self: Whether to exclude the movie itself from results
            
        Returns:
            List of dictionaries with movie_id, similarity_score, and explanation
        """
        # Get similar movies
        similar_movies = self.get_similar_movies(
            movie_id, edge_index, similarity_metric, top_k, exclude_self
        )
        
        # Add explanations if RAG system is available
        results = []
        for rec_movie_id, similarity_score in similar_movies:
            result = {
                'movie_id': rec_movie_id,
                'similarity_score': similarity_score,
                'explanation': None
            }
            
            # Generate explanation if RAG system is available
            if self.rag_system:
                try:
                    explanation = self.rag_system.explain_recommendation(movie_id, rec_movie_id)
                    result['explanation'] = explanation
                except Exception as e:
                    result['explanation'] = f"Explanation unavailable: {str(e)}"
            
            results.append(result)
        
        return results
    
    def get_similarity_matrix(
        self,
        edge_index: torch.Tensor,
        similarity_metric: str = 'cosine'
    ) -> torch.Tensor:
        """
        Calculate similarity matrix for all movie pairs.
        
        Args:
            edge_index: Tensor of shape (2, num_edges) containing user-item interactions
            similarity_metric: Similarity metric to use
            
        Returns:
            Similarity matrix of shape (num_items, num_items)
        """
        movie_embeddings = self.get_movie_embeddings(edge_index)
        
        if similarity_metric == 'cosine':
            return self.cosine_similarity(movie_embeddings)
        elif similarity_metric == 'dot_product':
            return self.dot_product_similarity(movie_embeddings)
        elif similarity_metric == 'euclidean':
            distance_matrix = self.euclidean_distance(movie_embeddings)
            return 1.0 / (1.0 + distance_matrix)
        elif similarity_metric == 'manhattan':
            distance_matrix = self.manhattan_distance(movie_embeddings)
            return 1.0 / (1.0 + distance_matrix)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
    
    def get_movie_clusters(
        self,
        edge_index: torch.Tensor,
        similarity_metric: str = 'cosine',
        similarity_threshold: float = 0.8,
        min_cluster_size: int = 2
    ) -> List[List[int]]:
        """
        Find clusters of similar movies based on similarity threshold.
        
        Args:
            edge_index: Tensor of shape (2, num_edges) containing user-item interactions
            similarity_metric: Similarity metric to use
            similarity_threshold: Minimum similarity for movies to be in same cluster
            min_cluster_size: Minimum number of movies in a cluster
            
        Returns:
            List of movie clusters, where each cluster is a list of movie IDs
        """
        similarity_matrix = self.get_similarity_matrix(edge_index, similarity_metric)
        
        # Create adjacency matrix based on similarity threshold
        adjacency_matrix = (similarity_matrix >= similarity_threshold).float()
        
        # Find connected components (clusters)
        clusters = self._find_connected_components(adjacency_matrix)
        
        # Filter clusters by minimum size
        clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_size]
        
        return clusters
    
    def _find_connected_components(self, adjacency_matrix: torch.Tensor) -> List[List[int]]:
        """
        Find connected components in the adjacency matrix using DFS.
        
        Args:
            adjacency_matrix: Binary adjacency matrix
            
        Returns:
            List of connected components
        """
        num_nodes = adjacency_matrix.shape[0]
        visited = [False] * num_nodes
        clusters = []
        
        def dfs(node: int, cluster: List[int]):
            visited[node] = True
            cluster.append(node)
            
            for neighbor in range(num_nodes):
                if (adjacency_matrix[node, neighbor] > 0 and not visited[neighbor]):
                    dfs(neighbor, cluster)
        
        for node in range(num_nodes):
            if not visited[node]:
                cluster = []
                dfs(node, cluster)
                clusters.append(cluster)
        
        return clusters
    
    def get_diverse_recommendations(
        self,
        movie_id: int,
        edge_index: torch.Tensor,
        similarity_metric: str = 'cosine',
        top_k: int = 10,
        diversity_weight: float = 0.3
    ) -> List[Tuple[int, float]]:
        """
        Get diverse recommendations by balancing similarity and diversity.
        
        Args:
            movie_id: ID of the target movie
            edge_index: Tensor of shape (2, num_edges) containing user-item interactions
            similarity_metric: Similarity metric to use
            top_k: Number of recommendations to return
            diversity_weight: Weight for diversity (0 = only similarity, 1 = only diversity)
            
        Returns:
            List of tuples (movie_id, score) sorted by balanced score
        """
        movie_embeddings = self.get_movie_embeddings(edge_index)
        similarity_matrix = self.get_similarity_matrix(edge_index, similarity_metric)
        
        # Get initial similarities
        movie_similarities = similarity_matrix[movie_id]
        
        # Initialize selected movies
        selected_movies = [movie_id]
        recommendations = []
        
        # Greedy selection with diversity penalty
        for _ in range(top_k):
            best_score = -float('inf')
            best_movie = None
            
            for movie_idx in range(len(movie_similarities)):
                if movie_idx in selected_movies:
                    continue
                
                # Similarity score
                similarity_score = movie_similarities[movie_idx].item()
                
                # Diversity penalty (average similarity to already selected movies)
                if len(selected_movies) > 1:
                    diversity_penalty = torch.mean(similarity_matrix[movie_idx, selected_movies[1:]]).item()
                else:
                    diversity_penalty = 0.0
                
                # Combined score
                combined_score = similarity_score - diversity_weight * diversity_penalty
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_movie = movie_idx
            
            if best_movie is not None:
                selected_movies.append(best_movie)
                recommendations.append((best_movie, best_score))
        
        return recommendations
    
    def clear_cache(self):
        """Clear cached embeddings."""
        self._cached_embeddings = None
        self._cached_edge_index = None 