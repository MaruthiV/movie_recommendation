import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch

from lightgcn_model import LightGCN
from movie_similarity import MovieSimilarityCalculator


class TestMovieSimilarityCalculator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.num_users = 50
        self.num_items = 100
        self.embedding_dim = 32
        self.device = 'cpu'
        
        # Create a mock LightGCN model
        self.model = LightGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_layers=2,
            device=self.device
        )
        
        # Create sample edge index
        self.edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3],  # users
            [0, 1, 1, 2, 2, 3, 3, 4]   # items (offset by num_users)
        ], dtype=torch.long)
        self.edge_index[1] += self.num_users
        
        # Create similarity calculator
        self.calculator = MovieSimilarityCalculator(self.model, self.device)

    def test_initialization(self):
        """Test calculator initialization."""
        self.assertEqual(self.calculator.device, self.device)
        self.assertEqual(self.calculator.model, self.model)
        self.assertIsNone(self.calculator._cached_embeddings)
        self.assertIsNone(self.calculator._cached_edge_index)

    def test_get_movie_embeddings(self):
        """Test getting movie embeddings."""
        embeddings = self.calculator.get_movie_embeddings(self.edge_index)
        
        # Check shape
        self.assertEqual(embeddings.shape, (self.num_items, self.embedding_dim))
        
        # Check that embeddings are not all zeros
        self.assertFalse(torch.allclose(embeddings, torch.zeros_like(embeddings)))
        
        # Check caching
        self.assertIsNotNone(self.calculator._cached_embeddings)
        self.assertIsNotNone(self.calculator._cached_edge_index)

    def test_embedding_caching(self):
        """Test that embeddings are cached correctly."""
        # First call should compute embeddings
        embeddings1 = self.calculator.get_movie_embeddings(self.edge_index)
        
        # Second call with same edge_index should use cache
        embeddings2 = self.calculator.get_movie_embeddings(self.edge_index)
        
        # Should be the same
        self.assertTrue(torch.equal(embeddings1, embeddings2))
        
        # Different edge_index should recompute
        different_edge_index = torch.tensor([[0], [self.num_users]], dtype=torch.long)
        embeddings3 = self.calculator.get_movie_embeddings(different_edge_index)
        
        # Should be different
        self.assertFalse(torch.equal(embeddings1, embeddings3))

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        movie_embeddings = self.calculator.get_movie_embeddings(self.edge_index)
        similarity_matrix = self.calculator.cosine_similarity(movie_embeddings)
        
        # Check shape
        self.assertEqual(similarity_matrix.shape, (self.num_items, self.num_items))
        
        # Check diagonal (self-similarity should be 1.0)
        diagonal = torch.diag(similarity_matrix)
        self.assertTrue(torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-6))
        
        # Check symmetry
        self.assertTrue(torch.allclose(similarity_matrix, similarity_matrix.t()))

    def test_dot_product_similarity(self):
        """Test dot product similarity calculation."""
        movie_embeddings = self.calculator.get_movie_embeddings(self.edge_index)
        similarity_matrix = self.calculator.dot_product_similarity(movie_embeddings)
        
        # Check shape
        self.assertEqual(similarity_matrix.shape, (self.num_items, self.num_items))
        
        # Check symmetry
        self.assertTrue(torch.allclose(similarity_matrix, similarity_matrix.t()))

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        movie_embeddings = self.calculator.get_movie_embeddings(self.edge_index)
        distance_matrix = self.calculator.euclidean_distance(movie_embeddings)
        
        # Check shape
        self.assertEqual(distance_matrix.shape, (self.num_items, self.num_items))
        
        # Check diagonal (self-distance should be 0.0)
        diagonal = torch.diag(distance_matrix)
        # Use a more lenient tolerance for numerical precision
        self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal), atol=2e-4))
        
        # Check symmetry
        self.assertTrue(torch.allclose(distance_matrix, distance_matrix.t()))

    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        movie_embeddings = self.calculator.get_movie_embeddings(self.edge_index)
        distance_matrix = self.calculator.manhattan_distance(movie_embeddings)
        
        # Check shape
        self.assertEqual(distance_matrix.shape, (self.num_items, self.num_items))
        
        # Check diagonal (self-distance should be 0.0)
        diagonal = torch.diag(distance_matrix)
        # Use a more lenient tolerance for numerical precision
        self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-4))
        
        # Check symmetry
        self.assertTrue(torch.allclose(distance_matrix, distance_matrix.t()))

    def test_get_similar_movies_cosine(self):
        """Test getting similar movies with cosine similarity."""
        similar_movies = self.calculator.get_similar_movies(
            movie_id=0,
            edge_index=self.edge_index,
            similarity_metric='cosine',
            top_k=5
        )
        
        # Check number of results
        self.assertLessEqual(len(similar_movies), 5)
        
        # Check format
        for movie_id, score in similar_movies:
            self.assertIsInstance(movie_id, int)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)
        
        # Check sorting (descending)
        scores = [score for _, score in similar_movies]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_get_similar_movies_dot_product(self):
        """Test getting similar movies with dot product similarity."""
        similar_movies = self.calculator.get_similar_movies(
            movie_id=0,
            edge_index=self.edge_index,
            similarity_metric='dot_product',
            top_k=5
        )
        
        # Check number of results
        self.assertLessEqual(len(similar_movies), 5)
        
        # Check format
        for movie_id, score in similar_movies:
            self.assertIsInstance(movie_id, int)
            self.assertIsInstance(score, float)

    def test_get_similar_movies_euclidean(self):
        """Test getting similar movies with Euclidean distance."""
        similar_movies = self.calculator.get_similar_movies(
            movie_id=0,
            edge_index=self.edge_index,
            similarity_metric='euclidean',
            top_k=5
        )
        
        # Check number of results
        self.assertLessEqual(len(similar_movies), 5)
        
        # Check format
        for movie_id, score in similar_movies:
            self.assertIsInstance(movie_id, int)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_get_similar_movies_exclude_self(self):
        """Test that exclude_self parameter works correctly."""
        # With exclude_self=True
        similar_movies_excluded = self.calculator.get_similar_movies(
            movie_id=0,
            edge_index=self.edge_index,
            similarity_metric='cosine',
            top_k=10,
            exclude_self=True
        )
        
        # With exclude_self=False
        similar_movies_included = self.calculator.get_similar_movies(
            movie_id=0,
            edge_index=self.edge_index,
            similarity_metric='cosine',
            top_k=10,
            exclude_self=False
        )
        
        # Check that movie_id=0 is in the included results
        movie_ids_included = [mid for mid, _ in similar_movies_included]
        self.assertIn(0, movie_ids_included)
        
        # Check that movie_id=0 is NOT in the excluded results
        movie_ids_excluded = [mid for mid, _ in similar_movies_excluded]
        self.assertNotIn(0, movie_ids_excluded)

    def test_get_similarity_matrix(self):
        """Test getting similarity matrix."""
        similarity_matrix = self.calculator.get_similarity_matrix(
            self.edge_index,
            similarity_metric='cosine'
        )
        
        # Check shape
        self.assertEqual(similarity_matrix.shape, (self.num_items, self.num_items))
        
        # Check that it's a valid similarity matrix
        self.assertTrue(torch.allclose(similarity_matrix, similarity_matrix.t()))

    def test_invalid_similarity_metric(self):
        """Test that invalid similarity metric raises error."""
        with self.assertRaises(ValueError):
            self.calculator.get_similar_movies(
                movie_id=0,
                edge_index=self.edge_index,
                similarity_metric='invalid_metric'
            )

    def test_get_movie_clusters(self):
        """Test movie clustering functionality."""
        clusters = self.calculator.get_movie_clusters(
            edge_index=self.edge_index,
            similarity_metric='cosine',
            similarity_threshold=0.9,
            min_cluster_size=2
        )
        
        # Check that clusters is a list
        self.assertIsInstance(clusters, list)
        
        # Check each cluster
        for cluster in clusters:
            self.assertIsInstance(cluster, list)
            self.assertGreaterEqual(len(cluster), 2)  # min_cluster_size
            
            # Check that cluster contains valid movie IDs
            for movie_id in cluster:
                self.assertIsInstance(movie_id, int)
                self.assertGreaterEqual(movie_id, 0)
                self.assertLess(movie_id, self.num_items)

    def test_get_diverse_recommendations(self):
        """Test diverse recommendations functionality."""
        diverse_recs = self.calculator.get_diverse_recommendations(
            movie_id=0,
            edge_index=self.edge_index,
            similarity_metric='cosine',
            top_k=5,
            diversity_weight=0.3
        )
        
        # Check number of results
        self.assertLessEqual(len(diverse_recs), 5)
        
        # Check format
        for movie_id, score in diverse_recs:
            self.assertIsInstance(movie_id, int)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(movie_id, 0)
            self.assertLess(movie_id, self.num_items)

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Populate cache
        self.calculator.get_movie_embeddings(self.edge_index)
        self.assertIsNotNone(self.calculator._cached_embeddings)
        self.assertIsNotNone(self.calculator._cached_edge_index)
        
        # Clear cache
        self.calculator.clear_cache()
        self.assertIsNone(self.calculator._cached_embeddings)
        self.assertIsNone(self.calculator._cached_edge_index)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with single movie
        single_edge_index = torch.tensor([[0], [self.num_users]], dtype=torch.long)
        similar_movies = self.calculator.get_similar_movies(
            movie_id=0,
            edge_index=single_edge_index,
            top_k=1
        )
        
        # Should return empty list or single item
        self.assertLessEqual(len(similar_movies), 1)
        
        # Test with empty edge index
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        similar_movies = self.calculator.get_similar_movies(
            movie_id=0,
            edge_index=empty_edge_index,
            top_k=5
        )
        
        # Should handle gracefully
        self.assertIsInstance(similar_movies, list)

    def test_different_top_k_values(self):
        """Test with different top_k values."""
        for top_k in [1, 5, 10, 20]:
            similar_movies = self.calculator.get_similar_movies(
                movie_id=0,
                edge_index=self.edge_index,
                similarity_metric='cosine',
                top_k=top_k
            )
            
            # Should not exceed top_k
            self.assertLessEqual(len(similar_movies), top_k)

    def test_similarity_metrics_consistency(self):
        """Test that different similarity metrics produce consistent results."""
        movie_id = 0
        top_k = 5
        
        # Get similar movies with different metrics
        cosine_results = self.calculator.get_similar_movies(
            movie_id, self.edge_index, 'cosine', top_k
        )
        dot_product_results = self.calculator.get_similar_movies(
            movie_id, self.edge_index, 'dot_product', top_k
        )
        
        # Both should return the same number of results
        self.assertEqual(len(cosine_results), len(dot_product_results))
        
        # Both should return valid movie IDs
        for results in [cosine_results, dot_product_results]:
            for movie_id_result, score in results:
                self.assertIsInstance(movie_id_result, int)
                self.assertIsInstance(score, float)


if __name__ == '__main__':
    unittest.main() 