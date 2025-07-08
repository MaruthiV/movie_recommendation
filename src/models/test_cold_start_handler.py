import unittest
import torch
import numpy as np
from lightgcn_model import LightGCN
from cold_start_handler import ColdStartHandler


class TestColdStartHandler(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.num_users = 10
        self.num_items = 20
        self.embedding_dim = 8
        self.device = 'cpu'
        
        # Create LightGCN model
        self.model = LightGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_layers=2,
            device=self.device
        )
        
        # Create edge index
        self.edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 1, 1, 2, 2, 3, 3, 4]
        ], dtype=torch.long)
        self.edge_index[1] += self.num_users
        
        # Create movie features (simple feature vectors)
        self.movie_features = {
            i: np.random.rand(5) for i in range(self.num_items)
        }
        
        # Create popularity scores
        self.movie_popularity = np.arange(self.num_items, 0, -1)
        
        # Create cold start handler
        self.handler = ColdStartHandler(
            model=self.model,
            edge_index=self.edge_index,
            movie_features=self.movie_features,
            movie_popularity=self.movie_popularity,
            device=self.device
        )

    def test_popularity_based_new_user(self):
        """Test popularity-based recommendations for new users."""
        recommendations = self.handler.recommend_for_new_user(
            strategy='popularity',
            top_k=5
        )
        
        self.assertEqual(len(recommendations), 5)
        self.assertTrue(all(isinstance(i, int) for i in recommendations))
        self.assertTrue(all(0 <= i < self.num_items for i in recommendations))
        
        # Should be sorted by popularity (descending)
        expected_order = np.argsort(-self.movie_popularity)[:5]
        self.assertEqual(recommendations, expected_order.tolist())

    def test_content_based_new_user(self):
        """Test content-based recommendations for new users."""
        initial_preferences = [0, 1, 2]
        recommendations = self.handler.recommend_for_new_user(
            initial_preferences=initial_preferences,
            strategy='content_based',
            top_k=5
        )
        
        self.assertEqual(len(recommendations), 5)
        self.assertTrue(all(isinstance(i, int) for i in recommendations))
        self.assertTrue(all(0 <= i < self.num_items for i in recommendations))

    def test_content_based_new_user_no_preferences(self):
        """Test content-based recommendations when no initial preferences."""
        recommendations = self.handler.recommend_for_new_user(
            strategy='content_based',
            top_k=5
        )
        
        # Should fall back to popularity-based
        self.assertEqual(len(recommendations), 5)
        expected_order = np.argsort(-self.movie_popularity)[:5]
        self.assertEqual(recommendations, expected_order.tolist())

    def test_hybrid_new_user(self):
        """Test hybrid recommendations for new users."""
        initial_preferences = [0, 1]
        user_demographics = {'age': 25, 'gender': 'M'}
        
        recommendations = self.handler.recommend_for_new_user(
            user_demographics=user_demographics,
            initial_preferences=initial_preferences,
            strategy='hybrid',
            top_k=10
        )
        
        self.assertEqual(len(recommendations), 10)
        self.assertTrue(all(isinstance(i, int) for i in recommendations))
        self.assertTrue(all(0 <= i < self.num_items for i in recommendations))

    def test_content_based_new_content(self):
        """Test content-based recommendations for new content."""
        movie_id = 5
        movie_features = np.random.rand(5)
        
        similar_movies = self.handler.recommend_for_new_content(
            movie_id=movie_id,
            movie_features=movie_features,
            strategy='content_based',
            top_k=5
        )
        
        self.assertEqual(len(similar_movies), 5)
        self.assertTrue(all(isinstance(i, int) for i in similar_movies))
        self.assertTrue(all(0 <= i < self.num_items for i in similar_movies))
        self.assertNotIn(movie_id, similar_movies)  # Should not include itself

    def test_content_based_new_content_no_features(self):
        """Test content-based recommendations when no features provided."""
        # Use a movie ID that doesn't exist in features to trigger fallback
        movie_id = 999  # This won't exist in movie_features
        
        similar_movies = self.handler.recommend_for_new_content(
            movie_id=movie_id,
            strategy='content_based',
            top_k=5
        )
        
        # Should fall back to popularity-based
        self.assertEqual(len(similar_movies), 5)
        self.assertTrue(all(isinstance(i, int) for i in similar_movies))
        self.assertTrue(all(0 <= i < self.num_items for i in similar_movies))
        # Check that it returns the most popular movies (not necessarily in exact order)
        expected_popular = set(np.argsort(-self.movie_popularity)[:5])
        self.assertEqual(set(similar_movies), expected_popular)

    def test_popularity_based_new_content(self):
        """Test popularity-based recommendations for new content."""
        similar_movies = self.handler.recommend_for_new_content(
            movie_id=5,
            strategy='popularity',
            top_k=5
        )
        
        self.assertEqual(len(similar_movies), 5)
        expected_order = np.argsort(-self.movie_popularity)[:5]
        self.assertEqual(similar_movies, expected_order.tolist())

    def test_hybrid_new_content(self):
        """Test hybrid recommendations for new content."""
        movie_id = 5
        movie_features = np.random.rand(5)
        
        similar_movies = self.handler.recommend_for_new_content(
            movie_id=movie_id,
            movie_features=movie_features,
            strategy='hybrid',
            top_k=10
        )
        
        self.assertEqual(len(similar_movies), 10)
        self.assertTrue(all(isinstance(i, int) for i in similar_movies))
        self.assertTrue(all(0 <= i < self.num_items for i in similar_movies))

    def test_get_user_embedding_for_new_user(self):
        """Test pseudo-embedding generation for new users."""
        initial_preferences = [0, 1, 2]
        
        embedding = self.handler.get_user_embedding_for_new_user(
            initial_preferences=initial_preferences
        )
        
        self.assertEqual(embedding.shape, (self.embedding_dim,))
        self.assertIsInstance(embedding, torch.Tensor)

    def test_get_user_embedding_no_preferences(self):
        """Test pseudo-embedding generation without preferences."""
        embedding = self.handler.get_user_embedding_for_new_user()
        
        self.assertEqual(embedding.shape, (self.embedding_dim,))
        self.assertTrue(torch.allclose(embedding, torch.zeros_like(embedding)))

    def test_recommend_with_pseudo_embedding(self):
        """Test recommendations using pseudo embedding."""
        initial_preferences = [0, 1, 2]
        pseudo_embedding = self.handler.get_user_embedding_for_new_user(
            initial_preferences=initial_preferences
        )
        
        recommendations = self.handler.recommend_with_pseudo_embedding(
            pseudo_embedding=pseudo_embedding,
            top_k=5
        )
        
        self.assertEqual(len(recommendations), 5)
        self.assertTrue(all(isinstance(i, int) for i in recommendations))
        self.assertTrue(all(0 <= i < self.num_items for i in recommendations))

    def test_recommend_with_pseudo_embedding_exclude_watched(self):
        """Test pseudo embedding recommendations with exclusion."""
        initial_preferences = [0, 1, 2]
        pseudo_embedding = self.handler.get_user_embedding_for_new_user(
            initial_preferences=initial_preferences
        )
        
        exclude_watched = [0, 1]
        recommendations = self.handler.recommend_with_pseudo_embedding(
            pseudo_embedding=pseudo_embedding,
            top_k=5,
            exclude_watched=exclude_watched
        )
        
        self.assertEqual(len(recommendations), 5)
        self.assertTrue(all(i not in exclude_watched for i in recommendations))

    def test_invalid_strategy_new_user(self):
        """Test that invalid strategy raises error for new users."""
        with self.assertRaises(ValueError):
            self.handler.recommend_for_new_user(strategy='invalid')

    def test_invalid_strategy_new_content(self):
        """Test that invalid strategy raises error for new content."""
        with self.assertRaises(ValueError):
            self.handler.recommend_for_new_content(movie_id=0, strategy='invalid')

    def test_top_k_larger_than_items(self):
        """Test behavior when top_k is larger than available items."""
        recommendations = self.handler.recommend_for_new_user(
            strategy='popularity',
            top_k=100
        )
        
        self.assertLessEqual(len(recommendations), self.num_items)

    def test_empty_initial_preferences(self):
        """Test behavior with empty initial preferences."""
        recommendations = self.handler.recommend_for_new_user(
            initial_preferences=[],
            strategy='content_based',
            top_k=5
        )
        
        # Should fall back to popularity-based
        self.assertEqual(len(recommendations), 5)
        expected_order = np.argsort(-self.movie_popularity)[:5]
        self.assertEqual(recommendations, expected_order.tolist())

    def test_handler_without_movie_features(self):
        """Test handler initialization without movie features."""
        handler_no_features = ColdStartHandler(
            model=self.model,
            edge_index=self.edge_index,
            movie_popularity=self.movie_popularity,
            device=self.device
        )
        
        recommendations = handler_no_features.recommend_for_new_user(
            strategy='content_based',
            top_k=5
        )
        
        # Should fall back to popularity-based
        self.assertEqual(len(recommendations), 5)
        expected_order = np.argsort(-self.movie_popularity)[:5]
        self.assertEqual(recommendations, expected_order.tolist())

    def test_handler_without_popularity(self):
        """Test handler initialization without popularity scores."""
        handler_no_popularity = ColdStartHandler(
            model=self.model,
            edge_index=self.edge_index,
            movie_features=self.movie_features,
            device=self.device
        )
        
        recommendations = handler_no_popularity.recommend_for_new_user(
            strategy='popularity',
            top_k=5
        )
        
        # Should use random selection as fallback
        self.assertEqual(len(recommendations), 5)
        self.assertTrue(all(isinstance(i, int) for i in recommendations))


if __name__ == '__main__':
    unittest.main() 