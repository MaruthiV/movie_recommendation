import unittest
import torch
import numpy as np
from lightgcn_model import LightGCN
from candidate_generator import CandidateGenerator

class TestCandidateGenerator(unittest.TestCase):
    def setUp(self):
        self.num_users = 10
        self.num_items = 20
        self.embedding_dim = 8
        self.device = 'cpu'
        self.model = LightGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_layers=2,
            device=self.device
        )
        # Create a simple edge index
        self.edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 1, 1, 2, 2, 3, 3, 4]
        ], dtype=torch.long)
        self.edge_index[1] += self.num_users
        # Fake popularity: item 0 is most popular, item 19 is least
        self.popularity = np.arange(self.num_items, 0, -1)
        self.generator = CandidateGenerator(
            model=self.model,
            edge_index=self.edge_index,
            movie_popularity=self.popularity,
            device=self.device
        )

    def test_lightgcn_generation(self):
        candidates = self.generator.generate_for_user_lightgcn(user_id=0, top_k=5)
        self.assertEqual(len(candidates), 5)
        self.assertTrue(all(isinstance(i, int) for i in candidates))
        self.assertTrue(all(0 <= i < self.num_items for i in candidates))

    def test_lightgcn_exclude_watched(self):
        watched = [0, 1, 2]
        candidates = self.generator.generate_for_user_lightgcn(user_id=0, top_k=10, exclude_watched=watched)
        self.assertTrue(all(i not in watched for i in candidates))

    def test_popularity_generation(self):
        candidates = self.generator.generate_for_user_popularity(top_k=7)
        self.assertEqual(len(candidates), 7)
        # Should be sorted by popularity (descending)
        self.assertEqual(candidates, sorted(candidates, key=lambda x: -self.popularity[x]))

    def test_popularity_exclude_watched(self):
        watched = [0, 1, 2, 3, 4]
        candidates = self.generator.generate_for_user_popularity(top_k=10, exclude_watched=watched)
        self.assertTrue(all(i not in watched for i in candidates))

    def test_hybrid_generation(self):
        candidates = self.generator.generate_for_user_hybrid(user_id=0, top_k=6, alpha=0.5)
        self.assertEqual(len(candidates), 6)
        self.assertTrue(all(isinstance(i, int) for i in candidates))

    def test_hybrid_exclude_watched(self):
        watched = [0, 1, 2, 3]
        candidates = self.generator.generate_for_user_hybrid(user_id=0, top_k=8, exclude_watched=watched, alpha=0.8)
        self.assertTrue(all(i not in watched for i in candidates))

    def test_batch_generation_lightgcn(self):
        user_ids = [0, 1, 2]
        results = self.generator.generate_for_users_batch(user_ids, strategy='lightgcn', top_k=4)
        self.assertEqual(set(results.keys()), set(user_ids))
        for cands in results.values():
            self.assertEqual(len(cands), 4)

    def test_batch_generation_popularity(self):
        user_ids = [0, 1]
        results = self.generator.generate_for_users_batch(user_ids, strategy='popularity', top_k=3)
        for cands in results.values():
            self.assertEqual(len(cands), 3)
            self.assertEqual(cands, sorted(cands, key=lambda x: -self.popularity[x]))

    def test_batch_generation_hybrid(self):
        user_ids = [0, 1]
        results = self.generator.generate_for_users_batch(user_ids, strategy='hybrid', top_k=5, alpha=0.6)
        for cands in results.values():
            self.assertEqual(len(cands), 5)

    def test_batch_exclude_watched(self):
        user_ids = [0, 1]
        exclude = {0: [0, 1], 1: [2, 3]}
        results = self.generator.generate_for_users_batch(user_ids, strategy='lightgcn', top_k=6, exclude_watched=exclude)
        self.assertTrue(all(i not in exclude[uid] for uid, cands in results.items() for i in cands))

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            self.generator.generate_for_users_batch([0], strategy='unknown', top_k=2)

    def test_top_k_larger_than_items(self):
        candidates = self.generator.generate_for_user_lightgcn(user_id=0, top_k=100)
        self.assertLessEqual(len(candidates), self.num_items)

    def test_empty_exclude_watched(self):
        candidates = self.generator.generate_for_user_lightgcn(user_id=0, top_k=5, exclude_watched=[])
        self.assertEqual(len(candidates), 5)

if __name__ == '__main__':
    unittest.main() 