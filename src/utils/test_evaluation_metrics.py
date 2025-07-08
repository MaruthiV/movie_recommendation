"""
Unit tests for evaluation metrics.
"""

import unittest
import numpy as np
from typing import List, Dict, Set

from src.utils.evaluation_metrics import (
    precision_at_k, recall_at_k, hit_ratio_at_k, ndcg_at_k,
    map_at_k, mrr_at_k, coverage, diversity, evaluate_ranking
)


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics functions."""
    
    def setUp(self):
        """Set up test data."""
        # Test data: user recommendations and ground truth
        self.recommendations = {
            1: [1, 2, 3, 4, 5],
            2: [2, 1, 3, 4, 5],
            3: [3, 4, 5, 1, 2]
        }
        
        self.ground_truth = {
            1: {1, 2, 3},  # 3 relevant items
            2: {1, 2},     # 2 relevant items
            3: {1, 2, 3, 4}  # 4 relevant items
        }
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        # User 1: [1, 2, 3, 4, 5], relevant: {1, 2, 3}
        # At k=3: precision = 3/3 = 1.0
        # At k=5: precision = 3/5 = 0.6
        self.assertEqual(precision_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 3), 1.0)
        self.assertEqual(precision_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 5), 0.6)
        
        # User 2: [2, 1, 3, 4, 5], relevant: {1, 2}
        # At k=2: precision = 2/2 = 1.0
        # At k=3: precision = 2/3 ≈ 0.667
        self.assertEqual(precision_at_k([2, 1, 3, 4, 5], {1, 2}, 2), 1.0)
        self.assertAlmostEqual(precision_at_k([2, 1, 3, 4, 5], {1, 2}, 3), 2/3, places=3)
        
        # Edge cases
        self.assertEqual(precision_at_k([], {1, 2}, 0), 0.0)
        self.assertEqual(precision_at_k([1, 2, 3], set(), 3), 0.0)
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        # User 1: [1, 2, 3, 4, 5], relevant: {1, 2, 3}
        # At k=3: recall = 3/3 = 1.0
        # At k=2: recall = 2/3 ≈ 0.667
        self.assertEqual(recall_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 3), 1.0)
        self.assertAlmostEqual(recall_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 2), 2/3, places=3)
        
        # User 2: [2, 1, 3, 4, 5], relevant: {1, 2}
        # At k=2: recall = 2/2 = 1.0
        # At k=1: recall = 1/2 = 0.5
        self.assertEqual(recall_at_k([2, 1, 3, 4, 5], {1, 2}, 2), 1.0)
        self.assertEqual(recall_at_k([2, 1, 3, 4, 5], {1, 2}, 1), 0.5)
        
        # Edge cases
        self.assertEqual(recall_at_k([], set(), 0), 0.0)
        self.assertEqual(recall_at_k([1, 2, 3], set(), 3), 0.0)
    
    def test_hit_ratio_at_k(self):
        """Test hit ratio@k calculation."""
        # User 1: [1, 2, 3, 4, 5], relevant: {1, 2, 3}
        # At k=1: hit_ratio = 1.0 (first item is relevant)
        # At k=5: hit_ratio = 1.0 (multiple relevant items)
        self.assertEqual(hit_ratio_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 1), 1.0)
        self.assertEqual(hit_ratio_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 5), 1.0)
        
        # User 3: [3, 4, 5, 1, 2], relevant: {1, 2, 3, 4}
        # At k=1: hit_ratio = 1.0 (first item is relevant)
        # At k=2: hit_ratio = 1.0 (second item is also relevant)
        self.assertEqual(hit_ratio_at_k([3, 4, 5, 1, 2], {1, 2, 3, 4}, 1), 1.0)
        self.assertEqual(hit_ratio_at_k([3, 4, 5, 1, 2], {1, 2, 3, 4}, 2), 1.0)
        
        # No hits
        self.assertEqual(hit_ratio_at_k([4, 5, 6], {1, 2, 3}, 3), 0.0)
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        # User 1: [1, 2, 3, 4, 5], relevant: {1, 2, 3}
        # Ideal ranking: [1, 2, 3, 4, 5] (already optimal)
        # DCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.631 + 0.5 = 2.131
        # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 2.131
        # NDCG = 2.131/2.131 = 1.0
        self.assertAlmostEqual(ndcg_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 3), 1.0, places=3)
        
        # User 2: [2, 1, 3, 4, 5], relevant: {1, 2}
        # DCG = 1/log2(3) + 1/log2(4) = 0.631 + 0.5 = 1.131
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.631 = 1.631
        # NDCG = 1.131/1.631 ≈ 0.693
        self.assertAlmostEqual(ndcg_at_k([2, 1, 3, 4, 5], {1, 2}, 2), 1.0, places=3)
        
        # Edge cases
        self.assertEqual(ndcg_at_k([], set(), 0), 0.0)
        self.assertEqual(ndcg_at_k([1, 2, 3], set(), 3), 0.0)
    
    def test_map_at_k(self):
        """Test MAP@k calculation."""
        # User 1: [1, 2, 3, 4, 5], relevant: {1, 2, 3}
        # At k=3: MAP = (1/1 + 2/2 + 3/3) / 3 = 1.0
        # At k=5: MAP = (1/1 + 2/2 + 3/3) / 3 = 1.0
        self.assertEqual(map_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 3), 1.0)
        self.assertEqual(map_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 5), 1.0)
        
        # User 2: [2, 1, 3, 4, 5], relevant: {1, 2}
        # At k=2: MAP = (1/1 + 2/2) / 2 = 1.0 (both items are relevant)
        # At k=3: MAP = (1/1 + 2/2) / 2 = 1.0
        self.assertAlmostEqual(map_at_k([2, 1, 3, 4, 5], {1, 2}, 2), 1.0, places=3)
        self.assertAlmostEqual(map_at_k([2, 1, 3, 4, 5], {1, 2}, 3), 1.0, places=3)
        
        # Edge cases
        self.assertEqual(map_at_k([], set(), 0), 0.0)
        self.assertEqual(map_at_k([1, 2, 3], set(), 3), 0.0)
    
    def test_mrr_at_k(self):
        """Test MRR@k calculation."""
        # User 1: [1, 2, 3, 4, 5], relevant: {1, 2, 3}
        # First relevant item at position 1: MRR = 1/1 = 1.0
        self.assertEqual(mrr_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 3), 1.0)
        self.assertEqual(mrr_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 5), 1.0)
        
        # User 2: [2, 1, 3, 4, 5], relevant: {1, 2}
        # First relevant item at position 1: MRR = 1/1 = 1.0
        self.assertEqual(mrr_at_k([2, 1, 3, 4, 5], {1, 2}, 2), 1.0)
        self.assertEqual(mrr_at_k([2, 1, 3, 4, 5], {1, 2}, 3), 1.0)
        
        # User 3: [3, 4, 5, 1, 2], relevant: {1, 2, 3, 4}
        # First relevant item at position 1: MRR = 1/1 = 1.0
        self.assertEqual(mrr_at_k([3, 4, 5, 1, 2], {1, 2, 3, 4}, 3), 1.0)
        
        # No relevant items in top-k
        self.assertEqual(mrr_at_k([4, 5, 6], {1, 2, 3}, 3), 0.0)
    
    def test_coverage(self):
        """Test coverage calculation."""
        recommendations = [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ]
        all_items = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        
        # Recommended items: {1, 2, 3, 4, 5}
        # Coverage = 5/10 = 0.5
        self.assertEqual(coverage(recommendations, all_items), 0.5)
        
        # Edge cases
        self.assertEqual(coverage([], all_items), 0.0)
        self.assertEqual(coverage(recommendations, set()), 0.0)
    
    def test_diversity(self):
        """Test diversity calculation."""
        # Create mock embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # item 0
            [0.0, 1.0, 0.0],  # item 1
            [0.0, 0.0, 1.0],  # item 2
            [0.5, 0.5, 0.0],  # item 3
            [0.0, 0.5, 0.5],  # item 4
        ])
        
        recommendations = [
            [0, 1, 2],  # Diverse items (orthogonal)
            [0, 3, 4],  # Less diverse items
        ]
        
        # Diversity should be higher for more diverse recommendations
        diversity_score = diversity(recommendations, embeddings)
        self.assertGreater(diversity_score, 0.0)
        self.assertLessEqual(diversity_score, 1.0)
        
        # Edge cases
        self.assertEqual(diversity([[0]], embeddings), 0.0)
        self.assertEqual(diversity([], embeddings), 0.0)
    
    def test_evaluate_ranking(self):
        """Test batch evaluation function."""
        k_list = [3, 5]
        metrics = evaluate_ranking(self.recommendations, self.ground_truth, k_list)
        
        # Check that all metrics are computed
        expected_metrics = ["precision", "recall", "ndcg", "hit_ratio", "map", "mrr"]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            for k in k_list:
                self.assertIn(k, metrics[metric])
                self.assertGreaterEqual(metrics[metric][k], 0.0)
                self.assertLessEqual(metrics[metric][k], 1.0)
        
        # Check specific values for k=3
        # User 1: precision=1.0, recall=1.0, hit_ratio=1.0
        # User 2: precision=0.667, recall=1.0, hit_ratio=1.0
        # User 3: precision=0.667, recall=0.5, hit_ratio=1.0
        # Average precision@3 = (1.0 + 0.667 + 0.667) / 3 ≈ 0.778
        self.assertAlmostEqual(metrics["precision"][3], 0.778, places=3)
        
        # Average recall@3 = (1.0 + 1.0 + 0.5) / 3 ≈ 0.833
        self.assertAlmostEqual(metrics["recall"][3], 0.833, places=3)
        
        # Average hit_ratio@3 = (1.0 + 1.0 + 1.0) / 3 = 1.0
        self.assertEqual(metrics["hit_ratio"][3], 1.0)
    
    def test_evaluate_ranking_empty(self):
        """Test evaluation with empty data."""
        empty_recs = {}
        empty_gt = {}
        k_list = [5, 10]
        
        metrics = evaluate_ranking(empty_recs, empty_gt, k_list)
        
        for metric in metrics:
            for k in k_list:
                self.assertEqual(metrics[metric][k], 0.0)
    
    def test_evaluate_ranking_no_relevant(self):
        """Test evaluation when no users have relevant items."""
        recs = {1: [1, 2, 3], 2: [4, 5, 6]}
        gt = {1: set(), 2: set()}  # No relevant items
        k_list = [3]
        
        metrics = evaluate_ranking(recs, gt, k_list)
        
        for metric in metrics:
            for k in k_list:
                self.assertEqual(metrics[metric][k], 0.0)


if __name__ == "__main__":
    unittest.main() 