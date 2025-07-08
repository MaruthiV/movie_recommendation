import unittest
import numpy as np
from diversity import DiversityConstraint

class TestDiversityConstraint(unittest.TestCase):
    def setUp(self):
        self.candidate_ids = list(range(10))
        self.relevance_scores = [10 - i for i in range(10)]  # 10, 9, ..., 1
        self.embeddings = np.random.rand(10, 4)
        self.movie_metadata = {
            0: {'genres': ['Action', 'Comedy']},
            1: {'genres': ['Drama']},
            2: {'genres': ['Action']},
            3: {'genres': ['Comedy']},
            4: {'genres': ['Drama', 'Romance']},
            5: {'genres': ['Sci-Fi']},
            6: {'genres': ['Action', 'Sci-Fi']},
            7: {'genres': ['Romance']},
            8: {'genres': ['Comedy']},
            9: {'genres': ['Action', 'Drama']},
        }
        self.diversity = DiversityConstraint(self.movie_metadata)

    def test_mmr_rerank_relevance_only(self):
        reranked = self.diversity.mmr_rerank(
            self.candidate_ids, self.relevance_scores, embeddings=None, lambda_diversity=0.0, top_k=5
        )
        # Should be sorted by relevance
        self.assertEqual(reranked, [0, 1, 2, 3, 4])

    def test_mmr_rerank_diversity_only(self):
        reranked = self.diversity.mmr_rerank(
            self.candidate_ids, self.relevance_scores, embeddings=self.embeddings, lambda_diversity=1.0, top_k=5
        )
        # Should be a permutation of candidate_ids
        self.assertEqual(len(reranked), 5)
        self.assertTrue(set(reranked).issubset(set(self.candidate_ids)))

    def test_mmr_rerank_mixed(self):
        reranked = self.diversity.mmr_rerank(
            self.candidate_ids, self.relevance_scores, embeddings=self.embeddings, lambda_diversity=0.5, top_k=5
        )
        self.assertEqual(len(reranked), 5)
        self.assertTrue(set(reranked).issubset(set(self.candidate_ids)))

    def test_genre_coverage(self):
        # Should cover as many genres as possible in the first top_k
        result = self.diversity.genre_coverage(self.candidate_ids, top_k=5)
        genres_covered = set()
        for mid in result:
            genres_covered.update(self.movie_metadata.get(mid, {}).get('genres', []))
        self.assertGreaterEqual(len(genres_covered), 5)  # At least 5 genres in top 5

    def test_genre_coverage_padding(self):
        # If not enough genres, should pad with remaining
        small_metadata = {0: {'genres': ['Action']}, 1: {'genres': ['Action']}}
        diversity = DiversityConstraint(small_metadata)
        result = diversity.genre_coverage([0, 1], top_k=4)
        self.assertEqual(len(result), 2)
        self.assertTrue(set(result).issubset({0, 1}))

    def test_embedding_diversity(self):
        result = self.diversity.embedding_diversity(self.candidate_ids, self.embeddings, top_k=5)
        self.assertEqual(len(result), 5)
        self.assertTrue(set(result).issubset(set(self.candidate_ids)))
        # Should not have duplicates
        self.assertEqual(len(result), len(set(result)))

    def test_embedding_diversity_small(self):
        # If fewer candidates than top_k, should return all
        result = self.diversity.embedding_diversity([0, 1], self.embeddings[:2], top_k=5)
        self.assertEqual(result, [0, 1])

    def test_empty_candidates(self):
        self.assertEqual(self.diversity.mmr_rerank([], [], embeddings=None, top_k=5), [])
        self.assertEqual(self.diversity.genre_coverage([], top_k=5), [])
        self.assertEqual(self.diversity.embedding_diversity([], np.empty((0, 4)), top_k=5), [])

if __name__ == '__main__':
    unittest.main() 