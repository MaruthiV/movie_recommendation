"""
Evaluation metrics for recommender systems (ranking-based).
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from sklearn.metrics import ndcg_score, precision_score, recall_score
import logging


def calculate_ndcg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
    method: str = 'standard'
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at k.
    
    Args:
        y_true: True relevance scores (binary or graded)
        y_pred: Predicted relevance scores
        k: Number of top items to consider
        method: NDCG calculation method ('standard' or 'sklearn')
        
    Returns:
        NDCG@k score
    """
    if method == 'sklearn':
        # Use sklearn's implementation
        try:
            # Reshape for sklearn (expects 2D array)
            y_true_2d = y_true.reshape(1, -1)
            y_pred_2d = y_pred.reshape(1, -1)
            return ndcg_score(y_true_2d, y_pred_2d, k=k)
        except Exception as e:
            logging.warning(f"sklearn NDCG failed: {e}, falling back to custom implementation")
            method = 'standard'
    
    if method == 'standard':
        # Custom implementation
        # Sort predictions in descending order
        sorted_indices = np.argsort(y_pred)[::-1]
        
        # Get top-k items
        top_k_indices = sorted_indices[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            dcg += y_true[idx] / np.log2(i + 2)  # log2(i+2) because i starts from 0
        
        # Calculate IDCG (ideal DCG)
        ideal_sorted_indices = np.argsort(y_true)[::-1]
        ideal_top_k_indices = ideal_sorted_indices[:k]
        
        idcg = 0.0
        for i, idx in enumerate(ideal_top_k_indices):
            idcg += y_true[idx] / np.log2(i + 2)
        
        # Calculate NDCG
        if idcg == 0:
            return 0.0
        
        return dcg / idcg


def calculate_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
    method: str = 'standard'
) -> float:
    """
    Calculate Recall at k.
    
    Args:
        y_true: True relevance scores (binary)
        y_pred: Predicted relevance scores
        k: Number of top items to consider
        method: Calculation method ('standard' or 'sklearn')
        
    Returns:
        Recall@k score
    """
    if method == 'sklearn':
        # Use sklearn's implementation
        try:
            # Create binary predictions for top-k
            sorted_indices = np.argsort(y_pred)[::-1]
            top_k_indices = sorted_indices[:k]
            
            y_pred_binary = np.zeros_like(y_true)
            y_pred_binary[top_k_indices] = 1
            
            return recall_score(y_true, y_pred_binary, zero_division=0)
        except Exception as e:
            logging.warning(f"sklearn recall failed: {e}, falling back to custom implementation")
            method = 'standard'
    
    if method == 'standard':
        # Custom implementation
        # Sort predictions in descending order
        sorted_indices = np.argsort(y_pred)[::-1]
        
        # Get top-k items
        top_k_indices = sorted_indices[:k]
        
        # Count relevant items in top-k
        relevant_in_top_k = np.sum(y_true[top_k_indices])
        
        # Count total relevant items
        total_relevant = np.sum(y_true)
        
        # Calculate recall
        if total_relevant == 0:
            return 0.0
        
        return relevant_in_top_k / total_relevant


def calculate_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
    method: str = 'standard'
) -> float:
    """
    Calculate Precision at k.
    
    Args:
        y_true: True relevance scores (binary)
        y_pred: Predicted relevance scores
        k: Number of top items to consider
        method: Calculation method ('standard' or 'sklearn')
        
    Returns:
        Precision@k score
    """
    if method == 'sklearn':
        # Use sklearn's implementation
        try:
            # Create binary predictions for top-k
            sorted_indices = np.argsort(y_pred)[::-1]
            top_k_indices = sorted_indices[:k]
            
            y_pred_binary = np.zeros_like(y_true)
            y_pred_binary[top_k_indices] = 1
            
            return precision_score(y_true, y_pred_binary, zero_division=0)
        except Exception as e:
            logging.warning(f"sklearn precision failed: {e}, falling back to custom implementation")
            method = 'standard'
    
    if method == 'standard':
        # Custom implementation
        # Sort predictions in descending order
        sorted_indices = np.argsort(y_pred)[::-1]
        
        # Get top-k items
        top_k_indices = sorted_indices[:k]
        
        # Count relevant items in top-k
        relevant_in_top_k = np.sum(y_true[top_k_indices])
        
        # Calculate precision
        return relevant_in_top_k / k


def calculate_hit_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10
) -> float:
    """
    Calculate Hit Ratio at k (whether at least one relevant item is in top-k).
    
    Args:
        y_true: True relevance scores (binary)
        y_pred: Predicted relevance scores
        k: Number of top items to consider
        
    Returns:
        Hit Ratio@k score
    """
    # Sort predictions in descending order
    sorted_indices = np.argsort(y_pred)[::-1]
    
    # Get top-k items
    top_k_indices = sorted_indices[:k]
    
    # Check if any relevant item is in top-k
    has_relevant = np.any(y_true[top_k_indices])
    
    return 1.0 if has_relevant else 0.0


def calculate_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: Optional[int] = None
) -> float:
    """
    Calculate Mean Average Precision (MAP).
    
    Args:
        y_true: True relevance scores (binary)
        y_pred: Predicted relevance scores
        k: Number of top items to consider (None for all)
        
    Returns:
        MAP score
    """
    if k is not None:
        # Sort predictions in descending order
        sorted_indices = np.argsort(y_pred)[::-1]
        top_k_indices = sorted_indices[:k]
        
        y_true = y_true[top_k_indices]
        y_pred = y_pred[top_k_indices]
    
    # Calculate average precision
    sorted_indices = np.argsort(y_pred)[::-1]
    
    precision_sum = 0.0
    relevant_count = 0
    total_relevant = np.sum(y_true)
    
    if total_relevant == 0:
        return 0.0
    
    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    return precision_sum / total_relevant


def calculate_mrr(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        y_true: True relevance scores (binary)
        y_pred: Predicted relevance scores
        
    Returns:
        MRR score
    """
    # Sort predictions in descending order
    sorted_indices = np.argsort(y_pred)[::-1]
    
    # Find the rank of the first relevant item
    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            return 1.0 / (i + 1)
    
    return 0.0


def calculate_diversity(
    recommendations: List[List[int]],
    item_features: Optional[np.ndarray] = None,
    method: str = 'intra_list'
) -> float:
    """
    Calculate diversity of recommendations.
    
    Args:
        recommendations: List of recommendation lists for each user
        item_features: Optional item features for feature-based diversity
        method: Diversity calculation method ('intra_list' or 'feature')
        
    Returns:
        Diversity score
    """
    if method == 'intra_list':
        # Intra-list diversity (average pairwise distance between items)
        total_diversity = 0.0
        total_pairs = 0
        
        for rec_list in recommendations:
            if len(rec_list) < 2:
                continue
            
            # Calculate pairwise distances (assuming items are indices)
            for i in range(len(rec_list)):
                for j in range(i + 1, len(rec_list)):
                    # Simple distance: 1 if different, 0 if same
                    distance = 1.0 if rec_list[i] != rec_list[j] else 0.0
                    total_diversity += distance
                    total_pairs += 1
        
        return total_diversity / total_pairs if total_pairs > 0 else 0.0
    
    elif method == 'feature' and item_features is not None:
        # Feature-based diversity
        total_diversity = 0.0
        total_pairs = 0
        
        for rec_list in recommendations:
            if len(rec_list) < 2:
                continue
            
            for i in range(len(rec_list)):
                for j in range(i + 1, len(rec_list)):
                    # Calculate cosine distance between item features
                    feat_i = item_features[rec_list[i]]
                    feat_j = item_features[rec_list[j]]
                    
                    # Cosine distance = 1 - cosine similarity
                    cosine_sim = np.dot(feat_i, feat_j) / (np.linalg.norm(feat_i) * np.linalg.norm(feat_j))
                    distance = 1.0 - cosine_sim
                    
                    total_diversity += distance
                    total_pairs += 1
        
        return total_diversity / total_pairs if total_pairs > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown diversity method: {method}")


def calculate_coverage(
    recommendations: List[List[int]],
    num_items: int
) -> float:
    """
    Calculate catalog coverage (percentage of items recommended).
    
    Args:
        recommendations: List of recommendation lists for each user
        num_items: Total number of items in catalog
        
    Returns:
        Coverage score
    """
    recommended_items = set()
    
    for rec_list in recommendations:
        recommended_items.update(rec_list)
    
    return len(recommended_items) / num_items


def evaluate_recommendations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    recommendations: Optional[List[List[int]]] = None,
    item_features: Optional[np.ndarray] = None,
    k_values: List[int] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of recommendations.
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted relevance scores
        recommendations: Optional list of recommendation lists
        item_features: Optional item features for diversity calculation
        k_values: List of k values for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]
    
    metrics = {}
    
    # Calculate metrics for each k value
    for k in k_values:
        metrics[f'ndcg@{k}'] = calculate_ndcg(y_true, y_pred, k=k)
        metrics[f'recall@{k}'] = calculate_recall(y_true, y_pred, k=k)
        metrics[f'precision@{k}'] = calculate_precision(y_true, y_pred, k=k)
        metrics[f'hit_ratio@{k}'] = calculate_hit_ratio(y_true, y_pred, k=k)
    
    # Calculate overall metrics
    metrics['map'] = calculate_map(y_true, y_pred)
    metrics['mrr'] = calculate_mrr(y_true, y_pred)
    
    # Calculate diversity and coverage if recommendations are provided
    if recommendations is not None:
        metrics['diversity'] = calculate_diversity(recommendations, item_features)
        metrics['coverage'] = calculate_coverage(recommendations, len(y_true))
    
    return metrics


def print_evaluation_results(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Print evaluation results in a formatted way.
    
    Args:
        metrics: Dictionary of evaluation metrics
        title: Title for the results
    """
    print(f"\n{title}")
    print("=" * 50)
    
    # Group metrics by type
    ranking_metrics = {k: v for k, v in metrics.items() if '@' in k}
    overall_metrics = {k: v for k, v in metrics.items() if '@' not in k}
    
    # Print ranking metrics
    if ranking_metrics:
        print("\nRanking Metrics:")
        for metric, value in sorted(ranking_metrics.items()):
            print(f"  {metric:15s}: {value:.4f}")
    
    # Print overall metrics
    if overall_metrics:
        print("\nOverall Metrics:")
        for metric, value in sorted(overall_metrics.items()):
            print(f"  {metric:15s}: {value:.4f}")
    
    print("=" * 50)


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if k == 0:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / len(relevant)


def hit_ratio_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    recommended_k = recommended[:k]
    return 1.0 if any(item in relevant for item in recommended_k) else 0.0


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    recommended_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    # Ideal DCG
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    recommended_k = recommended[:k]
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant) if relevant else 0.0


def mrr_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    recommended_k = recommended[:k]
    for i, item in enumerate(recommended_k):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def coverage(recommendations: List[List[int]], all_items: Set[int]) -> float:
    recommended_items = set(item for rec in recommendations for item in rec)
    return len(recommended_items) / len(all_items) if all_items else 0.0


def diversity(recommendations: List[List[int]], item_embeddings: np.ndarray) -> float:
    # Average pairwise cosine distance between recommended items
    from sklearn.metrics.pairwise import cosine_distances
    total_div = 0.0
    count = 0
    for rec in recommendations:
        if len(rec) < 2:
            continue
        emb = item_embeddings[rec]
        dists = cosine_distances(emb)
        # Only upper triangle, excluding diagonal
        triu_indices = np.triu_indices_from(dists, k=1)
        total_div += dists[triu_indices].sum()
        count += len(dists[triu_indices])
    return total_div / count if count > 0 else 0.0


def evaluate_ranking(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, Set[int]],
    k_list: List[int] = [5, 10, 20]
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate ranking metrics for a batch of users.
    Args:
        recommendations: user_id -> list of recommended item ids
        ground_truth: user_id -> set of relevant item ids
        k_list: list of cutoff values
    Returns:
        metrics: metric_name -> {k: value}
    """
    metrics = {m: {k: 0.0 for k in k_list} for m in ["precision", "recall", "ndcg", "hit_ratio", "map", "mrr"]}
    n_users = 0
    for user, recs in recommendations.items():
        relevant = ground_truth.get(user, set())
        if not relevant:
            continue
        n_users += 1
        for k in k_list:
            metrics["precision"][k] += precision_at_k(recs, relevant, k)
            metrics["recall"][k] += recall_at_k(recs, relevant, k)
            metrics["ndcg"][k] += ndcg_at_k(recs, relevant, k)
            metrics["hit_ratio"][k] += hit_ratio_at_k(recs, relevant, k)
            metrics["map"][k] += map_at_k(recs, relevant, k)
            metrics["mrr"][k] += mrr_at_k(recs, relevant, k)
    if n_users > 0:
        for m in metrics:
            for k in k_list:
                metrics[m][k] /= n_users
    return metrics 