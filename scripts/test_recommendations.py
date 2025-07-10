#!/usr/bin/env python3
"""
Test script for the trained LightGCN model to generate movie recommendations.
"""

import argparse
import json
import logging
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.append('.')

from src.models.lightgcn_model import LightGCN
from src.models.data_loader import create_data_loaders
from src.config.model_config import ModelConfig


def load_trained_model(checkpoint_path: str, config: ModelConfig, data_loader) -> LightGCN:
    """Load the trained model from checkpoint."""
    # Initialize model
    model = LightGCN(
        num_users=data_loader.dataset.graph_data.num_users,
        num_items=data_loader.dataset.graph_data.num_items,
        embedding_dim=config.lightgcn.embedding_dim,
        num_layers=config.lightgcn.num_layers,
        device=getattr(config.lightgcn, 'device', 'cpu')
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def get_user_recommendations(
    model: LightGCN,
    user_idx: int,
    data_loader,
    movies_df: pd.DataFrame,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Generate recommendations for a specific user."""
    model.eval()
    
    with torch.no_grad():
        # Get user and item embeddings
        user_embeddings, item_embeddings = model.get_user_item_embeddings(data_loader.dataset.graph_data.edge_index)
        
        # Get user embedding
        user_embedding = user_embeddings[user_idx].unsqueeze(0)  # [1, embedding_dim]
        
        # Calculate scores for all items
        scores = torch.mm(user_embedding, item_embeddings.t())  # [1, num_items]
        scores = scores.squeeze(0)  # [num_items]
        
        # Get top-k recommendations
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        # Convert to recommendations
        recommendations = []
        for i, (score, item_idx) in enumerate(zip(top_scores, top_indices)):
            item_idx = item_idx.item()
            
            # Get movie info
            movie_id = data_loader.dataset.idx_to_item[item_idx]
            movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            
            recommendations.append({
                'rank': i + 1,
                'movie_id': int(movie_id),
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'score': score.item()
            })
    
    return recommendations


def get_similar_movies(
    model: LightGCN,
    movie_id: int,
    data_loader,
    movies_df: pd.DataFrame,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Find movies similar to a given movie."""
    model.eval()
    
    # Get item index
    if movie_id not in data_loader.dataset.item_to_idx:
        raise ValueError(f"Movie ID {movie_id} not found in dataset")
    
    item_idx = data_loader.dataset.item_to_idx[movie_id]
    
    with torch.no_grad():
        # Get user and item embeddings
        _, item_embeddings = model.get_user_item_embeddings(data_loader.dataset.graph_data.edge_index)
        
        # Get target movie embedding
        target_embedding = item_embeddings[item_idx].unsqueeze(0)  # [1, embedding_dim]
        
        # Calculate similarities with all other movies
        similarities = torch.mm(target_embedding, item_embeddings.t())  # [1, num_items]
        similarities = similarities.squeeze(0)  # [num_items]
        
        # Get top-k similar movies (excluding the target movie)
        similarities[item_idx] = -float('inf')  # Exclude self
        top_scores, top_indices = torch.topk(similarities, k=top_k)
        
        # Convert to recommendations
        similar_movies = []
        for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
            idx = idx.item()
            
            # Get movie info
            movie_id = data_loader.dataset.idx_to_item[idx]
            movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            
            similar_movies.append({
                'rank': i + 1,
                'movie_id': int(movie_id),
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'similarity': score.item()
            })
    
    return similar_movies


def get_content_based_similar_movies(
    movie_id: int,
    movies_df: pd.DataFrame,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Find movies similar to a given movie based on genre overlap."""
    
    # Get target movie info
    target_movie = movies_df[movies_df['movieId'] == movie_id]
    if target_movie.empty:
        raise ValueError(f"Movie ID {movie_id} not found in dataset")
    
    target_movie = target_movie.iloc[0]
    target_genres = set(target_movie['genres'].split('|'))
    
    # Calculate genre similarity for all movies
    similarities = []
    for _, movie in movies_df.iterrows():
        if movie['movieId'] == movie_id:
            continue  # Skip the target movie
        
        movie_genres = set(movie['genres'].split('|'))
        
        # Calculate Jaccard similarity
        intersection = len(target_genres & movie_genres)
        union = len(target_genres | movie_genres)
        similarity = intersection / union if union > 0 else 0
        
        similarities.append({
            'movie_id': int(movie['movieId']),
            'title': movie['title'],
            'genres': movie['genres'],
            'similarity': similarity,
            'shared_genres': list(target_genres & movie_genres)
        })
    
    # Sort by similarity and get top-k
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    result = []
    for i, movie in enumerate(similarities[:top_k]):
        movie['rank'] = i + 1
        result.append(movie)
    return result


def get_hybrid_similar_movies(
    model: LightGCN,
    movie_id: int,
    data_loader,
    movies_df: pd.DataFrame,
    top_k: int = 10,
    collaborative_weight: float = 0.7,
    content_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """Find similar movies using both collaborative filtering and content-based similarity."""
    
    # Get collaborative filtering similarities
    try:
        cf_similarities = get_similar_movies(model, movie_id, data_loader, movies_df, top_k * 2)
        cf_scores = {item['movie_id']: item['similarity'] for item in cf_similarities}
    except:
        cf_scores = {}
    
    # Get content-based similarities
    cb_similarities = get_content_based_similar_movies(movie_id, movies_df, top_k * 2)
    cb_scores = {item['movie_id']: item['similarity'] for item in cb_similarities}
    
    # Combine scores
    all_movies = set(cf_scores.keys()) | set(cb_scores.keys())
    hybrid_scores = []
    
    for movie_id in all_movies:
        cf_score = cf_scores.get(movie_id, 0.0)
        cb_score = cb_scores.get(movie_id, 0.0)
        
        # Normalize scores to [0, 1] range
        cf_score = (cf_score + 1) / 2  # Assuming CF scores are roughly in [-1, 1]
        
        # Weighted combination
        hybrid_score = collaborative_weight * cf_score + content_weight * cb_score
        
        # Get movie info
        movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        
        hybrid_scores.append({
            'movie_id': int(movie_id),
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'hybrid_score': hybrid_score,
            'cf_score': cf_score,
            'cb_score': cb_score
        })
    
    # Sort by hybrid score and return top-k
    hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return hybrid_scores[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Test trained LightGCN model")
    parser.add_argument("--data-dir", type=str, default="data/ml-100k", help="Data directory")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/checkpoint_epoch_0.pt", help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="outputs/config.json", help="Config file path")
    parser.add_argument("--user-id", type=int, help="User ID to get recommendations for")
    parser.add_argument("--movie-id", type=int, help="Movie ID to find similar movies for")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--similarity-method", type=str, choices=['collaborative', 'content', 'hybrid'], 
                       default='hybrid', help="Method for finding similar movies")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)
    
    # Load data
    logger.info("Loading data...")
    data_loader, graph_data, metadata = create_data_loaders(
        data_dir=args.data_dir,
        config=config.data,
        cache_dir="outputs/cache"
    )
    
    # Load movies data
    movies_df = pd.read_csv(f"{args.data_dir}/movies.csv")
    
    # Load trained model
    logger.info("Loading trained model...")
    model = load_trained_model(args.checkpoint, config, data_loader)
    
    # Test user recommendations
    if args.user_id:
        logger.info(f"Generating recommendations for user {args.user_id}")
        
        if args.user_id not in data_loader.dataset.user_to_idx:
            logger.error(f"User ID {args.user_id} not found in dataset")
            return
        
        user_idx = data_loader.dataset.user_to_idx[args.user_id]
        recommendations = get_user_recommendations(model, user_idx, data_loader, movies_df, args.top_k)
        
        print(f"\n🎬 Top {args.top_k} Recommendations for User {args.user_id}:")
        print("=" * 80)
        for rec in recommendations:
            print(f"{rec['rank']:2d}. {rec['title']:<50} | {rec['genres']:<30} | Score: {rec['score']:.4f}")
    
    # Test similar movies
    if args.movie_id:
        logger.info(f"Finding movies similar to movie {args.movie_id} using {args.similarity_method} method")
        
        # Get target movie info
        target_movie = movies_df[movies_df['movieId'] == args.movie_id]
        if target_movie.empty:
            logger.error(f"Movie ID {args.movie_id} not found in dataset")
            return
        
        target_movie = target_movie.iloc[0]
        
        if args.similarity_method == 'collaborative':
            similar_movies = get_similar_movies(model, args.movie_id, data_loader, movies_df, args.top_k)
            print(f"\n🎭 Movies Similar to '{target_movie['title']}' (Collaborative Filtering):")
            print("=" * 80)
            for movie in similar_movies:
                print(f"{movie['rank']:2d}. {movie['title']:<50} | {movie['genres']:<30} | Similarity: {movie['similarity']:.4f}")
        
        elif args.similarity_method == 'content':
            similar_movies = get_content_based_similar_movies(args.movie_id, movies_df, args.top_k)
            print(f"\n🎭 Movies Similar to '{target_movie['title']}' (Content-Based):")
            print("=" * 80)
            for movie in similar_movies:
                shared_genres = ', '.join(movie['shared_genres']) if movie['shared_genres'] else 'None'
                print(f"{movie['rank']:2d}. {movie['title']:<50} | {movie['genres']:<30} | Similarity: {movie['similarity']:.4f} | Shared: {shared_genres}")
        
        else:  # hybrid
            similar_movies = get_hybrid_similar_movies(model, args.movie_id, data_loader, movies_df, args.top_k)
            print(f"\n🎭 Movies Similar to '{target_movie['title']}' (Hybrid - Collaborative + Content):")
            print("=" * 80)
            for movie in similar_movies:
                print(f"{movie['rank']:2d}. {movie['title']:<50} | {movie['genres']:<30} | Hybrid: {movie['hybrid_score']:.4f} | CF: {movie['cf_score']:.4f} | CB: {movie['cb_score']:.4f}")
    
    # If no specific user/movie provided, show some examples
    if not args.user_id and not args.movie_id:
        logger.info("No specific user or movie provided. Showing examples...")
        
        # Get recommendations for first few users
        for user_id in list(data_loader.dataset.user_to_idx.keys())[:3]:
            user_idx = data_loader.dataset.user_to_idx[user_id]
            recommendations = get_user_recommendations(model, user_idx, data_loader, movies_df, 5)
            
            print(f"\n🎬 Top 5 Recommendations for User {user_id}:")
            print("-" * 60)
            for rec in recommendations:
                print(f"{rec['rank']:2d}. {rec['title']:<40} | Score: {rec['score']:.4f}")


if __name__ == "__main__":
    main() 