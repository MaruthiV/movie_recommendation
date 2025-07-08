"""
Data loaders and preprocessing utilities for the movie recommendation system.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected, AddSelfLoops
from typing import Tuple, List, Dict, Optional, Union
import pickle
from sklearn.preprocessing import LabelEncoder
import logging

from config.model_config import DataConfig


class MovieLensDataset(Dataset):
    """
    Dataset class for MovieLens data with PyTorch Geometric integration.
    """
    
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None,
        config: DataConfig = None,
        transform=None
    ):
        """
        Initialize MovieLens dataset.
        
        Args:
            ratings_df: DataFrame with columns [userId, movieId, rating, timestamp]
            movies_df: DataFrame with movie metadata
            users_df: Optional DataFrame with user metadata
            config: Data configuration
            transform: Optional transform to apply
        """
        self.config = config or DataConfig()
        self.transform = transform
        
        # Preprocess data
        self.ratings_df = self._preprocess_ratings(ratings_df)
        self.movies_df = self._preprocess_movies(movies_df)
        self.users_df = self._preprocess_users(users_df) if users_df is not None else None
        
        # Create encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Fit encoders
        self.user_encoder.fit(self.ratings_df['userId'].unique())
        self.item_encoder.fit(self.ratings_df['movieId'].unique())
        
        # Create mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(self.user_encoder.classes_)}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.item_encoder.classes_)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Convert to indices
        self.ratings_df['user_idx'] = self.ratings_df['userId'].map(self.user_to_idx)
        self.ratings_df['item_idx'] = self.ratings_df['movieId'].map(self.item_to_idx)
        
        # Create positive interactions (ratings >= threshold)
        rating_threshold = (self.config.min_rating + self.config.max_rating) / 2
        self.positive_interactions = self.ratings_df[
            self.ratings_df['rating'] >= rating_threshold
        ][['user_idx', 'item_idx']].values
        
        # Create negative sampling pool
        self._create_negative_pool()
        
        # Create PyTorch Geometric data
        self.graph_data = self._create_graph_data()
        
        logging.info(f"Dataset created with {len(self.user_to_idx)} users, "
                    f"{len(self.item_to_idx)} items, {len(self.positive_interactions)} positive interactions")
    
    def _preprocess_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess ratings data."""
        # Filter by minimum interactions
        user_counts = ratings_df['userId'].value_counts()
        item_counts = ratings_df['movieId'].value_counts()
        
        valid_users = user_counts[user_counts >= self.config.min_interactions_per_user].index
        valid_items = item_counts[item_counts >= self.config.min_interactions_per_item].index
        
        filtered_ratings = ratings_df[
            (ratings_df['userId'].isin(valid_users)) &
            (ratings_df['movieId'].isin(valid_items))
        ].copy()
        
        # Sort by timestamp
        filtered_ratings = filtered_ratings.sort_values('timestamp')
        
        return filtered_ratings
    
    def _preprocess_movies(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess movies data."""
        # Extract year from title
        movies_df = movies_df.copy()
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
        
        # Process genres
        movies_df['genres'] = movies_df['genres'].fillna('Unknown')
        all_genres = set()
        for genres in movies_df['genres']:
            all_genres.update(genres.split('|'))
        
        # Create genre features
        for genre in all_genres:
            movies_df[f'genre_{genre}'] = movies_df['genres'].str.contains(genre).astype(int)
        
        return movies_df
    
    def _preprocess_users(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess users data."""
        if users_df is None:
            return None
        
        users_df = users_df.copy()
        
        # Encode categorical features
        if 'gender' in users_df.columns:
            users_df['gender_encoded'] = (users_df['gender'] == 'M').astype(int)
        
        if 'occupation' in users_df.columns:
            occupation_encoder = LabelEncoder()
            users_df['occupation_encoded'] = occupation_encoder.fit_transform(users_df['occupation'])
        
        return users_df
    
    def _create_negative_pool(self):
        """Create negative sampling pool."""
        all_items = set(range(len(self.item_to_idx)))
        self.negative_pool = {}
        
        for user_idx in range(len(self.user_to_idx)):
            user_items = set(self.ratings_df[
                self.ratings_df['user_idx'] == user_idx
            ]['item_idx'].values)
            self.negative_pool[user_idx] = list(all_items - user_items)
    
    def _create_graph_data(self) -> Data:
        """Create PyTorch Geometric data object."""
        # Create edge index for user-item interactions
        edge_index = torch.tensor(self.positive_interactions.T, dtype=torch.long)
        
        # Create node features (optional)
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        
        # Simple one-hot encoding for users and items
        user_features = torch.eye(num_users)
        item_features = torch.eye(num_items)
        
        # Combine features
        x = torch.cat([user_features, item_features], dim=0)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            num_users=num_users,
            num_items=num_items
        )
        
        return data
    
    def __len__(self) -> int:
        """Return number of positive interactions."""
        return len(self.positive_interactions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        user_idx, item_idx = self.positive_interactions[idx]
        
        # Sample negative items
        negative_items = np.random.choice(
            self.negative_pool[user_idx],
            size=1,
            replace=False
        )
        
        return {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'pos_item_idx': torch.tensor(item_idx, dtype=torch.long),
            'neg_item_idx': torch.tensor(negative_items[0], dtype=torch.long)
        }
    
    def get_graph_data(self) -> Data:
        """Get the PyTorch Geometric graph data."""
        return self.graph_data
    
    def get_user_item_mappings(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """Get user and item mappings."""
        return self.user_to_idx, self.item_to_idx, self.idx_to_user, self.idx_to_item


class MovieLensDataLoader:
    """
    Data loader for MovieLens dataset with train/val/test splits.
    """
    
    def __init__(
        self,
        data_dir: str,
        config: DataConfig = None,
        cache_dir: str = "cache"
    ):
        """
        Initialize MovieLens data loader.
        
        Args:
            data_dir: Directory containing MovieLens data files
            config: Data configuration
            cache_dir: Directory for caching processed data
        """
        self.data_dir = data_dir
        self.config = config or DataConfig()
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load and process data
        self.dataset = self._load_dataset()
        
        # Create train/val/test splits
        self.train_dataset, self.val_dataset, self.test_dataset = self._create_splits()
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def _load_dataset(self) -> MovieLensDataset:
        """Load and process MovieLens dataset."""
        cache_file = os.path.join(self.cache_dir, "processed_dataset.pkl")
        
        # Try to load from cache
        if self.config.use_cache and os.path.exists(cache_file):
            logging.info("Loading dataset from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Load raw data
        logging.info("Loading MovieLens data...")
        ratings_file = os.path.join(self.data_dir, "ratings.csv")
        movies_file = os.path.join(self.data_dir, "movies.csv")
        
        if not os.path.exists(ratings_file):
            raise FileNotFoundError(f"Ratings file not found: {ratings_file}")
        if not os.path.exists(movies_file):
            raise FileNotFoundError(f"Movies file not found: {movies_file}")
        
        ratings_df = pd.read_csv(ratings_file)
        movies_df = pd.read_csv(movies_file)
        
        # Check for users file
        users_file = os.path.join(self.data_dir, "users.csv")
        users_df = None
        if os.path.exists(users_file):
            users_df = pd.read_csv(users_file)
        
        # Create dataset
        dataset = MovieLensDataset(ratings_df, movies_df, users_df, self.config)
        
        # Save to cache
        if self.config.use_cache:
            logging.info("Saving dataset to cache...")
            with open(cache_file, 'wb') as f:
                pickle.dump(dataset, f)
        
        return dataset
    
    def _create_splits(self) -> Tuple[MovieLensDataset, MovieLensDataset, MovieLensDataset]:
        """Create train/val/test splits."""
        total_size = len(self.dataset)
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_graph_data(self) -> Data:
        """Get the PyTorch Geometric graph data."""
        return self.dataset.get_graph_data()
    
    def get_user_item_mappings(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """Get user and item mappings."""
        return self.dataset.get_user_item_mappings()


class HeterogeneousGraphData(HeteroData):
    """
    Heterogeneous graph data for movie recommendation with user-item-movie relationships.
    """
    
    def __init__(self, dataset: MovieLensDataset):
        """
        Initialize heterogeneous graph data.
        
        Args:
            dataset: MovieLens dataset
        """
        super().__init__()
        
        # Get user-item mappings
        user_to_idx, item_to_idx, _, _ = dataset.get_user_item_mappings()
        
        # Create node indices
        self['user'].num_nodes = len(user_to_idx)
        self['item'].num_nodes = len(item_to_idx)
        
        # Create user-item edges
        edge_index = torch.tensor(dataset.positive_interactions.T, dtype=torch.long)
        self['user', 'rates', 'item'].edge_index = edge_index
        
        # Create reverse edges
        self['item', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        
        # Add node features if available
        if hasattr(dataset, 'user_features') and dataset.user_features is not None:
            self['user'].x = dataset.user_features
        
        if hasattr(dataset, 'item_features') and dataset.item_features is not None:
            self['item'].x = dataset.item_features
        
        # Apply transforms
        self = ToUndirected()(self)
        self = AddSelfLoops()(self)


def create_data_loaders(
    data_dir: str,
    config: DataConfig = None,
    cache_dir: str = "cache"
) -> Tuple[MovieLensDataLoader, Data, Dict]:
    """
    Create data loaders and graph data for training.
    
    Args:
        data_dir: Directory containing MovieLens data
        config: Data configuration
        cache_dir: Directory for caching
        
    Returns:
        Tuple of (data_loader, graph_data, metadata)
    """
    # Create data loader
    data_loader = MovieLensDataLoader(data_dir, config, cache_dir)
    
    # Get graph data
    graph_data = data_loader.get_graph_data()
    
    # Get metadata
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = data_loader.get_user_item_mappings()
    metadata = {
        'num_users': len(user_to_idx),
        'num_items': len(item_to_idx),
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item
    }
    
    return data_loader, graph_data, metadata 