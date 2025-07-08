"""
Unit tests for the training pipeline with PyTorch Geometric integration.
"""

import unittest
import tempfile
import os
import shutil
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from models.training_pipeline import (
    TrainingPipeline, EarlyStopping, ModelCheckpoint, 
    TrainingMetrics, train_model
)
from models.data_loader import MovieLensDataset, MovieLensDataLoader, create_data_loaders
from config.model_config import ModelConfig, get_default_config, get_fast_config
from models.lightgcn_model import LightGCN
from utils.evaluation_metrics import print_evaluation_results


class TestEarlyStopping(unittest.TestCase):
    """Test EarlyStopping callback."""
    
    def setUp(self):
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    
    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode."""
        # Should not stop initially
        self.assertFalse(self.early_stopping(0.5))
        self.assertFalse(self.early_stopping(0.4))
        self.assertFalse(self.early_stopping(0.3))
        
        # Should stop after patience is exceeded
        self.assertFalse(self.early_stopping(0.35))  # Worse but within delta
        self.assertFalse(self.early_stopping(0.35))
        self.assertTrue(self.early_stopping(0.35))  # Should stop now
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        early_stopping = EarlyStopping(patience=2, mode='max')
        
        # Should not stop initially
        self.assertFalse(early_stopping(0.5))
        self.assertFalse(early_stopping(0.6))
        
        # Should stop after patience is exceeded
        self.assertFalse(early_stopping(0.4))
        self.assertTrue(early_stopping(0.4))
    
    def test_early_stopping_with_delta(self):
        """Test early stopping with minimum delta."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.1)
        
        # Should not stop for small improvements
        self.assertFalse(early_stopping(0.5))
        self.assertFalse(early_stopping(0.45))  # Improvement but less than delta
        self.assertTrue(early_stopping(0.45))


class TestModelCheckpoint(unittest.TestCase):
    """Test ModelCheckpoint callback."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=self.temp_dir,
            save_best_only=True,
            mode='min'
        )
        
        # Create a simple model
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        metrics = {'val_loss': 0.5}
        config = get_default_config()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_callback.save_checkpoint(
            self.model, self.optimizer, self.scheduler, 1, metrics, config
        )
        
        # Check that file was created
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Check that best model was saved
        best_path = os.path.join(self.temp_dir, "best_model.pt")
        self.assertTrue(os.path.exists(best_path))
    
    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        # Save a checkpoint first
        metrics = {'val_loss': 0.5}
        config = get_default_config()
        
        checkpoint_path = self.checkpoint_callback.save_checkpoint(
            self.model, self.optimizer, self.scheduler, 5, metrics, config
        )
        
        # Create new model and optimizer
        new_model = torch.nn.Linear(10, 1)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1)
        
        # Load checkpoint
        epoch, loaded_metrics = self.checkpoint_callback.load_checkpoint(
            new_model, new_optimizer, new_scheduler, checkpoint_path
        )
        
        # Check that values were loaded correctly
        self.assertEqual(epoch, 5)
        self.assertEqual(loaded_metrics, metrics)
    
    def test_save_best_only(self):
        """Test save_best_only functionality."""
        checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=self.temp_dir,
            save_best_only=True,
            mode='min'
        )
        
        config = get_default_config()
        
        # Save first checkpoint (should be saved as best)
        path1 = checkpoint_callback.save_checkpoint(
            self.model, self.optimizer, self.scheduler, 1, 
            {'val_loss': 0.5}, config
        )
        
        # Save worse checkpoint (should not be saved)
        path2 = checkpoint_callback.save_checkpoint(
            self.model, self.optimizer, self.scheduler, 2, 
            {'val_loss': 0.6}, config
        )
        
        # Save better checkpoint (should be saved as new best)
        path3 = checkpoint_callback.save_checkpoint(
            self.model, self.optimizer, self.scheduler, 3, 
            {'val_loss': 0.3}, config
        )
        
        # Check that only best checkpoints were saved
        self.assertTrue(path1 != "")
        self.assertTrue(path2 == "")  # Should not be saved
        self.assertTrue(path3 != "")


class TestTrainingMetrics(unittest.TestCase):
    """Test TrainingMetrics dataclass."""
    
    def test_training_metrics_creation(self):
        """Test creating TrainingMetrics object."""
        metrics = TrainingMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            train_metrics={'accuracy': 0.8},
            val_metrics={'accuracy': 0.75},
            learning_rate=0.001,
            time_taken=10.5
        )
        
        self.assertEqual(metrics.epoch, 1)
        self.assertEqual(metrics.train_loss, 0.5)
        self.assertEqual(metrics.val_loss, 0.4)
        self.assertEqual(metrics.learning_rate, 0.001)
        self.assertEqual(metrics.time_taken, 10.5)


class TestTrainingPipeline(unittest.TestCase):
    """Test TrainingPipeline class."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = get_fast_config()  # Use fast config for testing
        self.config.lightgcn.num_epochs = 2  # Short training for testing
        
        # Create mock data
        self.create_mock_data()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def create_mock_data(self):
        """Create mock MovieLens data for testing."""
        data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create ratings data
        ratings_data = {
            'userId': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2],
            'movieId': [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            'rating': [4.0, 3.5, 5.0, 4.0, 3.0, 4.5, 2.0, 5.0, 4.0, 3.5, 4.0, 3.0],
            'timestamp': [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011]
        }
        
        # Create movies data
        movies_data = {
            'movieId': [1, 2, 3, 4, 5, 6],
            'title': ['Movie1 (2000)', 'Movie2 (2001)', 'Movie3 (2002)', 
                     'Movie4 (2003)', 'Movie5 (2004)', 'Movie6 (2005)'],
            'genres': ['Action', 'Comedy', 'Drama', 'Action|Comedy', 'Drama', 'Comedy']
        }
        
        # Save to CSV files
        ratings_df = pd.DataFrame(ratings_data)
        movies_df = pd.DataFrame(movies_data)
        
        ratings_df.to_csv(os.path.join(data_dir, "ratings.csv"), index=False)
        movies_df.to_csv(os.path.join(data_dir, "movies.csv"), index=False)
        
        self.data_dir = data_dir
    
    @patch('models.training_pipeline.MovieLensDataLoader')
    @patch('models.training_pipeline.create_data_loaders')
    def test_training_pipeline_initialization(self, mock_create_loaders, mock_data_loader):
        """Test TrainingPipeline initialization."""
        # Mock the data loading
        mock_data_loader_instance = Mock()
        mock_graph_data = Mock()
        mock_metadata = {
            'num_users': 5,
            'num_items': 6,
            'user_to_idx': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
            'item_to_idx': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        }
        
        mock_create_loaders.return_value = (mock_data_loader_instance, mock_graph_data, mock_metadata)
        
        # Create pipeline
        pipeline = TrainingPipeline(self.config, self.data_dir, self.temp_dir)
        
        # Check that pipeline was initialized correctly
        self.assertIsNotNone(pipeline.model)
        self.assertIsNotNone(pipeline.optimizer)
        self.assertIsNotNone(pipeline.scheduler)
        self.assertIsNotNone(pipeline.early_stopping)
        self.assertIsNotNone(pipeline.checkpoint_callback)
    
    @patch('models.training_pipeline.MovieLensDataLoader')
    @patch('models.training_pipeline.create_data_loaders')
    def test_training_pipeline_device_setup(self, mock_create_loaders, mock_data_loader):
        """Test device setup in TrainingPipeline."""
        # Mock the data loading
        mock_data_loader_instance = Mock()
        mock_graph_data = Mock()
        mock_metadata = {
            'num_users': 5,
            'num_items': 6,
            'user_to_idx': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
            'item_to_idx': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        }
        
        mock_create_loaders.return_value = (mock_data_loader_instance, mock_graph_data, mock_metadata)
        
        # Test CPU device
        config = get_fast_config()
        config.lightgcn.device = 'cpu'
        
        pipeline = TrainingPipeline(config, self.data_dir, self.temp_dir)
        self.assertEqual(str(pipeline.device), 'cpu')
    
    @patch('models.training_pipeline.MovieLensDataLoader')
    @patch('models.training_pipeline.create_data_loaders')
    def test_training_pipeline_save_results(self, mock_create_loaders, mock_data_loader):
        """Test saving training results."""
        # Mock the data loading
        mock_data_loader_instance = Mock()
        mock_graph_data = Mock()
        mock_metadata = {
            'num_users': 5,
            'num_items': 6,
            'user_to_idx': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
            'item_to_idx': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        }
        
        mock_create_loaders.return_value = (mock_data_loader_instance, mock_graph_data, mock_metadata)
        
        # Create pipeline
        pipeline = TrainingPipeline(self.config, self.data_dir, self.temp_dir)
        
        # Add some training history
        pipeline.training_history = [
            TrainingMetrics(
                epoch=1,
                train_loss=0.5,
                val_loss=0.4,
                train_metrics={},
                val_metrics={},
                learning_rate=0.001,
                time_taken=10.0
            )
        ]
        
        # Save results
        pipeline._save_training_results()
        
        # Check that files were created
        history_file = os.path.join(self.temp_dir, "training_history.json")
        config_file = os.path.join(self.temp_dir, "config.json")
        metadata_file = os.path.join(self.temp_dir, "metadata.json")
        
        self.assertTrue(os.path.exists(history_file))
        self.assertTrue(os.path.exists(config_file))
        self.assertTrue(os.path.exists(metadata_file))


class TestDataLoader(unittest.TestCase):
    """Test data loader functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_mock_data()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def create_mock_data(self):
        """Create mock MovieLens data for testing."""
        data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create ratings data
        ratings_data = {
            'userId': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2],
            'movieId': [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            'rating': [4.0, 3.5, 5.0, 4.0, 3.0, 4.5, 2.0, 5.0, 4.0, 3.5, 4.0, 3.0],
            'timestamp': [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011]
        }
        
        # Create movies data
        movies_data = {
            'movieId': [1, 2, 3, 4, 5, 6],
            'title': ['Movie1 (2000)', 'Movie2 (2001)', 'Movie3 (2002)', 
                     'Movie4 (2003)', 'Movie5 (2004)', 'Movie6 (2005)'],
            'genres': ['Action', 'Comedy', 'Drama', 'Action|Comedy', 'Drama', 'Comedy']
        }
        
        # Save to CSV files
        ratings_df = pd.DataFrame(ratings_data)
        movies_df = pd.DataFrame(movies_data)
        
        ratings_df.to_csv(os.path.join(data_dir, "ratings.csv"), index=False)
        movies_df.to_csv(os.path.join(data_dir, "movies.csv"), index=False)
        
        self.data_dir = data_dir
    
    def test_movielens_dataset_creation(self):
        """Test MovieLensDataset creation."""
        # Load data
        ratings_df = pd.read_csv(os.path.join(self.data_dir, "ratings.csv"))
        movies_df = pd.read_csv(os.path.join(self.data_dir, "movies.csv"))
        
        # Create dataset
        from config.model_config import get_default_config
        config = get_default_config().data
        config.min_interactions_per_user = 1
        config.min_interactions_per_item = 1
        dataset = MovieLensDataset(ratings_df, movies_df, config=config)
        
        # Check basic properties
        self.assertGreater(len(dataset), 0)
        self.assertIsNotNone(dataset.user_to_idx)
        self.assertIsNotNone(dataset.item_to_idx)
        self.assertIsNotNone(dataset.graph_data)
        
        # Check that encoders were fitted
        self.assertEqual(len(dataset.user_encoder.classes_), len(dataset.user_to_idx))
        self.assertEqual(len(dataset.item_encoder.classes_), len(dataset.item_to_idx))
        self.assertEqual(len(dataset.item_to_idx), len(set(ratings_df['movieId'])))
        self.assertEqual(len(dataset.user_to_idx), len(set(ratings_df['userId'])))
    
    def test_movielens_dataset_getitem(self):
        """Test MovieLensDataset __getitem__ method."""
        # Load data
        ratings_df = pd.read_csv(os.path.join(self.data_dir, "ratings.csv"))
        movies_df = pd.read_csv(os.path.join(self.data_dir, "movies.csv"))
        
        # Create dataset
        from config.model_config import get_default_config
        config = get_default_config().data
        config.min_interactions_per_user = 1
        config.min_interactions_per_item = 1
        dataset = MovieLensDataset(ratings_df, movies_df, config=config)
        
        # Get a sample
        sample = dataset[0]
        
        # Check sample structure
        self.assertIn('user_idx', sample)
        self.assertIn('pos_item_idx', sample)
        self.assertIn('neg_item_idx', sample)
        
        # Check data types
        self.assertIsInstance(sample['user_idx'], torch.Tensor)
        self.assertIsInstance(sample['pos_item_idx'], torch.Tensor)
        self.assertIsInstance(sample['neg_item_idx'], torch.Tensor)
    
    def test_movielens_dataset_graph_data(self):
        """Test PyTorch Geometric graph data creation."""
        # Load data
        ratings_df = pd.read_csv(os.path.join(self.data_dir, "ratings.csv"))
        movies_df = pd.read_csv(os.path.join(self.data_dir, "movies.csv"))
        
        # Create dataset
        from config.model_config import get_default_config
        config = get_default_config().data
        config.min_interactions_per_user = 1
        config.min_interactions_per_item = 1
        dataset = MovieLensDataset(ratings_df, movies_df, config=config)
        
        # Get graph data
        graph_data = dataset.get_graph_data()
        
        # Check graph data properties
        self.assertIsNotNone(graph_data.x)
        self.assertIsNotNone(graph_data.edge_index)
        self.assertIsNotNone(graph_data.num_users)
        self.assertIsNotNone(graph_data.num_items)
        
        # Check dimensions
        expected_nodes = len(dataset.user_to_idx) + len(dataset.item_to_idx)
        self.assertEqual(graph_data.x.shape[0], expected_nodes)
        self.assertEqual(graph_data.edge_index.shape[0], 2)  # 2 rows for edge indices
        self.assertEqual(len(dataset.item_to_idx), len(set(ratings_df['movieId'])))
        self.assertEqual(len(dataset.user_to_idx), len(set(ratings_df['userId'])))


class TestTrainModelFunction(unittest.TestCase):
    """Test train_model convenience function."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = get_fast_config()
        self.config.lightgcn.num_epochs = 1  # Very short training for testing
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('models.training_pipeline.TrainingPipeline')
    def test_train_model_function(self, mock_pipeline_class):
        """Test train_model function."""
        # Mock the training pipeline
        mock_pipeline = Mock()
        mock_pipeline.train.return_value = {
            'best_metrics': {'val_loss': 0.5},
            'training_history': [],
            'total_time': 10.0
        }
        mock_pipeline.evaluate.return_value = {
            'test_loss': 0.4,
            'test_ndcg@10': 0.6
        }
        
        mock_pipeline_class.return_value = mock_pipeline
        
        # Call train_model
        results = train_model(
            config=self.config,
            data_dir="mock_data_dir",
            output_dir=self.temp_dir
        )
        
        # Check that pipeline was created and used
        mock_pipeline_class.assert_called_once()
        mock_pipeline.train.assert_called_once()
        mock_pipeline.evaluate.assert_called_once()
        
        # Check results structure
        self.assertIn('best_metrics', results)
        self.assertIn('training_history', results)
        self.assertIn('total_time', results)
        self.assertIn('test_metrics', results)


if __name__ == '__main__':
    unittest.main() 