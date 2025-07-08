"""
Unit tests for model versioning system.
"""

import unittest
import tempfile
import shutil
import json
import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import numpy as np

from src.models.model_versioning import (
    ModelRegistry, ModelMetadata, ModelStatus, ExperimentType
)
from src.models.lightgcn_model import LightGCN
from src.config.model_config import ModelConfig


class TestModelMetadata(unittest.TestCase):
    """Test ModelMetadata class."""
    
    def test_model_metadata_creation(self):
        """Test creating ModelMetadata instance."""
        metadata = ModelMetadata(
            model_id="test_123",
            version="1.0",
            name="Test Model",
            description="A test model",
            training_start_time=datetime.datetime.now(),
            training_end_time=datetime.datetime.now(),
            training_duration=3600.0,
            config={"test": "config"},
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7},
            data_version="1.0",
            data_hash="abc123",
            num_train_samples=1000,
            num_validation_samples=200,
            num_test_samples=300,
            num_users=100,
            num_items=200,
            model_size=10.5,
            num_parameters=1000000,
            status=ModelStatus.TRAINED,
            is_production=False,
            deployment_time=None,
            experiment_id="exp_001",
            experiment_type=ExperimentType.HYPERPARAMETER_TUNING,
            parent_model_id=None,
            tags=["test", "demo"],
            notes="Test notes"
        )
        
        self.assertEqual(metadata.model_id, "test_123")
        self.assertEqual(metadata.version, "1.0")
        self.assertEqual(metadata.name, "Test Model")
        self.assertEqual(metadata.status, ModelStatus.TRAINED)
        self.assertEqual(metadata.experiment_type, ExperimentType.HYPERPARAMETER_TUNING)
        self.assertEqual(len(metadata.tags), 2)
    
    def test_model_metadata_to_dict(self):
        """Test converting ModelMetadata to dictionary."""
        metadata = ModelMetadata(
            model_id="test_123",
            version="1.0",
            name="Test Model",
            description="A test model",
            training_start_time=datetime.datetime.now(),
            training_end_time=datetime.datetime.now(),
            training_duration=3600.0,
            config={"test": "config"},
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7},
            data_version="1.0",
            data_hash="abc123",
            num_train_samples=1000,
            num_validation_samples=200,
            num_test_samples=300,
            num_users=100,
            num_items=200,
            model_size=10.5,
            num_parameters=1000000,
            status=ModelStatus.TRAINED,
            is_production=False,
            deployment_time=None,
            experiment_id="exp_001",
            experiment_type=ExperimentType.HYPERPARAMETER_TUNING,
            parent_model_id=None,
            tags=["test", "demo"],
            notes="Test notes"
        )
        
        data = metadata.to_dict()
        
        self.assertEqual(data['model_id'], "test_123")
        self.assertEqual(data['status'], "trained")
        self.assertEqual(data['experiment_type'], "hyperparameter_tuning")
        self.assertIsInstance(data['tags'], list)
    
    def test_model_metadata_from_dict(self):
        """Test creating ModelMetadata from dictionary."""
        data = {
            'model_id': 'test_123',
            'version': '1.0',
            'name': 'Test Model',
            'description': 'A test model',
            'training_start_time': datetime.datetime.now(),
            'training_end_time': datetime.datetime.now(),
            'training_duration': 3600.0,
            'config': {'test': 'config'},
            'hyperparameters': {'lr': 0.001},
            'train_metrics': {'loss': 0.5},
            'validation_metrics': {'loss': 0.6},
            'test_metrics': {'loss': 0.7},
            'data_version': '1.0',
            'data_hash': 'abc123',
            'num_train_samples': 1000,
            'num_validation_samples': 200,
            'num_test_samples': 300,
            'num_users': 100,
            'num_items': 200,
            'model_size': 10.5,
            'num_parameters': 1000000,
            'status': 'trained',
            'is_production': False,
            'deployment_time': None,
            'experiment_id': 'exp_001',
            'experiment_type': 'hyperparameter_tuning',
            'parent_model_id': None,
            'tags': ['test', 'demo'],
            'notes': 'Test notes'
        }
        
        metadata = ModelMetadata.from_dict(data)
        
        self.assertEqual(metadata.model_id, "test_123")
        self.assertEqual(metadata.status, ModelStatus.TRAINED)
        self.assertEqual(metadata.experiment_type, ExperimentType.HYPERPARAMETER_TUNING)


class TestModelRegistry(unittest.TestCase):
    """Test ModelRegistry class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "test_registry"
        self.registry = ModelRegistry(registry_path=str(self.registry_path))
        
        # Create a simple test model
        self.test_model = LightGCN(
            num_users=100,
            num_items=200,
            embedding_dim=32,
            num_layers=2,
            device='cpu'
        )
        
        # Create test config
        self.test_config = ModelConfig()
        self.test_config.lightgcn.num_users = 100
        self.test_config.lightgcn.num_items = 200
        self.test_config.lightgcn.embedding_dim = 32
        self.test_config.lightgcn.num_layers = 2
        self.test_config.lightgcn.device = 'cpu'
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_registry_initialization(self):
        """Test ModelRegistry initialization."""
        self.assertTrue(self.registry.registry_path.exists())
        self.assertTrue(self.registry.models_path.exists())
        self.assertTrue(self.registry.metadata_path.exists())
        self.assertTrue(self.registry.experiments_path.exists())
        self.assertTrue(self.registry.db_path.exists())
    
    def test_register_model(self):
        """Test registering a model."""
        model_id = self.registry.register_model(
            model=self.test_model,
            name="Test Model",
            version="1.0",
            description="A test model",
            config=self.test_config,
            hyperparameters={"lr": 0.001, "batch_size": 1024},
            train_metrics={"loss": 0.5, "ndcg@10": 0.65},
            validation_metrics={"loss": 0.6, "ndcg@10": 0.63},
            test_metrics={"ndcg@10": 0.62, "recall@10": 0.45},
            experiment_id="exp_001",
            experiment_type=ExperimentType.HYPERPARAMETER_TUNING,
            tags=["test", "lightgcn"]
        )
        
        self.assertIsInstance(model_id, str)
        self.assertEqual(len(model_id), 12)  # MD5 hash truncated to 12 chars
        
        # Check that model file was saved
        model_dir = self.registry.models_path / model_id
        self.assertTrue(model_dir.exists())
        self.assertTrue((model_dir / "model_1.0.pt").exists())
        
        # Check that metadata was saved
        metadata_dir = self.registry.metadata_path / model_id
        self.assertTrue(metadata_dir.exists())
        self.assertTrue((metadata_dir / "metadata_1.0.json").exists())
    
    def test_get_model_metadata(self):
        """Test retrieving model metadata."""
        model_id = self.registry.register_model(
            model=self.test_model,
            name="Test Model",
            version="1.0",
            description="A test model",
            config=self.test_config,
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7},
            experiment_id="exp_001",
            experiment_type=ExperimentType.HYPERPARAMETER_TUNING
        )
        
        metadata = self.registry.get_metadata(model_id)
        
        self.assertEqual(metadata.model_id, model_id)
        self.assertEqual(metadata.name, "Test Model")
        self.assertEqual(metadata.version, "1.0")
        self.assertEqual(metadata.status, ModelStatus.TRAINED)
        self.assertEqual(metadata.experiment_type, ExperimentType.HYPERPARAMETER_TUNING)
        self.assertEqual(metadata.train_metrics["loss"], 0.5)
        self.assertEqual(metadata.validation_metrics["loss"], 0.6)
        self.assertEqual(metadata.test_metrics["loss"], 0.7)
    
    def test_get_model(self):
        """Test retrieving a model."""
        model_id = self.registry.register_model(
            model=self.test_model,
            name="Test Model",
            version="1.0",
            description="A test model",
            config=self.test_config,
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7}
        )
        
        model, metadata = self.registry.get_model(model_id)
        
        self.assertIsInstance(model, LightGCN)
        self.assertEqual(metadata.model_id, model_id)
        self.assertEqual(metadata.name, "Test Model")
        
        # Check that model parameters match
        self.assertEqual(model.num_users, 100)
        self.assertEqual(model.num_items, 200)
        self.assertEqual(model.embedding_dim, 32)
        self.assertEqual(model.num_layers, 2)
    
    def test_list_models(self):
        """Test listing models."""
        # Register multiple models
        model_id_1 = self.registry.register_model(
            model=self.test_model,
            name="Model 1",
            version="1.0",
            description="First model",
            config=self.test_config,
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7},
            experiment_id="exp_001"
        )
        
        model_id_2 = self.registry.register_model(
            model=self.test_model,
            name="Model 2",
            version="1.0",
            description="Second model",
            config=self.test_config,
            hyperparameters={"lr": 0.002},
            train_metrics={"loss": 0.4},
            validation_metrics={"loss": 0.5},
            test_metrics={"loss": 0.6},
            experiment_id="exp_002"
        )
        
        # List all models
        models = self.registry.list_models()
        self.assertEqual(len(models), 2)
        
        # Filter by experiment
        exp_models = self.registry.list_models(experiment_id="exp_001")
        self.assertEqual(len(exp_models), 1)
        self.assertEqual(exp_models[0].experiment_id, "exp_001")
        
        # Filter by name
        named_models = self.registry.list_models(name="Model 1")
        self.assertEqual(len(named_models), 1)
        self.assertEqual(named_models[0].name, "Model 1")
    
    def test_update_model_status(self):
        """Test updating model status."""
        model_id = self.registry.register_model(
            model=self.test_model,
            name="Test Model",
            version="1.0",
            description="A test model",
            config=self.test_config,
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7}
        )
        
        # Update status
        self.registry.update_model_status(model_id, ModelStatus.VALIDATED)
        
        metadata = self.registry.get_metadata(model_id)
        self.assertEqual(metadata.status, ModelStatus.VALIDATED)
    
    def test_deploy_model(self):
        """Test deploying a model."""
        model_id = self.registry.register_model(
            model=self.test_model,
            name="Test Model",
            version="1.0",
            description="A test model",
            config=self.test_config,
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7}
        )
        
        # Deploy model
        self.registry.deploy_model(model_id)
        
        metadata = self.registry.get_metadata(model_id)
        self.assertEqual(metadata.status, ModelStatus.DEPLOYED)
        self.assertTrue(metadata.is_production)
        self.assertIsNotNone(metadata.deployment_time)
        
        # Check that only one model is production
        production_models = self.registry.list_models(is_production=True)
        self.assertEqual(len(production_models), 1)
        self.assertEqual(production_models[0].model_id, model_id)
    
    def test_archive_model(self):
        """Test archiving a model."""
        model_id = self.registry.register_model(
            model=self.test_model,
            name="Test Model",
            version="1.0",
            description="A test model",
            config=self.test_config,
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7}
        )
        
        # Archive model
        self.registry.archive_model(model_id)
        
        metadata = self.registry.get_metadata(model_id)
        self.assertEqual(metadata.status, ModelStatus.ARCHIVED)
    
    def test_delete_model(self):
        """Test deleting a model."""
        model_id = self.registry.register_model(
            model=self.test_model,
            name="Test Model",
            version="1.0",
            description="A test model",
            config=self.test_config,
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7}
        )
        
        # Verify model exists
        metadata = self.registry.get_metadata(model_id)
        self.assertIsNotNone(metadata)
        
        # Delete model
        self.registry.delete_model(model_id)
        
        # Verify model is deleted
        with self.assertRaises(ValueError):
            self.registry.get_metadata(model_id)
    
    def test_model_size_calculation(self):
        """Test model size calculation."""
        model_id = self.registry.register_model(
            model=self.test_model,
            name="Test Model",
            version="1.0",
            description="A test model",
            config=self.test_config,
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7}
        )
        
        metadata = self.registry.get_metadata(model_id)
        
        # Check that model size is calculated
        self.assertIsNotNone(metadata.model_size)
        self.assertGreater(metadata.model_size, 0)
        
        # Check that parameter count is calculated
        self.assertIsNotNone(metadata.num_parameters)
        self.assertGreater(metadata.num_parameters, 0)
    
    def test_multiple_versions(self):
        """Test handling multiple model versions."""
        # Register first version
        model_id = self.registry.register_model(
            model=self.test_model,
            name="Test Model",
            version="1.0",
            description="First version",
            config=self.test_config,
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7}
        )
        
        # Register second version
        model_id_2 = self.registry.register_model(
            model=self.test_model,
            name="Test Model",
            version="2.0",
            description="Second version",
            config=self.test_config,
            hyperparameters={"lr": 0.002},
            train_metrics={"loss": 0.4},
            validation_metrics={"loss": 0.5},
            test_metrics={"loss": 0.6}
        )
        
        # Get latest version
        metadata_latest = self.registry.get_metadata(model_id_2)
        self.assertEqual(metadata_latest.version, "2.0")
        
        # Get specific version
        metadata_v1 = self.registry.get_metadata(model_id, version="1.0")
        self.assertEqual(metadata_v1.version, "1.0")
        
        # Get model with specific version
        model_v1, metadata_v1 = self.registry.get_model(model_id, version="1.0")
        self.assertEqual(metadata_v1.version, "1.0")


if __name__ == "__main__":
    unittest.main() 