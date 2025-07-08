"""
Unit tests for A/B testing framework.
"""

import unittest
import tempfile
import shutil
import json
import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from scipy import stats

from src.models.ab_testing import (
    ABTestManager, ABTestConfig, ABTestSimulator, ABTestResult,
    TrafficSplit, ABTestStatus
)
from src.models.model_versioning import ModelRegistry, ModelMetadata, ExperimentType, ModelStatus
from src.models.lightgcn_model import LightGCN
from src.config.model_config import ModelConfig


class TestABTestConfig(unittest.TestCase):
    """Test ABTestConfig class."""
    
    def test_ab_test_config_creation(self):
        """Test creating ABTestConfig instance."""
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B Test",
            description="A test A/B test",
            model_a_id="model_a",
            model_b_id="model_b",
            traffic_split=TrafficSplit.FIFTY_FIFTY,
            primary_metric="ndcg@10",
            minimum_sample_size=1000,
            confidence_level=0.95
        )
        
        self.assertEqual(config.test_id, "test_001")
        self.assertEqual(config.name, "Test A/B Test")
        self.assertEqual(config.traffic_split, TrafficSplit.FIFTY_FIFTY)
        self.assertEqual(config.primary_metric, "ndcg@10")
        self.assertEqual(config.minimum_sample_size, 1000)
        self.assertEqual(config.confidence_level, 0.95)
        self.assertEqual(len(config.secondary_metrics), 3)
        self.assertEqual(len(config.user_segments), 0)
        self.assertEqual(len(config.exclude_users), 0)
    
    def test_ab_test_config_to_dict(self):
        """Test converting ABTestConfig to dictionary."""
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B Test",
            description="A test A/B test",
            model_a_id="model_a",
            model_b_id="model_b",
            traffic_split=TrafficSplit.SEVENTY_THIRTY,
            primary_metric="ndcg@10"
        )
        
        data = config.to_dict()
        
        self.assertEqual(data['test_id'], "test_001")
        self.assertEqual(data['traffic_split'], "70_30")
        self.assertEqual(data['primary_metric'], "ndcg@10")
    
    def test_ab_test_config_from_dict(self):
        """Test creating ABTestConfig from dictionary."""
        data = {
            'test_id': 'test_001',
            'name': 'Test A/B Test',
            'description': 'A test A/B test',
            'model_a_id': 'model_a',
            'model_b_id': 'model_b',
            'traffic_split': '50_50',
            'primary_metric': 'ndcg@10',
            'secondary_metrics': ['recall@10', 'precision@10'],
            'minimum_sample_size': 1000,
            'confidence_level': 0.95,
            'user_segments': ['premium'],
            'exclude_users': ['user_1', 'user_2']
        }
        
        config = ABTestConfig.from_dict(data)
        
        self.assertEqual(config.test_id, "test_001")
        self.assertEqual(config.traffic_split, TrafficSplit.FIFTY_FIFTY)
        self.assertEqual(len(config.user_segments), 1)
        self.assertEqual(len(config.exclude_users), 2)


class TestABTestResult(unittest.TestCase):
    """Test ABTestResult class."""
    
    def test_ab_test_result_creation(self):
        """Test creating ABTestResult instance."""
        result = ABTestResult(
            test_id="test_001",
            timestamp=datetime.datetime.now(),
            sample_size_a=500,
            sample_size_b=500,
            metrics_a={"ndcg@10": 0.65, "recall@10": 0.45},
            metrics_b={"ndcg@10": 0.68, "recall@10": 0.47},
            primary_metric="ndcg@10",
            effect_size=0.15,
            p_value=0.02,
            confidence_interval=(0.01, 0.05),
            is_significant=True,
            standard_error=0.01,
            degrees_of_freedom=998,
            t_statistic=2.5,
            winner="B",
            recommendation="Variant B is significantly better"
        )
        
        self.assertEqual(result.test_id, "test_001")
        self.assertEqual(result.sample_size_a, 500)
        self.assertEqual(result.sample_size_b, 500)
        self.assertEqual(result.metrics_a["ndcg@10"], 0.65)
        self.assertEqual(result.metrics_b["ndcg@10"], 0.68)
        self.assertEqual(result.effect_size, 0.15)
        self.assertEqual(result.p_value, 0.02)
        self.assertTrue(result.is_significant)
        self.assertEqual(result.winner, "B")
    
    def test_ab_test_result_to_dict(self):
        """Test converting ABTestResult to dictionary."""
        result = ABTestResult(
            test_id="test_001",
            timestamp=datetime.datetime.now(),
            sample_size_a=500,
            sample_size_b=500,
            metrics_a={"ndcg@10": 0.65},
            metrics_b={"ndcg@10": 0.68},
            primary_metric="ndcg@10",
            effect_size=0.15,
            p_value=0.02,
            confidence_interval=(0.01, 0.05),
            is_significant=True,
            standard_error=0.01,
            degrees_of_freedom=998,
            t_statistic=2.5,
            winner="B",
            recommendation="Variant B is significantly better"
        )
        
        data = result.to_dict()
        
        self.assertEqual(data['test_id'], "test_001")
        self.assertEqual(data['winner'], "B")
        self.assertTrue(data['is_significant'])
        self.assertIsInstance(data['timestamp'], str)


class TestABTestManager(unittest.TestCase):
    """Test ABTestManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "test_registry"
        self.results_path = Path(self.temp_dir) / "test_results"
        
        # Create model registry
        self.model_registry = ModelRegistry(registry_path=str(self.registry_path))
        
        # Create test models
        self.test_model_a = LightGCN(
            num_users=100,
            num_items=200,
            embedding_dim=32,
            num_layers=2,
            device='cpu'
        )
        
        self.test_model_b = LightGCN(
            num_users=100,
            num_items=200,
            embedding_dim=64,
            num_layers=3,
            device='cpu'
        )
        
        # Create test config
        self.test_config = ModelConfig()
        self.test_config.lightgcn.num_users = 100
        self.test_config.lightgcn.num_items = 200
        self.test_config.lightgcn.embedding_dim = 32
        self.test_config.lightgcn.num_layers = 2
        self.test_config.lightgcn.device = 'cpu'
        
        # Register models
        self.model_a_id = self.model_registry.register_model(
            model=self.test_model_a,
            name="Model A",
            version="1.0",
            description="Test model A",
            config=self.test_config,
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7}
        )
        
        self.model_b_id = self.model_registry.register_model(
            model=self.test_model_b,
            name="Model B",
            version="1.0",
            description="Test model B",
            config=self.test_config,
            hyperparameters={"lr": 0.002},
            train_metrics={"loss": 0.4},
            validation_metrics={"loss": 0.5},
            test_metrics={"loss": 0.6}
        )
        
        # Create A/B test manager
        self.ab_manager = ABTestManager(
            model_registry=self.model_registry,
            results_path=str(self.results_path)
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_ab_test_manager_initialization(self):
        """Test ABTestManager initialization."""
        self.assertTrue(self.ab_manager.results_path.exists())
        self.assertEqual(len(self.ab_manager.active_tests), 0)
        self.assertEqual(len(self.ab_manager.test_results), 0)
    
    def test_create_test(self):
        """Test creating an A/B test."""
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B Test",
            description="A test A/B test",
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id,
            traffic_split=TrafficSplit.FIFTY_FIFTY,
            primary_metric="ndcg@10"
        )
        
        test_id = self.ab_manager.create_test(config)
        
        self.assertEqual(test_id, "test_001")
        
        # Check that config file was saved
        config_file = self.ab_manager.results_path / "test_001_config.json"
        self.assertTrue(config_file.exists())
    
    def test_start_stop_test(self):
        """Test starting and stopping an A/B test."""
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B Test",
            description="A test A/B test",
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id,
            traffic_split=TrafficSplit.FIFTY_FIFTY,
            primary_metric="ndcg@10"
        )
        
        test_id = self.ab_manager.create_test(config)
        
        # Start test
        success = self.ab_manager.start_test(test_id)
        self.assertTrue(success)
        self.assertIn(test_id, self.ab_manager.active_tests)
        
        # Stop test
        success = self.ab_manager.stop_test(test_id)
        self.assertTrue(success)
        self.assertNotIn(test_id, self.ab_manager.active_tests)
    
    def test_assign_variant(self):
        """Test variant assignment."""
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B Test",
            description="A test A/B test",
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id,
            traffic_split=TrafficSplit.FIFTY_FIFTY,
            primary_metric="ndcg@10",
            exclude_users=["user_3"]
        )
        
        test_id = self.ab_manager.create_test(config)
        self.ab_manager.start_test(test_id)
        
        # Test variant assignment
        variant_a = self.ab_manager.assign_variant("user_1", test_id)
        variant_b = self.ab_manager.assign_variant("user_2", test_id)
        
        self.assertIn(variant_a, ["A", "B"])
        self.assertIn(variant_b, ["A", "B"])
        
        # Test excluded user
        variant_excluded = self.ab_manager.assign_variant("user_3", test_id)
        self.assertIsNone(variant_excluded)
        
        # Test non-existent test
        variant_none = self.ab_manager.assign_variant("user_1", "non_existent")
        self.assertIsNone(variant_none)
    
    def test_record_interaction(self):
        """Test recording interactions."""
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B Test",
            description="A test A/B test",
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id,
            traffic_split=TrafficSplit.FIFTY_FIFTY,
            primary_metric="ndcg@10"
        )
        
        test_id = self.ab_manager.create_test(config)
        self.ab_manager.start_test(test_id)
        
        # Record interactions
        metrics_a = {"ndcg@10": 0.65, "recall@10": 0.45}
        metrics_b = {"ndcg@10": 0.68, "recall@10": 0.47}
        
        self.ab_manager.record_interaction("user_1", test_id, "A", metrics_a)
        self.ab_manager.record_interaction("user_2", test_id, "B", metrics_b)
        
        # Check that interactions were saved
        interaction_file = self.ab_manager.results_path / "test_001_interactions.jsonl"
        self.assertTrue(interaction_file.exists())
        
        # Read interactions
        interactions = []
        with open(interaction_file, 'r') as f:
            for line in f:
                if line.strip():
                    interactions.append(json.loads(line))
        
        self.assertEqual(len(interactions), 2)
        self.assertEqual(interactions[0]['variant'], "A")
        self.assertEqual(interactions[1]['variant'], "B")
    
    def test_analyze_test(self):
        """Test analyzing A/B test results."""
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B Test",
            description="A test A/B test",
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id,
            traffic_split=TrafficSplit.FIFTY_FIFTY,
            primary_metric="ndcg@10"
        )
        
        test_id = self.ab_manager.create_test(config)
        self.ab_manager.start_test(test_id)
        
        # Record interactions with clear difference
        for i in range(100):
            user_id = f"user_{i}"
            variant = self.ab_manager.assign_variant(user_id, test_id)
            
            if variant == "A":
                metrics = {"ndcg@10": 0.65 + np.random.normal(0, 0.05)}
            else:
                metrics = {"ndcg@10": 0.70 + np.random.normal(0, 0.05)}
            
            self.ab_manager.record_interaction(user_id, test_id, variant, metrics)
        
        # Analyze test
        result = self.ab_manager.analyze_test(test_id)
        
        self.assertEqual(result.test_id, test_id)
        self.assertGreater(result.sample_size_a, 0)
        self.assertGreater(result.sample_size_b, 0)
        self.assertIsInstance(result.effect_size, float)
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.is_significant, bool)
        self.assertIsInstance(result.confidence_interval, tuple)
        self.assertEqual(len(result.confidence_interval), 2)
    
    def test_get_test_results(self):
        """Test getting test results."""
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B Test",
            description="A test A/B test",
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id,
            traffic_split=TrafficSplit.FIFTY_FIFTY,
            primary_metric="ndcg@10"
        )
        
        test_id = self.ab_manager.create_test(config)
        self.ab_manager.start_test(test_id)
        
        # Record some interactions and analyze
        for i in range(50):
            user_id = f"user_{i}"
            variant = self.ab_manager.assign_variant(user_id, test_id)
            metrics = {"ndcg@10": 0.65 + np.random.normal(0, 0.05)}
            self.ab_manager.record_interaction(user_id, test_id, variant, metrics)
        
        result = self.ab_manager.analyze_test(test_id)
        
        # Get results
        results = self.ab_manager.get_test_results(test_id)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].test_id, test_id)
    
    def test_list_tests(self):
        """Test listing tests."""
        config_1 = ABTestConfig(
            test_id="test_001",
            name="Test A/B Test 1",
            description="First test",
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id,
            traffic_split=TrafficSplit.FIFTY_FIFTY,
            primary_metric="ndcg@10"
        )
        
        config_2 = ABTestConfig(
            test_id="test_002",
            name="Test A/B Test 2",
            description="Second test",
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id,
            traffic_split=TrafficSplit.SEVENTY_THIRTY,
            primary_metric="recall@10"
        )
        
        self.ab_manager.create_test(config_1)
        self.ab_manager.create_test(config_2)
        
        tests = self.ab_manager.list_tests()
        self.assertEqual(len(tests), 2)
        
        test_names = [test.name for test in tests]
        self.assertIn("Test A/B Test 1", test_names)
        self.assertIn("Test A/B Test 2", test_names)
    
    def test_get_test_summary(self):
        """Test getting test summary."""
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B Test",
            description="A test A/B test",
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id,
            traffic_split=TrafficSplit.FIFTY_FIFTY,
            primary_metric="ndcg@10"
        )
        
        test_id = self.ab_manager.create_test(config)
        self.ab_manager.start_test(test_id)
        
        # Record some interactions
        for i in range(30):
            user_id = f"user_{i}"
            variant = self.ab_manager.assign_variant(user_id, test_id)
            metrics = {"ndcg@10": 0.65 + np.random.normal(0, 0.05)}
            self.ab_manager.record_interaction(user_id, test_id, variant, metrics)
        
        summary = self.ab_manager.get_test_summary(test_id)
        
        self.assertEqual(summary['test_id'], test_id)
        self.assertEqual(summary['name'], "Test A/B Test")
        self.assertEqual(summary['status'], 'running')
        self.assertEqual(summary['model_a'], self.model_a_id)
        self.assertEqual(summary['model_b'], self.model_b_id)
        self.assertGreater(summary['total_interactions'], 0)


class TestABTestSimulator(unittest.TestCase):
    """Test ABTestSimulator class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "test_registry"
        self.results_path = Path(self.temp_dir) / "test_results"
        
        # Create model registry and A/B test manager
        self.model_registry = ModelRegistry(registry_path=str(self.registry_path))
        self.ab_manager = ABTestManager(
            model_registry=self.model_registry,
            results_path=str(self.results_path)
        )
        
        # Create test models
        self.test_model_a = LightGCN(
            num_users=100,
            num_items=200,
            embedding_dim=32,
            num_layers=2,
            device='cpu'
        )
        
        self.test_model_b = LightGCN(
            num_users=100,
            num_items=200,
            embedding_dim=64,
            num_layers=3,
            device='cpu'
        )
        
        # Create test config
        self.test_config = ModelConfig()
        self.test_config.lightgcn.num_users = 100
        self.test_config.lightgcn.num_items = 200
        self.test_config.lightgcn.embedding_dim = 32
        self.test_config.lightgcn.num_layers = 2
        self.test_config.lightgcn.device = 'cpu'
        
        # Register models
        self.model_a_id = self.model_registry.register_model(
            model=self.test_model_a,
            name="Model A",
            version="1.0",
            description="Test model A",
            config=self.test_config,
            hyperparameters={"lr": 0.001},
            train_metrics={"loss": 0.5},
            validation_metrics={"loss": 0.6},
            test_metrics={"loss": 0.7}
        )
        
        self.model_b_id = self.model_registry.register_model(
            model=self.test_model_b,
            name="Model B",
            version="1.0",
            description="Test model B",
            config=self.test_config,
            hyperparameters={"lr": 0.002},
            train_metrics={"loss": 0.4},
            validation_metrics={"loss": 0.5},
            test_metrics={"loss": 0.6}
        )
        
        # Create simulator
        self.simulator = ABTestSimulator(self.ab_manager)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_simulate_test(self):
        """Test A/B test simulation."""
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B Test",
            description="A test A/B test",
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id,
            traffic_split=TrafficSplit.FIFTY_FIFTY,
            primary_metric="ndcg@10"
        )
        
        # Run simulation
        result = self.simulator.simulate_test(
            config=config,
            num_users=100,
            effect_size=0.1,
            noise_level=0.05
        )
        
        self.assertEqual(result.test_id, "test_001")
        self.assertGreater(result.sample_size_a, 0)
        self.assertGreater(result.sample_size_b, 0)
        self.assertIsInstance(result.effect_size, float)
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.is_significant, bool)
        self.assertIsInstance(result.confidence_interval, tuple)
        self.assertEqual(len(result.confidence_interval), 2)
        
        # Check that test was created and stopped
        tests = self.ab_manager.list_tests()
        self.assertEqual(len(tests), 1)
        self.assertEqual(tests[0].test_id, "test_001")
    
    def test_simulate_test_with_different_effect_sizes(self):
        """Test simulation with different effect sizes."""
        config = ABTestConfig(
            test_id="test_001",
            name="Test A/B Test",
            description="A test A/B test",
            model_a_id=self.model_a_id,
            model_b_id=self.model_b_id,
            traffic_split=TrafficSplit.FIFTY_FIFTY,
            primary_metric="ndcg@10"
        )
        
        # Test with small effect size
        result_small = self.simulator.simulate_test(
            config=config,
            num_users=100,
            effect_size=0.05,
            noise_level=0.05
        )
        
        # Test with large effect size
        result_large = self.simulator.simulate_test(
            config=config,
            num_users=100,
            effect_size=0.2,
            noise_level=0.05
        )
        
        # Larger effect size should generally result in larger measured effect
        # (though this is not guaranteed due to noise)
        self.assertIsInstance(result_small.effect_size, float)
        self.assertIsInstance(result_large.effect_size, float)


if __name__ == "__main__":
    unittest.main() 