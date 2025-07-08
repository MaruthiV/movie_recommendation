"""
A/B testing framework for model comparison and evaluation.
"""

import uuid
import json
import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import random
import numpy as np
import pandas as pd
from scipy import stats
from enum import Enum

from .model_versioning import ModelRegistry, ModelMetadata, ExperimentType


class ABTestStatus(Enum):
    """A/B test status enumeration."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class TrafficSplit(Enum):
    """Traffic split enumeration."""
    FIFTY_FIFTY = "50_50"
    SEVENTY_THIRTY = "70_30"
    EIGHTY_TWENTY = "80_20"
    NINETY_TEN = "90_10"


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    
    # Basic information
    test_id: str
    name: str
    description: str
    
    # Model information
    model_a_id: str
    model_b_id: str
    traffic_split: TrafficSplit
    
    # Optional model versions
    model_a_version: Optional[str] = None
    model_b_version: Optional[str] = None
    
    # Traffic configuration
    total_traffic_percentage: float = 100.0  # Percentage of total traffic to use
    
    # Test parameters
    primary_metric: str = "ndcg@10"
    secondary_metrics: List[str] = None
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    statistical_power: float = 0.8
    
    # Duration and stopping criteria
    max_duration_days: int = 30
    early_stopping_enabled: bool = True
    min_detection_effect: float = 0.05  # Minimum effect size to detect
    
    # User targeting
    user_segments: List[str] = None
    exclude_users: List[str] = None
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = ["recall@10", "precision@10", "click_through_rate"]
        if self.user_segments is None:
            self.user_segments = []
        if self.exclude_users is None:
            self.exclude_users = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['traffic_split'] = self.traffic_split.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABTestConfig':
        """Create from dictionary."""
        data['traffic_split'] = TrafficSplit(data['traffic_split'])
        return cls(**data)


@dataclass
class ABTestResult:
    """A/B test result."""
    
    test_id: str
    timestamp: datetime.datetime
    
    # Sample sizes
    sample_size_a: int
    sample_size_b: int
    
    # Metrics for each variant
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    
    # Statistical analysis
    primary_metric: str
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    
    # Additional statistics
    standard_error: float
    degrees_of_freedom: int
    t_statistic: float
    
    # Recommendations
    winner: Optional[str]  # "A", "B", or None
    recommendation: str
    
    def __post_init__(self):
        # Ensure is_significant is always a native Python bool
        self.is_significant = bool(self.is_significant)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        # Cast numpy types to native Python types for JSON serialization
        for k, v in data.items():
            if isinstance(v, (np.generic, np.bool_)):
                data[k] = v.item()
            elif isinstance(v, tuple):
                data[k] = tuple(float(x) if isinstance(x, (np.generic, np.floating)) else x for x in v)
            elif isinstance(v, dict):
                data[k] = {kk: float(vv) if isinstance(vv, (np.generic, np.floating)) else vv for kk, vv in v.items()}
        return data


class ABTestManager:
    """
    A/B test manager for running and analyzing model comparison experiments.
    """
    
    def __init__(self, model_registry: ModelRegistry, results_path: str = "ab_test_results"):
        """
        Initialize A/B test manager.
        
        Args:
            model_registry: Model registry instance
            results_path: Path to store A/B test results
        """
        self.model_registry = model_registry
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Store active tests
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, List[ABTestResult]] = {}
    
    def create_test(self, config: ABTestConfig) -> str:
        """
        Create a new A/B test.
        
        Args:
            config: A/B test configuration
            
        Returns:
            Test ID
        """
        # Validate models exist
        try:
            self.model_registry.get_metadata(config.model_a_id, config.model_a_version)
            self.model_registry.get_metadata(config.model_b_id, config.model_b_version)
        except ValueError as e:
            raise ValueError(f"Invalid model in A/B test config: {e}")
        
        # Generate test ID if not provided
        if not config.test_id:
            config.test_id = str(uuid.uuid4())
        
        # Save test configuration
        test_file = self.results_path / f"{config.test_id}_config.json"
        with open(test_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        self.logger.info(f"Created A/B test {config.test_id}: {config.name}")
        
        return config.test_id
    
    def start_test(self, test_id: str) -> bool:
        """
        Start an A/B test.
        
        Args:
            test_id: Test ID
            
        Returns:
            True if test started successfully
        """
        config = self._load_test_config(test_id)
        if not config:
            raise ValueError(f"Test configuration not found: {test_id}")
        
        # Add to active tests
        self.active_tests[test_id] = config
        self.test_results[test_id] = []
        
        self.logger.info(f"Started A/B test {test_id}")
        return True
    
    def stop_test(self, test_id: str) -> bool:
        """
        Stop an A/B test.
        
        Args:
            test_id: Test ID
            
        Returns:
            True if test stopped successfully
        """
        if test_id in self.active_tests:
            del self.active_tests[test_id]
            self.logger.info(f"Stopped A/B test {test_id}")
            return True
        return False
    
    def assign_variant(self, user_id: str, test_id: str) -> Optional[str]:
        """
        Assign a user to a test variant (A or B).
        
        Args:
            user_id: User ID
            test_id: Test ID
            
        Returns:
            Variant ("A" or "B") or None if not in test
        """
        if test_id not in self.active_tests:
            return None
        
        config = self.active_tests[test_id]
        
        # Check if user should be excluded
        if user_id in config.exclude_users:
            return None
        
        # Check user segments (if specified)
        if config.user_segments and not self._user_in_segments(user_id, config.user_segments):
            return None
        
        # Determine traffic split
        split_ratio = self._get_split_ratio(config.traffic_split)
        
        # Use user ID hash for consistent assignment
        user_hash = hash(user_id) % 100
        
        if user_hash < split_ratio:
            return "A"
        else:
            return "B"
    
    def _get_split_ratio(self, traffic_split: TrafficSplit) -> int:
        """Get traffic split ratio for variant A."""
        if traffic_split == TrafficSplit.FIFTY_FIFTY:
            return 50
        elif traffic_split == TrafficSplit.SEVENTY_THIRTY:
            return 70
        elif traffic_split == TrafficSplit.EIGHTY_TWENTY:
            return 80
        elif traffic_split == TrafficSplit.NINETY_TEN:
            return 90
        else:
            return 50
    
    def _user_in_segments(self, user_id: str, segments: List[str]) -> bool:
        """Check if user belongs to specified segments."""
        # This is a placeholder implementation
        # In a real system, you would check user attributes against segments
        return True
    
    def record_interaction(self, 
                          user_id: str, 
                          test_id: str, 
                          variant: str,
                          metrics: Dict[str, float],
                          timestamp: Optional[datetime.datetime] = None):
        """
        Record user interaction for A/B test.
        
        Args:
            user_id: User ID
            test_id: Test ID
            variant: Variant ("A" or "B")
            metrics: Interaction metrics
            timestamp: Interaction timestamp
        """
        if test_id not in self.active_tests:
            return
        
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        # Store interaction data
        interaction_data = {
            'user_id': user_id,
            'test_id': test_id,
            'variant': variant,
            'metrics': metrics,
            'timestamp': timestamp.isoformat()
        }
        
        interaction_file = self.results_path / f"{test_id}_interactions.jsonl"
        with open(interaction_file, 'a') as f:
            f.write(json.dumps(interaction_data) + '\n')
    
    def analyze_test(self, test_id: str) -> ABTestResult:
        """
        Analyze A/B test results.
        
        Args:
            test_id: Test ID
            
        Returns:
            A/B test result
        """
        # Load interaction data
        interactions = self._load_interactions(test_id)
        if not interactions:
            raise ValueError(f"No interaction data found for test {test_id}")
        
        config = self.active_tests.get(test_id) or self._load_test_config(test_id)
        if not config:
            raise ValueError(f"Test configuration not found: {test_id}")
        
        # Separate data by variant
        variant_a_data = [i for i in interactions if i['variant'] == 'A']
        variant_b_data = [i for i in interactions if i['variant'] == 'B']
        
        if not variant_a_data or not variant_b_data:
            raise ValueError(f"Insufficient data for analysis: A={len(variant_a_data)}, B={len(variant_b_data)}")
        
        # Calculate metrics for each variant
        metrics_a = self._calculate_aggregate_metrics(variant_a_data)
        metrics_b = self._calculate_aggregate_metrics(variant_b_data)
        
        # Perform statistical analysis
        primary_metric_a = [i['metrics'].get(config.primary_metric, 0) for i in variant_a_data]
        primary_metric_b = [i['metrics'].get(config.primary_metric, 0) for i in variant_b_data]
        
        # T-test for statistical significance
        t_stat, p_value = stats.ttest_ind(primary_metric_a, primary_metric_b)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(primary_metric_a) - 1) * np.var(primary_metric_a, ddof=1) + 
                             (len(primary_metric_b) - 1) * np.var(primary_metric_b, ddof=1)) / 
                            (len(primary_metric_a) + len(primary_metric_b) - 2))
        
        effect_size = (np.mean(primary_metric_b) - np.mean(primary_metric_a)) / pooled_std
        
        # Calculate confidence interval
        mean_diff = np.mean(primary_metric_b) - np.mean(primary_metric_a)
        se_diff = np.sqrt(np.var(primary_metric_a) / len(primary_metric_a) + 
                         np.var(primary_metric_b) / len(primary_metric_b))
        
        confidence_interval = stats.t.interval(
            config.confidence_level, 
            len(primary_metric_a) + len(primary_metric_b) - 2,
            loc=mean_diff,
            scale=se_diff
        )
        
        # Determine significance
        is_significant = p_value < (1 - config.confidence_level)
        
        # Determine winner
        winner = None
        if is_significant:
            if np.mean(primary_metric_b) > np.mean(primary_metric_a):
                winner = "B"
            else:
                winner = "A"
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            is_significant, winner, effect_size, p_value, config
        )
        
        # Create result
        result = ABTestResult(
            test_id=test_id,
            timestamp=datetime.datetime.now(),
            sample_size_a=len(variant_a_data),
            sample_size_b=len(variant_b_data),
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            primary_metric=config.primary_metric,
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=confidence_interval,
            is_significant=is_significant,
            standard_error=se_diff,
            degrees_of_freedom=len(primary_metric_a) + len(primary_metric_b) - 2,
            t_statistic=t_stat,
            winner=winner,
            recommendation=recommendation
        )
        
        # Store result
        self.test_results[test_id].append(result)
        self._save_result(test_id, result)
        
        return result
    
    def _calculate_aggregate_metrics(self, interactions: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate metrics from interactions."""
        if not interactions:
            return {}
        
        # Get all metric keys
        all_metrics = set()
        for interaction in interactions:
            all_metrics.update(interaction['metrics'].keys())
        
        # Calculate means for each metric
        aggregate_metrics = {}
        for metric in all_metrics:
            values = [i['metrics'].get(metric, 0) for i in interactions]
            aggregate_metrics[metric] = np.mean(values)
        
        return aggregate_metrics
    
    def _generate_recommendation(self, 
                               is_significant: bool, 
                               winner: Optional[str], 
                               effect_size: float,
                               p_value: float,
                               config: ABTestConfig) -> str:
        """Generate recommendation based on test results."""
        if not is_significant:
            return "No significant difference detected. Continue monitoring or increase sample size."
        
        if winner == "B":
            if effect_size > config.min_detection_effect:
                return f"Variant B is significantly better (effect size: {effect_size:.3f}). Consider deploying B."
            else:
                return f"Variant B is better but effect size ({effect_size:.3f}) is below minimum threshold."
        elif winner == "A":
            if effect_size > config.min_detection_effect:
                return f"Variant A is significantly better (effect size: {effect_size:.3f}). Keep current model."
            else:
                return f"Variant A is better but effect size ({effect_size:.3f}) is below minimum threshold."
        else:
            return "No clear winner despite statistical significance."
    
    def _load_interactions(self, test_id: str) -> List[Dict]:
        """Load interaction data for a test."""
        interaction_file = self.results_path / f"{test_id}_interactions.jsonl"
        
        if not interaction_file.exists():
            return []
        
        interactions = []
        with open(interaction_file, 'r') as f:
            for line in f:
                if line.strip():
                    interactions.append(json.loads(line))
        
        return interactions
    
    def _load_test_config(self, test_id: str) -> Optional[ABTestConfig]:
        """Load test configuration."""
        config_file = self.results_path / f"{test_id}_config.json"
        
        if not config_file.exists():
            return None
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        return ABTestConfig.from_dict(config_data)
    
    def _save_result(self, test_id: str, result: ABTestResult):
        """Save test result."""
        result_file = self.results_path / f"{test_id}_results.jsonl"
        
        with open(result_file, 'a') as f:
            f.write(json.dumps(result.to_dict()) + '\n')
    
    def get_test_results(self, test_id: str) -> List[ABTestResult]:
        """Get all results for a test."""
        if test_id in self.test_results:
            return self.test_results[test_id]
        
        # Load from file
        result_file = self.results_path / f"{test_id}_results.jsonl"
        
        if not result_file.exists():
            return []
        
        results = []
        with open(result_file, 'r') as f:
            for line in f:
                if line.strip():
                    result_data = json.loads(line)
                    # Convert timestamp back to datetime
                    result_data['timestamp'] = datetime.datetime.fromisoformat(result_data['timestamp'])
                    results.append(ABTestResult(**result_data))
        
        return results
    
    def list_tests(self) -> List[ABTestConfig]:
        """List all A/B tests."""
        tests = []
        
        for config_file in self.results_path.glob("*_config.json"):
            test_id = config_file.stem.replace("_config", "")
            config = self._load_test_config(test_id)
            if config:
                tests.append(config)
        
        return tests
    
    def get_test_summary(self, test_id: str) -> Dict[str, Any]:
        """Get summary of A/B test."""
        config = self._load_test_config(test_id)
        if not config:
            return {}
        
        interactions = self._load_interactions(test_id)
        results = self.get_test_results(test_id)
        
        # Calculate summary statistics
        variant_a_count = len([i for i in interactions if i['variant'] == 'A'])
        variant_b_count = len([i for i in interactions if i['variant'] == 'B'])
        
        latest_result = results[-1] if results else None
        
        summary = {
            'test_id': test_id,
            'name': config.name,
            'status': 'running' if test_id in self.active_tests else 'stopped',
            'model_a': config.model_a_id,
            'model_b': config.model_b_id,
            'traffic_split': config.traffic_split.value,
            'total_interactions': len(interactions),
            'variant_a_interactions': variant_a_count,
            'variant_b_interactions': variant_b_count,
            'latest_result': latest_result.to_dict() if latest_result else None
        }
        
        return summary


class ABTestSimulator:
    """
    A/B test simulator for testing and validation.
    """
    
    def __init__(self, ab_manager: ABTestManager):
        """
        Initialize A/B test simulator.
        
        Args:
            ab_manager: A/B test manager
        """
        self.ab_manager = ab_manager
        self.logger = logging.getLogger(__name__)
    
    def simulate_test(self, 
                     config: ABTestConfig,
                     num_users: int = 1000,
                     effect_size: float = 0.1,
                     noise_level: float = 0.05) -> ABTestResult:
        """
        Simulate an A/B test with synthetic data.
        
        Args:
            config: A/B test configuration
            num_users: Number of users to simulate
            effect_size: Effect size (difference between variants)
            noise_level: Noise level in the data
            
        Returns:
            A/B test result
        """
        self.logger.info(f"Simulating A/B test {config.test_id} with {num_users} users")
        
        # Create test
        test_id = self.ab_manager.create_test(config)
        self.ab_manager.start_test(test_id)
        
        # Generate synthetic data
        base_metric_value = 0.5  # Base value for primary metric
        
        for i in range(num_users):
            user_id = f"user_{i}"
            variant = self.ab_manager.assign_variant(user_id, test_id)
            
            if variant is None:
                continue
            
            # Generate metrics with effect size and noise
            if variant == "A":
                metric_value = base_metric_value + np.random.normal(0, noise_level)
            else:
                metric_value = base_metric_value + effect_size + np.random.normal(0, noise_level)
            
            # Ensure metric is in valid range
            metric_value = np.clip(metric_value, 0, 1)
            
            metrics = {
                config.primary_metric: metric_value,
                'recall@10': metric_value * 0.8 + np.random.normal(0, noise_level * 0.5),
                'precision@10': metric_value * 0.9 + np.random.normal(0, noise_level * 0.5),
                'click_through_rate': metric_value * 0.3 + np.random.normal(0, noise_level * 0.3)
            }
            
            # Clip all metrics to valid range
            for key in metrics:
                metrics[key] = np.clip(metrics[key], 0, 1)
            
            self.ab_manager.record_interaction(user_id, test_id, variant, metrics)
        
        # Analyze results
        result = self.ab_manager.analyze_test(test_id)
        
        # Stop test
        self.ab_manager.stop_test(test_id)
        
        return result 