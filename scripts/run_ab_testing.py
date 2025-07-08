#!/usr/bin/env python3
"""
Script to demonstrate model versioning and A/B testing framework.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.model_versioning import ModelRegistry, ModelMetadata, ExperimentType, ModelStatus
from models.ab_testing import ABTestManager, ABTestConfig, ABTestSimulator, TrafficSplit
from models.lightgcn_model import LightGCN
from config.model_config import get_default_config, get_fast_config


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ab_testing.log')
        ]
    )


def create_sample_models(model_registry: ModelRegistry) -> Tuple[str, str]:
    """
    Create sample models for A/B testing demonstration.
    
    Args:
        model_registry: Model registry instance
        
    Returns:
        Tuple of (model_a_id, model_b_id)
    """
    logger = logging.getLogger(__name__)
    
    # Create two different model configurations
    config_a = get_default_config()
    config_a.lightgcn.embedding_dim = 64
    config_a.lightgcn.num_layers = 3
    config_a.lightgcn.learning_rate = 0.001
    
    config_b = get_default_config()
    config_b.lightgcn.embedding_dim = 128
    config_b.lightgcn.num_layers = 4
    config_b.lightgcn.learning_rate = 0.0005
    
    # Create sample models (in real scenario, these would be trained)
    model_a = LightGCN(
        num_users=1000,
        num_items=2000,
        embedding_dim=config_a.lightgcn.embedding_dim,
        num_layers=config_a.lightgcn.num_layers,
        device='cpu'
    )
    
    model_b = LightGCN(
        num_users=1000,
        num_items=2000,
        embedding_dim=config_b.lightgcn.embedding_dim,
        num_layers=config_b.lightgcn.num_layers,
        device='cpu'
    )
    
    # Register models
    logger.info("Registering model A...")
    model_a_id = model_registry.register_model(
        model=model_a,
        name="LightGCN-Base",
        version="1.0",
        description="Base LightGCN model with 64-dim embeddings and 3 layers",
        config=config_a,
        hyperparameters={
            'embedding_dim': 64,
            'num_layers': 3,
            'learning_rate': 0.001,
            'batch_size': 1024
        },
        train_metrics={'loss': 0.5, 'ndcg@10': 0.65, 'recall@10': 0.45},
        validation_metrics={'loss': 0.52, 'ndcg@10': 0.63, 'recall@10': 0.43},
        test_metrics={'ndcg@10': 0.62, 'recall@10': 0.42, 'precision@10': 0.38},
        experiment_id="exp_001",
        experiment_type=ExperimentType.ARCHITECTURE_COMPARISON,
        tags=['baseline', 'lightgcn']
    )
    
    logger.info("Registering model B...")
    model_b_id = model_registry.register_model(
        model=model_b,
        name="LightGCN-Enhanced",
        version="1.0",
        description="Enhanced LightGCN model with 128-dim embeddings and 4 layers",
        config=config_b,
        hyperparameters={
            'embedding_dim': 128,
            'num_layers': 4,
            'learning_rate': 0.0005,
            'batch_size': 1024
        },
        train_metrics={'loss': 0.48, 'ndcg@10': 0.68, 'recall@10': 0.47},
        validation_metrics={'loss': 0.50, 'ndcg@10': 0.66, 'recall@10': 0.45},
        test_metrics={'ndcg@10': 0.65, 'recall@10': 0.44, 'precision@10': 0.40},
        experiment_id="exp_001",
        experiment_type=ExperimentType.ARCHITECTURE_COMPARISON,
        parent_model_id=model_a_id,
        tags=['enhanced', 'lightgcn', 'larger_embeddings']
    )
    
    logger.info(f"Created models: A={model_a_id}, B={model_b_id}")
    return model_a_id, model_b_id


def run_ab_test_demo(model_a_id: str, model_b_id: str, output_dir: str = "ab_test_demo"):
    """
    Run A/B testing demonstration.
    
    Args:
        model_a_id: Model A ID
        model_b_id: Model B ID
        output_dir: Output directory for results
    """
    logger = logging.getLogger(__name__)
    
    # Initialize model registry and A/B test manager
    model_registry = ModelRegistry(registry_path=os.path.join(output_dir, "model_registry"))
    ab_manager = ABTestManager(model_registry, results_path=os.path.join(output_dir, "ab_test_results"))
    
    # Create A/B test configuration
    test_config = ABTestConfig(
        test_id="demo_test_001",
        name="LightGCN Architecture Comparison",
        description="Compare base LightGCN vs enhanced LightGCN with larger embeddings",
        model_a_id=model_a_id,
        model_b_id=model_b_id,
        traffic_split=TrafficSplit.FIFTY_FIFTY,
        primary_metric="ndcg@10",
        secondary_metrics=["recall@10", "precision@10", "click_through_rate"],
        minimum_sample_size=500,
        confidence_level=0.95,
        max_duration_days=7,
        early_stopping_enabled=True,
        min_detection_effect=0.05
    )
    
    logger.info("Creating A/B test...")
    test_id = ab_manager.create_test(test_config)
    
    # Simulate A/B test
    logger.info("Running A/B test simulation...")
    simulator = ABTestSimulator(ab_manager)
    
    # Run simulation with different effect sizes
    effect_sizes = [0.05, 0.1, 0.15]
    results = []
    
    for effect_size in effect_sizes:
        logger.info(f"Simulating with effect size: {effect_size}")
        result = simulator.simulate_test(
            config=test_config,
            num_users=1000,
            effect_size=effect_size,
            noise_level=0.05
        )
        results.append(result)
        
        logger.info(f"Effect size {effect_size}:")
        logger.info(f"  Sample sizes: A={result.sample_size_a}, B={result.sample_size_b}")
        logger.info(f"  Primary metric A: {result.metrics_a[result.primary_metric]:.4f}")
        logger.info(f"  Primary metric B: {result.metrics_b[result.primary_metric]:.4f}")
        logger.info(f"  Effect size: {result.effect_size:.4f}")
        logger.info(f"  P-value: {result.p_value:.4f}")
        logger.info(f"  Significant: {result.is_significant}")
        logger.info(f"  Winner: {result.winner}")
        logger.info(f"  Recommendation: {result.recommendation}")
        logger.info("")
    
    # List all tests
    logger.info("Listing all A/B tests:")
    tests = ab_manager.list_tests()
    for test in tests:
        logger.info(f"  {test.test_id}: {test.name}")
    
    # Get test summary
    logger.info("Test summary:")
    summary = ab_manager.get_test_summary(test_id)
    for key, value in summary.items():
        if key != 'latest_result':
            logger.info(f"  {key}: {value}")
    
    return results


def demonstrate_model_registry(output_dir: str = "model_registry_demo"):
    """
    Demonstrate model registry functionality.
    
    Args:
        output_dir: Output directory for registry
    """
    logger = logging.getLogger(__name__)
    
    # Initialize model registry
    model_registry = ModelRegistry(registry_path=output_dir)
    
    # Create sample models
    model_a_id, model_b_id = create_sample_models(model_registry)
    
    # List all models
    logger.info("Listing all models:")
    models = model_registry.list_models()
    for model in models:
        logger.info(f"  {model.model_id}: {model.name} v{model.version} ({model.status.value})")
    
    # Get model metadata
    logger.info("Model A metadata:")
    metadata_a = model_registry.get_metadata(model_a_id)
    logger.info(f"  Name: {metadata_a.name}")
    logger.info(f"  Version: {metadata_a.version}")
    logger.info(f"  Status: {metadata_a.status.value}")
    logger.info(f"  Parameters: {metadata_a.num_parameters:,}")
    logger.info(f"  Model size: {metadata_a.model_size:.2f} MB")
    logger.info(f"  Test NDCG@10: {metadata_a.test_metrics['ndcg@10']:.4f}")
    
    # Deploy a model
    logger.info("Deploying model A to production...")
    model_registry.deploy_model(model_a_id)
    
    # Check production status
    production_models = model_registry.list_models(is_production=True)
    logger.info(f"Production models: {len(production_models)}")
    for model in production_models:
        logger.info(f"  {model.name} v{model.version}")
    
    # Update model status
    logger.info("Updating model B status to validated...")
    model_registry.update_model_status(model_b_id, ModelStatus.VALIDATED)
    
    # Archive old model (if exists)
    logger.info("Archiving model A...")
    model_registry.archive_model(model_a_id)
    
    return model_a_id, model_b_id


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Model versioning and A/B testing demonstration")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo_output",
        help="Output directory for demo results"
    )
    parser.add_argument(
        "--demo-type",
        type=str,
        choices=["registry", "ab_test", "both"],
        default="both",
        help="Type of demonstration to run"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model versioning and A/B testing demonstration")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Demo type: {args.demo_type}")
    
    try:
        if args.demo_type in ["registry", "both"]:
            logger.info("=" * 60)
            logger.info("MODEL REGISTRY DEMONSTRATION")
            logger.info("=" * 60)
            model_a_id, model_b_id = demonstrate_model_registry(
                os.path.join(args.output_dir, "registry")
            )
        
        if args.demo_type in ["ab_test", "both"]:
            logger.info("=" * 60)
            logger.info("A/B TESTING DEMONSTRATION")
            logger.info("=" * 60)
            
            # Create models for A/B testing
            model_registry = ModelRegistry(registry_path=os.path.join(args.output_dir, "registry"))
            model_a_id, model_b_id = create_sample_models(model_registry)
            
            # Run A/B test demo
            results = run_ab_test_demo(
                model_a_id, 
                model_b_id, 
                os.path.join(args.output_dir, "ab_test")
            )
        
        logger.info("=" * 60)
        logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 