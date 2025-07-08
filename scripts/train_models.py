#!/usr/bin/env python3
"""
Training script for the movie recommendation system models.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.training_pipeline import train_model, TrainingPipeline
from config.model_config import get_default_config, get_gpu_config, get_fast_config, ModelConfig
from models.data_loader import create_data_loaders


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train movie recommendation models")
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ml-20m",
        help="Directory containing MovieLens data"
    )
    
    # Model arguments
    parser.add_argument(
        "--config",
        type=str,
        choices=["default", "gpu", "fast"],
        default="default",
        help="Configuration preset to use"
    )
    
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Embedding dimension for LightGCN"
    )
    
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of LightGCN layers"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Checkpoint path to resume training from"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use automatic mixed precision"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate model, don't train"
    )
    
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to checkpoint for evaluation"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> ModelConfig:
    """Create configuration from command line arguments."""
    # Get base config
    if args.config == "gpu":
        config = get_gpu_config()
    elif args.config == "fast":
        config = get_fast_config()
    else:
        config = get_default_config()
    
    # Override with command line arguments
    config.lightgcn.embedding_dim = args.embedding_dim
    config.lightgcn.num_layers = args.num_layers
    config.lightgcn.learning_rate = args.learning_rate
    config.lightgcn.batch_size = args.batch_size
    config.lightgcn.num_epochs = args.num_epochs
    config.lightgcn.device = args.device
    
    config.training.use_amp = args.use_amp
    
    return config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting movie recommendation model training")
    logger.info(f"Arguments: {args}")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.info("Please download MovieLens-20M data and place it in the data directory")
        return 1
    
    # Create configuration
    config = create_config_from_args(args)
    logger.info(f"Using configuration: {config.to_dict()}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.eval_only:
            # Evaluation only mode
            if not args.checkpoint_path:
                logger.error("Checkpoint path required for evaluation mode")
                return 1
            
            if not os.path.exists(args.checkpoint_path):
                logger.error(f"Checkpoint not found: {args.checkpoint_path}")
                return 1
            
            # Create training pipeline for evaluation
            pipeline = TrainingPipeline(config, args.data_dir, str(output_dir))
            
            # Load checkpoint
            pipeline.load_checkpoint(args.checkpoint_path)
            
            # Evaluate model
            logger.info("Evaluating model...")
            test_metrics = pipeline.evaluate()
            
            # Print results
            from utils.evaluation_metrics import print_evaluation_results
            print_evaluation_results(test_metrics, "Test Results")
            
        else:
            # Training mode
            logger.info("Starting model training...")
            
            # Train model
            results = train_model(
                config=config,
                data_dir=args.data_dir,
                output_dir=str(output_dir),
                resume_from=args.resume_from
            )
            
            # Print training summary
            logger.info("Training completed successfully!")
            logger.info(f"Best validation loss: {results['best_metrics'].get('val_loss', 'N/A')}")
            logger.info(f"Test metrics: {results.get('test_metrics', {})}")
            logger.info(f"Total training time: {results['total_time']:.2f} seconds")
            
            # Print final evaluation results
            if 'test_metrics' in results:
                from utils.evaluation_metrics import print_evaluation_results
                print_evaluation_results(results['test_metrics'], "Final Test Results")
        
        logger.info(f"Results saved to: {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main()) 