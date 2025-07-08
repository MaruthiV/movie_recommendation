"""
Training pipeline for the movie recommendation system with PyTorch Geometric integration.
"""

import os
import time
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

from .lightgcn_model import LightGCN
from .data_loader import MovieLensDataLoader, create_data_loaders
from config.model_config import ModelConfig, LightGCNConfig, TrainingConfig
from utils.evaluation_metrics import calculate_ndcg, calculate_recall, calculate_precision, calculate_hit_ratio


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    learning_rate: float
    time_taken: float


class EarlyStopping:
    """Early stopping callback for training."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.mode == 'min':
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
        else:
            if val_loss > self.best_loss + self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """Model checkpointing callback."""
    
    def __init__(self, checkpoint_dir: str, save_best_only: bool = True, mode: str = 'min'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        config: ModelConfig
    ) -> str:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': config.to_dict()
        }
        
        # Determine if this is the best model
        current_metric = metrics.get('val_loss', float('inf'))
        is_best = False
        
        if self.mode == 'min':
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                is_best = True
        else:
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                is_best = True
        
        # Save checkpoint
        if not self.save_best_only or is_best:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            if is_best:
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
            
            return str(checkpoint_path)
        
        return ""
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        checkpoint_path: str
    ) -> Tuple[int, Dict[str, float]]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']


class TrainingPipeline:
    """
    Training pipeline for LightGCN model with PyTorch Geometric integration.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        data_dir: str,
        output_dir: str = "outputs"
    ):
        """
        Initialize training pipeline.
        
        Args:
            config: Model configuration
            data_dir: Directory containing MovieLens data
            output_dir: Directory for outputs (logs, checkpoints, etc.)
        """
        self.config = config
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Set random seeds
        self._set_random_seeds()
        
        # Set up device
        self.device = self._setup_device()
        
        # Load data
        self.data_loader, self.graph_data, self.metadata = self._load_data()
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._initialize_optimizer()
        
        # Initialize callbacks
        self.early_stopping = EarlyStopping(
            patience=config.lightgcn.early_stopping_patience,
            mode='min'
        )
        self.checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=str(self.output_dir / "checkpoints"),
            save_best_only=True,
            mode='min'
        )
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history: List[TrainingMetrics] = []
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_file = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.training.seed)
        np.random.seed(self.config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.training.seed)
            torch.cuda.manual_seed_all(self.config.training.seed)
        
        if self.config.training.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_device(self) -> torch.device:
        """Set up device for training."""
        if self.config.lightgcn.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif self.config.lightgcn.device == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            self.logger.info("Using MPS device")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU device")
        
        return device
    
    def _load_data(self) -> Tuple[MovieLensDataLoader, Any, Dict]:
        """Load and prepare data."""
        self.logger.info("Loading data...")
        
        data_loader, graph_data, metadata = create_data_loaders(
            data_dir=self.data_dir,
            config=self.config.data,
            cache_dir=str(self.output_dir / "cache")
        )
        
        # Move graph data to device
        graph_data = graph_data.to(self.device)
        
        self.logger.info(f"Data loaded: {metadata['num_users']} users, "
                        f"{metadata['num_items']} items")
        
        return data_loader, graph_data, metadata
    
    def _initialize_model(self) -> LightGCN:
        """Initialize LightGCN model."""
        self.logger.info("Initializing LightGCN model...")
        
        model = LightGCN(
            num_users=self.metadata['num_users'],
            num_items=self.metadata['num_items'],
            embedding_dim=self.config.lightgcn.embedding_dim,
            num_layers=self.config.lightgcn.num_layers,
            device=str(self.device)
        )
        
        # Move model to device
        model = model.to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model initialized: {total_params:,} total parameters, "
                        f"{trainable_params:,} trainable parameters")
        
        return model
    
    def _initialize_optimizer(self) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
        """Initialize optimizer and scheduler."""
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lightgcn.learning_rate,
            weight_decay=self.config.lightgcn.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return optimizer, scheduler
    
    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.data_loader.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            users = batch['user_idx'].to(self.device)
            pos_items = batch['pos_item_idx'].to(self.device)
            neg_items = batch['neg_item_idx'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.training.use_amp:
                with autocast():
                    loss = self.model.bpr_loss(
                        users, pos_items, neg_items,
                        self.graph_data.edge_index,
                        reg_weight=self.config.lightgcn.l2_reg
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.lightgcn.clip_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.lightgcn.clip_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.model.bpr_loss(
                    users, pos_items, neg_items,
                    self.graph_data.edge_index,
                    reg_weight=self.config.lightgcn.l2_reg
                )
                
                loss.backward()
                
                # Gradient clipping
                if self.config.lightgcn.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.lightgcn.clip_grad_norm
                    )
                
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}"
            })
            
            # Log interval
            if batch_idx % self.config.training.log_interval == 0:
                self.logger.info(f"Epoch {self.current_epoch + 1}, Batch {batch_idx}, "
                               f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        metrics = {'train_loss': avg_loss}
        
        return avg_loss, metrics
    
    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loader.val_loader, desc="Validation"):
                # Move batch to device
                users = batch['user_idx'].to(self.device)
                pos_items = batch['pos_item_idx'].to(self.device)
                neg_items = batch['neg_item_idx'].to(self.device)
                
                # Forward pass
                loss = self.model.bpr_loss(
                    users, pos_items, neg_items,
                    self.graph_data.edge_index,
                    reg_weight=self.config.lightgcn.l2_reg
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions for evaluation
                pos_scores = self.model.predict(users, pos_items, self.graph_data.edge_index)
                neg_scores = self.model.predict(users, neg_items, self.graph_data.edge_index)
                
                all_predictions.extend(torch.cat([pos_scores, neg_scores]).cpu().numpy())
                all_targets.extend([1] * len(pos_scores) + [0] * len(neg_scores))
        
        avg_loss = total_loss / num_batches
        
        # Calculate evaluation metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        metrics = {
            'val_loss': avg_loss,
            'val_ndcg@10': calculate_ndcg(targets, predictions, k=10),
            'val_recall@10': calculate_recall(targets, predictions, k=10),
            'val_precision@10': calculate_precision(targets, predictions, k=10),
            'val_hit_ratio@10': calculate_hit_ratio(targets, predictions, k=10)
        }
        
        return avg_loss, metrics
    
    def train(self) -> Dict[str, Any]:
        """Train the model."""
        self.logger.info("Starting training...")
        self.logger.info(f"Configuration: {self.config.to_dict()}")
        
        start_time = time.time()
        
        for epoch in range(self.config.lightgcn.num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Training
            train_loss, train_metrics = self._train_epoch()
            
            # Validation
            val_loss, val_metrics = self._validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Calculate time taken
            epoch_time = time.time() - epoch_start_time
            
            # Create metrics object
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                time_taken=epoch_time
            )
            
            self.training_history.append(metrics)
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.lightgcn.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"NDCG@10: {val_metrics['val_ndcg@10']:.4f}, "
                f"Recall@10: {val_metrics['val_recall@10']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save checkpoint
            if epoch % self.config.training.save_interval == 0:
                checkpoint_path = self.checkpoint_callback.save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, val_metrics, self.config
                )
                if checkpoint_path:
                    self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Early stopping
            if self.early_stopping(val_loss):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model and training history
        self._save_training_results()
        
        return {
            'best_metrics': self.best_metrics,
            'training_history': self.training_history,
            'total_time': total_time
        }
    
    def _save_training_results(self):
        """Save training results and history."""
        # Save training history
        history_file = self.output_dir / "training_history.json"
        history_data = []
        
        for metrics in self.training_history:
            history_data.append({
                'epoch': metrics.epoch,
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'train_metrics': metrics.train_metrics,
                'val_metrics': metrics.val_metrics,
                'learning_rate': metrics.learning_rate,
                'time_taken': metrics.time_taken
            })
        
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # Save configuration
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save metadata
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self.logger.info(f"Training results saved to {self.output_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        epoch, metrics = self.checkpoint_callback.load_checkpoint(
            self.model, self.optimizer, self.scheduler, checkpoint_path
        )
        
        self.current_epoch = epoch
        self.best_metrics = metrics
        
        self.logger.info(f"Checkpoint loaded: epoch {epoch}, metrics {metrics}")
    
    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate the model on test set."""
        self.model.eval()
        
        if test_loader is None:
            test_loader = self.data_loader.test_loader
        
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # Move batch to device
                users = batch['user_idx'].to(self.device)
                pos_items = batch['pos_item_idx'].to(self.device)
                neg_items = batch['neg_item_idx'].to(self.device)
                
                # Forward pass
                loss = self.model.bpr_loss(
                    users, pos_items, neg_items,
                    self.graph_data.edge_index,
                    reg_weight=self.config.lightgcn.l2_reg
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions for evaluation
                pos_scores = self.model.predict(users, pos_items, self.graph_data.edge_index)
                neg_scores = self.model.predict(users, neg_items, self.graph_data.edge_index)
                
                all_predictions.extend(torch.cat([pos_scores, neg_scores]).cpu().numpy())
                all_targets.extend([1] * len(pos_scores) + [0] * len(neg_scores))
        
        avg_loss = total_loss / num_batches
        
        # Calculate evaluation metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        metrics = {
            'test_loss': avg_loss,
            'test_ndcg@10': calculate_ndcg(targets, predictions, k=10),
            'test_recall@10': calculate_recall(targets, predictions, k=10),
            'test_precision@10': calculate_precision(targets, predictions, k=10),
            'test_hit_ratio@10': calculate_hit_ratio(targets, predictions, k=10)
        }
        
        # Log results
        self.logger.info("Test Results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return metrics


def train_model(
    config: ModelConfig,
    data_dir: str,
    output_dir: str = "outputs",
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to train a model.
    
    Args:
        config: Model configuration
        data_dir: Directory containing MovieLens data
        output_dir: Directory for outputs
        resume_from: Optional checkpoint path to resume from
        
    Returns:
        Training results
    """
    # Create training pipeline
    pipeline = TrainingPipeline(config, data_dir, output_dir)
    
    # Resume from checkpoint if specified
    if resume_from:
        pipeline.load_checkpoint(resume_from)
    
    # Train model
    results = pipeline.train()
    
    # Evaluate on test set
    test_metrics = pipeline.evaluate()
    results['test_metrics'] = test_metrics
    
    return results 