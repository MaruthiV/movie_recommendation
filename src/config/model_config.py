"""
Model configuration for the movie recommendation system training pipeline.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch


@dataclass
class LightGCNConfig:
    """Configuration for LightGCN model training."""
    
    # Model architecture
    embedding_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 1024
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Loss function
    loss_type: str = "bpr"  # "bpr", "nce", "sampled_softmax"
    negative_sampling_ratio: int = 1  # Number of negative samples per positive
    
    # Regularization
    l2_reg: float = 1e-4
    clip_grad_norm: Optional[float] = 1.0
    
    # Device
    device: str = "cpu"  # "cpu", "cuda", "mps"
    
    # Data processing
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    min_interactions_per_user: int = 5
    min_interactions_per_item: int = 5
    
    # Evaluation
    eval_metrics: List[str] = None
    top_k_values: List[int] = None
    
    def __post_init__(self):
        if self.eval_metrics is None:
            self.eval_metrics = ["ndcg", "recall", "precision", "hit_ratio"]
        if self.top_k_values is None:
            self.top_k_values = [5, 10, 20, 50]


@dataclass
class TrainingConfig:
    """General training configuration."""
    
    # Experiment tracking
    experiment_name: str = "movie_recommendation"
    run_name: Optional[str] = None
    log_dir: str = "logs/training"
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 5
    save_interval: int = 10
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Multi-GPU training
    num_gpus: int = 0
    distributed: bool = False
    
    # Mixed precision
    use_amp: bool = False
    amp_dtype: str = "float16"
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    load_optimizer: bool = True
    load_scheduler: bool = True


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    
    # Data sources
    movielens_path: str = "data/ml-20m"
    tmdb_path: str = "data/tmdb"
    
    # Database connections
    postgres_connection: Dict[str, str] = None
    neo4j_connection: Dict[str, str] = None
    milvus_connection: Dict[str, str] = None
    
    # Preprocessing
    max_sequence_length: int = 100
    min_rating: float = 1.0
    max_rating: float = 5.0
    min_interactions_per_user: int = 5
    min_interactions_per_item: int = 5
    
    # Feature engineering
    use_movie_features: bool = True
    use_user_features: bool = True
    feature_dim: int = 128
    
    # Caching
    cache_dir: str = "cache"
    use_cache: bool = True
    
    def __post_init__(self):
        if self.postgres_connection is None:
            self.postgres_connection = {
                "host": "localhost",
                "port": 5432,
                "database": "movie_recommendation",
                "user": "postgres",
                "password": "password"
            }
        if self.neo4j_connection is None:
            self.neo4j_connection = {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "password"
            }
        if self.milvus_connection is None:
            self.milvus_connection = {
                "host": "localhost",
                "port": 19530
            }


@dataclass
class EvaluationConfig:
    """Evaluation and testing configuration."""
    
    # Test set
    test_users_ratio: float = 0.2
    test_items_ratio: float = 0.2
    
    # Metrics
    primary_metric: str = "ndcg@10"
    secondary_metrics: List[str] = None
    
    # Evaluation settings
    eval_batch_size: int = 1024
    num_test_users: Optional[int] = None  # None means all test users
    num_candidates_per_user: int = 100
    
    # A/B testing
    ab_test_enabled: bool = False
    ab_test_ratio: float = 0.1
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = ["recall@10", "precision@10", "hit_ratio@10"]


@dataclass
class ModelConfig:
    """Main configuration class that combines all sub-configs."""
    
    lightgcn: LightGCNConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    evaluation: EvaluationConfig = None
    
    def __post_init__(self):
        if self.lightgcn is None:
            self.lightgcn = LightGCNConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "lightgcn": self.lightgcn.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "evaluation": self.evaluation.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        # Filter out dataset-specific fields
        lightgcn_dict = dict(config_dict.get("lightgcn", {}))
        lightgcn_dict.pop("num_users", None)
        lightgcn_dict.pop("num_items", None)
        lightgcn = LightGCNConfig(**lightgcn_dict)
        training = TrainingConfig(**config_dict.get("training", {}))
        data = DataConfig(**config_dict.get("data", {}))
        evaluation = EvaluationConfig(**config_dict.get("evaluation", {}))
        
        return cls(
            lightgcn=lightgcn,
            training=training,
            data=data,
            evaluation=evaluation
        )


def get_default_config() -> ModelConfig:
    """Get default configuration for the movie recommendation system."""
    return ModelConfig()


def get_gpu_config() -> ModelConfig:
    """Get configuration optimized for GPU training."""
    config = get_default_config()
    config.lightgcn.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.training.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    config.training.use_amp = True
    config.lightgcn.batch_size = 2048
    return config


def get_fast_config() -> ModelConfig:
    """Get configuration for fast development and testing."""
    config = get_default_config()
    config.lightgcn.num_epochs = 10
    config.lightgcn.batch_size = 512
    config.training.log_interval = 50
    config.training.eval_interval = 2
    config.evaluation.num_test_users = 1000
    return config 