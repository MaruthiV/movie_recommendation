"""
Model versioning and registry system for the movie recommendation system.
"""

import os
import json
import hashlib
import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import shutil
import logging
from enum import Enum

import torch
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from src.config.model_config import ModelConfig


class ModelStatus(Enum):
    """Model status enumeration."""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class ExperimentType(Enum):
    """Experiment type enumeration."""
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    ARCHITECTURE_COMPARISON = "architecture_comparison"
    FEATURE_ENGINEERING = "feature_engineering"
    A_B_TESTING = "a_b_testing"
    PRODUCTION_UPDATE = "production_update"


@dataclass
class ModelMetadata:
    """Model metadata for versioning."""
    
    # Basic information
    model_id: str
    version: str
    name: str
    description: str
    
    # Training information
    training_start_time: datetime.datetime
    training_end_time: Optional[datetime.datetime]
    training_duration: Optional[float]  # in seconds
    
    # Model configuration
    config: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    
    # Performance metrics
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]]
    
    # Data information
    data_version: str
    data_hash: str
    num_train_samples: int
    num_validation_samples: int
    num_test_samples: int
    num_users: int
    num_items: int
    
    # Model information
    model_size: Optional[float]  # in MB
    num_parameters: Optional[int]
    
    # Status and deployment
    status: ModelStatus
    is_production: bool
    deployment_time: Optional[datetime.datetime]
    
    # Experiment information
    experiment_id: Optional[str]
    experiment_type: Optional[ExperimentType]
    parent_model_id: Optional[str]
    
    # Additional metadata
    tags: List[str]
    notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        data['experiment_type'] = self.experiment_type.value if self.experiment_type else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        data['status'] = ModelStatus(data['status'])
        if data.get('experiment_type'):
            data['experiment_type'] = ExperimentType(data['experiment_type'])
        return cls(**data)


class ModelRegistry:
    """
    Model registry for versioning and managing models.
    """
    
    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to store model registry
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_path = self.registry_path / "models"
        self.metadata_path = self.registry_path / "metadata"
        self.experiments_path = self.registry_path / "experiments"
        
        for path in [self.models_path, self.metadata_path, self.experiments_path]:
            path.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.registry_path / "registry.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.Base = declarative_base()
        self._create_tables()
        self.Session = sessionmaker(bind=self.engine)
        
        self.logger = logging.getLogger(__name__)
    
    def _create_tables(self):
        """Create database tables."""
        class ModelRecord(self.Base):
            __tablename__ = 'models'
            
            model_id = Column(String, primary_key=True)
            version = Column(String, nullable=False)
            name = Column(String, nullable=False)
            description = Column(Text)
            training_start_time = Column(DateTime, nullable=False)
            training_end_time = Column(DateTime)
            training_duration = Column(Float)
            config = Column(Text)  # JSON string
            hyperparameters = Column(Text)  # JSON string
            train_metrics = Column(Text)  # JSON string
            validation_metrics = Column(Text)  # JSON string
            test_metrics = Column(Text)  # JSON string
            data_version = Column(String)
            data_hash = Column(String)
            num_train_samples = Column(Integer)
            num_validation_samples = Column(Integer)
            num_test_samples = Column(Integer)
            num_users = Column(Integer)
            num_items = Column(Integer)
            model_size = Column(Float)
            num_parameters = Column(Integer)
            status = Column(String, nullable=False)
            is_production = Column(Boolean, default=False)
            deployment_time = Column(DateTime)
            experiment_id = Column(String)
            experiment_type = Column(String)
            parent_model_id = Column(String)
            tags = Column(Text)  # JSON string
            notes = Column(Text)
            created_at = Column(DateTime, default=datetime.datetime.utcnow)
            updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
        
        self.ModelRecord = ModelRecord
        self.Base.metadata.create_all(self.engine)
    
    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.datetime.now().isoformat()
        unique_string = f"{name}_{version}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def _save_model_file(self, model: torch.nn.Module, model_id: str, version: str) -> str:
        """Save model file and return path."""
        model_dir = self.models_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"model_{version}.pt"
        torch.save(model.state_dict(), model_path)
        
        return str(model_path)
    
    def _save_metadata(self, metadata: ModelMetadata) -> str:
        """Save metadata to file."""
        metadata_dir = self.metadata_path / metadata.model_id
        metadata_dir.mkdir(exist_ok=True)
        
        metadata_path = metadata_dir / f"metadata_{metadata.version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        
        return str(metadata_path)
    
    def register_model(
        self,
        model: torch.nn.Module,
        name: str,
        version: str,
        description: str,
        config: ModelConfig,
        hyperparameters: Dict[str, Any],
        train_metrics: Dict[str, float],
        validation_metrics: Dict[str, float],
        test_metrics: Optional[Dict[str, float]] = None,
        data_version: str = "1.0",
        data_hash: str = "",
        num_train_samples: int = 0,
        num_validation_samples: int = 0,
        num_test_samples: int = 0,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
        experiment_id: Optional[str] = None,
        experiment_type: Optional[ExperimentType] = None,
        parent_model_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: str = ""
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model: PyTorch model to register
            name: Model name
            version: Model version
            description: Model description
            config: Model configuration
            hyperparameters: Training hyperparameters
            train_metrics: Training metrics
            validation_metrics: Validation metrics
            test_metrics: Test metrics (optional)
            data_version: Data version used for training
            data_hash: Hash of training data
            num_train_samples: Number of training samples
            num_validation_samples: Number of validation samples
            num_test_samples: Number of test samples
            experiment_id: Experiment ID (optional)
            experiment_type: Type of experiment (optional)
            parent_model_id: Parent model ID (optional)
            tags: Model tags (optional)
            notes: Additional notes (optional)
            
        Returns:
            Model ID
        """
        # Generate model ID
        model_id = self._generate_model_id(name, version)
        
        # Calculate model size and parameters
        model_size = self._calculate_model_size(model)
        num_parameters = sum(p.numel() for p in model.parameters())
        
        # Extract num_users and num_items from config if not provided
        if num_users is None:
            num_users = config.lightgcn.num_users
        if num_items is None:
            num_items = config.lightgcn.num_items
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=name,
            description=description,
            training_start_time=datetime.datetime.now(),
            training_end_time=datetime.datetime.now(),
            training_duration=0.0,
            config=config.to_dict(),
            hyperparameters=hyperparameters,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            data_version=data_version,
            data_hash=data_hash,
            num_train_samples=num_train_samples,
            num_validation_samples=num_validation_samples,
            num_test_samples=num_test_samples,
            num_users=num_users,
            num_items=num_items,
            model_size=model_size,
            num_parameters=num_parameters,
            status=ModelStatus.TRAINED,
            is_production=False,
            deployment_time=None,
            experiment_id=experiment_id,
            experiment_type=experiment_type,
            parent_model_id=parent_model_id,
            tags=tags or [],
            notes=notes
        )
        
        # Save model and metadata
        model_path = self._save_model_file(model, model_id, version)
        metadata_path = self._save_metadata(metadata)
        
        # Save to database
        self._save_to_database(metadata)
        
        self.logger.info(f"Registered model {name} v{version} with ID {model_id}")
        
        return model_id
    
    def _calculate_model_size(self, model: torch.nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def _save_to_database(self, metadata: ModelMetadata):
        """Save metadata to database."""
        session = self.Session()
        try:
            record = self.ModelRecord(
                model_id=metadata.model_id,
                version=metadata.version,
                name=metadata.name,
                description=metadata.description,
                training_start_time=metadata.training_start_time,
                training_end_time=metadata.training_end_time,
                training_duration=metadata.training_duration,
                config=json.dumps(metadata.config),
                hyperparameters=json.dumps(metadata.hyperparameters),
                train_metrics=json.dumps(metadata.train_metrics),
                validation_metrics=json.dumps(metadata.validation_metrics),
                test_metrics=json.dumps(metadata.test_metrics) if metadata.test_metrics else None,
                data_version=metadata.data_version,
                data_hash=metadata.data_hash,
                num_train_samples=metadata.num_train_samples,
                num_validation_samples=metadata.num_validation_samples,
                num_test_samples=metadata.num_test_samples,
                num_users=metadata.num_users,
                num_items=metadata.num_items,
                model_size=metadata.model_size,
                num_parameters=metadata.num_parameters,
                status=metadata.status.value,
                is_production=metadata.is_production,
                deployment_time=metadata.deployment_time,
                experiment_id=metadata.experiment_id,
                experiment_type=metadata.experiment_type.value if metadata.experiment_type else None,
                parent_model_id=metadata.parent_model_id,
                tags=json.dumps(metadata.tags),
                notes=metadata.notes
            )
            
            session.add(record)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_model(self, model_id: str, version: Optional[str] = None) -> Tuple[torch.nn.Module, ModelMetadata]:
        """
        Load model and metadata.
        
        Args:
            model_id: Model ID
            version: Model version (optional, uses latest if not specified)
            
        Returns:
            Tuple of (model, metadata)
        """
        # Get metadata
        metadata = self.get_metadata(model_id, version)
        
        # Load model
        model_path = self.models_path / model_id / f"model_{metadata.version}.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model instance (this would need to be adapted based on your model architecture)
        from .lightgcn_model import LightGCN
        
        # Extract model parameters from config and metadata
        config = ModelConfig.from_dict(metadata.config)
        model = LightGCN(
            num_users=metadata.num_users,
            num_items=metadata.num_items,
            embedding_dim=config.lightgcn.embedding_dim,
            num_layers=config.lightgcn.num_layers,
            device=config.lightgcn.device
        )
        
        # Load state dict
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        return model, metadata
    
    def get_metadata(self, model_id: str, version: Optional[str] = None) -> ModelMetadata:
        """
        Get model metadata.
        
        Args:
            model_id: Model ID
            version: Model version (optional, uses latest if not specified)
            
        Returns:
            Model metadata
        """
        session = self.Session()
        try:
            query = session.query(self.ModelRecord).filter(self.ModelRecord.model_id == model_id)
            
            if version:
                query = query.filter(self.ModelRecord.version == version)
            else:
                # Get latest version
                query = query.order_by(self.ModelRecord.created_at.desc()).limit(1)
            
            record = query.first()
            
            if not record:
                raise ValueError(f"Model not found: {model_id} v{version}")
            
            # Convert to ModelMetadata
            metadata = ModelMetadata(
                model_id=record.model_id,
                version=record.version,
                name=record.name,
                description=record.description,
                training_start_time=record.training_start_time,
                training_end_time=record.training_end_time,
                training_duration=record.training_duration,
                config=json.loads(record.config),
                hyperparameters=json.loads(record.hyperparameters),
                train_metrics=json.loads(record.train_metrics),
                validation_metrics=json.loads(record.validation_metrics),
                test_metrics=json.loads(record.test_metrics) if record.test_metrics else None,
                data_version=record.data_version,
                data_hash=record.data_hash,
                num_train_samples=record.num_train_samples,
                num_validation_samples=record.num_validation_samples,
                num_test_samples=record.num_test_samples,
                num_users=record.num_users,
                num_items=record.num_items,
                model_size=record.model_size,
                num_parameters=record.num_parameters,
                status=ModelStatus(record.status),
                is_production=record.is_production,
                deployment_time=record.deployment_time,
                experiment_id=record.experiment_id,
                experiment_type=ExperimentType(record.experiment_type) if record.experiment_type else None,
                parent_model_id=record.parent_model_id,
                tags=json.loads(record.tags),
                notes=record.notes
            )
            
            return metadata
            
        finally:
            session.close()
    
    def list_models(self, 
                   name: Optional[str] = None,
                   status: Optional[ModelStatus] = None,
                   experiment_id: Optional[str] = None,
                   is_production: Optional[bool] = None) -> List[ModelMetadata]:
        """
        List models with optional filters.
        
        Args:
            name: Filter by model name
            status: Filter by model status
            experiment_id: Filter by experiment ID
            is_production: Filter by production status
            
        Returns:
            List of model metadata
        """
        session = self.Session()
        try:
            query = session.query(self.ModelRecord)
            
            if name:
                query = query.filter(self.ModelRecord.name == name)
            if status:
                query = query.filter(self.ModelRecord.status == status.value)
            if experiment_id:
                query = query.filter(self.ModelRecord.experiment_id == experiment_id)
            if is_production is not None:
                query = query.filter(self.ModelRecord.is_production == is_production)
            
            records = query.order_by(self.ModelRecord.created_at.desc()).all()
            
            metadata_list = []
            for record in records:
                metadata = ModelMetadata(
                    model_id=record.model_id,
                    version=record.version,
                    name=record.name,
                    description=record.description,
                    training_start_time=record.training_start_time,
                    training_end_time=record.training_end_time,
                    training_duration=record.training_duration,
                    config=json.loads(record.config),
                    hyperparameters=json.loads(record.hyperparameters),
                    train_metrics=json.loads(record.train_metrics),
                    validation_metrics=json.loads(record.validation_metrics),
                    test_metrics=json.loads(record.test_metrics) if record.test_metrics else None,
                    data_version=record.data_version,
                    data_hash=record.data_hash,
                    num_train_samples=record.num_train_samples,
                    num_validation_samples=record.num_validation_samples,
                    num_test_samples=record.num_test_samples,
                    num_users=record.num_users,
                    num_items=record.num_items,
                    model_size=record.model_size,
                    num_parameters=record.num_parameters,
                    status=ModelStatus(record.status),
                    is_production=record.is_production,
                    deployment_time=record.deployment_time,
                    experiment_id=record.experiment_id,
                    experiment_type=ExperimentType(record.experiment_type) if record.experiment_type else None,
                    parent_model_id=record.parent_model_id,
                    tags=json.loads(record.tags),
                    notes=record.notes
                )
                metadata_list.append(metadata)
            
            return metadata_list
            
        finally:
            session.close()
    
    def update_model_status(self, model_id: str, status: ModelStatus, version: Optional[str] = None):
        """
        Update model status.
        
        Args:
            model_id: Model ID
            status: New status
            version: Model version (optional)
        """
        session = self.Session()
        try:
            query = session.query(self.ModelRecord).filter(self.ModelRecord.model_id == model_id)
            
            if version:
                query = query.filter(self.ModelRecord.version == version)
            
            record = query.first()
            if not record:
                raise ValueError(f"Model not found: {model_id} v{version}")
            
            record.status = status.value
            record.updated_at = datetime.datetime.utcnow()
            
            if status == ModelStatus.DEPLOYED:
                record.is_production = True
                record.deployment_time = datetime.datetime.utcnow()
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def deploy_model(self, model_id: str, version: Optional[str] = None):
        """
        Deploy model to production.
        
        Args:
            model_id: Model ID
            version: Model version (optional)
        """
        # First, unset all other models as production
        session = self.Session()
        try:
            session.query(self.ModelRecord).update({"is_production": False})
            session.commit()
        finally:
            session.close()
        
        # Set this model as production
        self.update_model_status(model_id, ModelStatus.DEPLOYED, version)
        
        self.logger.info(f"Deployed model {model_id} v{version} to production")
    
    def archive_model(self, model_id: str, version: Optional[str] = None):
        """
        Archive model.
        
        Args:
            model_id: Model ID
            version: Model version (optional)
        """
        self.update_model_status(model_id, ModelStatus.ARCHIVED, version)
        
        self.logger.info(f"Archived model {model_id} v{version}")
    
    def delete_model(self, model_id: str, version: Optional[str] = None):
        """
        Delete model (use with caution).
        
        Args:
            model_id: Model ID
            version: Model version (optional)
        """
        session = self.Session()
        try:
            query = session.query(self.ModelRecord).filter(self.ModelRecord.model_id == model_id)
            
            if version:
                query = query.filter(self.ModelRecord.version == version)
            
            record = query.first()
            if not record:
                raise ValueError(f"Model not found: {model_id} v{version}")
            
            # Delete from database
            session.delete(record)
            session.commit()
            
            # Delete files
            model_dir = self.models_path / model_id
            metadata_dir = self.metadata_path / model_id
            
            if version:
                model_file = model_dir / f"model_{version}.pt"
                metadata_file = metadata_dir / f"metadata_{version}.json"
                
                if model_file.exists():
                    model_file.unlink()
                if metadata_file.exists():
                    metadata_file.unlink()
            else:
                # Delete entire model directory
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                if metadata_dir.exists():
                    shutil.rmtree(metadata_dir)
            
            self.logger.info(f"Deleted model {model_id} v{version}")
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close() 