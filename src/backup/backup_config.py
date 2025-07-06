"""
Backup Configuration Module

Defines backup strategies, schedules, and storage configurations for the movie recommendation system.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from pathlib import Path


class BackupType(Enum):
    """Types of backups supported."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupFrequency(Enum):
    """Backup frequency options."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class StorageType(Enum):
    """Storage types for backups."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class CompressionType(Enum):
    """Compression types for backups."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


@dataclass
class DatabaseBackupConfig:
    """Configuration for database-specific backups."""
    enabled: bool = True
    backup_type: BackupType = BackupType.FULL
    frequency: BackupFrequency = BackupFrequency.DAILY
    retention_days: int = 30
    compression: CompressionType = CompressionType.GZIP
    include_schema: bool = True
    include_data: bool = True
    include_indexes: bool = True
    parallel_jobs: int = 4
    timeout_minutes: int = 60


@dataclass
class FileBackupConfig:
    """Configuration for file-based backups."""
    enabled: bool = True
    backup_type: BackupType = BackupType.FULL
    frequency: BackupFrequency = BackupFrequency.DAILY
    retention_days: int = 30
    compression: CompressionType = CompressionType.GZIP
    include_patterns: List[str] = field(default_factory=lambda: ["*.parquet", "*.json", "*.csv"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["*.tmp", "*.log", "*.lock"])
    max_file_size_mb: int = 1000


@dataclass
class StorageConfig:
    """Configuration for backup storage."""
    storage_type: StorageType = StorageType.LOCAL
    local_path: str = "backups"
    s3_bucket: Optional[str] = None
    s3_prefix: str = "movie-recommendation-backups"
    gcs_bucket: Optional[str] = None
    gcs_prefix: str = "movie-recommendation-backups"
    azure_container: Optional[str] = None
    azure_prefix: str = "movie-recommendation-backups"
    credentials_file: Optional[str] = None
    encryption_enabled: bool = True
    encryption_key: Optional[str] = None


@dataclass
class NotificationConfig:
    """Configuration for backup notifications."""
    enabled: bool = True
    email_recipients: List[str] = field(default_factory=list)
    slack_webhook: Optional[str] = None
    teams_webhook: Optional[str] = None
    on_success: bool = True
    on_failure: bool = True
    on_warning: bool = False


@dataclass
class BackupConfig:
    """Main backup configuration for the movie recommendation system."""
    
    # General settings
    backup_name: str = "movie-recommendation-system"
    description: str = "Backup configuration for movie recommendation system"
    version: str = "1.0"
    
    # Database configurations
    postgresql: DatabaseBackupConfig = field(default_factory=lambda: DatabaseBackupConfig(
        enabled=True,
        backup_type=BackupType.FULL,
        frequency=BackupFrequency.DAILY,
        retention_days=30,
        compression=CompressionType.GZIP
    ))
    
    neo4j: DatabaseBackupConfig = field(default_factory=lambda: DatabaseBackupConfig(
        enabled=True,
        backup_type=BackupType.FULL,
        frequency=BackupFrequency.DAILY,
        retention_days=30,
        compression=CompressionType.GZIP
    ))
    
    milvus: DatabaseBackupConfig = field(default_factory=lambda: DatabaseBackupConfig(
        enabled=True,
        backup_type=BackupType.FULL,
        frequency=BackupFrequency.WEEKLY,
        retention_days=90,
        compression=CompressionType.LZMA
    ))
    
    # File configurations
    data_files: FileBackupConfig = field(default_factory=lambda: FileBackupConfig(
        enabled=True,
        backup_type=BackupType.FULL,
        frequency=BackupFrequency.DAILY,
        retention_days=30,
        compression=CompressionType.GZIP,
        include_patterns=["*.parquet", "*.json", "*.csv", "*.pkl"]
    ))
    
    logs: FileBackupConfig = field(default_factory=lambda: FileBackupConfig(
        enabled=True,
        backup_type=BackupType.FULL,
        frequency=BackupFrequency.WEEKLY,
        retention_days=90,
        compression=CompressionType.GZIP,
        include_patterns=["*.log", "*.log.*"]
    ))
    
    quality_reports: FileBackupConfig = field(default_factory=lambda: FileBackupConfig(
        enabled=True,
        backup_type=BackupType.FULL,
        frequency=BackupFrequency.DAILY,
        retention_days=30,
        compression=CompressionType.GZIP,
        include_patterns=["*quality_report*.json", "*quality_history*.json"]
    ))
    
    # Storage configuration
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Notification configuration
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    
    # Advanced settings
    max_concurrent_backups: int = 3
    backup_timeout_minutes: int = 120
    verification_enabled: bool = True
    checksum_verification: bool = True
    backup_metadata: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'backup_name': self.backup_name,
            'description': self.description,
            'version': self.version,
            'postgresql': {
                'enabled': self.postgresql.enabled,
                'backup_type': self.postgresql.backup_type.value,
                'frequency': self.postgresql.frequency.value,
                'retention_days': self.postgresql.retention_days,
                'compression': self.postgresql.compression.value,
                'include_schema': self.postgresql.include_schema,
                'include_data': self.postgresql.include_data,
                'include_indexes': self.postgresql.include_indexes,
                'parallel_jobs': self.postgresql.parallel_jobs,
                'timeout_minutes': self.postgresql.timeout_minutes
            },
            'neo4j': {
                'enabled': self.neo4j.enabled,
                'backup_type': self.neo4j.backup_type.value,
                'frequency': self.neo4j.frequency.value,
                'retention_days': self.neo4j.retention_days,
                'compression': self.neo4j.compression.value,
                'include_schema': self.neo4j.include_schema,
                'include_data': self.neo4j.include_data,
                'include_indexes': self.neo4j.include_indexes,
                'parallel_jobs': self.neo4j.parallel_jobs,
                'timeout_minutes': self.neo4j.timeout_minutes
            },
            'milvus': {
                'enabled': self.milvus.enabled,
                'backup_type': self.milvus.backup_type.value,
                'frequency': self.milvus.frequency.value,
                'retention_days': self.milvus.retention_days,
                'compression': self.milvus.compression.value,
                'include_schema': self.milvus.include_schema,
                'include_data': self.milvus.include_data,
                'include_indexes': self.milvus.include_indexes,
                'parallel_jobs': self.milvus.parallel_jobs,
                'timeout_minutes': self.milvus.timeout_minutes
            },
            'data_files': {
                'enabled': self.data_files.enabled,
                'backup_type': self.data_files.backup_type.value,
                'frequency': self.data_files.frequency.value,
                'retention_days': self.data_files.retention_days,
                'compression': self.data_files.compression.value,
                'include_patterns': self.data_files.include_patterns,
                'exclude_patterns': self.data_files.exclude_patterns,
                'max_file_size_mb': self.data_files.max_file_size_mb
            },
            'logs': {
                'enabled': self.logs.enabled,
                'backup_type': self.logs.backup_type.value,
                'frequency': self.logs.frequency.value,
                'retention_days': self.logs.retention_days,
                'compression': self.logs.compression.value,
                'include_patterns': self.logs.include_patterns,
                'exclude_patterns': self.logs.exclude_patterns,
                'max_file_size_mb': self.logs.max_file_size_mb
            },
            'quality_reports': {
                'enabled': self.quality_reports.enabled,
                'backup_type': self.quality_reports.backup_type.value,
                'frequency': self.quality_reports.frequency.value,
                'retention_days': self.quality_reports.retention_days,
                'compression': self.quality_reports.compression.value,
                'include_patterns': self.quality_reports.include_patterns,
                'exclude_patterns': self.quality_reports.exclude_patterns,
                'max_file_size_mb': self.quality_reports.max_file_size_mb
            },
            'storage': {
                'storage_type': self.storage.storage_type.value,
                'local_path': self.storage.local_path,
                's3_bucket': self.storage.s3_bucket,
                's3_prefix': self.storage.s3_prefix,
                'gcs_bucket': self.storage.gcs_bucket,
                'gcs_prefix': self.storage.gcs_prefix,
                'azure_container': self.storage.azure_container,
                'azure_prefix': self.storage.azure_prefix,
                'credentials_file': self.storage.credentials_file,
                'encryption_enabled': self.storage.encryption_enabled,
                'encryption_key': self.storage.encryption_key
            },
            'notifications': {
                'enabled': self.notifications.enabled,
                'email_recipients': self.notifications.email_recipients,
                'slack_webhook': self.notifications.slack_webhook,
                'teams_webhook': self.notifications.teams_webhook,
                'on_success': self.notifications.on_success,
                'on_failure': self.notifications.on_failure,
                'on_warning': self.notifications.on_warning
            },
            'max_concurrent_backups': self.max_concurrent_backups,
            'backup_timeout_minutes': self.backup_timeout_minutes,
            'verification_enabled': self.verification_enabled,
            'checksum_verification': self.checksum_verification,
            'backup_metadata': self.backup_metadata
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BackupConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update general settings
        config.backup_name = config_dict.get('backup_name', config.backup_name)
        config.description = config_dict.get('description', config.description)
        config.version = config_dict.get('version', config.version)
        
        # Update database configurations
        if 'postgresql' in config_dict:
            pg_config = config_dict['postgresql']
            config.postgresql.enabled = pg_config.get('enabled', config.postgresql.enabled)
            config.postgresql.backup_type = BackupType(pg_config.get('backup_type', config.postgresql.backup_type.value))
            config.postgresql.frequency = BackupFrequency(pg_config.get('frequency', config.postgresql.frequency.value))
            config.postgresql.retention_days = pg_config.get('retention_days', config.postgresql.retention_days)
            config.postgresql.compression = CompressionType(pg_config.get('compression', config.postgresql.compression.value))
        
        if 'neo4j' in config_dict:
            neo4j_config = config_dict['neo4j']
            config.neo4j.enabled = neo4j_config.get('enabled', config.neo4j.enabled)
            config.neo4j.backup_type = BackupType(neo4j_config.get('backup_type', config.neo4j.backup_type.value))
            config.neo4j.frequency = BackupFrequency(neo4j_config.get('frequency', config.neo4j.frequency.value))
            config.neo4j.retention_days = neo4j_config.get('retention_days', config.neo4j.retention_days)
            config.neo4j.compression = CompressionType(neo4j_config.get('compression', config.neo4j.compression.value))
        
        if 'milvus' in config_dict:
            milvus_config = config_dict['milvus']
            config.milvus.enabled = milvus_config.get('enabled', config.milvus.enabled)
            config.milvus.backup_type = BackupType(milvus_config.get('backup_type', config.milvus.backup_type.value))
            config.milvus.frequency = BackupFrequency(milvus_config.get('frequency', config.milvus.frequency.value))
            config.milvus.retention_days = milvus_config.get('retention_days', config.milvus.retention_days)
            config.milvus.compression = CompressionType(milvus_config.get('compression', config.milvus.compression.value))
        
        # Update file configurations
        if 'data_files' in config_dict:
            data_config = config_dict['data_files']
            config.data_files.enabled = data_config.get('enabled', config.data_files.enabled)
            config.data_files.backup_type = BackupType(data_config.get('backup_type', config.data_files.backup_type.value))
            config.data_files.frequency = BackupFrequency(data_config.get('frequency', config.data_files.frequency.value))
            config.data_files.retention_days = data_config.get('retention_days', config.data_files.retention_days)
            config.data_files.compression = CompressionType(data_config.get('compression', config.data_files.compression.value))
            config.data_files.include_patterns = data_config.get('include_patterns', config.data_files.include_patterns)
            config.data_files.exclude_patterns = data_config.get('exclude_patterns', config.data_files.exclude_patterns)
        
        # Update storage configuration
        if 'storage' in config_dict:
            storage_config = config_dict['storage']
            config.storage.storage_type = StorageType(storage_config.get('storage_type', config.storage.storage_type.value))
            config.storage.local_path = storage_config.get('local_path', config.storage.local_path)
            config.storage.s3_bucket = storage_config.get('s3_bucket', config.storage.s3_bucket)
            config.storage.s3_prefix = storage_config.get('s3_prefix', config.storage.s3_prefix)
            config.storage.gcs_bucket = storage_config.get('gcs_bucket', config.storage.gcs_bucket)
            config.storage.gcs_prefix = storage_config.get('gcs_prefix', config.storage.gcs_prefix)
            config.storage.azure_container = storage_config.get('azure_container', config.storage.azure_container)
            config.storage.azure_prefix = storage_config.get('azure_prefix', config.storage.azure_prefix)
            config.storage.encryption_enabled = storage_config.get('encryption_enabled', config.storage.encryption_enabled)
        
        # Update notification configuration
        if 'notifications' in config_dict:
            notif_config = config_dict['notifications']
            config.notifications.enabled = notif_config.get('enabled', config.notifications.enabled)
            config.notifications.email_recipients = notif_config.get('email_recipients', config.notifications.email_recipients)
            config.notifications.slack_webhook = notif_config.get('slack_webhook', config.notifications.slack_webhook)
            config.notifications.teams_webhook = notif_config.get('teams_webhook', config.notifications.teams_webhook)
            config.notifications.on_success = notif_config.get('on_success', config.notifications.on_success)
            config.notifications.on_failure = notif_config.get('on_failure', config.notifications.on_failure)
            config.notifications.on_warning = notif_config.get('on_warning', config.notifications.on_warning)
        
        # Update advanced settings
        config.max_concurrent_backups = config_dict.get('max_concurrent_backups', config.max_concurrent_backups)
        config.backup_timeout_minutes = config_dict.get('backup_timeout_minutes', config.backup_timeout_minutes)
        config.verification_enabled = config_dict.get('verification_enabled', config.verification_enabled)
        config.checksum_verification = config_dict.get('checksum_verification', config.checksum_verification)
        config.backup_metadata = config_dict.get('backup_metadata', config.backup_metadata)
        
        return config
    
    def save_to_file(self, filepath: str):
        """Save configuration to file."""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'BackupConfig':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate database configurations
        if self.postgresql.enabled and self.postgresql.retention_days < 1:
            errors.append("PostgreSQL retention days must be at least 1")
        
        if self.neo4j.enabled and self.neo4j.retention_days < 1:
            errors.append("Neo4j retention days must be at least 1")
        
        if self.milvus.enabled and self.milvus.retention_days < 1:
            errors.append("Milvus retention days must be at least 1")
        
        # Validate file configurations
        if self.data_files.enabled and self.data_files.retention_days < 1:
            errors.append("Data files retention days must be at least 1")
        
        if self.logs.enabled and self.logs.retention_days < 1:
            errors.append("Logs retention days must be at least 1")
        
        # Validate storage configuration
        if self.storage.storage_type == StorageType.S3 and not self.storage.s3_bucket:
            errors.append("S3 bucket must be specified for S3 storage type")
        
        if self.storage.storage_type == StorageType.GCS and not self.storage.gcs_bucket:
            errors.append("GCS bucket must be specified for GCS storage type")
        
        if self.storage.storage_type == StorageType.AZURE and not self.storage.azure_container:
            errors.append("Azure container must be specified for Azure storage type")
        
        # Validate notification configuration
        if self.notifications.enabled:
            if not self.notifications.email_recipients and not self.notifications.slack_webhook and not self.notifications.teams_webhook:
                errors.append("At least one notification method must be configured")
        
        # Validate advanced settings
        if self.max_concurrent_backups < 1:
            errors.append("Max concurrent backups must be at least 1")
        
        if self.backup_timeout_minutes < 1:
            errors.append("Backup timeout must be at least 1 minute")
        
        return errors 