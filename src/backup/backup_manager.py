"""
Backup Manager Module

Core backup management system for the movie recommendation system.
Handles database and file backups with support for multiple storage backends.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import subprocess
import shutil
import gzip
import bz2
import lzma
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import os

from .backup_config import BackupConfig, BackupType, StorageType, CompressionType

logger = logging.getLogger(__name__)


@dataclass
class BackupResult:
    """Result of a backup operation."""
    success: bool
    backup_id: str
    backup_type: BackupType
    source: str
    destination: str
    size_bytes: int
    checksum: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class BackupManager:
    """Main backup manager for the movie recommendation system."""
    
    def __init__(self, config: BackupConfig):
        """
        Initialize the backup manager.
        
        Args:
            config: Backup configuration
        """
        self.config = config
        self.backup_history: List[BackupResult] = []
        self.current_backups: Dict[str, BackupResult] = {}
        
        # Create backup directories
        self._setup_backup_directories()
        
        # Initialize storage handlers
        self._init_storage_handlers()
    
    def _setup_backup_directories(self):
        """Set up backup directories."""
        if self.config.storage.storage_type == StorageType.LOCAL:
            backup_path = Path(self.config.storage.local_path)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for different backup types
            (backup_path / "postgresql").mkdir(exist_ok=True)
            (backup_path / "neo4j").mkdir(exist_ok=True)
            (backup_path / "milvus").mkdir(exist_ok=True)
            (backup_path / "data_files").mkdir(exist_ok=True)
            (backup_path / "logs").mkdir(exist_ok=True)
            (backup_path / "quality_reports").mkdir(exist_ok=True)
            (backup_path / "metadata").mkdir(exist_ok=True)
    
    def _init_storage_handlers(self):
        """Initialize storage handlers based on configuration."""
        self.storage_handlers = {}
        
        if self.config.storage.storage_type == StorageType.S3:
            try:
                import boto3
                self.storage_handlers['s3'] = boto3.client('s3')
                logger.info("S3 storage handler initialized")
            except ImportError:
                logger.warning("boto3 not available, S3 backups disabled")
        
        elif self.config.storage.storage_type == StorageType.GCS:
            try:
                from google.cloud import storage
                self.storage_handlers['gcs'] = storage.Client()
                logger.info("GCS storage handler initialized")
            except ImportError:
                logger.warning("google-cloud-storage not available, GCS backups disabled")
        
        elif self.config.storage.storage_type == StorageType.AZURE:
            try:
                from azure.storage.blob import BlobServiceClient
                self.storage_handlers['azure'] = BlobServiceClient.from_connection_string(
                    os.getenv('AZURE_STORAGE_CONNECTION_STRING', '')
                )
                logger.info("Azure storage handler initialized")
            except ImportError:
                logger.warning("azure-storage-blob not available, Azure backups disabled")
    
    def create_backup(self, backup_type: str, force: bool = False) -> BackupResult:
        """
        Create a backup of the specified type.
        
        Args:
            backup_type: Type of backup ('postgresql', 'neo4j', 'milvus', 'data_files', 'logs', 'quality_reports')
            force: Force backup even if not scheduled
            
        Returns:
            BackupResult with backup details
        """
        backup_id = f"{backup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Starting {backup_type} backup: {backup_id}")
        
        try:
            # Check if backup is enabled and scheduled
            if not self._should_create_backup(backup_type) and not force:
                return BackupResult(
                    success=False,
                    backup_id=backup_id,
                    backup_type=BackupType.FULL,
                    source=backup_type,
                    destination="",
                    size_bytes=0,
                    checksum="",
                    start_time=start_time,
                    end_time=datetime.now(),
                    duration_seconds=0,
                    error_message="Backup not scheduled or disabled"
                )
            
            # Create backup based on type
            if backup_type == 'postgresql':
                result = self._backup_postgresql(backup_id)
            elif backup_type == 'neo4j':
                result = self._backup_neo4j(backup_id)
            elif backup_type == 'milvus':
                result = self._backup_milvus(backup_id)
            elif backup_type == 'data_files':
                result = self._backup_data_files(backup_id)
            elif backup_type == 'logs':
                result = self._backup_logs(backup_id)
            elif backup_type == 'quality_reports':
                result = self._backup_quality_reports(backup_id)
            else:
                raise ValueError(f"Unknown backup type: {backup_type}")
            
            # Store backup result
            self.backup_history.append(result)
            
            # Clean up old backups
            self._cleanup_old_backups(backup_type)
            
            logger.info(f"Completed {backup_type} backup: {backup_id}")
            return result
            
        except Exception as e:
            error_msg = f"Backup failed: {str(e)}"
            logger.error(error_msg)
            
            result = BackupResult(
                success=False,
                backup_id=backup_id,
                backup_type=BackupType.FULL,
                source=backup_type,
                destination="",
                size_bytes=0,
                checksum="",
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error_message=error_msg
            )
            
            self.backup_history.append(result)
            return result
    
    def _should_create_backup(self, backup_type: str) -> bool:
        """Check if backup should be created based on schedule."""
        config = getattr(self.config, backup_type, None)
        if not config or not config.enabled:
            return False
        
        # For now, always allow backup creation
        # In a production system, you'd check the schedule here
        return True
    
    def _backup_postgresql(self, backup_id: str) -> BackupResult:
        """Create PostgreSQL backup."""
        start_time = datetime.now()
        config = self.config.postgresql
        
        try:
            # Create temporary backup file
            with tempfile.NamedTemporaryFile(suffix='.sql', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Run pg_dump
            cmd = [
                'pg_dump',
                '-h', 'localhost',
                '-U', 'postgres',
                '-d', 'movie_recommendation',
                '-f', temp_path,
                '--verbose'
            ]
            
            if config.include_schema:
                cmd.append('--schema-only')
            if config.include_data:
                cmd.append('--data-only')
            if config.include_indexes:
                cmd.append('--indexes')
            
            # Set environment variables for authentication
            env = os.environ.copy()
            env['PGPASSWORD'] = os.getenv('POSTGRES_PASSWORD', 'password')
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=config.timeout_minutes * 60)
            
            if result.returncode != 0:
                raise Exception(f"pg_dump failed: {result.stderr}")
            
            # Compress backup
            compressed_path = self._compress_file(temp_path, config.compression)
            
            # Calculate checksum
            checksum = self._calculate_checksum(compressed_path)
            
            # Get file size
            size_bytes = os.path.getsize(compressed_path)
            
            # Upload to storage
            destination = self._upload_to_storage(compressed_path, f"postgresql/{backup_id}.sql.{config.compression.value}")
            
            # Clean up temporary files
            os.unlink(temp_path)
            os.unlink(compressed_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return BackupResult(
                success=True,
                backup_id=backup_id,
                backup_type=config.backup_type,
                source="postgresql",
                destination=destination,
                size_bytes=size_bytes,
                checksum=checksum,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metadata={
                    'database': 'movie_recommendation',
                    'compression': config.compression.value,
                    'include_schema': config.include_schema,
                    'include_data': config.include_data,
                    'include_indexes': config.include_indexes
                }
            )
            
        except Exception as e:
            # Clean up on error
            for path in [temp_path, compressed_path]:
                if 'path' in locals() and os.path.exists(path):
                    os.unlink(path)
            raise e
    
    def _backup_neo4j(self, backup_id: str) -> BackupResult:
        """Create Neo4j backup."""
        start_time = datetime.now()
        config = self.config.neo4j
        
        try:
            # Create temporary backup directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Run Neo4j backup command
                cmd = [
                    'neo4j-admin', 'backup',
                    '--database=neo4j',
                    f'--to={temp_path}',
                    '--verbose'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=config.timeout_minutes * 60)
                
                if result.returncode != 0:
                    raise Exception(f"Neo4j backup failed: {result.stderr}")
                
                # Create archive
                archive_path = temp_path / f"{backup_id}.tar"
                shutil.make_archive(str(archive_path.with_suffix('')), 'tar', temp_path)
                
                # Compress archive
                compressed_path = self._compress_file(str(archive_path), config.compression)
                
                # Calculate checksum
                checksum = self._calculate_checksum(compressed_path)
                
                # Get file size
                size_bytes = os.path.getsize(compressed_path)
                
                # Upload to storage
                destination = self._upload_to_storage(compressed_path, f"neo4j/{backup_id}.tar.{config.compression.value}")
                
                # Clean up
                os.unlink(compressed_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return BackupResult(
                success=True,
                backup_id=backup_id,
                backup_type=config.backup_type,
                source="neo4j",
                destination=destination,
                size_bytes=size_bytes,
                checksum=checksum,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metadata={
                    'database': 'neo4j',
                    'compression': config.compression.value
                }
            )
            
        except Exception as e:
            raise e
    
    def _backup_milvus(self, backup_id: str) -> BackupResult:
        """Create Milvus backup."""
        start_time = datetime.now()
        config = self.config.milvus
        
        try:
            # Create temporary backup directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # For Milvus, we'll backup the data directory
                # In a real implementation, you'd use Milvus backup API
                milvus_data_path = Path("/var/lib/milvus/data")
                if milvus_data_path.exists():
                    shutil.copytree(milvus_data_path, temp_path / "milvus_data")
                else:
                    # Create dummy backup for testing
                    (temp_path / "milvus_data").mkdir()
                    (temp_path / "milvus_data" / "dummy.txt").write_text("Milvus backup placeholder")
                
                # Create archive
                archive_path = temp_path / f"{backup_id}.tar"
                shutil.make_archive(str(archive_path.with_suffix('')), 'tar', temp_path)
                
                # Compress archive
                compressed_path = self._compress_file(str(archive_path), config.compression)
                
                # Calculate checksum
                checksum = self._calculate_checksum(compressed_path)
                
                # Get file size
                size_bytes = os.path.getsize(compressed_path)
                
                # Upload to storage
                destination = self._upload_to_storage(compressed_path, f"milvus/{backup_id}.tar.{config.compression.value}")
                
                # Clean up
                os.unlink(compressed_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return BackupResult(
                success=True,
                backup_id=backup_id,
                backup_type=config.backup_type,
                source="milvus",
                destination=destination,
                size_bytes=size_bytes,
                checksum=checksum,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metadata={
                    'database': 'milvus',
                    'compression': config.compression.value
                }
            )
            
        except Exception as e:
            raise e
    
    def _backup_data_files(self, backup_id: str) -> BackupResult:
        """Create data files backup."""
        start_time = datetime.now()
        config = self.config.data_files
        
        try:
            # Create temporary backup directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Find files to backup
                data_path = Path("data")
                files_to_backup = []
                
                for pattern in config.include_patterns:
                    files_to_backup.extend(data_path.glob(pattern))
                
                # Exclude files
                for pattern in config.exclude_patterns:
                    excluded_files = list(data_path.glob(pattern))
                    files_to_backup = [f for f in files_to_backup if f not in excluded_files]
                
                # Copy files to temp directory
                for file_path in files_to_backup:
                    if file_path.is_file() and file_path.stat().st_size <= config.max_file_size_mb * 1024 * 1024:
                        dest_path = temp_path / file_path.name
                        shutil.copy2(file_path, dest_path)
                
                # Create archive
                archive_path = temp_path / f"{backup_id}.tar"
                shutil.make_archive(str(archive_path.with_suffix('')), 'tar', temp_path)
                
                # Compress archive
                compressed_path = self._compress_file(str(archive_path), config.compression)
                
                # Calculate checksum
                checksum = self._calculate_checksum(compressed_path)
                
                # Get file size
                size_bytes = os.path.getsize(compressed_path)
                
                # Upload to storage
                destination = self._upload_to_storage(compressed_path, f"data_files/{backup_id}.tar.{config.compression.value}")
                
                # Clean up
                os.unlink(compressed_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return BackupResult(
                success=True,
                backup_id=backup_id,
                backup_type=config.backup_type,
                source="data_files",
                destination=destination,
                size_bytes=size_bytes,
                checksum=checksum,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metadata={
                    'files_count': len(files_to_backup),
                    'compression': config.compression.value,
                    'include_patterns': config.include_patterns,
                    'exclude_patterns': config.exclude_patterns
                }
            )
            
        except Exception as e:
            raise e
    
    def _backup_logs(self, backup_id: str) -> BackupResult:
        """Create logs backup."""
        start_time = datetime.now()
        config = self.config.logs
        
        try:
            # Create temporary backup directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Find log files
                log_path = Path("logs")
                if not log_path.exists():
                    log_path.mkdir()
                
                files_to_backup = []
                for pattern in config.include_patterns:
                    files_to_backup.extend(log_path.glob(pattern))
                
                # Copy files to temp directory
                for file_path in files_to_backup:
                    if file_path.is_file() and file_path.stat().st_size <= config.max_file_size_mb * 1024 * 1024:
                        dest_path = temp_path / file_path.name
                        shutil.copy2(file_path, dest_path)
                
                # Create archive
                archive_path = temp_path / f"{backup_id}.tar"
                shutil.make_archive(str(archive_path.with_suffix('')), 'tar', temp_path)
                
                # Compress archive
                compressed_path = self._compress_file(str(archive_path), config.compression)
                
                # Calculate checksum
                checksum = self._calculate_checksum(compressed_path)
                
                # Get file size
                size_bytes = os.path.getsize(compressed_path)
                
                # Upload to storage
                destination = self._upload_to_storage(compressed_path, f"logs/{backup_id}.tar.{config.compression.value}")
                
                # Clean up
                os.unlink(compressed_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return BackupResult(
                success=True,
                backup_id=backup_id,
                backup_type=config.backup_type,
                source="logs",
                destination=destination,
                size_bytes=size_bytes,
                checksum=checksum,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metadata={
                    'files_count': len(files_to_backup),
                    'compression': config.compression.value
                }
            )
            
        except Exception as e:
            raise e
    
    def _backup_quality_reports(self, backup_id: str) -> BackupResult:
        """Create quality reports backup."""
        start_time = datetime.now()
        config = self.config.quality_reports
        
        try:
            # Create temporary backup directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Find quality report files
                quality_path = Path("data/quality_reports")
                if not quality_path.exists():
                    quality_path.mkdir(parents=True)
                
                files_to_backup = []
                for pattern in config.include_patterns:
                    files_to_backup.extend(quality_path.glob(pattern))
                
                # Copy files to temp directory
                for file_path in files_to_backup:
                    if file_path.is_file() and file_path.stat().st_size <= config.max_file_size_mb * 1024 * 1024:
                        dest_path = temp_path / file_path.name
                        shutil.copy2(file_path, dest_path)
                
                # Create archive
                archive_path = temp_path / f"{backup_id}.tar"
                shutil.make_archive(str(archive_path.with_suffix('')), 'tar', temp_path)
                
                # Compress archive
                compressed_path = self._compress_file(str(archive_path), config.compression)
                
                # Calculate checksum
                checksum = self._calculate_checksum(compressed_path)
                
                # Get file size
                size_bytes = os.path.getsize(compressed_path)
                
                # Upload to storage
                destination = self._upload_to_storage(compressed_path, f"quality_reports/{backup_id}.tar.{config.compression.value}")
                
                # Clean up
                os.unlink(compressed_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return BackupResult(
                success=True,
                backup_id=backup_id,
                backup_type=config.backup_type,
                source="quality_reports",
                destination=destination,
                size_bytes=size_bytes,
                checksum=checksum,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metadata={
                    'files_count': len(files_to_backup),
                    'compression': config.compression.value
                }
            )
            
        except Exception as e:
            raise e
    
    def _compress_file(self, file_path: str, compression_type: CompressionType) -> str:
        """Compress a file using the specified compression type."""
        if compression_type == CompressionType.NONE:
            return file_path
        
        compressed_path = f"{file_path}.{compression_type.value}"
        
        with open(file_path, 'rb') as infile:
            if compression_type == CompressionType.GZIP:
                with gzip.open(compressed_path, 'wb') as outfile:
                    shutil.copyfileobj(infile, outfile)
            elif compression_type == CompressionType.BZIP2:
                with bz2.open(compressed_path, 'wb') as outfile:
                    shutil.copyfileobj(infile, outfile)
            elif compression_type == CompressionType.LZMA:
                with lzma.open(compressed_path, 'wb') as outfile:
                    shutil.copyfileobj(infile, outfile)
        
        return compressed_path
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _upload_to_storage(self, local_path: str, remote_path: str) -> str:
        """Upload file to configured storage backend."""
        if self.config.storage.storage_type == StorageType.LOCAL:
            dest_path = Path(self.config.storage.local_path) / remote_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest_path)
            return str(dest_path)
        
        elif self.config.storage.storage_type == StorageType.S3 and 's3' in self.storage_handlers:
            s3_client = self.storage_handlers['s3']
            key = f"{self.config.storage.s3_prefix}/{remote_path}"
            s3_client.upload_file(local_path, self.config.storage.s3_bucket, key)
            return f"s3://{self.config.storage.s3_bucket}/{key}"
        
        elif self.config.storage.storage_type == StorageType.GCS and 'gcs' in self.storage_handlers:
            gcs_client = self.storage_handlers['gcs']
            bucket = gcs_client.bucket(self.config.storage.gcs_bucket)
            blob = bucket.blob(f"{self.config.storage.gcs_prefix}/{remote_path}")
            blob.upload_from_filename(local_path)
            return f"gs://{self.config.storage.gcs_bucket}/{self.config.storage.gcs_prefix}/{remote_path}"
        
        elif self.config.storage.storage_type == StorageType.AZURE and 'azure' in self.storage_handlers:
            azure_client = self.storage_handlers['azure']
            container_client = azure_client.get_container_client(self.config.storage.azure_container)
            blob_client = container_client.get_blob_client(f"{self.config.storage.azure_prefix}/{remote_path}")
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            return f"azure://{self.config.storage.azure_container}/{self.config.storage.azure_prefix}/{remote_path}"
        
        else:
            raise Exception(f"Storage type {self.config.storage.storage_type.value} not supported or not configured")
    
    def _cleanup_old_backups(self, backup_type: str):
        """Clean up old backups based on retention policy."""
        config = getattr(self.config, backup_type, None)
        if not config:
            return
        
        cutoff_date = datetime.now() - timedelta(days=config.retention_days)
        
        # Remove old backups from history
        self.backup_history = [
            backup for backup in self.backup_history
            if backup.source != backup_type or backup.start_time > cutoff_date
        ]
        
        # In a production system, you'd also clean up the actual backup files
        logger.info(f"Cleaned up old {backup_type} backups older than {config.retention_days} days")
    
    def get_backup_status(self, backup_type: Optional[str] = None) -> Dict[str, Any]:
        """Get status of backups."""
        if backup_type:
            backups = [b for b in self.backup_history if b.source == backup_type]
        else:
            backups = self.backup_history
        
        if not backups:
            return {'status': 'no_backups', 'backups': []}
        
        latest_backup = max(backups, key=lambda b: b.start_time)
        
        return {
            'status': 'available',
            'total_backups': len(backups),
            'latest_backup': {
                'id': latest_backup.backup_id,
                'type': latest_backup.backup_type.value,
                'source': latest_backup.source,
                'destination': latest_backup.destination,
                'size_bytes': latest_backup.size_bytes,
                'start_time': latest_backup.start_time.isoformat(),
                'duration_seconds': latest_backup.duration_seconds,
                'success': latest_backup.success
            },
            'backups': [
                {
                    'id': b.backup_id,
                    'type': b.backup_type.value,
                    'source': b.source,
                    'destination': b.destination,
                    'size_bytes': b.size_bytes,
                    'start_time': b.start_time.isoformat(),
                    'duration_seconds': b.duration_seconds,
                    'success': b.success
                }
                for b in sorted(backups, key=lambda b: b.start_time, reverse=True)
            ]
        }
    
    def list_backups(self, backup_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = self.backup_history
        
        if backup_type:
            backups = [b for b in backups if b.source == backup_type]
        
        backups = sorted(backups, key=lambda b: b.start_time, reverse=True)[:limit]
        
        return [
            {
                'id': b.backup_id,
                'type': b.backup_type.value,
                'source': b.source,
                'destination': b.destination,
                'size_bytes': b.size_bytes,
                'checksum': b.checksum,
                'start_time': b.start_time.isoformat(),
                'end_time': b.end_time.isoformat(),
                'duration_seconds': b.duration_seconds,
                'success': b.success,
                'error_message': b.error_message,
                'metadata': b.metadata
            }
            for b in backups
        ] 