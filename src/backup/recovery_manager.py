"""
Recovery Manager Module

Handles data recovery and restoration from backups for the movie recommendation system.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
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
from .backup_manager import BackupResult

logger = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    success: bool
    recovery_id: str
    backup_id: str
    source: str
    destination: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class RecoveryManager:
    """Recovery manager for restoring data from backups."""
    
    def __init__(self, config: BackupConfig):
        """
        Initialize the recovery manager.
        
        Args:
            config: Backup configuration
        """
        self.config = config
        self.recovery_history: List[RecoveryResult] = []
        self.current_recoveries: Dict[str, RecoveryResult] = {}
        
        # Initialize storage handlers (same as backup manager)
        self._init_storage_handlers()
    
    def _init_storage_handlers(self):
        """Initialize storage handlers based on configuration."""
        self.storage_handlers = {}
        
        if self.config.storage.storage_type == StorageType.S3:
            try:
                import boto3
                self.storage_handlers['s3'] = boto3.client('s3')
                logger.info("S3 storage handler initialized")
            except ImportError:
                logger.warning("boto3 not available, S3 recovery disabled")
        
        elif self.config.storage.storage_type == StorageType.GCS:
            try:
                from google.cloud import storage
                self.storage_handlers['gcs'] = storage.Client()
                logger.info("GCS storage handler initialized")
            except ImportError:
                logger.warning("google-cloud-storage not available, GCS recovery disabled")
        
        elif self.config.storage.storage_type == StorageType.AZURE:
            try:
                from azure.storage.blob import BlobServiceClient
                self.storage_handlers['azure'] = BlobServiceClient.from_connection_string(
                    os.getenv('AZURE_STORAGE_CONNECTION_STRING', '')
                )
                logger.info("Azure storage handler initialized")
            except ImportError:
                logger.warning("azure-storage-blob not available, Azure recovery disabled")
    
    def list_available_backups(self, backup_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backups for recovery."""
        if self.config.storage.storage_type == StorageType.LOCAL:
            return self._list_local_backups(backup_type)
        else:
            return self._list_remote_backups(backup_type)
    
    def _list_local_backups(self, backup_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List backups stored locally."""
        backup_path = Path(self.config.storage.local_path)
        if not backup_path.exists():
            return []
        
        backups = []
        
        if backup_type:
            type_paths = [backup_path / backup_type]
        else:
            type_paths = [backup_path / t for t in ['postgresql', 'neo4j', 'milvus', 'data_files', 'logs', 'quality_reports']]
        
        for type_path in type_paths:
            if type_path.exists():
                for backup_file in type_path.glob("*"):
                    if backup_file.is_file():
                        backup_info = self._parse_backup_filename(backup_file.name)
                        if backup_info:
                            backup_info['source'] = type_path.name
                            backup_info['path'] = str(backup_file)
                            backup_info['size_bytes'] = backup_file.stat().st_size
                            backup_info['modified_time'] = datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat()
                            backups.append(backup_info)
        
        return sorted(backups, key=lambda b: b['timestamp'], reverse=True)
    
    def _list_remote_backups(self, backup_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List backups stored remotely."""
        backups = []
        
        if self.config.storage.storage_type == StorageType.S3 and 's3' in self.storage_handlers:
            s3_client = self.storage_handlers['s3']
            prefix = f"{self.config.storage.s3_prefix}/"
            
            if backup_type:
                prefix += f"{backup_type}/"
            
            response = s3_client.list_objects_v2(
                Bucket=self.config.storage.s3_bucket,
                Prefix=prefix
            )
            
            for obj in response.get('Contents', []):
                backup_info = self._parse_backup_filename(Path(obj['Key']).name)
                if backup_info:
                    backup_info['source'] = Path(obj['Key']).parent.name
                    backup_info['path'] = f"s3://{self.config.storage.s3_bucket}/{obj['Key']}"
                    backup_info['size_bytes'] = obj['Size']
                    backup_info['modified_time'] = obj['LastModified'].isoformat()
                    backups.append(backup_info)
        
        elif self.config.storage.storage_type == StorageType.GCS and 'gcs' in self.storage_handlers:
            gcs_client = self.storage_handlers['gcs']
            bucket = gcs_client.bucket(self.config.storage.gcs_bucket)
            prefix = f"{self.config.storage.gcs_prefix}/"
            
            if backup_type:
                prefix += f"{backup_type}/"
            
            blobs = bucket.list_blobs(prefix=prefix)
            
            for blob in blobs:
                backup_info = self._parse_backup_filename(Path(blob.name).name)
                if backup_info:
                    backup_info['source'] = Path(blob.name).parent.name
                    backup_info['path'] = f"gs://{self.config.storage.gcs_bucket}/{blob.name}"
                    backup_info['size_bytes'] = blob.size
                    backup_info['modified_time'] = blob.updated.isoformat()
                    backups.append(backup_info)
        
        return sorted(backups, key=lambda b: b['timestamp'], reverse=True)
    
    def _parse_backup_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Parse backup filename to extract metadata."""
        try:
            # Expected format: {backup_type}_{timestamp}.{ext}.{compression}
            parts = filename.split('_')
            if len(parts) < 2:
                return None
            
            backup_type = parts[0]
            timestamp_str = parts[1]
            
            # Parse timestamp (format: YYYYMMDD_HHMMSS)
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
            # Determine compression type
            compression = CompressionType.NONE
            if filename.endswith('.gz'):
                compression = CompressionType.GZIP
            elif filename.endswith('.bz2'):
                compression = CompressionType.BZIP2
            elif filename.endswith('.xz'):
                compression = CompressionType.LZMA
            
            return {
                'backup_type': backup_type,
                'timestamp': timestamp,
                'compression': compression.value,
                'filename': filename
            }
        except Exception:
            return None
    
    def recover_data(self, backup_path: str, destination: str, verify_checksum: bool = True) -> RecoveryResult:
        """
        Recover data from a backup.
        
        Args:
            backup_path: Path to the backup file
            destination: Destination path for recovery
            verify_checksum: Whether to verify backup checksum
            
        Returns:
            RecoveryResult with recovery details
        """
        recovery_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Starting recovery: {recovery_id} from {backup_path}")
        
        try:
            # Download backup if it's remote
            local_backup_path = self._download_backup(backup_path)
            
            # Verify checksum if requested
            if verify_checksum:
                self._verify_backup_checksum(local_backup_path)
            
            # Determine backup type and perform recovery
            backup_info = self._parse_backup_filename(Path(backup_path).name)
            if not backup_info:
                raise ValueError(f"Could not parse backup filename: {backup_path}")
            
            backup_type = backup_info['backup_type']
            
            if backup_type == 'postgresql':
                result = self._recover_postgresql(local_backup_path, destination, recovery_id, start_time)
            elif backup_type == 'neo4j':
                result = self._recover_neo4j(local_backup_path, destination, recovery_id, start_time)
            elif backup_type == 'milvus':
                result = self._recover_milvus(local_backup_path, destination, recovery_id, start_time)
            elif backup_type in ['data_files', 'logs', 'quality_reports']:
                result = self._recover_files(local_backup_path, destination, recovery_id, start_time)
            else:
                raise ValueError(f"Unknown backup type: {backup_type}")
            
            # Clean up temporary files
            if local_backup_path != backup_path:  # Only delete if it was downloaded
                os.unlink(local_backup_path)
            
            # Store recovery result
            self.recovery_history.append(result)
            
            logger.info(f"Completed recovery: {recovery_id}")
            return result
            
        except Exception as e:
            error_msg = f"Recovery failed: {str(e)}"
            logger.error(error_msg)
            
            result = RecoveryResult(
                success=False,
                recovery_id=recovery_id,
                backup_id=Path(backup_path).name,
                source=backup_path,
                destination=destination,
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error_message=error_msg
            )
            
            self.recovery_history.append(result)
            return result
    
    def _download_backup(self, backup_path: str) -> str:
        """Download backup from remote storage if needed."""
        if backup_path.startswith('s3://'):
            return self._download_from_s3(backup_path)
        elif backup_path.startswith('gs://'):
            return self._download_from_gcs(backup_path)
        elif backup_path.startswith('azure://'):
            return self._download_from_azure(backup_path)
        else:
            return backup_path  # Local file
    
    def _download_from_s3(self, s3_path: str) -> str:
        """Download backup from S3."""
        if 's3' not in self.storage_handlers:
            raise Exception("S3 storage handler not available")
        
        # Parse S3 path: s3://bucket/key
        parts = s3_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Download file
        s3_client = self.storage_handlers['s3']
        s3_client.download_file(bucket, key, temp_path)
        
        return temp_path
    
    def _download_from_gcs(self, gcs_path: str) -> str:
        """Download backup from GCS."""
        if 'gcs' not in self.storage_handlers:
            raise Exception("GCS storage handler not available")
        
        # Parse GCS path: gs://bucket/key
        parts = gcs_path.replace('gs://', '').split('/', 1)
        bucket_name = parts[0]
        blob_name = parts[1]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Download file
        gcs_client = self.storage_handlers['gcs']
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(temp_path)
        
        return temp_path
    
    def _download_from_azure(self, azure_path: str) -> str:
        """Download backup from Azure."""
        if 'azure' not in self.storage_handlers:
            raise Exception("Azure storage handler not available")
        
        # Parse Azure path: azure://container/blob
        parts = azure_path.replace('azure://', '').split('/', 1)
        container_name = parts[0]
        blob_name = parts[1]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Download file
        azure_client = self.storage_handlers['azure']
        container_client = azure_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        
        with open(temp_path, 'wb') as f:
            blob_data = blob_client.download_blob()
            f.write(blob_data.readall())
        
        return temp_path
    
    def _verify_backup_checksum(self, backup_path: str):
        """Verify backup file checksum."""
        # In a real implementation, you'd compare against stored checksum
        # For now, just verify the file exists and is readable
        if not os.path.exists(backup_path):
            raise Exception(f"Backup file not found: {backup_path}")
        
        # Calculate checksum for verification
        checksum = self._calculate_checksum(backup_path)
        logger.info(f"Backup checksum: {checksum}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _recover_postgresql(self, backup_path: str, destination: str, recovery_id: str, start_time: datetime) -> RecoveryResult:
        """Recover PostgreSQL database."""
        try:
            # Decompress if needed
            decompressed_path = self._decompress_file(backup_path)
            
            # Create destination directory
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)
            
            # Run pg_restore or psql depending on backup format
            if backup_path.endswith('.sql'):
                # SQL dump format
                cmd = [
                    'psql',
                    '-h', 'localhost',
                    '-U', 'postgres',
                    '-d', 'movie_recommendation',
                    '-f', decompressed_path
                ]
            else:
                # Custom format
                cmd = [
                    'pg_restore',
                    '-h', 'localhost',
                    '-U', 'postgres',
                    '-d', 'movie_recommendation',
                    '--verbose',
                    '--clean',
                    '--if-exists',
                    decompressed_path
                ]
            
            # Set environment variables for authentication
            env = os.environ.copy()
            env['PGPASSWORD'] = os.getenv('POSTGRES_PASSWORD', 'password')
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                raise Exception(f"PostgreSQL recovery failed: {result.stderr}")
            
            # Clean up decompressed file
            if decompressed_path != backup_path:
                os.unlink(decompressed_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return RecoveryResult(
                success=True,
                recovery_id=recovery_id,
                backup_id=Path(backup_path).name,
                source=backup_path,
                destination=destination,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metadata={
                    'database': 'postgresql',
                    'restored_tables': 'all'  # In a real implementation, you'd track specific tables
                }
            )
            
        except Exception as e:
            raise e
    
    def _recover_neo4j(self, backup_path: str, destination: str, recovery_id: str, start_time: datetime) -> RecoveryResult:
        """Recover Neo4j database."""
        try:
            # Decompress if needed
            decompressed_path = self._decompress_file(backup_path)
            
            # Extract archive
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract tar archive
                shutil.unpack_archive(decompressed_path, temp_path, 'tar')
                
                # For Neo4j, you'd typically restore to the data directory
                # In a real implementation, you'd use Neo4j restore commands
                neo4j_data_path = Path(destination)
                neo4j_data_path.mkdir(parents=True, exist_ok=True)
                
                # Copy extracted data
                extracted_data = next(temp_path.iterdir())
                if extracted_data.is_dir():
                    shutil.copytree(extracted_data, neo4j_data_path, dirs_exist_ok=True)
            
            # Clean up decompressed file
            if decompressed_path != backup_path:
                os.unlink(decompressed_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return RecoveryResult(
                success=True,
                recovery_id=recovery_id,
                backup_id=Path(backup_path).name,
                source=backup_path,
                destination=destination,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metadata={
                    'database': 'neo4j',
                    'restored_nodes': 'all',
                    'restored_relationships': 'all'
                }
            )
            
        except Exception as e:
            raise e
    
    def _recover_milvus(self, backup_path: str, destination: str, recovery_id: str, start_time: datetime) -> RecoveryResult:
        """Recover Milvus database."""
        try:
            # Decompress if needed
            decompressed_path = self._decompress_file(backup_path)
            
            # Extract archive
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract tar archive
                shutil.unpack_archive(decompressed_path, temp_path, 'tar')
                
                # For Milvus, restore to data directory
                milvus_data_path = Path(destination)
                milvus_data_path.mkdir(parents=True, exist_ok=True)
                
                # Copy extracted data
                extracted_data = next(temp_path.iterdir())
                if extracted_data.is_dir():
                    shutil.copytree(extracted_data, milvus_data_path, dirs_exist_ok=True)
            
            # Clean up decompressed file
            if decompressed_path != backup_path:
                os.unlink(decompressed_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return RecoveryResult(
                success=True,
                recovery_id=recovery_id,
                backup_id=Path(backup_path).name,
                source=backup_path,
                destination=destination,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metadata={
                    'database': 'milvus',
                    'restored_collections': 'all',
                    'restored_vectors': 'all'
                }
            )
            
        except Exception as e:
            raise e
    
    def _recover_files(self, backup_path: str, destination: str, recovery_id: str, start_time: datetime) -> RecoveryResult:
        """Recover file-based data."""
        try:
            # Decompress if needed
            decompressed_path = self._decompress_file(backup_path)
            
            # Extract archive
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)
            
            # Extract tar archive
            shutil.unpack_archive(decompressed_path, dest_path, 'tar')
            
            # Count restored files
            restored_files = list(dest_path.rglob('*'))
            file_count = len([f for f in restored_files if f.is_file()])
            
            # Clean up decompressed file
            if decompressed_path != backup_path:
                os.unlink(decompressed_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return RecoveryResult(
                success=True,
                recovery_id=recovery_id,
                backup_id=Path(backup_path).name,
                source=backup_path,
                destination=destination,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metadata={
                    'file_type': 'data_files',
                    'restored_files': file_count,
                    'destination_path': str(dest_path)
                }
            )
            
        except Exception as e:
            raise e
    
    def _decompress_file(self, file_path: str) -> str:
        """Decompress a file if needed."""
        if file_path.endswith('.gz'):
            decompressed_path = file_path[:-3]
            with gzip.open(file_path, 'rb') as infile:
                with open(decompressed_path, 'wb') as outfile:
                    shutil.copyfileobj(infile, outfile)
            return decompressed_path
        elif file_path.endswith('.bz2'):
            decompressed_path = file_path[:-4]
            with bz2.open(file_path, 'rb') as infile:
                with open(decompressed_path, 'wb') as outfile:
                    shutil.copyfileobj(infile, outfile)
            return decompressed_path
        elif file_path.endswith('.xz'):
            decompressed_path = file_path[:-3]
            with lzma.open(file_path, 'rb') as infile:
                with open(decompressed_path, 'wb') as outfile:
                    shutil.copyfileobj(infile, outfile)
            return decompressed_path
        else:
            return file_path  # No compression
    
    def get_recovery_status(self, recovery_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of recovery operations."""
        if recovery_id:
            recoveries = [r for r in self.recovery_history if r.recovery_id == recovery_id]
        else:
            recoveries = self.recovery_history
        
        if not recoveries:
            return {'status': 'no_recoveries', 'recoveries': []}
        
        latest_recovery = max(recoveries, key=lambda r: r.start_time)
        
        return {
            'status': 'available',
            'total_recoveries': len(recoveries),
            'latest_recovery': {
                'id': latest_recovery.recovery_id,
                'backup_id': latest_recovery.backup_id,
                'source': latest_recovery.source,
                'destination': latest_recovery.destination,
                'start_time': latest_recovery.start_time.isoformat(),
                'duration_seconds': latest_recovery.duration_seconds,
                'success': latest_recovery.success
            },
            'recoveries': [
                {
                    'id': r.recovery_id,
                    'backup_id': r.backup_id,
                    'source': r.source,
                    'destination': r.destination,
                    'start_time': r.start_time.isoformat(),
                    'duration_seconds': r.duration_seconds,
                    'success': r.success
                }
                for r in sorted(recoveries, key=lambda r: r.start_time, reverse=True)
            ]
        } 