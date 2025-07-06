"""
Backup Validator Module

Validates backup integrity and completeness for the movie recommendation system.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import os
import gzip
import bz2
import lzma
import shutil

from .backup_config import BackupConfig, CompressionType
from .backup_manager import BackupResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a backup validation."""
    success: bool
    backup_id: str
    validation_type: str
    checksum_valid: bool
    size_valid: bool
    format_valid: bool
    content_valid: bool
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = None


class BackupValidator:
    """Validator for backup integrity and completeness."""
    
    def __init__(self, config: BackupConfig):
        """
        Initialize the backup validator.
        
        Args:
            config: Backup configuration
        """
        self.config = config
        self.validation_history: List[ValidationResult] = []
    
    def validate_backup(self, backup_result: BackupResult) -> ValidationResult:
        """
        Validate a backup result.
        
        Args:
            backup_result: Backup result to validate
            
        Returns:
            ValidationResult with validation details
        """
        validation_id = f"validation_{backup_result.backup_id}"
        start_time = datetime.now()
        
        logger.info(f"Starting backup validation: {validation_id}")
        
        try:
            # Perform different types of validation
            checksum_valid = self._validate_checksum(backup_result)
            size_valid = self._validate_size(backup_result)
            format_valid = self._validate_format(backup_result)
            content_valid = self._validate_content(backup_result)
            
            # Overall validation success
            success = checksum_valid and size_valid and format_valid and content_valid
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = ValidationResult(
                success=success,
                backup_id=backup_result.backup_id,
                validation_type="comprehensive",
                checksum_valid=checksum_valid,
                size_valid=size_valid,
                format_valid=format_valid,
                content_valid=content_valid,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                details={
                    'backup_type': backup_result.backup_type.value,
                    'source': backup_result.source,
                    'destination': backup_result.destination,
                    'size_bytes': backup_result.size_bytes,
                    'checksum': backup_result.checksum
                }
            )
            
            self.validation_history.append(result)
            
            if success:
                logger.info(f"Backup validation successful: {validation_id}")
            else:
                logger.warning(f"Backup validation failed: {validation_id}")
            
            return result
            
        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            logger.error(error_msg)
            
            result = ValidationResult(
                success=False,
                backup_id=backup_result.backup_id,
                validation_type="comprehensive",
                checksum_valid=False,
                size_valid=False,
                format_valid=False,
                content_valid=False,
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error_message=error_msg
            )
            
            self.validation_history.append(result)
            return result
    
    def _validate_checksum(self, backup_result: BackupResult) -> bool:
        """Validate backup file checksum."""
        try:
            if not backup_result.checksum:
                logger.warning("No checksum provided for validation")
                return False
            
            # Get backup file path
            backup_path = self._get_backup_file_path(backup_result)
            if not backup_path or not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Calculate current checksum
            current_checksum = self._calculate_checksum(backup_path)
            
            # Compare checksums
            checksum_valid = current_checksum == backup_result.checksum
            
            if not checksum_valid:
                logger.error(f"Checksum mismatch: expected {backup_result.checksum}, got {current_checksum}")
            
            return checksum_valid
            
        except Exception as e:
            logger.error(f"Checksum validation failed: {e}")
            return False
    
    def _validate_size(self, backup_result: BackupResult) -> bool:
        """Validate backup file size."""
        try:
            backup_path = self._get_backup_file_path(backup_result)
            if not backup_path or not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Get actual file size
            actual_size = os.path.getsize(backup_path)
            
            # Compare sizes (allow small tolerance for metadata)
            size_tolerance = 1024  # 1KB tolerance
            size_valid = abs(actual_size - backup_result.size_bytes) <= size_tolerance
            
            if not size_valid:
                logger.error(f"Size mismatch: expected {backup_result.size_bytes}, got {actual_size}")
            
            return size_valid
            
        except Exception as e:
            logger.error(f"Size validation failed: {e}")
            return False
    
    def _validate_format(self, backup_result: BackupResult) -> bool:
        """Validate backup file format."""
        try:
            backup_path = self._get_backup_file_path(backup_result)
            if not backup_path or not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Check file extension and format based on backup type
            if backup_result.source == 'postgresql':
                return self._validate_postgresql_format(backup_path)
            elif backup_result.source == 'neo4j':
                return self._validate_neo4j_format(backup_path)
            elif backup_result.source == 'milvus':
                return self._validate_milvus_format(backup_path)
            elif backup_result.source in ['data_files', 'logs', 'quality_reports']:
                return self._validate_archive_format(backup_path)
            else:
                logger.warning(f"Unknown backup type for format validation: {backup_result.source}")
                return True  # Assume valid for unknown types
            
        except Exception as e:
            logger.error(f"Format validation failed: {e}")
            return False
    
    def _validate_content(self, backup_result: BackupResult) -> bool:
        """Validate backup content integrity."""
        try:
            backup_path = self._get_backup_file_path(backup_result)
            if not backup_path or not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Check content based on backup type
            if backup_result.source == 'postgresql':
                return self._validate_postgresql_content(backup_path)
            elif backup_result.source == 'neo4j':
                return self._validate_neo4j_content(backup_path)
            elif backup_result.source == 'milvus':
                return self._validate_milvus_content(backup_path)
            elif backup_result.source in ['data_files', 'logs', 'quality_reports']:
                return self._validate_archive_content(backup_path)
            else:
                logger.warning(f"Unknown backup type for content validation: {backup_result.source}")
                return True  # Assume valid for unknown types
            
        except Exception as e:
            logger.error(f"Content validation failed: {e}")
            return False
    
    def _get_backup_file_path(self, backup_result: BackupResult) -> Optional[str]:
        """Get the local file path for a backup."""
        if backup_result.destination.startswith('s3://'):
            # For S3, we'd need to download the file for validation
            # In a real implementation, you'd download to a temp location
            logger.warning("S3 backup validation not implemented")
            return None
        elif backup_result.destination.startswith('gs://'):
            # For GCS, we'd need to download the file for validation
            logger.warning("GCS backup validation not implemented")
            return None
        elif backup_result.destination.startswith('azure://'):
            # For Azure, we'd need to download the file for validation
            logger.warning("Azure backup validation not implemented")
            return None
        else:
            # Local file
            return backup_result.destination
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _validate_postgresql_format(self, backup_path: str) -> bool:
        """Validate PostgreSQL backup format."""
        try:
            # Check if it's a compressed file
            if backup_path.endswith('.gz'):
                with gzip.open(backup_path, 'rt') as f:
                    first_line = f.readline().strip()
            elif backup_path.endswith('.bz2'):
                with bz2.open(backup_path, 'rt') as f:
                    first_line = f.readline().strip()
            elif backup_path.endswith('.xz'):
                with lzma.open(backup_path, 'rt') as f:
                    first_line = f.readline().strip()
            else:
                with open(backup_path, 'r') as f:
                    first_line = f.readline().strip()
            
            # Check for PostgreSQL dump header
            if first_line.startswith('-- PostgreSQL database dump'):
                return True
            elif first_line.startswith('-- pg_dump'):
                return True
            else:
                logger.error(f"Invalid PostgreSQL backup format: {first_line}")
                return False
                
        except Exception as e:
            logger.error(f"PostgreSQL format validation failed: {e}")
            return False
    
    def _validate_neo4j_format(self, backup_path: str) -> bool:
        """Validate Neo4j backup format."""
        try:
            # Neo4j backups are typically tar archives
            if not backup_path.endswith('.tar'):
                logger.error("Neo4j backup should be a tar archive")
                return False
            
            # Try to read tar header
            with open(backup_path, 'rb') as f:
                header = f.read(512)
                if len(header) < 512:
                    logger.error("Invalid tar archive: too short")
                    return False
                
                # Check tar magic number
                if not header[257:262] == b'ustar':
                    logger.error("Invalid tar archive: missing ustar magic")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Neo4j format validation failed: {e}")
            return False
    
    def _validate_milvus_format(self, backup_path: str) -> bool:
        """Validate Milvus backup format."""
        try:
            # Milvus backups are typically tar archives
            if not backup_path.endswith('.tar'):
                logger.error("Milvus backup should be a tar archive")
                return False
            
            # Try to read tar header
            with open(backup_path, 'rb') as f:
                header = f.read(512)
                if len(header) < 512:
                    logger.error("Invalid tar archive: too short")
                    return False
                
                # Check tar magic number
                if not header[257:262] == b'ustar':
                    logger.error("Invalid tar archive: missing ustar magic")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Milvus format validation failed: {e}")
            return False
    
    def _validate_archive_format(self, backup_path: str) -> bool:
        """Validate archive format for file backups."""
        try:
            # File backups are typically tar archives
            if not backup_path.endswith('.tar'):
                logger.error("File backup should be a tar archive")
                return False
            
            # Try to read tar header
            with open(backup_path, 'rb') as f:
                header = f.read(512)
                if len(header) < 512:
                    logger.error("Invalid tar archive: too short")
                    return False
                
                # Check tar magic number
                if not header[257:262] == b'ustar':
                    logger.error("Invalid tar archive: missing ustar magic")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Archive format validation failed: {e}")
            return False
    
    def _validate_postgresql_content(self, backup_path: str) -> bool:
        """Validate PostgreSQL backup content."""
        try:
            # Check if backup contains essential PostgreSQL elements
            essential_elements = [
                'CREATE TABLE',
                'INSERT INTO',
                '-- PostgreSQL database dump'
            ]
            
            found_elements = 0
            
            # Read backup content
            if backup_path.endswith('.gz'):
                with gzip.open(backup_path, 'rt') as f:
                    content = f.read(8192)  # Read first 8KB
            elif backup_path.endswith('.bz2'):
                with bz2.open(backup_path, 'rt') as f:
                    content = f.read(8192)
            elif backup_path.endswith('.xz'):
                with lzma.open(backup_path, 'rt') as f:
                    content = f.read(8192)
            else:
                with open(backup_path, 'r') as f:
                    content = f.read(8192)
            
            for element in essential_elements:
                if element in content:
                    found_elements += 1
            
            # At least one essential element should be present
            content_valid = found_elements > 0
            
            if not content_valid:
                logger.error(f"PostgreSQL backup content validation failed: found {found_elements} essential elements")
            
            return content_valid
            
        except Exception as e:
            logger.error(f"PostgreSQL content validation failed: {e}")
            return False
    
    def _validate_neo4j_content(self, backup_path: str) -> bool:
        """Validate Neo4j backup content."""
        try:
            # Extract and check tar archive contents
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract archive
                shutil.unpack_archive(backup_path, temp_path, 'tar')
                
                # Check for Neo4j-specific files/directories
                neo4j_files = list(temp_path.rglob('*'))
                
                # Should have some files
                content_valid = len(neo4j_files) > 0
                
                if not content_valid:
                    logger.error("Neo4j backup is empty")
                
                return content_valid
                
        except Exception as e:
            logger.error(f"Neo4j content validation failed: {e}")
            return False
    
    def _validate_milvus_content(self, backup_path: str) -> bool:
        """Validate Milvus backup content."""
        try:
            # Extract and check tar archive contents
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract archive
                shutil.unpack_archive(backup_path, temp_path, 'tar')
                
                # Check for Milvus-specific files/directories
                milvus_files = list(temp_path.rglob('*'))
                
                # Should have some files
                content_valid = len(milvus_files) > 0
                
                if not content_valid:
                    logger.error("Milvus backup is empty")
                
                return content_valid
                
        except Exception as e:
            logger.error(f"Milvus content validation failed: {e}")
            return False
    
    def _validate_archive_content(self, backup_path: str) -> bool:
        """Validate archive content for file backups."""
        try:
            # Extract and check tar archive contents
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract archive
                shutil.unpack_archive(backup_path, temp_path, 'tar')
                
                # Check for files
                archive_files = list(temp_path.rglob('*'))
                file_count = len([f for f in archive_files if f.is_file()])
                
                # Should have some files
                content_valid = file_count > 0
                
                if not content_valid:
                    logger.error("Archive backup is empty")
                
                return content_valid
                
        except Exception as e:
            logger.error(f"Archive content validation failed: {e}")
            return False
    
    def validate_backup_file(self, backup_path: str, backup_type: str) -> ValidationResult:
        """
        Validate a backup file directly.
        
        Args:
            backup_path: Path to backup file
            backup_type: Type of backup
            
        Returns:
            ValidationResult with validation details
        """
        validation_id = f"validation_{Path(backup_path).stem}"
        start_time = datetime.now()
        
        logger.info(f"Starting file validation: {validation_id}")
        
        try:
            # Create a mock backup result for validation
            backup_result = BackupResult(
                success=True,
                backup_id=Path(backup_path).stem,
                backup_type=None,  # Will be determined by backup_type
                source=backup_type,
                destination=backup_path,
                size_bytes=os.path.getsize(backup_path) if os.path.exists(backup_path) else 0,
                checksum=self._calculate_checksum(backup_path) if os.path.exists(backup_path) else "",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0
            )
            
            # Perform validation
            return self.validate_backup(backup_result)
            
        except Exception as e:
            error_msg = f"File validation failed: {str(e)}"
            logger.error(error_msg)
            
            result = ValidationResult(
                success=False,
                backup_id=Path(backup_path).stem,
                validation_type="file",
                checksum_valid=False,
                size_valid=False,
                format_valid=False,
                content_valid=False,
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error_message=error_msg
            )
            
            self.validation_history.append(result)
            return result
    
    def get_validation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get validation history."""
        validations = sorted(self.validation_history, key=lambda v: v.start_time, reverse=True)[:limit]
        
        return [
            {
                'backup_id': v.backup_id,
                'validation_type': v.validation_type,
                'success': v.success,
                'checksum_valid': v.checksum_valid,
                'size_valid': v.size_valid,
                'format_valid': v.format_valid,
                'content_valid': v.content_valid,
                'start_time': v.start_time.isoformat(),
                'duration_seconds': v.duration_seconds,
                'error_message': v.error_message,
                'details': v.details
            }
            for v in validations
        ]
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary statistics."""
        if not self.validation_history:
            return {
                'total_validations': 0,
                'successful_validations': 0,
                'failed_validations': 0,
                'success_rate': 0.0
            }
        
        total = len(self.validation_history)
        successful = len([v for v in self.validation_history if v.success])
        failed = total - successful
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        return {
            'total_validations': total,
            'successful_validations': successful,
            'failed_validations': failed,
            'success_rate': round(success_rate, 2)
        } 