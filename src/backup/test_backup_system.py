"""
Test Backup System Module

Comprehensive tests for the backup and recovery system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

from .backup_config import BackupConfig, BackupType, BackupFrequency, StorageType, CompressionType
from .backup_manager import BackupManager, BackupResult
from .recovery_manager import RecoveryManager, RecoveryResult
from .backup_scheduler import BackupScheduler, ScheduledBackup
from .backup_validator import BackupValidator, ValidationResult


class TestBackupConfig(unittest.TestCase):
    """Test backup configuration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BackupConfig()
    
    def test_default_config(self):
        """Test default configuration values."""
        self.assertEqual(self.config.backup_name, "movie-recommendation-system")
        self.assertTrue(self.config.postgresql.enabled)
        self.assertEqual(self.config.postgresql.frequency, BackupFrequency.DAILY)
        self.assertEqual(self.config.storage.storage_type, StorageType.LOCAL)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Set up valid config with notification email
        self.config.notifications.email_recipients = ["test@example.com"]
        
        # Valid config should have no errors
        errors = self.config.validate()
        self.assertEqual(len(errors), 0)
        
        # Test invalid retention days
        self.config.postgresql.retention_days = 0
        errors = self.config.validate()
        self.assertIn("PostgreSQL retention days must be at least 1", errors)
        
        # Reset for next test
        self.config.postgresql.retention_days = 30
        
        # Test S3 without bucket
        self.config.storage.storage_type = StorageType.S3
        errors = self.config.validate()
        self.assertIn("S3 bucket must be specified for S3 storage type", errors)
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        config_dict = self.config.to_dict()
        restored_config = BackupConfig.from_dict(config_dict)
        
        self.assertEqual(self.config.backup_name, restored_config.backup_name)
        self.assertEqual(self.config.postgresql.enabled, restored_config.postgresql.enabled)
        self.assertEqual(self.config.postgresql.frequency, restored_config.postgresql.frequency)
    
    def test_config_file_operations(self):
        """Test configuration file save and load."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            # Save config
            self.config.save_to_file(config_file)
            
            # Load config
            loaded_config = BackupConfig.load_from_file(config_file)
            
            self.assertEqual(self.config.backup_name, loaded_config.backup_name)
            self.assertEqual(self.config.postgresql.enabled, loaded_config.postgresql.enabled)
            
        finally:
            os.unlink(config_file)


class TestBackupManager(unittest.TestCase):
    """Test backup manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BackupConfig()
        self.config.storage.local_path = tempfile.mkdtemp()
        self.backup_manager = BackupManager(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.config.storage.local_path, ignore_errors=True)
    
    def test_backup_manager_initialization(self):
        """Test backup manager initialization."""
        self.assertIsNotNone(self.backup_manager.config)
        self.assertEqual(len(self.backup_manager.backup_history), 0)
        self.assertEqual(len(self.backup_manager.current_backups), 0)
    
    def test_backup_directories_creation(self):
        """Test backup directories creation."""
        backup_path = Path(self.config.storage.local_path)
        
        # Check that subdirectories were created
        self.assertTrue((backup_path / "postgresql").exists())
        self.assertTrue((backup_path / "neo4j").exists())
        self.assertTrue((backup_path / "milvus").exists())
        self.assertTrue((backup_path / "data_files").exists())
        self.assertTrue((backup_path / "logs").exists())
        self.assertTrue((backup_path / "quality_reports").exists())
        self.assertTrue((backup_path / "metadata").exists())
    
    @patch('subprocess.run')
    def test_postgresql_backup(self, mock_run):
        """Test PostgreSQL backup creation."""
        # Mock successful pg_dump
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        
        # Create a temporary SQL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write("-- PostgreSQL database dump\nCREATE TABLE test (id INT);")
            sql_file = f.name
        
        try:
            # Mock file operations
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = sql_file
                mock_temp.return_value.__exit__.return_value = None
                
                # Mock file existence and size
                with patch('os.path.getsize', return_value=1024):
                    with patch('os.unlink'):
                        result = self.backup_manager._backup_postgresql("test_backup")
                        
                        self.assertTrue(result.success)
                        self.assertEqual(result.backup_id, "test_backup")
                        self.assertEqual(result.source, "postgresql")
                        
        finally:
            os.unlink(sql_file)
    
    def test_data_files_backup(self):
        """Test data files backup creation."""
        # Create test data files
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        test_file = data_dir / "test_data.json"
        test_file.write_text('{"test": "data"}')
        
        try:
            result = self.backup_manager._backup_data_files("test_backup")
            
            self.assertTrue(result.success)
            self.assertEqual(result.backup_id, "test_backup")
            self.assertEqual(result.source, "data_files")
            
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)
    
    def test_compression(self):
        """Test file compression functionality."""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            test_file = f.name
        
        try:
            # Test gzip compression
            compressed_path = self.backup_manager._compress_file(test_file, CompressionType.GZIP)
            self.assertTrue(compressed_path.endswith('.gzip'))
            self.assertTrue(os.path.exists(compressed_path))
            
            # Clean up
            if os.path.exists(compressed_path):
                os.unlink(compressed_path)
            
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)
    
    def test_checksum_calculation(self):
        """Test checksum calculation."""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            test_file = f.name
        
        try:
            checksum = self.backup_manager._calculate_checksum(test_file)
            self.assertIsInstance(checksum, str)
            self.assertEqual(len(checksum), 64)  # SHA256 hex length
            
        finally:
            os.unlink(test_file)
    
    def test_backup_status(self):
        """Test backup status retrieval."""
        # Add a mock backup to history
        mock_backup = BackupResult(
            success=True,
            backup_id="test_backup",
            backup_type=BackupType.FULL,
            source="postgresql",
            destination="/tmp/test",
            size_bytes=1024,
            checksum="test_checksum",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=1.0
        )
        self.backup_manager.backup_history.append(mock_backup)
        
        status = self.backup_manager.get_backup_status()
        self.assertEqual(status['status'], 'available')
        self.assertEqual(status['total_backups'], 1)
    
    def test_list_backups(self):
        """Test backup listing."""
        # Add mock backups
        for i in range(3):
            mock_backup = BackupResult(
                success=True,
                backup_id=f"test_backup_{i}",
                backup_type=BackupType.FULL,
                source="postgresql",
                destination=f"/tmp/test_{i}",
                size_bytes=1024,
                checksum=f"test_checksum_{i}",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.0
            )
            self.backup_manager.backup_history.append(mock_backup)
        
        backups = self.backup_manager.list_backups(limit=2)
        self.assertEqual(len(backups), 2)


class TestRecoveryManager(unittest.TestCase):
    """Test recovery manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BackupConfig()
        self.recovery_manager = RecoveryManager(self.config)
    
    def test_recovery_manager_initialization(self):
        """Test recovery manager initialization."""
        self.assertIsNotNone(self.recovery_manager.config)
        self.assertEqual(len(self.recovery_manager.recovery_history), 0)
        self.assertEqual(len(self.recovery_manager.current_recoveries), 0)
    
    def test_parse_backup_filename(self):
        """Test backup filename parsing."""
        filename = "postgresql_20240101_120000.sql.gz"
        parsed = self.recovery_manager._parse_backup_filename(filename)
        
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['backup_type'], 'postgresql')
        self.assertEqual(parsed['compression'], 'gzip')
        self.assertEqual(parsed['timestamp'].year, 2024)
        self.assertEqual(parsed['timestamp'].month, 1)
        self.assertEqual(parsed['timestamp'].day, 1)
        
        # Test with different compression
        filename2 = "neo4j_20240101_120000.tar.bz2"
        parsed2 = self.recovery_manager._parse_backup_filename(filename2)
        self.assertIsNotNone(parsed2)
        self.assertEqual(parsed2['compression'], 'bzip2')
    
    def test_parse_invalid_filename(self):
        """Test parsing invalid backup filename."""
        filename = "invalid_filename"
        parsed = self.recovery_manager._parse_backup_filename(filename)
        
        self.assertIsNone(parsed)
    
    def test_checksum_calculation(self):
        """Test checksum calculation."""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            test_file = f.name
        
        try:
            checksum = self.recovery_manager._calculate_checksum(test_file)
            self.assertIsInstance(checksum, str)
            self.assertEqual(len(checksum), 64)  # SHA256 hex length
            
        finally:
            os.unlink(test_file)
    
    def test_decompress_file(self):
        """Test file decompression."""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            test_file = f.name
        
        try:
            # Test gzip compression and decompression
            import gzip
            compressed_file = test_file + '.gz'
            with gzip.open(compressed_file, 'wt') as f:
                f.write("test content")
            
            decompressed_file = self.recovery_manager._decompress_file(compressed_file)
            self.assertEqual(decompressed_file, compressed_file[:-3])  # Remove .gz
            
            # Clean up
            if os.path.exists(compressed_file):
                os.unlink(compressed_file)
            if os.path.exists(decompressed_file):
                os.unlink(decompressed_file)
            
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)
    
    def test_recovery_status(self):
        """Test recovery status retrieval."""
        # Add a mock recovery to history
        mock_recovery = RecoveryResult(
            success=True,
            recovery_id="test_recovery",
            backup_id="test_backup",
            source="/tmp/test_backup",
            destination="/tmp/recovered",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=1.0
        )
        self.recovery_manager.recovery_history.append(mock_recovery)
        
        status = self.recovery_manager.get_recovery_status()
        self.assertEqual(status['status'], 'available')
        self.assertEqual(status['total_recoveries'], 1)


class TestBackupScheduler(unittest.TestCase):
    """Test backup scheduler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BackupConfig()
        self.backup_manager = Mock()
        self.scheduler = BackupScheduler(self.config, self.backup_manager)
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        self.assertIsNotNone(self.scheduler.config)
        self.assertIsNotNone(self.scheduler.backup_manager)
        self.assertFalse(self.scheduler.running)
    
    def test_calculate_next_run(self):
        """Test next run time calculation."""
        # Test daily frequency
        next_run = self.scheduler._calculate_next_run(BackupFrequency.DAILY)
        self.assertIsInstance(next_run, datetime)
        self.assertGreater(next_run, datetime.now())
        
        # Test weekly frequency
        next_run = self.scheduler._calculate_next_run(BackupFrequency.WEEKLY)
        self.assertIsInstance(next_run, datetime)
        self.assertGreater(next_run, datetime.now())
    
    def test_add_backup_schedule(self):
        """Test adding backup schedule."""
        success = self.scheduler.add_backup_schedule(
            "test_backup",
            BackupFrequency.DAILY
        )
        
        self.assertTrue(success)
        self.assertIn("test_backup", self.scheduler.scheduled_backups)
        
        scheduled = self.scheduler.scheduled_backups["test_backup"]
        self.assertEqual(scheduled.backup_type, "test_backup")
        self.assertEqual(scheduled.frequency, BackupFrequency.DAILY)
        self.assertTrue(scheduled.enabled)
    
    def test_remove_backup_schedule(self):
        """Test removing backup schedule."""
        # Add a schedule first
        self.scheduler.add_backup_schedule("test_backup", BackupFrequency.DAILY)
        
        # Remove it
        success = self.scheduler.remove_backup_schedule("test_backup")
        
        self.assertTrue(success)
        self.assertNotIn("test_backup", self.scheduler.scheduled_backups)
    
    def test_enable_disable_schedule(self):
        """Test enabling and disabling schedules."""
        # Add a schedule
        self.scheduler.add_backup_schedule("test_backup", BackupFrequency.DAILY)
        
        # Disable it
        success = self.scheduler.disable_backup_schedule("test_backup")
        self.assertTrue(success)
        self.assertFalse(self.scheduler.scheduled_backups["test_backup"].enabled)
        
        # Enable it
        success = self.scheduler.enable_backup_schedule("test_backup")
        self.assertTrue(success)
        self.assertTrue(self.scheduler.scheduled_backups["test_backup"].enabled)
    
    def test_get_schedule(self):
        """Test schedule retrieval."""
        # Clear existing schedules for this test
        self.scheduler.scheduled_backups.clear()
        
        # Add a schedule
        self.scheduler.add_backup_schedule("test_backup", BackupFrequency.DAILY)
        
        # Get specific schedule
        schedule = self.scheduler.get_schedule("test_backup")
        self.assertEqual(schedule['backup_type'], "test_backup")
        self.assertEqual(schedule['frequency'], "daily")
        
        # Get all schedules
        all_schedules = self.scheduler.get_schedule()
        self.assertIn("test_backup", all_schedules['schedules'])
        self.assertEqual(all_schedules['total_schedules'], 1)
    
    def test_scheduler_status(self):
        """Test scheduler status retrieval."""
        status = self.scheduler.get_scheduler_status()
        
        self.assertIn('running', status)
        self.assertIn('total_schedules', status)
        self.assertIn('enabled_schedules', status)
        self.assertIn('next_backups', status)


class TestBackupValidator(unittest.TestCase):
    """Test backup validator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BackupConfig()
        self.validator = BackupValidator(self.config)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertIsNotNone(self.validator.config)
        self.assertEqual(len(self.validator.validation_history), 0)
    
    def test_checksum_calculation(self):
        """Test checksum calculation."""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            test_file = f.name
        
        try:
            checksum = self.validator._calculate_checksum(test_file)
            self.assertIsInstance(checksum, str)
            self.assertEqual(len(checksum), 64)  # SHA256 hex length
            
        finally:
            os.unlink(test_file)
    
    def test_validate_checksum(self):
        """Test checksum validation."""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            test_file = f.name
        
        try:
            # Calculate expected checksum
            expected_checksum = self.validator._calculate_checksum(test_file)
            
            # Create mock backup result
            backup_result = BackupResult(
                success=True,
                backup_id="test_backup",
                backup_type=BackupType.FULL,
                source="postgresql",
                destination=test_file,
                size_bytes=os.path.getsize(test_file),
                checksum=expected_checksum,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.0
            )
            
            # Test valid checksum
            with patch.object(self.validator, '_get_backup_file_path', return_value=test_file):
                is_valid = self.validator._validate_checksum(backup_result)
                self.assertTrue(is_valid)
            
            # Test invalid checksum
            backup_result.checksum = "invalid_checksum"
            with patch.object(self.validator, '_get_backup_file_path', return_value=test_file):
                is_valid = self.validator._validate_checksum(backup_result)
                self.assertFalse(is_valid)
            
        finally:
            os.unlink(test_file)
    
    def test_validate_size(self):
        """Test size validation."""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            test_file = f.name
        
        try:
            file_size = os.path.getsize(test_file)
            
            # Create mock backup result
            backup_result = BackupResult(
                success=True,
                backup_id="test_backup",
                backup_type=BackupType.FULL,
                source="postgresql",
                destination=test_file,
                size_bytes=file_size,
                checksum="test_checksum",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.0
            )
            
            # Test valid size
            with patch.object(self.validator, '_get_backup_file_path', return_value=test_file):
                is_valid = self.validator._validate_size(backup_result)
                self.assertTrue(is_valid)
            
            # Test invalid size (outside tolerance)
            backup_result.size_bytes = file_size + 2000  # More than 1KB tolerance
            with patch.object(self.validator, '_get_backup_file_path', return_value=test_file):
                is_valid = self.validator._validate_size(backup_result)
                self.assertFalse(is_valid)
            
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)
    
    def test_validate_postgresql_format(self):
        """Test PostgreSQL format validation."""
        # Create test PostgreSQL dump
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("-- PostgreSQL database dump\nCREATE TABLE test (id INT);")
            test_file = f.name
        
        try:
            is_valid = self.validator._validate_postgresql_format(test_file)
            self.assertTrue(is_valid)
            
        finally:
            os.unlink(test_file)
    
    def test_validate_archive_format(self):
        """Test archive format validation."""
        # Create test tar archive
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.txt"
            test_file.write_text("test content")
            
            tar_file = temp_path / "test.tar"
            shutil.make_archive(str(tar_file.with_suffix('')), 'tar', temp_path)
            
            is_valid = self.validator._validate_archive_format(str(tar_file))
            self.assertTrue(is_valid)
    
    def test_validation_summary(self):
        """Test validation summary."""
        # Add mock validations
        for i in range(3):
            validation = ValidationResult(
                success=i < 2,  # 2 successful, 1 failed
                backup_id=f"test_backup_{i}",
                validation_type="comprehensive",
                checksum_valid=True,
                size_valid=True,
                format_valid=True,
                content_valid=True,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.0
            )
            self.validator.validation_history.append(validation)
        
        summary = self.validator.get_validation_summary()
        
        self.assertEqual(summary['total_validations'], 3)
        self.assertEqual(summary['successful_validations'], 2)
        self.assertEqual(summary['failed_validations'], 1)
        self.assertEqual(summary['success_rate'], 66.67)


if __name__ == '__main__':
    unittest.main() 