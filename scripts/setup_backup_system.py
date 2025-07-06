#!/usr/bin/env python3
"""
Setup Backup System Script

Sets up the backup and recovery system for the movie recommendation system.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backup.backup_config import BackupConfig, BackupType, BackupFrequency, StorageType, CompressionType
from backup.backup_manager import BackupManager
from backup.recovery_manager import RecoveryManager
from backup.backup_scheduler import BackupScheduler
from backup.backup_validator import BackupValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_default_config() -> BackupConfig:
    """Create default backup configuration."""
    config = BackupConfig()
    
    # Customize configuration for movie recommendation system
    config.backup_name = "movie-recommendation-system"
    config.description = "Backup configuration for movie recommendation system with PostgreSQL, Neo4j, Milvus, and data files"
    
    # PostgreSQL configuration
    config.postgresql.enabled = True
    config.postgresql.backup_type = BackupType.FULL
    config.postgresql.frequency = BackupFrequency.DAILY
    config.postgresql.retention_days = 30
    config.postgresql.compression = CompressionType.GZIP
    config.postgresql.include_schema = True
    config.postgresql.include_data = True
    config.postgresql.include_indexes = True
    config.postgresql.parallel_jobs = 4
    config.postgresql.timeout_minutes = 60
    
    # Neo4j configuration
    config.neo4j.enabled = True
    config.neo4j.backup_type = BackupType.FULL
    config.neo4j.frequency = BackupFrequency.DAILY
    config.neo4j.retention_days = 30
    config.neo4j.compression = CompressionType.GZIP
    config.neo4j.timeout_minutes = 60
    
    # Milvus configuration
    config.milvus.enabled = True
    config.milvus.backup_type = BackupType.FULL
    config.milvus.frequency = BackupFrequency.WEEKLY
    config.milvus.retention_days = 90
    config.milvus.compression = CompressionType.LZMA
    config.milvus.timeout_minutes = 120
    
    # Data files configuration
    config.data_files.enabled = True
    config.data_files.backup_type = BackupType.FULL
    config.data_files.frequency = BackupFrequency.DAILY
    config.data_files.retention_days = 30
    config.data_files.compression = CompressionType.GZIP
    config.data_files.include_patterns = [
        "*.parquet", "*.json", "*.csv", "*.pkl", "*.h5", "*.feather"
    ]
    config.data_files.exclude_patterns = [
        "*.tmp", "*.log", "*.lock", "*.swp", "*.bak"
    ]
    config.data_files.max_file_size_mb = 1000
    
    # Logs configuration
    config.logs.enabled = True
    config.logs.backup_type = BackupType.FULL
    config.logs.frequency = BackupFrequency.WEEKLY
    config.logs.retention_days = 90
    config.logs.compression = CompressionType.GZIP
    config.logs.include_patterns = ["*.log", "*.log.*"]
    config.logs.max_file_size_mb = 500
    
    # Quality reports configuration
    config.quality_reports.enabled = True
    config.quality_reports.backup_type = BackupType.FULL
    config.quality_reports.frequency = BackupFrequency.DAILY
    config.quality_reports.retention_days = 30
    config.quality_reports.compression = CompressionType.GZIP
    config.quality_reports.include_patterns = [
        "*quality_report*.json", "*quality_history*.json", "*quality_metrics*.json"
    ]
    
    # Storage configuration
    config.storage.storage_type = StorageType.LOCAL
    config.storage.local_path = "backups"
    config.storage.encryption_enabled = True
    
    # Notification configuration
    config.notifications.enabled = True
    config.notifications.email_recipients = ["cynthia.petgrooming@gmail.com"]
    config.notifications.on_success = True
    config.notifications.on_failure = True
    config.notifications.on_warning = False
    
    # Advanced settings
    config.max_concurrent_backups = 3
    config.backup_timeout_minutes = 120
    config.verification_enabled = True
    config.checksum_verification = True
    config.backup_metadata = True
    
    return config


def setup_backup_directories(config: BackupConfig):
    """Set up backup directories."""
    logger.info("Setting up backup directories...")
    
    # Create main backup directory
    backup_path = Path(config.storage.local_path)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "postgresql", "neo4j", "milvus", "data_files", 
        "logs", "quality_reports", "metadata", "temp"
    ]
    
    for subdir in subdirs:
        (backup_path / subdir).mkdir(exist_ok=True)
        logger.info(f"Created directory: {backup_path / subdir}")
    
    # Create logs directory if it doesn't exist
    logs_path = Path("logs")
    logs_path.mkdir(exist_ok=True)
    logger.info(f"Created logs directory: {logs_path}")


def validate_configuration(config: BackupConfig):
    """Validate backup configuration."""
    logger.info("Validating backup configuration...")
    
    errors = config.validate()
    
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    else:
        logger.info("Configuration validation passed")
        return True


def create_backup_managers(config: BackupConfig):
    """Create and test backup managers."""
    logger.info("Creating backup managers...")
    
    try:
        # Create backup manager
        backup_manager = BackupManager(config)
        logger.info("Backup manager created successfully")
        
        # Create recovery manager
        recovery_manager = RecoveryManager(config)
        logger.info("Recovery manager created successfully")
        
        # Create backup scheduler
        scheduler = BackupScheduler(config, backup_manager)
        logger.info("Backup scheduler created successfully")
        
        # Create backup validator
        validator = BackupValidator(config)
        logger.info("Backup validator created successfully")
        
        return backup_manager, recovery_manager, scheduler, validator
        
    except Exception as e:
        logger.error(f"Failed to create backup managers: {e}")
        return None, None, None, None


def test_backup_functionality(backup_manager: BackupManager, validator: BackupValidator):
    """Test backup functionality."""
    logger.info("Testing backup functionality...")
    
    try:
        # Test data files backup
        logger.info("Testing data files backup...")
        result = backup_manager.create_backup("data_files", force=True)
        
        if result.success:
            logger.info(f"Data files backup successful: {result.backup_id}")
            
            # Test validation
            validation_result = validator.validate_backup(result)
            if validation_result.success:
                logger.info("Backup validation successful")
            else:
                logger.warning("Backup validation failed")
        else:
            logger.error(f"Data files backup failed: {result.error_message}")
        
        return result.success
        
    except Exception as e:
        logger.error(f"Backup test failed: {e}")
        return False


def setup_scheduled_backups(scheduler: BackupScheduler):
    """Set up scheduled backups."""
    logger.info("Setting up scheduled backups...")
    
    try:
        # Get current schedule
        schedule_info = scheduler.get_schedule()
        logger.info(f"Current schedules: {schedule_info['total_schedules']}")
        
        # Show next backup times
        status = scheduler.get_scheduler_status()
        logger.info("Next scheduled backups:")
        for backup in status['next_backups']:
            logger.info(f"  - {backup['backup_type']}: {backup['next_run']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to set up scheduled backups: {e}")
        return False


def create_backup_scripts():
    """Create backup utility scripts."""
    logger.info("Creating backup utility scripts...")
    
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Create backup script
    backup_script = scripts_dir / "run_backup.py"
    backup_script_content = '''#!/usr/bin/env python3
"""
Run Backup Script

Executes backups for the movie recommendation system.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backup.backup_config import BackupConfig
from backup.backup_manager import BackupManager

def main():
    parser = argparse.ArgumentParser(description="Run movie recommendation system backup")
    parser.add_argument("--type", choices=["postgresql", "neo4j", "milvus", "data_files", "logs", "quality_reports", "all"], 
                       default="all", help="Type of backup to run")
    parser.add_argument("--config", default="backup_config.json", help="Backup configuration file")
    parser.add_argument("--force", action="store_true", help="Force backup even if not scheduled")
    
    args = parser.parse_args()
    
    # Load configuration
    config = BackupConfig.load_from_file(args.config)
    
    # Create backup manager
    backup_manager = BackupManager(config)
    
    # Run backup
    if args.type == "all":
        backup_types = ["postgresql", "neo4j", "milvus", "data_files", "logs", "quality_reports"]
        for backup_type in backup_types:
            if getattr(config, backup_type).enabled:
                print(f"Running {backup_type} backup...")
                result = backup_manager.create_backup(backup_type, force=args.force)
                if result.success:
                    print(f"{backup_type} backup successful: {result.backup_id}")
                else:
                    print(f"{backup_type} backup failed: {result.error_message}")
    else:
        result = backup_manager.create_backup(args.type, force=args.force)
        if result.success:
            print(f"Backup successful: {result.backup_id}")
        else:
            print(f"Backup failed: {result.error_message}")

if __name__ == "__main__":
    main()
'''
    
    with open(backup_script, 'w') as f:
        f.write(backup_script_content)
    
    # Make executable
    backup_script.chmod(0o755)
    logger.info(f"Created backup script: {backup_script}")
    
    # Create recovery script
    recovery_script = scripts_dir / "run_recovery.py"
    recovery_script_content = '''#!/usr/bin/env python3
"""
Run Recovery Script

Executes data recovery for the movie recommendation system.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backup.backup_config import BackupConfig
from backup.recovery_manager import RecoveryManager

def main():
    parser = argparse.ArgumentParser(description="Run movie recommendation system recovery")
    parser.add_argument("--backup-path", required=True, help="Path to backup file")
    parser.add_argument("--destination", required=True, help="Recovery destination")
    parser.add_argument("--config", default="backup_config.json", help="Backup configuration file")
    parser.add_argument("--verify-checksum", action="store_true", help="Verify backup checksum")
    
    args = parser.parse_args()
    
    # Load configuration
    config = BackupConfig.load_from_file(args.config)
    
    # Create recovery manager
    recovery_manager = RecoveryManager(config)
    
    # Run recovery
    result = recovery_manager.recover_data(args.backup_path, args.destination, args.verify_checksum)
    
    if result.success:
        print(f"Recovery successful: {result.recovery_id}")
    else:
        print(f"Recovery failed: {result.error_message}")

if __name__ == "__main__":
    main()
'''
    
    with open(recovery_script, 'w') as f:
        f.write(recovery_script_content)
    
    # Make executable
    recovery_script.chmod(0o755)
    logger.info(f"Created recovery script: {recovery_script}")


def create_documentation():
    """Create backup system documentation."""
    logger.info("Creating backup system documentation...")
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Create README for backup system
    backup_readme = docs_dir / "backup_system.md"
    backup_readme_content = '''# Backup and Recovery System

This document describes the backup and recovery system for the movie recommendation system.

## Overview

The backup system provides comprehensive data protection for:
- PostgreSQL database (user data, movie metadata)
- Neo4j graph database (knowledge graph)
- Milvus vector database (embeddings)
- Data files (processed datasets, models)
- Logs (system and application logs)
- Quality reports (data quality metrics)

## Configuration

The backup system is configured through `backup_config.json`. Key settings include:

### Backup Types
- **PostgreSQL**: Daily full backups with 30-day retention
- **Neo4j**: Daily full backups with 30-day retention  
- **Milvus**: Weekly full backups with 90-day retention
- **Data Files**: Daily full backups with 30-day retention
- **Logs**: Weekly full backups with 90-day retention
- **Quality Reports**: Daily full backups with 30-day retention

### Storage
- **Local**: Backups stored in `backups/` directory
- **S3**: Cloud storage (requires AWS credentials)
- **GCS**: Google Cloud Storage (requires GCP credentials)
- **Azure**: Azure Blob Storage (requires Azure credentials)

### Compression
- **GZIP**: Fast compression, good for most data
- **BZIP2**: Better compression, slower
- **LZMA**: Best compression, slowest
- **None**: No compression

## Usage

### Manual Backup
```bash
# Run all backups
python scripts/run_backup.py --type all

# Run specific backup
python scripts/run_backup.py --type postgresql

# Force backup (ignore schedule)
python scripts/run_backup.py --type postgresql --force
```

### Manual Recovery
```bash
# Recover PostgreSQL database
python scripts/run_recovery.py --backup-path backups/postgresql/postgresql_20240101_120000.sql.gz --destination /var/lib/postgresql/data

# Recover data files
python scripts/run_recovery.py --backup-path backups/data_files/data_files_20240101_120000.tar.gz --destination data/
```

### Scheduled Backups
The backup scheduler runs automatically and executes backups according to the configured schedule.

## Monitoring

### Backup Status
```python
from src.backup.backup_manager import BackupManager
from src.backup.backup_config import BackupConfig

config = BackupConfig.load_from_file("backup_config.json")
manager = BackupManager(config)

# Get backup status
status = manager.get_backup_status("postgresql")
print(status)

# List recent backups
backups = manager.list_backups("postgresql", limit=10)
for backup in backups:
    print(f"{backup['id']}: {backup['success']}")
```

### Validation
```python
from src.backup.backup_validator import BackupValidator

validator = BackupValidator(config)

# Validate a backup
validation_result = validator.validate_backup(backup_result)
if validation_result.success:
    print("Backup is valid")
else:
    print(f"Backup validation failed: {validation_result.error_message}")
```

## Troubleshooting

### Common Issues

1. **PostgreSQL backup fails**
   - Check PostgreSQL service is running
   - Verify database credentials
   - Ensure sufficient disk space

2. **Neo4j backup fails**
   - Check Neo4j service is running
   - Verify Neo4j backup directory permissions
   - Ensure sufficient disk space

3. **Milvus backup fails**
   - Check Milvus service is running
   - Verify Milvus data directory permissions
   - Ensure sufficient disk space

4. **Storage upload fails**
   - Check network connectivity
   - Verify cloud credentials
   - Ensure sufficient storage quota

### Logs
Backup system logs are written to:
- Application logs: `logs/backup_system.log`
- System logs: Check system journal

### Recovery Testing
Regularly test recovery procedures to ensure backups are working correctly:

```bash
# Test PostgreSQL recovery
python scripts/run_recovery.py --backup-path backups/postgresql/latest.sql.gz --destination /tmp/test_recovery

# Verify recovered data
psql -d test_db -c "SELECT COUNT(*) FROM movies;"
```

## Security

### Encryption
- Backup files can be encrypted using AES-256
- Encryption keys should be stored securely
- Consider using cloud KMS for key management

### Access Control
- Limit access to backup files
- Use role-based access control
- Monitor backup access logs

### Compliance
- Ensure backups meet data retention requirements
- Implement data classification
- Regular security audits

## Performance

### Optimization
- Use parallel compression for large files
- Schedule backups during low-traffic periods
- Monitor backup performance metrics

### Monitoring
- Track backup duration and size
- Monitor storage usage
- Alert on backup failures

## Maintenance

### Regular Tasks
- Review backup logs weekly
- Test recovery procedures monthly
- Update backup configuration as needed
- Clean up old backup files

### Health Checks
- Verify backup integrity
- Check storage space
- Monitor backup success rates
- Validate recovery procedures
'''
    
    with open(backup_readme, 'w') as f:
        f.write(backup_readme_content)
    
    logger.info(f"Created backup documentation: {backup_readme}")


def main():
    """Main setup function."""
    logger.info("Setting up backup and recovery system...")
    
    try:
        # Create default configuration
        config = create_default_config()
        logger.info("Created default backup configuration")
        
        # Validate configuration
        if not validate_configuration(config):
            logger.error("Configuration validation failed")
            return False
        
        # Save configuration
        config.save_to_file("backup_config.json")
        logger.info("Saved backup configuration to backup_config.json")
        
        # Set up backup directories
        setup_backup_directories(config)
        
        # Create backup managers
        backup_manager, recovery_manager, scheduler, validator = create_backup_managers(config)
        
        if not all([backup_manager, recovery_manager, scheduler, validator]):
            logger.error("Failed to create backup managers")
            return False
        
        # Test backup functionality
        if test_backup_functionality(backup_manager, validator):
            logger.info("Backup functionality test passed")
        else:
            logger.warning("Backup functionality test failed")
        
        # Set up scheduled backups
        if setup_scheduled_backups(scheduler):
            logger.info("Scheduled backups set up successfully")
        else:
            logger.warning("Failed to set up scheduled backups")
        
        # Create utility scripts
        create_backup_scripts()
        
        # Create documentation
        create_documentation()
        
        logger.info("Backup and recovery system setup completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Review backup_config.json and adjust settings as needed")
        logger.info("2. Test backup functionality: python scripts/run_backup.py --type data_files")
        logger.info("3. Start the backup scheduler for automated backups")
        logger.info("4. Review documentation in docs/backup_system.md")
        
        return True
        
    except Exception as e:
        logger.error(f"Backup system setup failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 