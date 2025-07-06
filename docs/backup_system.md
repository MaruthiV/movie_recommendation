# Backup and Recovery System

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
