#!/usr/bin/env python3
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
