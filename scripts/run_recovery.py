#!/usr/bin/env python3
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
