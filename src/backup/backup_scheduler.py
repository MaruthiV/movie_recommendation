"""
Backup Scheduler Module

Manages automated backup scheduling and execution for the movie recommendation system.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import threading
import logging
from pathlib import Path
import json
# import schedule  # Not used in this implementation

from .backup_config import BackupConfig, BackupFrequency
from .backup_manager import BackupManager

logger = logging.getLogger(__name__)


@dataclass
class ScheduledBackup:
    """Information about a scheduled backup."""
    backup_type: str
    frequency: BackupFrequency
    next_run: datetime
    last_run: Optional[datetime] = None
    enabled: bool = True
    config: Dict[str, Any] = None


class BackupScheduler:
    """Scheduler for automated backups."""
    
    def __init__(self, config: BackupConfig, backup_manager: BackupManager):
        """
        Initialize the backup scheduler.
        
        Args:
            config: Backup configuration
            backup_manager: Backup manager instance
        """
        self.config = config
        self.backup_manager = backup_manager
        self.scheduled_backups: Dict[str, ScheduledBackup] = {}
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running = False
        self.schedule_file = Path("backup_schedule.json")
        
        # Load existing schedule
        self._load_schedule()
        
        # Set up scheduled backups
        self._setup_scheduled_backups()
    
    def _load_schedule(self):
        """Load schedule from file."""
        if self.schedule_file.exists():
            try:
                with open(self.schedule_file, 'r') as f:
                    schedule_data = json.load(f)
                
                for backup_type, data in schedule_data.items():
                    self.scheduled_backups[backup_type] = ScheduledBackup(
                        backup_type=backup_type,
                        frequency=BackupFrequency(data['frequency']),
                        next_run=datetime.fromisoformat(data['next_run']),
                        last_run=datetime.fromisoformat(data['last_run']) if data.get('last_run') else None,
                        enabled=data.get('enabled', True),
                        config=data.get('config', {})
                    )
                logger.info(f"Loaded {len(self.scheduled_backups)} scheduled backups")
            except Exception as e:
                logger.error(f"Failed to load schedule: {e}")
    
    def _save_schedule(self):
        """Save schedule to file."""
        try:
            schedule_data = {}
            for backup_type, scheduled_backup in self.scheduled_backups.items():
                schedule_data[backup_type] = {
                    'frequency': scheduled_backup.frequency.value,
                    'next_run': scheduled_backup.next_run.isoformat(),
                    'last_run': scheduled_backup.last_run.isoformat() if scheduled_backup.last_run else None,
                    'enabled': scheduled_backup.enabled,
                    'config': scheduled_backup.config or {}
                }
            
            with open(self.schedule_file, 'w') as f:
                json.dump(schedule_data, f, indent=2)
            
            logger.debug("Schedule saved to file")
        except Exception as e:
            logger.error(f"Failed to save schedule: {e}")
    
    def _setup_scheduled_backups(self):
        """Set up scheduled backups based on configuration."""
        backup_types = ['postgresql', 'neo4j', 'milvus', 'data_files', 'logs', 'quality_reports']
        
        for backup_type in backup_types:
            config = getattr(self.config, backup_type, None)
            if config and config.enabled:
                # Check if already scheduled
                if backup_type not in self.scheduled_backups:
                    # Create new scheduled backup
                    next_run = self._calculate_next_run(config.frequency)
                    self.scheduled_backups[backup_type] = ScheduledBackup(
                        backup_type=backup_type,
                        frequency=config.frequency,
                        next_run=next_run,
                        config={
                            'backup_type': config.backup_type.value,
                            'retention_days': config.retention_days,
                            'compression': config.compression.value
                        }
                    )
                else:
                    # Update existing scheduled backup
                    scheduled = self.scheduled_backups[backup_type]
                    scheduled.frequency = config.frequency
                    scheduled.enabled = True
                    scheduled.config = {
                        'backup_type': config.backup_type.value,
                        'retention_days': config.retention_days,
                        'compression': config.compression.value
                    }
        
        # Save updated schedule
        self._save_schedule()
        logger.info(f"Set up {len(self.scheduled_backups)} scheduled backups")
    
    def _calculate_next_run(self, frequency: BackupFrequency) -> datetime:
        """Calculate next run time based on frequency."""
        now = datetime.now()
        
        if frequency == BackupFrequency.HOURLY:
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif frequency == BackupFrequency.DAILY:
            return now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif frequency == BackupFrequency.WEEKLY:
            # Next Sunday at 2 AM
            days_ahead = 6 - now.weekday()  # 6 = Sunday
            if days_ahead <= 0:
                days_ahead += 7
            return now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)
        elif frequency == BackupFrequency.MONTHLY:
            # First day of next month at 2 AM
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=1, hour=2, minute=0, second=0, microsecond=0)
            else:
                next_month = now.replace(month=now.month + 1, day=1, hour=2, minute=0, second=0, microsecond=0)
            return next_month
        else:
            return now + timedelta(days=1)  # Default to daily
    
    def start(self):
        """Start the backup scheduler."""
        if self.running:
            logger.warning("Backup scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info("Backup scheduler started")
    
    def stop(self):
        """Stop the backup scheduler."""
        if not self.running:
            logger.warning("Backup scheduler is not running")
            return
        
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        logger.info("Backup scheduler stopped")
    
    def _run_scheduler(self):
        """Main scheduler loop."""
        while self.running:
            try:
                now = datetime.now()
                
                # Check for backups that need to run
                for backup_type, scheduled_backup in self.scheduled_backups.items():
                    if (scheduled_backup.enabled and 
                        scheduled_backup.next_run <= now):
                        
                        logger.info(f"Running scheduled backup: {backup_type}")
                        
                        # Run backup
                        try:
                            result = self.backup_manager.create_backup(backup_type, force=True)
                            
                            # Update schedule
                            scheduled_backup.last_run = now
                            scheduled_backup.next_run = self._calculate_next_run(scheduled_backup.frequency)
                            
                            if result.success:
                                logger.info(f"Scheduled backup completed: {backup_type}")
                            else:
                                logger.error(f"Scheduled backup failed: {backup_type} - {result.error_message}")
                            
                        except Exception as e:
                            logger.error(f"Scheduled backup failed: {backup_type} - {e}")
                            scheduled_backup.last_run = now
                            scheduled_backup.next_run = self._calculate_next_run(scheduled_backup.frequency)
                        
                        # Save updated schedule
                        self._save_schedule()
                
                # Sleep for a minute before checking again
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Continue running despite errors
    
    def add_backup_schedule(self, backup_type: str, frequency: BackupFrequency, 
                           next_run: Optional[datetime] = None, enabled: bool = True) -> bool:
        """
        Add or update a backup schedule.
        
        Args:
            backup_type: Type of backup to schedule
            frequency: How often to run the backup
            next_run: When to run the next backup (defaults to calculated time)
            enabled: Whether the schedule is enabled
            
        Returns:
            True if schedule was added/updated successfully
        """
        try:
            if next_run is None:
                next_run = self._calculate_next_run(frequency)
            
            self.scheduled_backups[backup_type] = ScheduledBackup(
                backup_type=backup_type,
                frequency=frequency,
                next_run=next_run,
                enabled=enabled
            )
            
            self._save_schedule()
            logger.info(f"Added/updated backup schedule: {backup_type} ({frequency.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add backup schedule: {e}")
            return False
    
    def remove_backup_schedule(self, backup_type: str) -> bool:
        """
        Remove a backup schedule.
        
        Args:
            backup_type: Type of backup to remove from schedule
            
        Returns:
            True if schedule was removed successfully
        """
        try:
            if backup_type in self.scheduled_backups:
                del self.scheduled_backups[backup_type]
                self._save_schedule()
                logger.info(f"Removed backup schedule: {backup_type}")
                return True
            else:
                logger.warning(f"Backup schedule not found: {backup_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove backup schedule: {e}")
            return False
    
    def enable_backup_schedule(self, backup_type: str) -> bool:
        """Enable a backup schedule."""
        if backup_type in self.scheduled_backups:
            self.scheduled_backups[backup_type].enabled = True
            self._save_schedule()
            logger.info(f"Enabled backup schedule: {backup_type}")
            return True
        else:
            logger.warning(f"Backup schedule not found: {backup_type}")
            return False
    
    def disable_backup_schedule(self, backup_type: str) -> bool:
        """Disable a backup schedule."""
        if backup_type in self.scheduled_backups:
            self.scheduled_backups[backup_type].enabled = False
            self._save_schedule()
            logger.info(f"Disabled backup schedule: {backup_type}")
            return True
        else:
            logger.warning(f"Backup schedule not found: {backup_type}")
            return False
    
    def get_schedule(self, backup_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current backup schedule.
        
        Args:
            backup_type: Specific backup type to get schedule for
            
        Returns:
            Schedule information
        """
        if backup_type:
            if backup_type in self.scheduled_backups:
                scheduled = self.scheduled_backups[backup_type]
                return {
                    'backup_type': scheduled.backup_type,
                    'frequency': scheduled.frequency.value,
                    'next_run': scheduled.next_run.isoformat(),
                    'last_run': scheduled.last_run.isoformat() if scheduled.last_run else None,
                    'enabled': scheduled.enabled,
                    'config': scheduled.config
                }
            else:
                return {'error': f'Backup schedule not found: {backup_type}'}
        else:
            return {
                'schedules': {
                    bt: {
                        'frequency': scheduled.frequency.value,
                        'next_run': scheduled.next_run.isoformat(),
                        'last_run': scheduled.last_run.isoformat() if scheduled.last_run else None,
                        'enabled': scheduled.enabled,
                        'config': scheduled.config
                    }
                    for bt, scheduled in self.scheduled_backups.items()
                },
                'total_schedules': len(self.scheduled_backups),
                'enabled_schedules': len([s for s in self.scheduled_backups.values() if s.enabled])
            }
    
    def run_backup_now(self, backup_type: str) -> bool:
        """
        Run a backup immediately, regardless of schedule.
        
        Args:
            backup_type: Type of backup to run
            
        Returns:
            True if backup was started successfully
        """
        try:
            logger.info(f"Running immediate backup: {backup_type}")
            result = self.backup_manager.create_backup(backup_type, force=True)
            
            if result.success:
                logger.info(f"Immediate backup completed: {backup_type}")
                return True
            else:
                logger.error(f"Immediate backup failed: {backup_type} - {result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Immediate backup failed: {backup_type} - {e}")
            return False
    
    def get_next_backup_time(self, backup_type: str) -> Optional[datetime]:
        """Get the next scheduled backup time for a specific type."""
        if backup_type in self.scheduled_backups:
            return self.scheduled_backups[backup_type].next_run
        return None
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get overall scheduler status."""
        now = datetime.now()
        
        # Find next backup time
        next_backups = []
        for backup_type, scheduled in self.scheduled_backups.items():
            if scheduled.enabled and scheduled.next_run > now:
                next_backups.append({
                    'backup_type': backup_type,
                    'next_run': scheduled.next_run.isoformat(),
                    'time_until': (scheduled.next_run - now).total_seconds()
                })
        
        # Sort by next run time
        next_backups.sort(key=lambda x: x['time_until'])
        
        return {
            'running': self.running,
            'total_schedules': len(self.scheduled_backups),
            'enabled_schedules': len([s for s in self.scheduled_backups.values() if s.enabled]),
            'next_backups': next_backups[:5],  # Show next 5 backups
            'last_updated': now.isoformat()
        } 