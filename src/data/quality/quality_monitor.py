"""
Quality Monitor Module

Provides automated quality monitoring and alerting capabilities for data quality management.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import pandas as pd

from .data_validator import DataValidator
from .quality_metrics import QualityMetrics, QualityScore, QualityLevel
from .validation_rules import IssueSeverity

logger = logging.getLogger(__name__)


@dataclass
class QualityAlert:
    """Represents a quality alert."""
    id: str
    severity: IssueSeverity
    message: str
    table: str
    rule: str
    timestamp: datetime
    data: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class QualityMonitor:
    """Automated quality monitoring and alerting system."""
    
    def __init__(self, 
                 data_validator: Optional[DataValidator] = None,
                 quality_metrics: Optional[QualityMetrics] = None,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 storage_path: Optional[str] = None):
        """
        Initialize the quality monitor.
        
        Args:
            data_validator: Data validator instance
            quality_metrics: Quality metrics calculator
            alert_thresholds: Custom alert thresholds
            storage_path: Path for storing monitoring data
        """
        self.data_validator = data_validator or DataValidator()
        self.quality_metrics = quality_metrics or QualityMetrics()
        self.storage_path = Path(storage_path) if storage_path else Path("data/quality_monitoring")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'quality_score_min': 85.0,
            'critical_issues_max': 10,
            'warning_issues_max': 50,
            'completeness_min': 90.0,
            'consistency_min': 90.0
        }
        
        # Monitoring state
        self.alerts: List[QualityAlert] = []
        self.quality_history: List[QualityScore] = []
        self.monitoring_enabled = True
        self.alert_callbacks: List[Callable] = []
        
        # Load existing monitoring data
        self._load_monitoring_data()
    
    def monitor_data_quality(self, 
                           data: Dict[str, Any], 
                           tmdb_data: Optional[Dict[int, Dict]] = None,
                           trigger_alerts: bool = True) -> QualityScore:
        """
        Monitor data quality and generate alerts if needed.
        
        Args:
            data: Dictionary containing DataFrames for validation
            tmdb_data: Optional TMDB metadata
            trigger_alerts: Whether to trigger alerts for issues
            
        Returns:
            QualityScore object with current quality assessment
        """
        if not self.monitoring_enabled:
            logger.warning("Quality monitoring is disabled")
            return None
        
        logger.info("Starting quality monitoring...")
        
        # Validate data
        validation_results = self.data_validator.validate_data(data, tmdb_data)
        
        # Calculate data statistics
        data_stats = {table: len(df) for table, df in data.items() if isinstance(df, pd.DataFrame)}
        
        # Calculate quality score
        quality_score = self.quality_metrics.calculate_quality_score(
            validation_results, data_stats, tmdb_data
        )
        
        # Store in history
        self.quality_history.append(quality_score)
        
        # Check for alerts
        if trigger_alerts:
            self._check_alerts(quality_score, validation_results, data_stats)
        
        # Save monitoring data
        self._save_monitoring_data()
        
        logger.info(f"Quality monitoring completed. Score: {quality_score.score:.2f} ({quality_score.level.value})")
        
        return quality_score
    
    def _check_alerts(self, quality_score: QualityScore, validation_results: Dict[str, List[Dict[str, Any]]], data_stats: Dict[str, int]):
        """Check for quality alerts based on thresholds."""
        alerts_generated = []
        
        # Quality score alert
        if quality_score.score < self.alert_thresholds['quality_score_min']:
            alert = QualityAlert(
                id=f"quality_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity=IssueSeverity.CRITICAL if quality_score.score < 70 else IssueSeverity.WARNING,
                message=f"Quality score {quality_score.score:.2f} below threshold {self.alert_thresholds['quality_score_min']}",
                table="overall",
                rule="quality_score_threshold",
                timestamp=datetime.now(),
                data={'score': quality_score.score, 'threshold': self.alert_thresholds['quality_score_min']}
            )
            alerts_generated.append(alert)
        
        # Critical issues alert
        if quality_score.critical_issues > self.alert_thresholds['critical_issues_max']:
            alert = QualityAlert(
                id=f"critical_issues_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity=IssueSeverity.CRITICAL,
                message=f"Critical issues count {quality_score.critical_issues} exceeds threshold {self.alert_thresholds['critical_issues_max']}",
                table="overall",
                rule="critical_issues_threshold",
                timestamp=datetime.now(),
                data={'critical_issues': quality_score.critical_issues, 'threshold': self.alert_thresholds['critical_issues_max']}
            )
            alerts_generated.append(alert)
        
        # Warning issues alert
        if quality_score.warning_issues > self.alert_thresholds['warning_issues_max']:
            alert = QualityAlert(
                id=f"warning_issues_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity=IssueSeverity.WARNING,
                message=f"Warning issues count {quality_score.warning_issues} exceeds threshold {self.alert_thresholds['warning_issues_max']}",
                table="overall",
                rule="warning_issues_threshold",
                timestamp=datetime.now(),
                data={'warning_issues': quality_score.warning_issues, 'threshold': self.alert_thresholds['warning_issues_max']}
            )
            alerts_generated.append(alert)
        
        # Completeness alert
        completeness_score = quality_score.metadata.get('completeness_score', 100)
        if completeness_score < self.alert_thresholds['completeness_min']:
            alert = QualityAlert(
                id=f"completeness_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity=IssueSeverity.WARNING,
                message=f"Completeness score {completeness_score:.2f} below threshold {self.alert_thresholds['completeness_min']}",
                table="overall",
                rule="completeness_threshold",
                timestamp=datetime.now(),
                data={'completeness_score': completeness_score, 'threshold': self.alert_thresholds['completeness_min']}
            )
            alerts_generated.append(alert)
        
        # Consistency alert
        consistency_score = quality_score.metadata.get('consistency_score', 100)
        if consistency_score < self.alert_thresholds['consistency_min']:
            alert = QualityAlert(
                id=f"consistency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity=IssueSeverity.WARNING,
                message=f"Consistency score {consistency_score:.2f} below threshold {self.alert_thresholds['consistency_min']}",
                table="overall",
                rule="consistency_threshold",
                timestamp=datetime.now(),
                data={'consistency_score': consistency_score, 'threshold': self.alert_thresholds['consistency_min']}
            )
            alerts_generated.append(alert)
        
        # Table-specific alerts for critical issues
        for table, issues in validation_results.items():
            critical_count = len([i for i in issues if i.get('severity') == IssueSeverity.CRITICAL])
            if critical_count > 0:
                alert = QualityAlert(
                    id=f"table_critical_{table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    severity=IssueSeverity.CRITICAL,
                    message=f"Table '{table}' has {critical_count} critical issues",
                    table=table,
                    rule="table_critical_issues",
                    timestamp=datetime.now(),
                    data={'critical_issues': critical_count, 'total_issues': len(issues)}
                )
                alerts_generated.append(alert)
        
        # Add alerts to monitoring state
        self.alerts.extend(alerts_generated)
        
        # Trigger alert callbacks
        for alert in alerts_generated:
            self._trigger_alert_callbacks(alert)
        
        if alerts_generated:
            logger.warning(f"Generated {len(alerts_generated)} quality alerts")
    
    def _trigger_alert_callbacks(self, alert: QualityAlert):
        """Trigger registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[QualityAlert], None]):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
        logger.info("Added alert callback")
    
    def get_active_alerts(self, severity: Optional[IssueSeverity] = None) -> List[QualityAlert]:
        """Get active (unresolved) alerts."""
        alerts = [alert for alert in self.alerts if not alert.resolved]
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        return alerts
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Resolved alert: {alert_id}")
                break
    
    def get_quality_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get quality trends over the specified number of days."""
        if not self.quality_history:
            return {'trend': 'no_data', 'scores': []}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_scores = [score for score in self.quality_history if score.timestamp >= cutoff_date]
        
        if not recent_scores:
            return {'trend': 'no_recent_data', 'scores': []}
        
        return self.quality_metrics.calculate_trend_score(recent_scores[-1], recent_scores)
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of current quality status."""
        if not self.quality_history:
            return {'status': 'no_data'}
        
        current_score = self.quality_history[-1]
        active_alerts = self.get_active_alerts()
        
        return {
            'current_score': current_score.score,
            'quality_level': current_score.level.value,
            'total_records': current_score.total_records,
            'active_alerts': len(active_alerts),
            'critical_alerts': len(self.get_active_alerts(IssueSeverity.CRITICAL)),
            'warning_alerts': len(self.get_active_alerts(IssueSeverity.WARNING)),
            'last_updated': current_score.timestamp.isoformat(),
            'trends': self.get_quality_trends(7)  # Last 7 days
        }
    
    def set_alert_threshold(self, threshold_name: str, value: float):
        """Set a custom alert threshold."""
        if threshold_name in self.alert_thresholds:
            self.alert_thresholds[threshold_name] = value
            logger.info(f"Updated alert threshold {threshold_name}: {value}")
        else:
            logger.warning(f"Unknown threshold name: {threshold_name}")
    
    def enable_monitoring(self):
        """Enable quality monitoring."""
        self.monitoring_enabled = True
        logger.info("Quality monitoring enabled")
    
    def disable_monitoring(self):
        """Disable quality monitoring."""
        self.monitoring_enabled = False
        logger.info("Quality monitoring disabled")
    
    def _save_monitoring_data(self):
        """Save monitoring data to storage."""
        try:
            # Save quality history
            history_file = self.storage_path / "quality_history.json"
            history_data = []
            for score in self.quality_history[-100:]:  # Keep last 100 scores
                history_data.append({
                    'score': score.score,
                    'level': score.level.value,
                    'total_records': score.total_records,
                    'issues_count': score.issues_count,
                    'critical_issues': score.critical_issues,
                    'warning_issues': score.warning_issues,
                    'info_issues': score.info_issues,
                    'timestamp': score.timestamp.isoformat(),
                    'metadata': score.metadata
                })
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Save alerts
            alerts_file = self.storage_path / "alerts.json"
            alerts_data = []
            for alert in self.alerts[-1000:]:  # Keep last 1000 alerts
                alerts_data.append({
                    'id': alert.id,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'table': alert.table,
                    'rule': alert.rule,
                    'timestamp': alert.timestamp.isoformat(),
                    'data': alert.data,
                    'resolved': alert.resolved,
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
                })
            
            with open(alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2)
            
            # Save configuration
            config_file = self.storage_path / "monitor_config.json"
            config_data = {
                'alert_thresholds': self.alert_thresholds,
                'monitoring_enabled': self.monitoring_enabled,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    def _load_monitoring_data(self):
        """Load monitoring data from storage."""
        try:
            # Load quality history
            history_file = self.storage_path / "quality_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                for item in history_data:
                    score = QualityScore(
                        score=item['score'],
                        level=QualityLevel(item['level']),
                        total_records=item['total_records'],
                        issues_count=item['issues_count'],
                        critical_issues=item['critical_issues'],
                        warning_issues=item['warning_issues'],
                        info_issues=item['info_issues'],
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        metadata=item['metadata']
                    )
                    self.quality_history.append(score)
            
            # Load alerts
            alerts_file = self.storage_path / "alerts.json"
            if alerts_file.exists():
                with open(alerts_file, 'r') as f:
                    alerts_data = json.load(f)
                
                for item in alerts_data:
                    alert = QualityAlert(
                        id=item['id'],
                        severity=IssueSeverity(item['severity']),
                        message=item['message'],
                        table=item['table'],
                        rule=item['rule'],
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        data=item['data'],
                        resolved=item['resolved'],
                        resolved_at=datetime.fromisoformat(item['resolved_at']) if item['resolved_at'] else None
                    )
                    self.alerts.append(alert)
            
            # Load configuration
            config_file = self.storage_path / "monitor_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                self.alert_thresholds.update(config_data.get('alert_thresholds', {}))
                self.monitoring_enabled = config_data.get('monitoring_enabled', True)
                
        except Exception as e:
            logger.error(f"Error loading monitoring data: {e}")
    
    def clear_history(self, days: Optional[int] = None):
        """Clear monitoring history."""
        if days is None:
            self.quality_history.clear()
            self.alerts.clear()
            logger.info("Cleared all monitoring history")
        else:
            cutoff_date = datetime.now() - timedelta(days=days)
            self.quality_history = [score for score in self.quality_history if score.timestamp >= cutoff_date]
            self.alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_date]
            logger.info(f"Cleared monitoring history older than {days} days")
        
        self._save_monitoring_data() 