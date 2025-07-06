"""
Quality Reporter Module

Generates comprehensive quality reports and dashboards for data quality assessment.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import pandas as pd

from .data_validator import DataValidator
from .quality_metrics import QualityMetrics, QualityScore, QualityLevel
from .quality_monitor import QualityMonitor
from .validation_rules import IssueSeverity

logger = logging.getLogger(__name__)


class QualityReporter:
    """Generates comprehensive quality reports and dashboards."""
    
    def __init__(self, 
                 data_validator: Optional[DataValidator] = None,
                 quality_metrics: Optional[QualityMetrics] = None,
                 quality_monitor: Optional[QualityMonitor] = None,
                 output_path: Optional[str] = None):
        """
        Initialize the quality reporter.
        
        Args:
            data_validator: Data validator instance
            quality_metrics: Quality metrics calculator
            quality_monitor: Quality monitor instance
            output_path: Path for storing reports
        """
        self.data_validator = data_validator or DataValidator()
        self.quality_metrics = quality_metrics or QualityMetrics()
        self.quality_monitor = quality_monitor
        self.output_path = Path(output_path) if output_path else Path("data/quality_reports")
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, 
                                    data: Dict[str, Any],
                                    tmdb_data: Optional[Dict[int, Dict]] = None,
                                    include_trends: bool = True,
                                    include_recommendations: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report.
        
        Args:
            data: Dictionary containing DataFrames for validation
            tmdb_data: Optional TMDB metadata
            include_trends: Whether to include trend analysis
            include_recommendations: Whether to include recommendations
            
        Returns:
            Comprehensive quality report dictionary
        """
        logger.info("Generating comprehensive quality report...")
        
        # Validate data and calculate quality score
        validation_results = self.data_validator.validate_data(data, tmdb_data)
        data_stats = {table: len(df) for table, df in data.items() if isinstance(df, pd.DataFrame)}
        quality_score = self.quality_metrics.calculate_quality_score(validation_results, data_stats, tmdb_data)
        
        # Generate report sections
        report = {
            'report_metadata': self._generate_report_metadata(),
            'executive_summary': self._generate_executive_summary(quality_score, validation_results),
            'quality_score': self._generate_quality_score_section(quality_score),
            'validation_results': self._generate_validation_results_section(validation_results),
            'table_analysis': self._generate_table_analysis_section(validation_results, data_stats),
            'issue_breakdown': self._generate_issue_breakdown_section(validation_results),
            'data_statistics': self._generate_data_statistics_section(data_stats, tmdb_data)
        }
        
        # Add optional sections
        if include_trends and self.quality_monitor:
            report['trends'] = self._generate_trends_section()
        
        if include_recommendations:
            report['recommendations'] = self._generate_recommendations_section(quality_score, validation_results)
        
        # Save report
        self._save_report(report, "comprehensive_report")
        
        logger.info("Comprehensive quality report generated successfully")
        return report
    
    def generate_executive_summary(self, quality_score: QualityScore, validation_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate an executive summary of quality assessment."""
        return {
            'overall_quality_score': quality_score.score,
            'quality_level': quality_score.level.value,
            'total_records': quality_score.total_records,
            'total_issues': quality_score.issues_count,
            'critical_issues': quality_score.critical_issues,
            'warning_issues': quality_score.warning_issues,
            'info_issues': quality_score.info_issues,
            'assessment_date': quality_score.timestamp.isoformat(),
            'key_insights': self.quality_metrics.generate_quality_insights(quality_score, validation_results),
            'status': self._determine_overall_status(quality_score)
        }
    
    def generate_table_specific_report(self, table_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report for a specific table."""
        if table_name not in data:
            return {'error': f'Table {table_name} not found in data'}
        
        # Validate specific table
        table_data = {table_name: data[table_name]}
        validation_results = self.data_validator.validate_data(table_data)
        data_stats = {table_name: len(data[table_name])}
        
        # Calculate table-specific quality score
        quality_score = self.quality_metrics.calculate_quality_score(validation_results, data_stats)
        
        report = {
            'table_name': table_name,
            'report_metadata': self._generate_report_metadata(),
            'quality_score': self._generate_quality_score_section(quality_score),
            'validation_results': validation_results.get(table_name, []),
            'data_statistics': {
                'total_records': len(data[table_name]),
                'columns': list(data[table_name].columns),
                'data_types': data[table_name].dtypes.to_dict(),
                'missing_values': data[table_name].isnull().sum().to_dict()
            },
            'recommendations': self.quality_metrics.get_quality_recommendations(quality_score, validation_results)
        }
        
        # Save table-specific report
        self._save_report(report, f"table_report_{table_name}")
        
        return report
    
    def generate_issue_report(self, validation_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate a detailed issue report."""
        issues_by_severity = {
            'critical': [],
            'warning': [],
            'info': []
        }
        
        issues_by_rule = {}
        issues_by_table = {}
        
        for table, issues in validation_results.items():
            if table not in issues_by_table:
                issues_by_table[table] = []
            
            for issue in issues:
                severity = issue.get('severity', IssueSeverity.INFO)
                if isinstance(severity, IssueSeverity):
                    severity = severity.value
                
                issues_by_severity[severity].append(issue)
                issues_by_table[table].append(issue)
                
                rule = issue.get('rule', 'unknown')
                if rule not in issues_by_rule:
                    issues_by_rule[rule] = []
                issues_by_rule[rule].append(issue)
        
        report = {
            'report_metadata': self._generate_report_metadata(),
            'summary': {
                'total_issues': sum(len(issues) for issues in issues_by_severity.values()),
                'critical_issues': len(issues_by_severity['critical']),
                'warning_issues': len(issues_by_severity['warning']),
                'info_issues': len(issues_by_severity['info'])
            },
            'issues_by_severity': issues_by_severity,
            'issues_by_rule': issues_by_rule,
            'issues_by_table': issues_by_table,
            'priority_issues': self._identify_priority_issues(validation_results)
        }
        
        # Save issue report
        self._save_report(report, "issue_report")
        
        return report
    
    def generate_trend_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate a trend analysis report."""
        if not self.quality_monitor:
            return {'error': 'Quality monitor not available for trend analysis'}
        
        trends = self.quality_monitor.get_quality_trends(days)
        quality_summary = self.quality_monitor.get_quality_summary()
        
        report = {
            'report_metadata': self._generate_report_metadata(),
            'analysis_period': f"Last {days} days",
            'trends': trends,
            'current_status': quality_summary,
            'trend_insights': self._generate_trend_insights(trends),
            'historical_data': self._get_historical_data(days)
        }
        
        # Save trend report
        self._save_report(report, f"trend_report_{days}d")
        
        return report
    
    def _generate_report_metadata(self) -> Dict[str, Any]:
        """Generate report metadata."""
        return {
            'report_type': 'quality_assessment',
            'generated_at': datetime.now().isoformat(),
            'version': '1.0',
            'generator': 'QualityReporter'
        }
    
    def _generate_executive_summary(self, quality_score: QualityScore, validation_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate executive summary section."""
        return {
            'overall_quality_score': quality_score.score,
            'quality_level': quality_score.level.value,
            'total_records': quality_score.total_records,
            'total_issues': quality_score.issues_count,
            'critical_issues': quality_score.critical_issues,
            'warning_issues': quality_score.warning_issues,
            'info_issues': quality_score.info_issues,
            'assessment_date': quality_score.timestamp.isoformat(),
            'key_insights': self.quality_metrics.generate_quality_insights(quality_score, validation_results),
            'status': self._determine_overall_status(quality_score)
        }
    
    def _generate_quality_score_section(self, quality_score: QualityScore) -> Dict[str, Any]:
        """Generate quality score section."""
        return {
            'overall_score': quality_score.score,
            'quality_level': quality_score.level.value,
            'component_scores': {
                'base_score': quality_score.metadata.get('base_score', 0),
                'completeness_score': quality_score.metadata.get('completeness_score', 0),
                'consistency_score': quality_score.metadata.get('consistency_score', 0),
                'accuracy_score': quality_score.metadata.get('accuracy_score', 0),
                'enrichment_score': quality_score.metadata.get('enrichment_score', 0)
            },
            'issue_counts': {
                'critical': quality_score.critical_issues,
                'warning': quality_score.warning_issues,
                'info': quality_score.info_issues
            }
        }
    
    def _generate_validation_results_section(self, validation_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate validation results section."""
        return {
            'total_issues': sum(len(issues) for issues in validation_results.values()),
            'issues_by_table': {table: len(issues) for table, issues in validation_results.items()},
            'issues_by_severity': self._count_issues_by_severity(validation_results),
            'detailed_issues': validation_results
        }
    
    def _generate_table_analysis_section(self, validation_results: Dict[str, List[Dict[str, Any]]], data_stats: Dict[str, int]) -> Dict[str, Any]:
        """Generate table analysis section."""
        table_scores = self.quality_metrics.calculate_table_scores(validation_results, data_stats)
        
        analysis = {}
        for table, score in table_scores.items():
            analysis[table] = {
                'quality_score': score.score,
                'quality_level': score.level.value,
                'total_records': score.total_records,
                'issues_count': score.issues_count,
                'critical_issues': score.critical_issues,
                'warning_issues': score.warning_issues,
                'info_issues': score.info_issues
            }
        
        return analysis
    
    def _generate_issue_breakdown_section(self, validation_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate issue breakdown section."""
        issues_by_rule = {}
        issues_by_severity = {'critical': 0, 'warning': 0, 'info': 0}
        
        for table, issues in validation_results.items():
            for issue in issues:
                rule = issue.get('rule', 'unknown')
                if rule not in issues_by_rule:
                    issues_by_rule[rule] = 0
                issues_by_rule[rule] += 1
                
                severity = issue.get('severity', IssueSeverity.INFO)
                if isinstance(severity, IssueSeverity):
                    severity = severity.value
                issues_by_severity[severity] += 1
        
        return {
            'issues_by_rule': issues_by_rule,
            'issues_by_severity': issues_by_severity,
            'most_common_issues': sorted(issues_by_rule.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _generate_data_statistics_section(self, data_stats: Dict[str, int], tmdb_data: Optional[Dict[int, Dict]] = None) -> Dict[str, Any]:
        """Generate data statistics section."""
        stats = {
            'total_records': sum(data_stats.values()),
            'records_by_table': data_stats,
            'tmdb_enrichment': {
                'enriched_movies': len(tmdb_data) if tmdb_data else 0,
                'total_movies': data_stats.get('movies', 0),
                'enrichment_percentage': (len(tmdb_data) / data_stats.get('movies', 1) * 100) if tmdb_data and data_stats.get('movies', 0) > 0 else 0
            }
        }
        
        return stats
    
    def _generate_trends_section(self) -> Dict[str, Any]:
        """Generate trends section."""
        if not self.quality_monitor:
            return {'error': 'Quality monitor not available'}
        
        return {
            'recent_trends': self.quality_monitor.get_quality_trends(7),
            'monthly_trends': self.quality_monitor.get_quality_trends(30),
            'quality_summary': self.quality_monitor.get_quality_summary()
        }
    
    def _generate_recommendations_section(self, quality_score: QualityScore, validation_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate recommendations section."""
        recommendations = self.quality_metrics.get_quality_recommendations(quality_score, validation_results)
        
        return {
            'priority_recommendations': recommendations[:5],
            'all_recommendations': recommendations,
            'implementation_priority': self._prioritize_recommendations(recommendations, quality_score)
        }
    
    def _determine_overall_status(self, quality_score: QualityScore) -> str:
        """Determine overall status based on quality score."""
        if quality_score.level == QualityLevel.EXCELLENT:
            return "Excellent - No immediate action required"
        elif quality_score.level == QualityLevel.GOOD:
            return "Good - Minor improvements recommended"
        elif quality_score.level == QualityLevel.FAIR:
            return "Fair - Attention needed"
        elif quality_score.level == QualityLevel.POOR:
            return "Poor - Immediate action required"
        else:
            return "Critical - System may be compromised"
    
    def _count_issues_by_severity(self, validation_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        """Count issues by severity across all tables."""
        counts = {'critical': 0, 'warning': 0, 'info': 0}
        
        for table, issues in validation_results.items():
            for issue in issues:
                severity = issue.get('severity', IssueSeverity.INFO)
                if isinstance(severity, IssueSeverity):
                    severity = severity.value
                counts[severity] += 1
        
        return counts
    
    def _identify_priority_issues(self, validation_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify priority issues that need immediate attention."""
        priority_issues = []
        
        for table, issues in validation_results.items():
            for issue in issues:
                if issue.get('severity') == IssueSeverity.CRITICAL:
                    priority_issues.append({
                        'table': table,
                        'rule': issue.get('rule'),
                        'message': issue.get('message'),
                        'severity': 'critical',
                        'data': issue.get('row_data', {})
                    })
        
        return priority_issues
    
    def _generate_trend_insights(self, trends: Dict[str, Any]) -> List[str]:
        """Generate insights from trend analysis."""
        insights = []
        
        trend = trends.get('trend', 'no_data')
        if trend == 'improving':
            insights.append("Data quality is showing positive trends")
        elif trend == 'declining':
            insights.append("Data quality is declining - investigate root causes")
        elif trend == 'stable':
            insights.append("Data quality is stable - maintain current practices")
        
        change = trends.get('change', 0)
        if abs(change) > 5:
            insights.append(f"Significant quality change detected: {change:+.2f} points")
        
        volatility = trends.get('volatility', 0)
        if volatility > 10:
            insights.append("High quality volatility detected - consider stabilizing processes")
        
        return insights
    
    def _get_historical_data(self, days: int) -> List[Dict[str, Any]]:
        """Get historical quality data."""
        if not self.quality_monitor:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_scores = [score for score in self.quality_monitor.quality_history if score.timestamp >= cutoff_date]
        
        return [
            {
                'date': score.timestamp.isoformat(),
                'score': score.score,
                'level': score.level.value,
                'issues_count': score.issues_count
            }
            for score in recent_scores
        ]
    
    def _prioritize_recommendations(self, recommendations: List[str], quality_score: QualityScore) -> List[Dict[str, Any]]:
        """Prioritize recommendations based on quality score and impact."""
        priorities = []
        
        for i, recommendation in enumerate(recommendations):
            priority = 'high' if quality_score.score < 70 else 'medium' if quality_score.score < 85 else 'low'
            priorities.append({
                'recommendation': recommendation,
                'priority': priority,
                'estimated_impact': 'high' if 'critical' in recommendation.lower() else 'medium',
                'order': i + 1
            })
        
        return priorities
    
    def _save_report(self, report: Dict[str, Any], report_name: str):
        """Save report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{report_name}_{timestamp}.json"
        filepath = self.output_path / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def export_to_csv(self, report: Dict[str, Any], report_name: str):
        """Export report data to CSV format."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export issues to CSV
        if 'validation_results' in report:
            issues_data = []
            for table, issues in report['validation_results'].items():
                for issue in issues:
                    issues_data.append({
                        'table': table,
                        'rule': issue.get('rule'),
                        'severity': issue.get('severity'),
                        'message': issue.get('message'),
                        'timestamp': issue.get('timestamp', datetime.now().isoformat())
                    })
            
            if issues_data:
                issues_df = pd.DataFrame(issues_data)
                issues_file = self.output_path / f"{report_name}_issues_{timestamp}.csv"
                issues_df.to_csv(issues_file, index=False)
                logger.info(f"Issues exported to {issues_file}")
        
        # Export quality scores to CSV
        if 'quality_score' in report:
            scores_data = [{
                'score': report['quality_score']['overall_score'],
                'level': report['quality_score']['quality_level'],
                'timestamp': report.get('report_metadata', {}).get('generated_at', datetime.now().isoformat())
            }]
            
            scores_df = pd.DataFrame(scores_data)
            scores_file = self.output_path / f"{report_name}_scores_{timestamp}.csv"
            scores_df.to_csv(scores_file, index=False)
            logger.info(f"Scores exported to {scores_file}") 