"""
Quality Metrics Module

Provides quality scoring algorithms and comprehensive quality assessment metrics.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .validation_rules import IssueSeverity

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Quality level classifications."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class QualityScore:
    """Represents a quality score with metadata."""
    score: float  # 0-100
    level: QualityLevel
    total_records: int
    issues_count: int
    critical_issues: int
    warning_issues: int
    info_issues: int
    timestamp: datetime
    metadata: Dict[str, Any]


class QualityMetrics:
    """Quality metrics calculator and scorer for movie recommendation data."""
    
    def __init__(self):
        """Initialize the quality metrics calculator."""
        self.severity_weights = {
            IssueSeverity.CRITICAL: 10.0,
            IssueSeverity.WARNING: 3.0,
            IssueSeverity.INFO: 1.0
        }
        
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 95.0,
            QualityLevel.GOOD: 85.0,
            QualityLevel.FAIR: 70.0,
            QualityLevel.POOR: 50.0,
            QualityLevel.CRITICAL: 0.0
        }
    
    def calculate_quality_score(self, 
                              validation_results: Dict[str, List[Dict[str, Any]]],
                              data_stats: Dict[str, int],
                              tmdb_data: Optional[Dict[int, Dict]] = None) -> QualityScore:
        """
        Calculate overall quality score based on validation results.
        
        Args:
            validation_results: Dictionary containing validation issues by table
            data_stats: Dictionary containing record counts by table
            tmdb_data: Optional TMDB metadata for additional scoring
            
        Returns:
            QualityScore object with comprehensive quality assessment
        """
        logger.info("Calculating quality score...")
        
        # Calculate base quality score from validation issues
        base_score = self._calculate_base_score(validation_results, data_stats)
        
        # Calculate additional quality metrics
        completeness_score = self._calculate_completeness_score(validation_results, data_stats)
        consistency_score = self._calculate_consistency_score(validation_results, data_stats)
        accuracy_score = self._calculate_accuracy_score(validation_results, data_stats)
        
        # Calculate TMDB enrichment score if available
        enrichment_score = 0.0
        if tmdb_data:
            enrichment_score = self._calculate_enrichment_score(tmdb_data, data_stats)
        
        # Weighted combination of scores
        final_score = (
            base_score * 0.4 +
            completeness_score * 0.25 +
            consistency_score * 0.2 +
            accuracy_score * 0.1 +
            enrichment_score * 0.05
        )
        
        # Ensure score is within bounds
        final_score = max(0.0, min(100.0, final_score))
        
        # Determine quality level
        quality_level = self._determine_quality_level(final_score)
        
        # Count issues by severity
        critical_count = sum(
            len([issue for issue in issues if issue.get('severity') == IssueSeverity.CRITICAL])
            for issues in validation_results.values()
        )
        warning_count = sum(
            len([issue for issue in issues if issue.get('severity') == IssueSeverity.WARNING])
            for issues in validation_results.values()
        )
        info_count = sum(
            len([issue for issue in issues if issue.get('severity') == IssueSeverity.INFO])
            for issues in validation_results.values()
        )
        
        total_issues = critical_count + warning_count + info_count
        total_records = sum(data_stats.values())
        
        # Prepare metadata
        metadata = {
            'base_score': base_score,
            'completeness_score': completeness_score,
            'consistency_score': consistency_score,
            'accuracy_score': accuracy_score,
            'enrichment_score': enrichment_score,
            'issues_by_table': {table: len(issues) for table, issues in validation_results.items()},
            'data_volume': data_stats
        }
        
        return QualityScore(
            score=final_score,
            level=quality_level,
            total_records=total_records,
            issues_count=total_issues,
            critical_issues=critical_count,
            warning_issues=warning_count,
            info_issues=info_count,
            timestamp=datetime.now(),
            metadata=metadata
        )
    
    def _calculate_base_score(self, validation_results: Dict[str, List[Dict[str, Any]]], data_stats: Dict[str, int]) -> float:
        """Calculate base quality score from validation issues."""
        total_weighted_penalty = 0.0
        total_records = sum(data_stats.values())
        
        if total_records == 0:
            return 100.0
        
        for table, issues in validation_results.items():
            table_records = data_stats.get(table, 0)
            if table_records == 0:
                continue
            
            for issue in issues:
                severity = issue.get('severity', IssueSeverity.INFO)
                weight = self.severity_weights.get(severity, 1.0)
                
                # Penalty based on severity and table size
                penalty = weight / table_records
                total_weighted_penalty += penalty
        
        # Convert penalty to score (higher penalty = lower score)
        base_score = max(0.0, 100.0 - (total_weighted_penalty * 100))
        
        return base_score
    
    def _calculate_completeness_score(self, validation_results: Dict[str, List[Dict[str, Any]]], data_stats: Dict[str, int]) -> float:
        """Calculate data completeness score."""
        completeness_issues = 0
        total_records = sum(data_stats.values())
        
        if total_records == 0:
            return 100.0
        
        # Count completeness-related issues
        for table, issues in validation_results.items():
            for issue in issues:
                rule = issue.get('rule', '')
                if any(keyword in rule.lower() for keyword in ['empty', 'missing', 'null', 'na']):
                    completeness_issues += 1
        
        # Calculate completeness score
        if completeness_issues == 0:
            return 100.0
        
        # Penalty based on completeness issues
        penalty = min(completeness_issues / total_records * 100, 50.0)  # Cap penalty at 50%
        completeness_score = max(50.0, 100.0 - penalty)
        
        return completeness_score
    
    def _calculate_consistency_score(self, validation_results: Dict[str, List[Dict[str, Any]]], data_stats: Dict[str, int]) -> float:
        """Calculate data consistency score."""
        consistency_issues = 0
        total_records = sum(data_stats.values())
        
        if total_records == 0:
            return 100.0
        
        # Count consistency-related issues
        for table, issues in validation_results.items():
            for issue in issues:
                rule = issue.get('rule', '')
                if any(keyword in rule.lower() for keyword in ['duplicate', 'unique', 'reference', 'foreign']):
                    consistency_issues += 1
        
        # Calculate consistency score
        if consistency_issues == 0:
            return 100.0
        
        # Penalty based on consistency issues
        penalty = min(consistency_issues / total_records * 100, 40.0)  # Cap penalty at 40%
        consistency_score = max(60.0, 100.0 - penalty)
        
        return consistency_score
    
    def _calculate_accuracy_score(self, validation_results: Dict[str, List[Dict[str, Any]]], data_stats: Dict[str, int]) -> float:
        """Calculate data accuracy score."""
        accuracy_issues = 0
        total_records = sum(data_stats.values())
        
        if total_records == 0:
            return 100.0
        
        # Count accuracy-related issues
        for table, issues in validation_results.items():
            for issue in issues:
                rule = issue.get('rule', '')
                if any(keyword in rule.lower() for keyword in ['range', 'invalid', 'type', 'format']):
                    accuracy_issues += 1
        
        # Calculate accuracy score
        if accuracy_issues == 0:
            return 100.0
        
        # Penalty based on accuracy issues
        penalty = min(accuracy_issues / total_records * 100, 30.0)  # Cap penalty at 30%
        accuracy_score = max(70.0, 100.0 - penalty)
        
        return accuracy_score
    
    def _calculate_enrichment_score(self, tmdb_data: Dict[int, Dict], data_stats: Dict[str, int]) -> float:
        """Calculate TMDB data enrichment score."""
        movies_count = data_stats.get('movies', 0)
        
        if movies_count == 0:
            return 0.0
        
        # Calculate enrichment percentage
        enriched_movies = len(tmdb_data)
        enrichment_percentage = (enriched_movies / movies_count) * 100
        
        # Score based on enrichment percentage
        if enrichment_percentage >= 80:
            return 100.0
        elif enrichment_percentage >= 60:
            return 80.0
        elif enrichment_percentage >= 40:
            return 60.0
        elif enrichment_percentage >= 20:
            return 40.0
        else:
            return 20.0
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score."""
        if score >= self.quality_thresholds[QualityLevel.EXCELLENT]:
            return QualityLevel.EXCELLENT
        elif score >= self.quality_thresholds[QualityLevel.GOOD]:
            return QualityLevel.GOOD
        elif score >= self.quality_thresholds[QualityLevel.FAIR]:
            return QualityLevel.FAIR
        elif score >= self.quality_thresholds[QualityLevel.POOR]:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def calculate_table_scores(self, validation_results: Dict[str, List[Dict[str, Any]]], data_stats: Dict[str, int]) -> Dict[str, QualityScore]:
        """Calculate quality scores for individual tables."""
        table_scores = {}
        
        for table in data_stats.keys():
            table_issues = validation_results.get(table, [])
            table_records = data_stats.get(table, 0)
            
            # Create single-table validation results
            single_table_results = {table: table_issues}
            single_table_stats = {table: table_records}
            
            # Calculate score for this table
            table_score = self.calculate_quality_score(single_table_results, single_table_stats)
            table_scores[table] = table_score
        
        return table_scores
    
    def calculate_trend_score(self, current_score: QualityScore, historical_scores: List[QualityScore]) -> Dict[str, Any]:
        """Calculate quality trend analysis."""
        if not historical_scores:
            return {
                'trend': 'no_data',
                'change': 0.0,
                'trend_direction': 'stable',
                'volatility': 0.0
            }
        
        # Calculate trend metrics
        recent_scores = [score.score for score in historical_scores[-10:]]  # Last 10 scores
        current_value = current_score.score
        
        if len(recent_scores) < 2:
            change = 0.0
        else:
            change = current_value - recent_scores[-1]
        
        # Determine trend direction
        if change > 2.0:
            trend_direction = 'improving'
        elif change < -2.0:
            trend_direction = 'declining'
        else:
            trend_direction = 'stable'
        
        # Calculate volatility (standard deviation of recent scores)
        if len(recent_scores) >= 2:
            volatility = np.std(recent_scores)
        else:
            volatility = 0.0
        
        # Determine overall trend
        if len(recent_scores) >= 3:
            # Simple linear trend
            x = np.arange(len(recent_scores))
            slope = np.polyfit(x, recent_scores, 1)[0]
            
            if slope > 1.0:
                trend = 'strongly_improving'
            elif slope > 0.1:
                trend = 'improving'
            elif slope < -1.0:
                trend = 'strongly_declining'
            elif slope < -0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'change': change,
            'trend_direction': trend_direction,
            'volatility': volatility,
            'recent_scores': recent_scores[-5:]  # Last 5 scores for context
        }
    
    def generate_quality_insights(self, quality_score: QualityScore, validation_results: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate actionable insights from quality assessment."""
        insights = []
        
        # Overall quality insights
        if quality_score.level == QualityLevel.EXCELLENT:
            insights.append("Data quality is excellent - no immediate action required")
        elif quality_score.level == QualityLevel.GOOD:
            insights.append("Data quality is good with minor issues to address")
        elif quality_score.level == QualityLevel.FAIR:
            insights.append("Data quality needs improvement - review critical issues")
        elif quality_score.level == QualityLevel.POOR:
            insights.append("Data quality is poor - immediate attention required")
        else:
            insights.append("Data quality is critical - system may be compromised")
        
        # Critical issues insights
        if quality_score.critical_issues > 0:
            insights.append(f"Found {quality_score.critical_issues} critical issues that must be resolved")
        
        # Table-specific insights
        for table, issues in validation_results.items():
            if issues:
                critical_count = len([i for i in issues if i.get('severity') == IssueSeverity.CRITICAL])
                if critical_count > 0:
                    insights.append(f"Table '{table}' has {critical_count} critical issues")
        
        # Completeness insights
        completeness_score = quality_score.metadata.get('completeness_score', 100)
        if completeness_score < 90:
            insights.append("Data completeness needs improvement - consider data enrichment")
        
        # Consistency insights
        consistency_score = quality_score.metadata.get('consistency_score', 100)
        if consistency_score < 90:
            insights.append("Data consistency issues detected - review duplicate handling")
        
        # Enrichment insights
        enrichment_score = quality_score.metadata.get('enrichment_score', 0)
        if enrichment_score < 50:
            insights.append("Low TMDB enrichment - consider expanding metadata collection")
        
        return insights
    
    def get_quality_recommendations(self, quality_score: QualityScore, validation_results: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate recommendations for improving data quality."""
        recommendations = []
        
        # Critical issues recommendations
        if quality_score.critical_issues > 0:
            recommendations.append("Prioritize fixing critical validation issues")
            recommendations.append("Review data ingestion pipeline for error handling")
        
        # Completeness recommendations
        completeness_score = quality_score.metadata.get('completeness_score', 100)
        if completeness_score < 85:
            recommendations.append("Implement data completeness checks in ingestion pipeline")
            recommendations.append("Consider data enrichment from additional sources")
        
        # Consistency recommendations
        consistency_score = quality_score.metadata.get('consistency_score', 100)
        if consistency_score < 85:
            recommendations.append("Implement referential integrity checks")
            recommendations.append("Add duplicate detection and resolution logic")
        
        # Accuracy recommendations
        accuracy_score = quality_score.metadata.get('accuracy_score', 100)
        if accuracy_score < 85:
            recommendations.append("Implement data type validation in ingestion")
            recommendations.append("Add range validation for numeric fields")
        
        # Enrichment recommendations
        enrichment_score = quality_score.metadata.get('enrichment_score', 0)
        if enrichment_score < 60:
            recommendations.append("Expand TMDB data collection coverage")
            recommendations.append("Consider additional metadata sources")
        
        # Monitoring recommendations
        if quality_score.score < 90:
            recommendations.append("Implement automated quality monitoring")
            recommendations.append("Set up quality alerting for critical issues")
        
        return recommendations 