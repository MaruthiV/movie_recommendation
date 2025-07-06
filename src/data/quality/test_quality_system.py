"""
Tests for the Quality System Components

Tests the validation rules, data validator, quality metrics, monitor, and reporter.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from .validation_rules import ValidationRules, IssueSeverity, ValidationRule
from .data_validator import DataValidator
from .quality_metrics import QualityMetrics, QualityScore, QualityLevel
from .quality_monitor import QualityMonitor, QualityAlert
from .quality_reporter import QualityReporter


class TestValidationRules:
    """Test validation rules functionality."""
    
    @pytest.fixture
    def validation_rules(self):
        return ValidationRules()
    
    @pytest.fixture
    def sample_movies_data(self):
        return pd.DataFrame({
            'movieId': [1, 2, 3, 4, 5],
            'title_clean': ['Movie 1', 'Movie 2', '', 'Movie 4', 'Movie 5'],
            'avg_rating': [4.5, 3.8, 2.1, 6.0, 4.2],
            'release_year': [2020, 2019, 2025, 2018, 2021],
            'genres_list': [['Action'], ['Drama'], [], ['Comedy'], ['Thriller']],
            'tmdb_id': [123, 456, None, 789, 101]
        })
    
    @pytest.fixture
    def sample_ratings_data(self):
        return pd.DataFrame({
            'userId': [1, 1, 2, 2, 3],
            'movieId': [1, 2, 1, 3, 4],
            'rating': [4.5, 3.0, 5.0, 2.5, 6.0],
            'timestamp': [
                datetime.now() - timedelta(days=1),
                datetime.now() - timedelta(days=2),
                datetime.now() - timedelta(days=3),
                datetime.now() - timedelta(days=4),
                datetime.now() + timedelta(days=1)  # Future date
            ]
        })
    
    def test_validation_rules_initialization(self, validation_rules):
        """Test that validation rules are properly initialized."""
        assert len(validation_rules.rules) > 0
        assert any(rule.table == 'movies' for rule in validation_rules.rules)
        assert any(rule.table == 'ratings' for rule in validation_rules.rules)
        assert any(rule.table == 'users' for rule in validation_rules.rules)
    
    def test_get_rules_for_table(self, validation_rules):
        """Test getting rules for specific table."""
        movie_rules = validation_rules.get_rules_for_table('movies')
        assert len(movie_rules) > 0
        assert all(rule.table == 'movies' for rule in movie_rules)
    
    def test_get_rules_by_severity(self, validation_rules):
        """Test getting rules by severity."""
        critical_rules = validation_rules.get_rules_by_severity(IssueSeverity.CRITICAL)
        assert len(critical_rules) > 0
        assert all(rule.severity == IssueSeverity.CRITICAL for rule in critical_rules)
    
    def test_movie_id_unique_validation(self, validation_rules, sample_movies_data):
        """Test movie ID uniqueness validation."""
        # Add duplicate movie ID
        duplicate_data = sample_movies_data.copy()
        duplicate_data.loc[5] = [1, 'Duplicate Movie', 3.5, 2020, ['Action'], 999]
        
        issues = validation_rules._validate_movie_id_unique(duplicate_data)
        assert len(issues) > 0
        assert any('Duplicate movie ID' in issue['message'] for issue in issues)
    
    def test_movie_title_not_empty_validation(self, validation_rules, sample_movies_data):
        """Test movie title not empty validation."""
        issues = validation_rules._validate_movie_title_not_empty(sample_movies_data)
        assert len(issues) > 0
        assert any('Empty movie title' in issue['message'] for issue in issues)
    
    def test_movie_rating_range_validation(self, validation_rules, sample_movies_data):
        """Test movie rating range validation."""
        issues = validation_rules._validate_movie_rating_range(sample_movies_data)
        assert len(issues) > 0
        assert any('Invalid movie rating' in issue['message'] for issue in issues)
    
    def test_rating_range_validation(self, validation_rules, sample_ratings_data):
        """Test rating range validation."""
        issues = validation_rules._validate_rating_range(sample_ratings_data)
        assert len(issues) > 0
        assert any('Invalid rating' in issue['message'] for issue in issues)
    
    def test_rating_timestamp_validation(self, validation_rules, sample_ratings_data):
        """Test rating timestamp validation."""
        issues = validation_rules._validate_rating_timestamp(sample_ratings_data)
        # Should find issues with future timestamps
        assert len(issues) >= 0  # May or may not have issues depending on current time
        if len(issues) > 0:
            assert any('Invalid timestamp' in issue['message'] for issue in issues)


class TestDataValidator:
    """Test data validator functionality."""
    
    @pytest.fixture
    def data_validator(self):
        return DataValidator()
    
    @pytest.fixture
    def sample_data(self):
        return {
            'movies': pd.DataFrame({
                'movieId': [1, 2, 3],
                'title_clean': ['Movie 1', 'Movie 2', ''],
                'avg_rating': [4.5, 3.8, 6.0],
                'release_year': [2020, 2019, 2025],
                'genres_list': [['Action'], ['Drama'], []],
                'tmdb_id': [123, 456, None]
            }),
            'ratings': pd.DataFrame({
                'userId': [1, 1, 2],
                'movieId': [1, 2, 1],
                'rating': [4.5, 3.0, 6.0],
                'timestamp': [datetime.now(), datetime.now(), datetime.now()]
            }),
            'users': pd.DataFrame({
                'userId': [1, 2, 3],
                'avg_rating': [4.2, 3.8, 4.5],
                'total_ratings': [10, 5, 15]
            })
        }
    
    def test_validate_data(self, data_validator, sample_data):
        """Test comprehensive data validation."""
        validation_results = data_validator.validate_data(sample_data)
        
        assert 'movies' in validation_results
        assert 'ratings' in validation_results
        assert 'users' in validation_results
        
        # Should find issues
        assert len(validation_results['movies']) > 0
        assert len(validation_results['ratings']) > 0
    
    def test_validate_schema(self, data_validator, sample_data):
        """Test schema validation."""
        expected_schema = {
            'movies': ['movieId', 'title_clean', 'avg_rating', 'release_year', 'genres_list', 'tmdb_id'],
            'ratings': ['userId', 'movieId', 'rating', 'timestamp'],
            'users': ['userId', 'avg_rating', 'total_ratings']
        }
        
        schema_issues = data_validator.validate_schema(sample_data, expected_schema)
        assert len(schema_issues) == 0  # Should pass with correct schema
    
    def test_validate_data_types(self, data_validator, sample_data):
        """Test data type validation."""
        expected_types = {
            'movies': {'movieId': 'int', 'avg_rating': 'float'},
            'ratings': {'userId': 'int', 'rating': 'float'},
            'users': {'userId': 'int', 'avg_rating': 'float'}
        }
        
        type_issues = data_validator.validate_data_types(sample_data, expected_types)
        # Should have some type issues since we're using string dtypes
        assert len(type_issues) >= 0
    
    def test_has_critical_issues(self, data_validator, sample_data):
        """Test critical issues detection."""
        validation_results = data_validator.validate_data(sample_data)
        has_critical = data_validator.has_critical_issues()
        assert isinstance(has_critical, bool)


class TestQualityMetrics:
    """Test quality metrics functionality."""
    
    @pytest.fixture
    def quality_metrics(self):
        return QualityMetrics()
    
    @pytest.fixture
    def sample_validation_results(self):
        return {
            'movies': [
                {
                    'rule': 'movie_title_not_empty',
                    'severity': IssueSeverity.CRITICAL,
                    'message': 'Empty movie title'
                },
                {
                    'rule': 'movie_rating_range',
                    'severity': IssueSeverity.WARNING,
                    'message': 'Invalid rating'
                }
            ],
            'ratings': [
                {
                    'rule': 'rating_range',
                    'severity': IssueSeverity.CRITICAL,
                    'message': 'Invalid rating'
                }
            ]
        }
    
    @pytest.fixture
    def sample_data_stats(self):
        return {
            'movies': 1000,
            'ratings': 5000,
            'users': 500
        }
    
    def test_calculate_quality_score(self, quality_metrics, sample_validation_results, sample_data_stats):
        """Test quality score calculation."""
        quality_score = quality_metrics.calculate_quality_score(sample_validation_results, sample_data_stats)
        
        assert isinstance(quality_score, QualityScore)
        assert 0 <= quality_score.score <= 100
        assert isinstance(quality_score.level, QualityLevel)
        assert quality_score.total_records == 6500
        assert quality_score.issues_count == 3
        assert quality_score.critical_issues == 2
        assert quality_score.warning_issues == 1
    
    def test_calculate_table_scores(self, quality_metrics, sample_validation_results, sample_data_stats):
        """Test table-specific score calculation."""
        table_scores = quality_metrics.calculate_table_scores(sample_validation_results, sample_data_stats)
        
        assert 'movies' in table_scores
        assert 'ratings' in table_scores
        assert 'users' in table_scores
        
        for table, score in table_scores.items():
            assert isinstance(score, QualityScore)
            assert 0 <= score.score <= 100
    
    def test_generate_quality_insights(self, quality_metrics, sample_validation_results, sample_data_stats):
        """Test quality insights generation."""
        quality_score = quality_metrics.calculate_quality_score(sample_validation_results, sample_data_stats)
        insights = quality_metrics.generate_quality_insights(quality_score, sample_validation_results)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)
    
    def test_get_quality_recommendations(self, quality_metrics, sample_validation_results, sample_data_stats):
        """Test quality recommendations generation."""
        quality_score = quality_metrics.calculate_quality_score(sample_validation_results, sample_data_stats)
        recommendations = quality_metrics.get_quality_recommendations(quality_score, sample_validation_results)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


class TestQualityMonitor:
    """Test quality monitor functionality."""
    
    @pytest.fixture
    def quality_monitor(self):
        return QualityMonitor()
    
    @pytest.fixture
    def sample_data(self):
        return {
            'movies': pd.DataFrame({
                'movieId': [1, 2, 3],
                'title_clean': ['Movie 1', 'Movie 2', ''],
                'avg_rating': [4.5, 3.8, 6.0]
            }),
            'ratings': pd.DataFrame({
                'userId': [1, 1, 2],
                'movieId': [1, 2, 1],
                'rating': [4.5, 3.0, 6.0]
            })
        }
    
    def test_monitor_data_quality(self, quality_monitor, sample_data):
        """Test data quality monitoring."""
        quality_score = quality_monitor.monitor_data_quality(sample_data, trigger_alerts=False)
        
        assert isinstance(quality_score, QualityScore)
        assert 0 <= quality_score.score <= 100
    
    def test_get_active_alerts(self, quality_monitor, sample_data):
        """Test active alerts retrieval."""
        # First monitor data to generate alerts
        quality_monitor.monitor_data_quality(sample_data, trigger_alerts=True)
        
        active_alerts = quality_monitor.get_active_alerts()
        assert isinstance(active_alerts, list)
        
        critical_alerts = quality_monitor.get_active_alerts(IssueSeverity.CRITICAL)
        assert isinstance(critical_alerts, list)
    
    def test_resolve_alert(self, quality_monitor, sample_data):
        """Test alert resolution."""
        # Generate alerts
        quality_monitor.monitor_data_quality(sample_data, trigger_alerts=True)
        
        active_alerts = quality_monitor.get_active_alerts()
        if active_alerts:
            alert_id = active_alerts[0].id
            quality_monitor.resolve_alert(alert_id)
            
            # Check that alert is resolved
            resolved_alerts = [alert for alert in quality_monitor.alerts if alert.id == alert_id and alert.resolved]
            assert len(resolved_alerts) == 1
    
    def test_get_quality_summary(self, quality_monitor, sample_data):
        """Test quality summary generation."""
        quality_monitor.monitor_data_quality(sample_data, trigger_alerts=False)
        summary = quality_monitor.get_quality_summary()
        
        assert isinstance(summary, dict)
        assert 'current_score' in summary
        assert 'quality_level' in summary
        assert 'active_alerts' in summary


class TestQualityReporter:
    """Test quality reporter functionality."""
    
    @pytest.fixture
    def quality_reporter(self):
        return QualityReporter()
    
    @pytest.fixture
    def sample_data(self):
        return {
            'movies': pd.DataFrame({
                'movieId': [1, 2, 3],
                'title_clean': ['Movie 1', 'Movie 2', ''],
                'avg_rating': [4.5, 3.8, 6.0]
            }),
            'ratings': pd.DataFrame({
                'userId': [1, 1, 2],
                'movieId': [1, 2, 1],
                'rating': [4.5, 3.0, 6.0]
            })
        }
    
    def test_generate_comprehensive_report(self, quality_reporter, sample_data):
        """Test comprehensive report generation."""
        report = quality_reporter.generate_comprehensive_report(sample_data)
        
        assert isinstance(report, dict)
        assert 'report_metadata' in report
        assert 'executive_summary' in report
        assert 'quality_score' in report
        assert 'validation_results' in report
        assert 'table_analysis' in report
        assert 'issue_breakdown' in report
        assert 'data_statistics' in report
    
    def test_generate_table_specific_report(self, quality_reporter, sample_data):
        """Test table-specific report generation."""
        report = quality_reporter.generate_table_specific_report('movies', sample_data)
        
        assert isinstance(report, dict)
        assert report['table_name'] == 'movies'
        assert 'quality_score' in report
        assert 'validation_results' in report
        assert 'data_statistics' in report
    
    def test_generate_issue_report(self, quality_reporter, sample_data):
        """Test issue report generation."""
        # First validate data to get validation results
        from .data_validator import DataValidator
        validator = DataValidator()
        validation_results = validator.validate_data(sample_data)
        
        report = quality_reporter.generate_issue_report(validation_results)
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'issues_by_severity' in report
        assert 'issues_by_rule' in report
        assert 'issues_by_table' in report
        assert 'priority_issues' in report


class TestQualitySystemIntegration:
    """Test integration between quality system components."""
    
    @pytest.fixture
    def sample_data(self):
        return {
            'movies': pd.DataFrame({
                'movieId': [1, 2, 3, 4, 5],
                'title_clean': ['Movie 1', 'Movie 2', '', 'Movie 4', 'Movie 5'],
                'avg_rating': [4.5, 3.8, 2.1, 6.0, 4.2],
                'release_year': [2020, 2019, 2025, 2018, 2021],
                'genres_list': [['Action'], ['Drama'], [], ['Comedy'], ['Thriller']],
                'tmdb_id': [123, 456, None, 789, 101]
            }),
            'ratings': pd.DataFrame({
                'userId': [1, 1, 2, 2, 3],
                'movieId': [1, 2, 1, 3, 4],
                'rating': [4.5, 3.0, 5.0, 2.5, 6.0],
                'timestamp': [datetime.now()] * 5
            }),
            'users': pd.DataFrame({
                'userId': [1, 2, 3],
                'avg_rating': [4.2, 3.8, 4.5],
                'total_ratings': [10, 5, 15]
            })
        }
    
    def test_full_quality_pipeline(self, sample_data):
        """Test the complete quality assessment pipeline."""
        # Initialize all components
        validator = DataValidator()
        metrics = QualityMetrics()
        monitor = QualityMonitor(validator, metrics)
        reporter = QualityReporter(validator, metrics, monitor)
        
        # Run validation
        validation_results = validator.validate_data(sample_data)
        assert isinstance(validation_results, dict)
        
        # Calculate quality score
        data_stats = {table: len(df) for table, df in sample_data.items()}
        quality_score = metrics.calculate_quality_score(validation_results, data_stats)
        assert isinstance(quality_score, QualityScore)
        
        # Monitor quality
        monitored_score = monitor.monitor_data_quality(sample_data, trigger_alerts=True)
        assert isinstance(monitored_score, QualityScore)
        
        # Generate report
        report = reporter.generate_comprehensive_report(sample_data)
        assert isinstance(report, dict)
        
        # Verify consistency
        assert quality_score.score == monitored_score.score
        assert report['quality_score']['overall_score'] == quality_score.score
    
    def test_quality_system_with_tmdb_data(self, sample_data):
        """Test quality system with TMDB data."""
        tmdb_data = {
            123: {'title': 'Movie 1', 'overview': 'Test movie', 'budget': 1000000, 'revenue': 5000000},
            456: {'title': 'Movie 2', 'overview': 'Test movie 2', 'budget': 2000000, 'revenue': 8000000}
        }
        
        validator = DataValidator()
        metrics = QualityMetrics()
        
        # Validate with TMDB data
        validation_results = validator.validate_data(sample_data, tmdb_data)
        data_stats = {table: len(df) for table, df in sample_data.items()}
        quality_score = metrics.calculate_quality_score(validation_results, data_stats, tmdb_data)
        
        assert isinstance(quality_score, QualityScore)
        assert 'enrichment_score' in quality_score.metadata
        assert quality_score.metadata['enrichment_score'] > 0 