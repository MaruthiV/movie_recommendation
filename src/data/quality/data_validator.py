"""
Data Validator Module

Core validation engine that applies validation rules to data and generates quality reports.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .validation_rules import ValidationRules, IssueSeverity, ValidationRule

logger = logging.getLogger(__name__)


class DataValidator:
    """Core data validation engine for movie recommendation data."""
    
    def __init__(self, validation_rules: Optional[ValidationRules] = None):
        """
        Initialize the data validator.
        
        Args:
            validation_rules: Custom validation rules. If None, uses default rules.
        """
        self.validation_rules = validation_rules or ValidationRules()
        self.validation_results: Dict[str, List[Dict[str, Any]]] = {}
        self.validation_stats: Dict[str, Any] = {}
    
    def validate_data(self, data: Dict[str, pd.DataFrame], tmdb_data: Optional[Dict[int, Dict]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Validate all data tables using configured validation rules.
        
        Args:
            data: Dictionary containing DataFrames for movies, ratings, and users
            tmdb_data: Optional TMDB metadata dictionary
            
        Returns:
            Dictionary containing validation issues organized by table
        """
        logger.info("Starting comprehensive data validation...")
        
        self.validation_results = {
            'movies': [],
            'ratings': [],
            'users': [],
            'tmdb': []
        }
        
        # Validate each table
        if 'movies' in data:
            self._validate_movies(data['movies'])
        
        if 'ratings' in data:
            self._validate_ratings(data['ratings'], data.get('movies'), data.get('users'))
        
        if 'users' in data:
            self._validate_users(data['users'])
        
        if tmdb_data:
            self._validate_tmdb_data(tmdb_data)
        
        # Calculate validation statistics
        self._calculate_validation_stats()
        
        logger.info(f"Data validation completed. Found {sum(len(issues) for issues in self.validation_results.values())} issues.")
        
        return self.validation_results
    
    def _validate_movies(self, movies_df: pd.DataFrame):
        """Validate movie data."""
        logger.info(f"Validating {len(movies_df)} movies...")
        
        rules = self.validation_rules.get_rules_for_table('movies')
        
        for rule in rules:
            try:
                if rule.name == 'rating_user_movie_exists':
                    # Skip this rule for movies validation
                    continue
                
                issues = rule.validator(movies_df)
                self.validation_results['movies'].extend(issues)
                
                if issues:
                    logger.debug(f"Rule '{rule.name}' found {len(issues)} issues")
                    
            except Exception as e:
                logger.error(f"Error applying rule '{rule.name}': {e}")
                self.validation_results['movies'].append({
                    'rule': rule.name,
                    'severity': IssueSeverity.CRITICAL,
                    'message': f"Validation rule error: {str(e)}",
                    'row_data': {}
                })
    
    def _validate_ratings(self, ratings_df: pd.DataFrame, movies_df: Optional[pd.DataFrame] = None, users_df: Optional[pd.DataFrame] = None):
        """Validate rating data."""
        logger.info(f"Validating {len(ratings_df)} ratings...")
        
        rules = self.validation_rules.get_rules_for_table('ratings')
        
        for rule in rules:
            try:
                if rule.name == 'rating_user_movie_exists':
                    if movies_df is not None and users_df is not None:
                        issues = rule.validator(ratings_df, movies_df, users_df)
                    else:
                        logger.warning("Skipping rating reference validation - missing movies or users data")
                        continue
                else:
                    issues = rule.validator(ratings_df)
                
                self.validation_results['ratings'].extend(issues)
                
                if issues:
                    logger.debug(f"Rule '{rule.name}' found {len(issues)} issues")
                    
            except Exception as e:
                logger.error(f"Error applying rule '{rule.name}': {e}")
                self.validation_results['ratings'].append({
                    'rule': rule.name,
                    'severity': IssueSeverity.CRITICAL,
                    'message': f"Validation rule error: {str(e)}",
                    'row_data': {}
                })
    
    def _validate_users(self, users_df: pd.DataFrame):
        """Validate user data."""
        logger.info(f"Validating {len(users_df)} users...")
        
        rules = self.validation_rules.get_rules_for_table('users')
        
        for rule in rules:
            try:
                issues = rule.validator(users_df)
                self.validation_results['users'].extend(issues)
                
                if issues:
                    logger.debug(f"Rule '{rule.name}' found {len(issues)} issues")
                    
            except Exception as e:
                logger.error(f"Error applying rule '{rule.name}': {e}")
                self.validation_results['users'].append({
                    'rule': rule.name,
                    'severity': IssueSeverity.CRITICAL,
                    'message': f"Validation rule error: {str(e)}",
                    'row_data': {}
                })
    
    def _validate_tmdb_data(self, tmdb_data: Dict[int, Dict]):
        """Validate TMDB metadata."""
        logger.info(f"Validating {len(tmdb_data)} TMDB records...")
        
        rules = self.validation_rules.get_rules_for_table('tmdb')
        
        for rule in rules:
            try:
                issues = rule.validator(tmdb_data)
                self.validation_results['tmdb'].extend(issues)
                
                if issues:
                    logger.debug(f"Rule '{rule.name}' found {len(issues)} issues")
                    
            except Exception as e:
                logger.error(f"Error applying rule '{rule.name}': {e}")
                self.validation_results['tmdb'].append({
                    'rule': rule.name,
                    'severity': IssueSeverity.CRITICAL,
                    'message': f"Validation rule error: {str(e)}",
                    'row_data': {}
                })
    
    def _calculate_validation_stats(self):
        """Calculate validation statistics."""
        self.validation_stats = {
            'total_issues': sum(len(issues) for issues in self.validation_results.values()),
            'issues_by_table': {table: len(issues) for table, issues in self.validation_results.items()},
            'issues_by_severity': {
                'critical': 0,
                'warning': 0,
                'info': 0
            },
            'issues_by_rule': {},
            'validation_timestamp': datetime.now()
        }
        
        # Count issues by severity and rule
        for table, issues in self.validation_results.items():
            for issue in issues:
                severity = issue.get('severity', IssueSeverity.INFO)
                if isinstance(severity, IssueSeverity):
                    severity = severity.value
                
                self.validation_stats['issues_by_severity'][severity] += 1
                
                rule = issue.get('rule', 'unknown')
                self.validation_stats['issues_by_rule'][rule] = self.validation_stats['issues_by_rule'].get(rule, 0) + 1
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            'total_issues': self.validation_stats.get('total_issues', 0),
            'critical_issues': self.validation_stats.get('issues_by_severity', {}).get('critical', 0),
            'warning_issues': self.validation_stats.get('issues_by_severity', {}).get('warning', 0),
            'info_issues': self.validation_stats.get('issues_by_severity', {}).get('info', 0),
            'issues_by_table': self.validation_stats.get('issues_by_table', {}),
            'validation_timestamp': self.validation_stats.get('validation_timestamp')
        }
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical validation issues."""
        return self.validation_stats.get('issues_by_severity', {}).get('critical', 0) > 0
    
    def get_issues_by_severity(self, severity: IssueSeverity) -> List[Dict[str, Any]]:
        """Get all issues of a specific severity level."""
        all_issues = []
        for table_issues in self.validation_results.values():
            for issue in table_issues:
                if issue.get('severity') == severity:
                    all_issues.append(issue)
        return all_issues
    
    def get_issues_by_rule(self, rule_name: str) -> List[Dict[str, Any]]:
        """Get all issues for a specific validation rule."""
        all_issues = []
        for table_issues in self.validation_results.values():
            for issue in table_issues:
                if issue.get('rule') == rule_name:
                    all_issues.append(issue)
        return all_issues
    
    def add_custom_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self.validation_rules.rules.append(rule)
        logger.info(f"Added custom validation rule: {rule.name}")
    
    def disable_rule(self, rule_name: str):
        """Disable a validation rule by name."""
        for rule in self.validation_rules.rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"Disabled validation rule: {rule_name}")
                break
    
    def enable_rule(self, rule_name: str):
        """Enable a validation rule by name."""
        for rule in self.validation_rules.rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"Enabled validation rule: {rule_name}")
                break
    
    def validate_schema(self, data: Dict[str, pd.DataFrame], expected_schema: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Validate data schema against expected column structure.
        
        Args:
            data: Dictionary containing DataFrames
            expected_schema: Dictionary mapping table names to expected column lists
            
        Returns:
            Dictionary containing schema validation issues
        """
        schema_issues = {}
        
        for table_name, expected_columns in expected_schema.items():
            if table_name not in data:
                schema_issues[table_name] = [{
                    'rule': 'schema_missing_table',
                    'severity': IssueSeverity.CRITICAL,
                    'message': f"Missing table: {table_name}",
                    'row_data': {}
                }]
                continue
            
            df = data[table_name]
            actual_columns = set(df.columns)
            expected_columns_set = set(expected_columns)
            
            missing_columns = expected_columns_set - actual_columns
            extra_columns = actual_columns - expected_columns_set
            
            table_issues = []
            
            if missing_columns:
                table_issues.append({
                    'rule': 'schema_missing_columns',
                    'severity': IssueSeverity.CRITICAL,
                    'message': f"Missing columns: {list(missing_columns)}",
                    'row_data': {'missing_columns': list(missing_columns)}
                })
            
            if extra_columns:
                table_issues.append({
                    'rule': 'schema_extra_columns',
                    'severity': IssueSeverity.WARNING,
                    'message': f"Extra columns: {list(extra_columns)}",
                    'row_data': {'extra_columns': list(extra_columns)}
                })
            
            if table_issues:
                schema_issues[table_name] = table_issues
        
        return schema_issues
    
    def validate_data_types(self, data: Dict[str, pd.DataFrame], expected_types: Dict[str, Dict[str, str]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Validate data types against expected types.
        
        Args:
            data: Dictionary containing DataFrames
            expected_types: Dictionary mapping table names to column type mappings
            
        Returns:
            Dictionary containing data type validation issues
        """
        type_issues = {}
        
        for table_name, column_types in expected_types.items():
            if table_name not in data:
                continue
            
            df = data[table_name]
            table_issues = []
            
            for column, expected_type in column_types.items():
                if column not in df.columns:
                    continue
                
                actual_type = str(df[column].dtype)
                
                # Simple type checking (can be enhanced for more complex types)
                if expected_type == 'int' and not pd.api.types.is_integer_dtype(df[column]):
                    table_issues.append({
                        'rule': 'data_type_mismatch',
                        'severity': IssueSeverity.WARNING,
                        'message': f"Column '{column}' should be {expected_type}, got {actual_type}",
                        'row_data': {'column': column, 'expected_type': expected_type, 'actual_type': actual_type}
                    })
                elif expected_type == 'float' and not pd.api.types.is_float_dtype(df[column]):
                    table_issues.append({
                        'rule': 'data_type_mismatch',
                        'severity': IssueSeverity.WARNING,
                        'message': f"Column '{column}' should be {expected_type}, got {actual_type}",
                        'row_data': {'column': column, 'expected_type': expected_type, 'actual_type': actual_type}
                    })
                elif expected_type == 'datetime' and not pd.api.types.is_datetime64_any_dtype(df[column]):
                    table_issues.append({
                        'rule': 'data_type_mismatch',
                        'severity': IssueSeverity.WARNING,
                        'message': f"Column '{column}' should be {expected_type}, got {actual_type}",
                        'row_data': {'column': column, 'expected_type': expected_type, 'actual_type': actual_type}
                    })
            
            if table_issues:
                type_issues[table_name] = table_issues
        
        return type_issues 