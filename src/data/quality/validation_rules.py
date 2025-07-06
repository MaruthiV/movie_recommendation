"""
Validation Rules Module

Defines business rules and constraints for data validation in the movie recommendation system.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class IssueSeverity(Enum):
    """Severity levels for data quality issues."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationRule:
    """Represents a validation rule with its configuration."""
    name: str
    description: str
    severity: IssueSeverity
    validator: Callable
    table: str
    field: Optional[str] = None
    weight: float = 1.0
    enabled: bool = True


class ValidationRules:
    """Collection of validation rules for movie recommendation data."""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize all validation rules."""
        
        # Movie validation rules
        self.rules.extend([
            ValidationRule(
                name="movie_id_unique",
                description="Movie IDs must be unique",
                severity=IssueSeverity.CRITICAL,
                validator=self._validate_movie_id_unique,
                table="movies",
                weight=2.0
            ),
            ValidationRule(
                name="movie_title_not_empty",
                description="Movie titles must not be empty",
                severity=IssueSeverity.CRITICAL,
                validator=self._validate_movie_title_not_empty,
                table="movies",
                field="title_clean",
                weight=2.0
            ),
            ValidationRule(
                name="movie_rating_range",
                description="Movie average ratings must be between 0 and 5",
                severity=IssueSeverity.WARNING,
                validator=self._validate_movie_rating_range,
                table="movies",
                field="avg_rating",
                weight=1.5
            ),
            ValidationRule(
                name="movie_release_year_valid",
                description="Movie release years must be reasonable (1888-2024)",
                severity=IssueSeverity.WARNING,
                validator=self._validate_movie_release_year,
                table="movies",
                field="release_year",
                weight=1.0
            ),
            ValidationRule(
                name="movie_genres_not_empty",
                description="Movies must have at least one genre",
                severity=IssueSeverity.WARNING,
                validator=self._validate_movie_genres_not_empty,
                table="movies",
                field="genres_list",
                weight=1.0
            ),
            ValidationRule(
                name="movie_tmdb_id_unique",
                description="TMDB IDs must be unique when present",
                severity=IssueSeverity.WARNING,
                validator=self._validate_tmdb_id_unique,
                table="movies",
                field="tmdb_id",
                weight=1.0
            )
        ])
        
        # Rating validation rules
        self.rules.extend([
            ValidationRule(
                name="rating_range",
                description="Ratings must be between 0.5 and 5.0",
                severity=IssueSeverity.CRITICAL,
                validator=self._validate_rating_range,
                table="ratings",
                field="rating",
                weight=2.0
            ),
            ValidationRule(
                name="rating_timestamp_valid",
                description="Rating timestamps must be reasonable",
                severity=IssueSeverity.WARNING,
                validator=self._validate_rating_timestamp,
                table="ratings",
                field="timestamp",
                weight=1.0
            ),
            ValidationRule(
                name="rating_user_movie_exists",
                description="Rating must reference existing user and movie",
                severity=IssueSeverity.CRITICAL,
                validator=self._validate_rating_references,
                table="ratings",
                weight=2.0
            ),
            ValidationRule(
                name="rating_no_duplicates",
                description="No duplicate ratings from same user for same movie",
                severity=IssueSeverity.WARNING,
                validator=self._validate_rating_no_duplicates,
                table="ratings",
                weight=1.5
            )
        ])
        
        # User validation rules
        self.rules.extend([
            ValidationRule(
                name="user_id_unique",
                description="User IDs must be unique",
                severity=IssueSeverity.CRITICAL,
                validator=self._validate_user_id_unique,
                table="users",
                weight=2.0
            ),
            ValidationRule(
                name="user_rating_range",
                description="User average ratings must be between 0 and 5",
                severity=IssueSeverity.WARNING,
                validator=self._validate_user_rating_range,
                table="users",
                field="avg_rating",
                weight=1.0
            ),
            ValidationRule(
                name="user_activity_positive",
                description="User total ratings must be positive",
                severity=IssueSeverity.WARNING,
                validator=self._validate_user_activity_positive,
                table="users",
                field="total_ratings",
                weight=1.0
            )
        ])
        
        # TMDB data validation rules
        self.rules.extend([
            ValidationRule(
                name="tmdb_data_completeness",
                description="TMDB data should have essential fields",
                severity=IssueSeverity.INFO,
                validator=self._validate_tmdb_completeness,
                table="tmdb",
                weight=0.5
            ),
            ValidationRule(
                name="tmdb_budget_revenue_consistency",
                description="TMDB budget and revenue should be consistent",
                severity=IssueSeverity.INFO,
                validator=self._validate_tmdb_budget_revenue,
                table="tmdb",
                weight=0.5
            )
        ])
    
    def get_rules_for_table(self, table: str) -> List[ValidationRule]:
        """Get all validation rules for a specific table."""
        return [rule for rule in self.rules if rule.table == table and rule.enabled]
    
    def get_rules_by_severity(self, severity: IssueSeverity) -> List[ValidationRule]:
        """Get all validation rules of a specific severity."""
        return [rule for rule in self.rules if rule.severity == severity and rule.enabled]
    
    # Movie validation methods
    def _validate_movie_id_unique(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that movie IDs are unique."""
        issues = []
        duplicates = df[df.duplicated(subset=['movieId'], keep=False)]
        if not duplicates.empty:
            for _, row in duplicates.iterrows():
                issues.append({
                    'rule': 'movie_id_unique',
                    'severity': IssueSeverity.CRITICAL,
                    'message': f"Duplicate movie ID: {row['movieId']}",
                    'row_data': row.to_dict()
                })
        return issues
    
    def _validate_movie_title_not_empty(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that movie titles are not empty."""
        issues = []
        empty_titles = df[df['title_clean'].isna() | (df['title_clean'] == '')]
        if not empty_titles.empty:
            for _, row in empty_titles.iterrows():
                issues.append({
                    'rule': 'movie_title_not_empty',
                    'severity': IssueSeverity.CRITICAL,
                    'message': f"Empty movie title for movie ID: {row['movieId']}",
                    'row_data': row.to_dict()
                })
        return issues
    
    def _validate_movie_rating_range(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that movie average ratings are within valid range."""
        issues = []
        invalid_ratings = df[
            (df['avg_rating'] < 0) | 
            (df['avg_rating'] > 5) | 
            (df['avg_rating'].isna())
        ]
        if not invalid_ratings.empty:
            for _, row in invalid_ratings.iterrows():
                issues.append({
                    'rule': 'movie_rating_range',
                    'severity': IssueSeverity.WARNING,
                    'message': f"Invalid movie rating {row['avg_rating']} for movie ID: {row['movieId']}",
                    'row_data': row.to_dict()
                })
        return issues
    
    def _validate_movie_release_year(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that movie release years are reasonable."""
        issues = []
        current_year = datetime.now().year
        invalid_years = df[
            (df['release_year'] < 1888) | 
            (df['release_year'] > current_year + 1) |
            (df['release_year'].isna())
        ]
        if not invalid_years.empty:
            for _, row in invalid_years.iterrows():
                issues.append({
                    'rule': 'movie_release_year_valid',
                    'severity': IssueSeverity.WARNING,
                    'message': f"Invalid release year {row['release_year']} for movie ID: {row['movieId']}",
                    'row_data': row.to_dict()
                })
        return issues
    
    def _validate_movie_genres_not_empty(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that movies have at least one genre."""
        issues = []
        empty_genres = df[
            (df['genres_list'].isna()) | 
            (df['genres_list'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True))
        ]
        if not empty_genres.empty:
            for _, row in empty_genres.iterrows():
                issues.append({
                    'rule': 'movie_genres_not_empty',
                    'severity': IssueSeverity.WARNING,
                    'message': f"No genres for movie ID: {row['movieId']}",
                    'row_data': row.to_dict()
                })
        return issues
    
    def _validate_tmdb_id_unique(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that TMDB IDs are unique when present."""
        issues = []
        tmdb_data = df[df['tmdb_id'].notna()]
        duplicates = tmdb_data[tmdb_data.duplicated(subset=['tmdb_id'], keep=False)]
        if not duplicates.empty:
            for _, row in duplicates.iterrows():
                issues.append({
                    'rule': 'tmdb_id_unique',
                    'severity': IssueSeverity.WARNING,
                    'message': f"Duplicate TMDB ID: {row['tmdb_id']}",
                    'row_data': row.to_dict()
                })
        return issues
    
    # Rating validation methods
    def _validate_rating_range(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that ratings are within valid range."""
        issues = []
        invalid_ratings = df[
            (df['rating'] < 0.5) | 
            (df['rating'] > 5.0) | 
            (df['rating'].isna())
        ]
        if not invalid_ratings.empty:
            for _, row in invalid_ratings.iterrows():
                issues.append({
                    'rule': 'rating_range',
                    'severity': IssueSeverity.CRITICAL,
                    'message': f"Invalid rating {row['rating']} for user {row['userId']}, movie {row['movieId']}",
                    'row_data': row.to_dict()
                })
        return issues
    
    def _validate_rating_timestamp(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that rating timestamps are reasonable."""
        issues = []
        current_time = datetime.now()
        min_time = datetime(1995, 1, 1)  # MovieLens started around 1995
        max_time = current_time + timedelta(days=1)  # Allow future dates within 1 day
        
        invalid_timestamps = df[
            (df['timestamp'] < min_time) | 
            (df['timestamp'] > max_time) |
            (df['timestamp'].isna())
        ]
        if not invalid_timestamps.empty:
            for _, row in invalid_timestamps.iterrows():
                issues.append({
                    'rule': 'rating_timestamp_valid',
                    'severity': IssueSeverity.WARNING,
                    'message': f"Invalid timestamp {row['timestamp']} for user {row['userId']}, movie {row['movieId']}",
                    'row_data': row.to_dict()
                })
        return issues
    
    def _validate_rating_references(self, df: pd.DataFrame, movies_df: pd.DataFrame, users_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that ratings reference existing users and movies."""
        issues = []
        valid_movie_ids = set(movies_df['movieId'].unique())
        valid_user_ids = set(users_df['userId'].unique())
        
        invalid_movies = df[~df['movieId'].isin(valid_movie_ids)]
        invalid_users = df[~df['userId'].isin(valid_user_ids)]
        
        for _, row in invalid_movies.iterrows():
            issues.append({
                'rule': 'rating_user_movie_exists',
                'severity': IssueSeverity.CRITICAL,
                'message': f"Rating references non-existent movie ID: {row['movieId']}",
                'row_data': row.to_dict()
            })
        
        for _, row in invalid_users.iterrows():
            issues.append({
                'rule': 'rating_user_movie_exists',
                'severity': IssueSeverity.CRITICAL,
                'message': f"Rating references non-existent user ID: {row['userId']}",
                'row_data': row.to_dict()
            })
        
        return issues
    
    def _validate_rating_no_duplicates(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that there are no duplicate ratings from the same user for the same movie."""
        issues = []
        duplicates = df[df.duplicated(subset=['userId', 'movieId'], keep=False)]
        if not duplicates.empty:
            for _, row in duplicates.iterrows():
                issues.append({
                    'rule': 'rating_no_duplicates',
                    'severity': IssueSeverity.WARNING,
                    'message': f"Duplicate rating from user {row['userId']} for movie {row['movieId']}",
                    'row_data': row.to_dict()
                })
        return issues
    
    # User validation methods
    def _validate_user_id_unique(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that user IDs are unique."""
        issues = []
        duplicates = df[df.duplicated(subset=['userId'], keep=False)]
        if not duplicates.empty:
            for _, row in duplicates.iterrows():
                issues.append({
                    'rule': 'user_id_unique',
                    'severity': IssueSeverity.CRITICAL,
                    'message': f"Duplicate user ID: {row['userId']}",
                    'row_data': row.to_dict()
                })
        return issues
    
    def _validate_user_rating_range(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that user average ratings are within valid range."""
        issues = []
        invalid_ratings = df[
            (df['avg_rating'] < 0) | 
            (df['avg_rating'] > 5) | 
            (df['avg_rating'].isna())
        ]
        if not invalid_ratings.empty:
            for _, row in invalid_ratings.iterrows():
                issues.append({
                    'rule': 'user_rating_range',
                    'severity': IssueSeverity.WARNING,
                    'message': f"Invalid user rating {row['avg_rating']} for user ID: {row['userId']}",
                    'row_data': row.to_dict()
                })
        return issues
    
    def _validate_user_activity_positive(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate that user total ratings are positive."""
        issues = []
        invalid_activity = df[
            (df['total_ratings'] < 0) | 
            (df['total_ratings'].isna())
        ]
        if not invalid_activity.empty:
            for _, row in invalid_activity.iterrows():
                issues.append({
                    'rule': 'user_activity_positive',
                    'severity': IssueSeverity.WARNING,
                    'message': f"Invalid total ratings {row['total_ratings']} for user ID: {row['userId']}",
                    'row_data': row.to_dict()
                })
        return issues
    
    # TMDB validation methods
    def _validate_tmdb_completeness(self, tmdb_data: Dict[int, Dict]) -> List[Dict[str, Any]]:
        """Validate that TMDB data has essential fields."""
        issues = []
        essential_fields = ['title', 'overview', 'release_date']
        
        for movie_id, data in tmdb_data.items():
            for field in essential_fields:
                if field not in data or not data[field]:
                    issues.append({
                        'rule': 'tmdb_data_completeness',
                        'severity': IssueSeverity.INFO,
                        'message': f"Missing {field} for TMDB movie ID: {movie_id}",
                        'row_data': {'tmdb_id': movie_id, 'field': field}
                    })
        return issues
    
    def _validate_tmdb_budget_revenue(self, tmdb_data: Dict[int, Dict]) -> List[Dict[str, Any]]:
        """Validate that TMDB budget and revenue are consistent."""
        issues = []
        
        for movie_id, data in tmdb_data.items():
            budget = data.get('budget', 0)
            revenue = data.get('revenue', 0)
            
            if budget > 0 and revenue > 0 and revenue < budget * 0.1:
                issues.append({
                    'rule': 'tmdb_budget_revenue_consistency',
                    'severity': IssueSeverity.INFO,
                    'message': f"Unusual revenue/budget ratio for TMDB movie ID: {movie_id}",
                    'row_data': {'tmdb_id': movie_id, 'budget': budget, 'revenue': revenue}
                })
        return issues 