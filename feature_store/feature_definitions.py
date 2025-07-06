from datetime import timedelta
from feast import (
    Entity, FeatureService, FeatureView, Field, FileSource, PushSource,
    RequestSource, ValueType, RepoConfig, FeatureStore
)
from feast.types import Float32, Int64, String
import pandas as pd

# Define entities
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="User identifier",
    join_keys=["user_id"],
)

movie = Entity(
    name="movie_id", 
    value_type=ValueType.INT64,
    description="Movie identifier",
    join_keys=["movie_id"],
)

# Define data sources
user_stats_source = FileSource(
    name="user_stats_source",
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
)

movie_stats_source = FileSource(
    name="movie_stats_source", 
    path="data/movie_stats.parquet",
    timestamp_field="event_timestamp",
)

user_movie_interactions_source = FileSource(
    name="user_movie_interactions_source",
    path="data/user_movie_interactions.parquet", 
    timestamp_field="event_timestamp",
)

# Define feature views
user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=90),
    schema=[
        Field(name="user_avg_rating", dtype=Float32),
        Field(name="user_total_ratings", dtype=Int64),
        Field(name="user_preferred_genres", dtype=String),
        Field(name="user_activity_level", dtype=String),
        Field(name="user_avg_watch_duration", dtype=Float32),
    ],
    source=user_stats_source,
    online=True,
)

movie_features = FeatureView(
    name="movie_features",
    entities=[movie],
    ttl=timedelta(days=365),
    schema=[
        Field(name="movie_avg_rating", dtype=Float32),
        Field(name="movie_total_ratings", dtype=Int64),
        Field(name="movie_popularity", dtype=Float32),
        Field(name="movie_genres", dtype=String),
        Field(name="movie_release_year", dtype=Int64),
        Field(name="movie_budget", dtype=Int64),
        Field(name="movie_revenue", dtype=Int64),
    ],
    source=movie_stats_source,
    online=True,
)

user_movie_interaction_features = FeatureView(
    name="user_movie_interaction_features",
    entities=[user, movie],
    ttl=timedelta(days=30),
    schema=[
        Field(name="user_movie_rating", dtype=Float32),
        Field(name="user_movie_watch_duration", dtype=Int64),
        Field(name="user_movie_completion_rate", dtype=Float32),
        Field(name="user_movie_liked", dtype=Int64),
        Field(name="user_movie_would_recommend", dtype=Int64),
    ],
    source=user_movie_interactions_source,
    online=True,
)

# Define feature service
recommendation_features = FeatureService(
    name="recommendation_features",
    features=[
        user_features,
        movie_features,
        user_movie_interaction_features,
    ],
) 