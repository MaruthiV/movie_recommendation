import os

class StreamingConfig:
    """Configuration for Kafka and Flink streaming."""
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "movie_recommendation_group")
    
    # Kafka Topics
    USER_EVENTS_TOPIC = "user_events"
    MOVIE_RATINGS_TOPIC = "movie_ratings"
    MOVIE_VIEWS_TOPIC = "movie_views"
    SEARCH_QUERIES_TOPIC = "search_queries"
    RECOMMENDATION_REQUESTS_TOPIC = "recommendation_requests"
    
    # Flink Configuration
    FLINK_JOBMANAGER_HOST = os.getenv("FLINK_JOBMANAGER_HOST", "localhost")
    FLINK_JOBMANAGER_PORT = int(os.getenv("FLINK_JOBMANAGER_PORT", "8081"))
    
    # Event Schema
    EVENT_SCHEMAS = {
        "user_event": {
            "user_id": "int",
            "event_type": "string",  # "view", "rating", "search", "like"
            "movie_id": "int",
            "timestamp": "datetime",
            "metadata": "json"
        },
        "movie_rating": {
            "user_id": "int",
            "movie_id": "int",
            "rating": "float",
            "timestamp": "datetime"
        },
        "movie_view": {
            "user_id": "int",
            "movie_id": "int",
            "duration": "int",  # seconds watched
            "completion_rate": "float",
            "timestamp": "datetime"
        },
        "search_query": {
            "user_id": "int",
            "query": "string",
            "results_count": "int",
            "clicked_movie_id": "int",
            "timestamp": "datetime"
        }
    } 