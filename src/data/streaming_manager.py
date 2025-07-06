import json
import logging
from datetime import datetime
from typing import Dict, Any, Callable, Optional
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
from src.config.streaming_config import StreamingConfig

logger = logging.getLogger(__name__)

class StreamingManager:
    """Kafka streaming manager for event handling."""
    
    def __init__(self):
        self.producer = None
        self.consumers = {}
        self.admin_client = None
        self._setup_producer()
        self._setup_admin_client()
    
    def _setup_producer(self):
        """Setup Kafka producer."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=StreamingConfig.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8') if k else None
            )
            logger.info("Kafka producer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def _setup_admin_client(self):
        """Setup Kafka admin client for topic management."""
        try:
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=StreamingConfig.KAFKA_BOOTSTRAP_SERVERS
            )
            logger.info("Kafka admin client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka admin client: {e}")
            raise
    
    def create_topics(self, topics: list):
        """Create Kafka topics if they don't exist."""
        topic_list = []
        for topic_name in topics:
            topic_list.append(NewTopic(
                name=topic_name,
                num_partitions=3,
                replication_factor=1
            ))
        
        try:
            self.admin_client.create_topics(topic_list)
            logger.info(f"Created topics: {topics}")
        except TopicAlreadyExistsError:
            logger.info(f"Topics already exist: {topics}")
        except Exception as e:
            logger.error(f"Failed to create topics: {e}")
            raise
    
    def send_user_event(self, user_id: int, event_type: str, movie_id: Optional[int] = None, metadata: Optional[Dict] = None):
        """Send a user event to Kafka."""
        event = {
            "user_id": user_id,
            "event_type": event_type,
            "movie_id": movie_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.producer.send(
            StreamingConfig.USER_EVENTS_TOPIC,
            key=user_id,
            value=event
        )
        logger.debug(f"Sent user event: {event}")
    
    def send_movie_rating(self, user_id: int, movie_id: int, rating: float):
        """Send a movie rating event to Kafka."""
        event = {
            "user_id": user_id,
            "movie_id": movie_id,
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        }
        
        self.producer.send(
            StreamingConfig.MOVIE_RATINGS_TOPIC,
            key=user_id,
            value=event
        )
        logger.debug(f"Sent movie rating: {event}")
    
    def send_movie_view(self, user_id: int, movie_id: int, duration: int, completion_rate: float):
        """Send a movie view event to Kafka."""
        event = {
            "user_id": user_id,
            "movie_id": movie_id,
            "duration": duration,
            "completion_rate": completion_rate,
            "timestamp": datetime.now().isoformat()
        }
        
        self.producer.send(
            StreamingConfig.MOVIE_VIEWS_TOPIC,
            key=user_id,
            value=event
        )
        logger.debug(f"Sent movie view: {event}")
    
    def send_search_query(self, user_id: int, query: str, results_count: int, clicked_movie_id: Optional[int] = None):
        """Send a search query event to Kafka."""
        event = {
            "user_id": user_id,
            "query": query,
            "results_count": results_count,
            "clicked_movie_id": clicked_movie_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.producer.send(
            StreamingConfig.SEARCH_QUERIES_TOPIC,
            key=user_id,
            value=event
        )
        logger.debug(f"Sent search query: {event}")
    
    def create_consumer(self, topic: str, group_id: Optional[str] = None) -> KafkaConsumer:
        """Create a Kafka consumer for a specific topic."""
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=StreamingConfig.KAFKA_BOOTSTRAP_SERVERS,
            group_id=group_id or StreamingConfig.KAFKA_GROUP_ID,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            key_deserializer=lambda x: int(x.decode('utf-8')) if x else None
        )
        
        self.consumers[topic] = consumer
        logger.info(f"Created consumer for topic: {topic}")
        return consumer
    
    def consume_events(self, topic: str, callback: Callable[[Dict[str, Any]], None], timeout_ms: int = 1000):
        """Consume events from a topic and process them with a callback function."""
        consumer = self.create_consumer(topic)
        
        try:
            for message in consumer:
                try:
                    event = message.value
                    callback(event)
                    logger.debug(f"Processed event from {topic}: {event}")
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
        except KeyboardInterrupt:
            logger.info("Stopping event consumption")
        finally:
            consumer.close()
    
    def get_topic_info(self, topic: str) -> Dict[str, Any]:
        """Get information about a Kafka topic."""
        try:
            cluster_metadata = self.admin_client.describe_topics([topic])
            return cluster_metadata[0]
        except Exception as e:
            logger.error(f"Failed to get topic info for {topic}: {e}")
            return {}
    
    def list_topics(self) -> list:
        """List all Kafka topics."""
        try:
            return list(self.admin_client.list_topics())
        except Exception as e:
            logger.error(f"Failed to list topics: {e}")
            return []
    
    def close(self):
        """Close all connections."""
        if self.producer:
            self.producer.close()
        
        for consumer in self.consumers.values():
            consumer.close()
        
        if self.admin_client:
            self.admin_client.close()
        
        logger.info("Closed all streaming connections") 