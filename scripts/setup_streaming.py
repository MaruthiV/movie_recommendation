#!/usr/bin/env python3
"""
Script to initialize Kafka topics and test streaming setup.
"""
import logging
import time
from src.data.streaming_manager import StreamingManager
from src.config.streaming_config import StreamingConfig

def test_event_processing(event):
    """Test callback function for event processing."""
    print(f"Received event: {event}")

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("Setting up Kafka streaming...")
    
    # Wait for Kafka to be ready
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            streaming_manager = StreamingManager()
            print("Connected to Kafka successfully!")
            break
        except Exception as e:
            retry_count += 1
            print(f"Attempt {retry_count}/{max_retries}: Kafka not ready yet. Waiting...")
            time.sleep(2)
    
    if retry_count >= max_retries:
        print("Failed to connect to Kafka after maximum retries.")
        return
    
    try:
        # Create topics
        topics = [
            StreamingConfig.USER_EVENTS_TOPIC,
            StreamingConfig.MOVIE_RATINGS_TOPIC,
            StreamingConfig.MOVIE_VIEWS_TOPIC,
            StreamingConfig.SEARCH_QUERIES_TOPIC,
            StreamingConfig.RECOMMENDATION_REQUESTS_TOPIC
        ]
        
        print("Creating Kafka topics...")
        streaming_manager.create_topics(topics)
        
        # List topics
        print("Available topics:")
        for topic in streaming_manager.list_topics():
            print(f"  - {topic}")
        
        # Test event sending
        print("Testing event sending...")
        streaming_manager.send_user_event(
            user_id=1,
            event_type="view",
            movie_id=123,
            metadata={"platform": "web"}
        )
        
        streaming_manager.send_movie_rating(
            user_id=1,
            movie_id=123,
            rating=4.5
        )
        
        streaming_manager.send_search_query(
            user_id=1,
            query="action movies",
            results_count=20,
            clicked_movie_id=456
        )
        
        # Flush producer to ensure messages are sent
        streaming_manager.producer.flush()
        
        print("Test events sent successfully!")
        
        # Test event consumption (non-blocking)
        print("Testing event consumption (will consume for 5 seconds)...")
        import threading
        import time
        
        def consume_test_events():
            streaming_manager.consume_events(
                StreamingConfig.USER_EVENTS_TOPIC,
                test_event_processing
            )
        
        # Start consumer in a separate thread
        consumer_thread = threading.Thread(target=consume_test_events)
        consumer_thread.daemon = True
        consumer_thread.start()
        
        # Wait for a few seconds to see events
        time.sleep(5)
        
        print("Streaming setup completed successfully!")
        
    except Exception as e:
        print(f"Error during streaming setup: {e}")
    finally:
        streaming_manager.close()

if __name__ == "__main__":
    main() 