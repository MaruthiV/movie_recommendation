from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from src.config.vector_config import VectorConfig
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """Milvus vector database manager for embeddings."""
    
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Connect to Milvus."""
        try:
            connections.connect(
                alias="default",
                host=VectorConfig.MILVUS_HOST,
                port=VectorConfig.MILVUS_PORT
            )
            self.connection = connections.get_connection("default")
            logger.info("Connected to Milvus successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from Milvus."""
        if self.connection:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
    
    def create_movie_embeddings_collection(self, vector_dim: int = VectorConfig.DEFAULT_VECTOR_DIM):
        """Create collection for movie embeddings."""
        collection_name = VectorConfig.MOVIE_EMBEDDINGS_COLLECTION
        
        if utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="movie_id", dtype=DataType.INT64, description="TMDB movie ID"),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
        ]
        
        schema = CollectionSchema(fields, description="Movie embeddings collection")
        collection = Collection(collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": VectorConfig.METRIC_TYPE,
            "index_type": VectorConfig.INDEX_TYPE,
            "params": {"nlist": VectorConfig.NLIST}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        logger.info(f"Created collection {collection_name} with index")
        return collection
    
    def create_user_embeddings_collection(self, vector_dim: int = VectorConfig.DEFAULT_VECTOR_DIM):
        """Create collection for user embeddings."""
        collection_name = VectorConfig.USER_EMBEDDINGS_COLLECTION
        
        if utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.INT64, description="User ID"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
        ]
        
        schema = CollectionSchema(fields, description="User embeddings collection")
        collection = Collection(collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": VectorConfig.METRIC_TYPE,
            "index_type": VectorConfig.INDEX_TYPE,
            "params": {"nlist": VectorConfig.NLIST}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        logger.info(f"Created collection {collection_name} with index")
        return collection
    
    def insert_movie_embeddings(self, movie_data: List[Dict[str, Any]]):
        """Insert movie embeddings into the collection."""
        collection = Collection(VectorConfig.MOVIE_EMBEDDINGS_COLLECTION)
        
        # Prepare data
        movie_ids = [item["movie_id"] for item in movie_data]
        titles = [item["title"] for item in movie_data]
        embeddings = [item["embedding"] for item in movie_data]
        
        # Insert data
        collection.insert([movie_ids, titles, embeddings])
        collection.flush()
        
        logger.info(f"Inserted {len(movie_data)} movie embeddings")
    
    def search_similar_movies(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar movies using vector similarity."""
        collection = Collection(VectorConfig.MOVIE_EMBEDDINGS_COLLECTION)
        collection.load()
        
        search_params = {
            "metric_type": VectorConfig.METRIC_TYPE,
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["movie_id", "title"]
        )
        
        similar_movies = []
        for hits in results:
            for hit in hits:
                similar_movies.append({
                    "movie_id": hit.entity.get("movie_id"),
                    "title": hit.entity.get("title"),
                    "score": hit.score
                })
        
        collection.release()
        return similar_movies
    
    def insert_user_embeddings(self, user_data: List[Dict[str, Any]]):
        """Insert user embeddings into the collection."""
        collection = Collection(VectorConfig.USER_EMBEDDINGS_COLLECTION)
        
        # Prepare data
        user_ids = [item["user_id"] for item in user_data]
        embeddings = [item["embedding"] for item in user_data]
        
        # Insert data
        collection.insert([user_ids, embeddings])
        collection.flush()
        
        logger.info(f"Inserted {len(user_data)} user embeddings")
    
    def search_similar_users(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar users using vector similarity."""
        collection = Collection(VectorConfig.USER_EMBEDDINGS_COLLECTION)
        collection.load()
        
        search_params = {
            "metric_type": VectorConfig.METRIC_TYPE,
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["user_id"]
        )
        
        similar_users = []
        for hits in results:
            for hit in hits:
                similar_users.append({
                    "user_id": hit.entity.get("user_id"),
                    "score": hit.score
                })
        
        collection.release()
        return similar_users
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        collection = Collection(collection_name)
        stats = {
            "num_entities": collection.num_entities,
            "schema": collection.schema.to_dict()
        }
        return stats 