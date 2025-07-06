import os

class VectorConfig:
    """Configuration for Milvus vector database."""
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    
    # Collection configuration
    MOVIE_EMBEDDINGS_COLLECTION = "movie_embeddings"
    USER_EMBEDDINGS_COLLECTION = "user_embeddings"
    
    # Vector dimensions (will be set based on the embedding model)
    DEFAULT_VECTOR_DIM = 512  # CLIP embedding dimension
    
    # Index configuration
    INDEX_TYPE = "IVF_FLAT"  # Inverted File with Flat index
    METRIC_TYPE = "COSINE"   # Cosine similarity
    NLIST = 1024            # Number of clusters for IVF 