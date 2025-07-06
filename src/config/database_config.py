"""
Database configuration for the Movie Recommendation System.
Handles PostgreSQL connection settings and environment variables.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Configuration class for database connections."""
    
    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "movie_recommendation"
    postgres_user: str = "movie_user"
    postgres_password: str = "movie_password"
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 30
    
    # Connection timeout settings
    connection_timeout: int = 30
    command_timeout: int = 60
    
    # SSL settings
    ssl_mode: str = "prefer"
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create configuration from environment variables."""
        return cls(
            postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
            postgres_database=os.getenv("POSTGRES_DB", "movie_recommendation"),
            postgres_user=os.getenv("POSTGRES_USER", "movie_user"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", "movie_password"),
            postgres_pool_size=int(os.getenv("POSTGRES_POOL_SIZE", "20")),
            postgres_max_overflow=int(os.getenv("POSTGRES_MAX_OVERFLOW", "30")),
            connection_timeout=int(os.getenv("DB_CONNECTION_TIMEOUT", "30")),
            command_timeout=int(os.getenv("DB_COMMAND_TIMEOUT", "60")),
            ssl_mode=os.getenv("POSTGRES_SSL_MODE", "prefer")
        )
    
    def get_connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
            f"?sslmode={self.ssl_mode}"
        )
    
    def get_async_connection_string(self) -> str:
        """Get async SQLAlchemy connection string."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
            f"?sslmode={self.ssl_mode}"
        )


# Global configuration instance
db_config = DatabaseConfig.from_env() 