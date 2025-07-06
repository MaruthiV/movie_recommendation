-- PostgreSQL initialization script for Movie Recommendation System
-- This script runs when the PostgreSQL container starts for the first time

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search
CREATE EXTENSION IF NOT EXISTS "btree_gin";  -- For GIN indexes on arrays

-- Set timezone
SET timezone = 'UTC';

-- Create database if it doesn't exist (handled by Docker environment)
-- The database 'movie_recommendation' is created by POSTGRES_DB environment variable

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE movie_recommendation TO movie_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO movie_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO movie_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO movie_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO movie_user;

-- Performance tuning for recommendation system
-- Adjust shared_buffers (25% of available RAM, but not more than 1GB)
ALTER SYSTEM SET shared_buffers = '256MB';

-- Adjust effective_cache_size (75% of available RAM)
ALTER SYSTEM SET effective_cache_size = '1GB';

-- Adjust work_mem for complex queries
ALTER SYSTEM SET work_mem = '4MB';

-- Adjust maintenance_work_mem for index creation
ALTER SYSTEM SET maintenance_work_mem = '64MB';

-- Enable parallel query execution
ALTER SYSTEM SET max_parallel_workers_per_gather = 2;
ALTER SYSTEM SET max_parallel_workers = 4;

-- Optimize for read-heavy workload (recommendation queries)
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Logging configuration
ALTER SYSTEM SET log_statement = 'none';
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries taking more than 1 second
ALTER SYSTEM SET log_checkpoints = on;
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;

-- Reload configuration
SELECT pg_reload_conf();

-- Create indexes for better performance (these will be created by SQLAlchemy, but we can add additional ones)
-- Note: The actual table creation and indexing will be handled by SQLAlchemy models

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a function to calculate movie similarity (will be used later)
CREATE OR REPLACE FUNCTION calculate_movie_similarity(movie1_id INTEGER, movie2_id INTEGER)
RETURNS FLOAT AS $$
DECLARE
    similarity FLOAT;
BEGIN
    -- This is a placeholder for similarity calculation
    -- Will be implemented with actual similarity algorithms later
    similarity := 0.0;
    RETURN similarity;
END;
$$ LANGUAGE plpgsql;

-- Create a function to get movie recommendations for a user (placeholder)
CREATE OR REPLACE FUNCTION get_movie_recommendations(user_id INTEGER, limit_count INTEGER DEFAULT 10)
RETURNS TABLE(movie_id INTEGER, title TEXT, similarity FLOAT) AS $$
BEGIN
    -- This is a placeholder for recommendation logic
    -- Will be implemented with actual ML model integration later
    RETURN QUERY
    SELECT m.id, m.title, 0.0::FLOAT as similarity
    FROM movies m
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Create a function to log user interactions
CREATE OR REPLACE FUNCTION log_user_interaction(
    p_user_id INTEGER,
    p_movie_id INTEGER,
    p_interaction_type TEXT,
    p_metadata JSONB DEFAULT '{}'::JSONB
)
RETURNS VOID AS $$
BEGIN
    -- This function will be used to log various user interactions
    -- Implementation will be added as needed
    INSERT INTO user_interaction_log (user_id, movie_id, interaction_type, metadata, created_at)
    VALUES (p_user_id, p_movie_id, p_interaction_type, p_metadata, CURRENT_TIMESTAMP);
END;
$$ LANGUAGE plpgsql;

-- Create a table for user interaction logging (if not created by SQLAlchemy)
CREATE TABLE IF NOT EXISTS user_interaction_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    interaction_type TEXT NOT NULL,  -- 'view', 'rate', 'search', 'click', etc.
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for the interaction log
CREATE INDEX IF NOT EXISTS idx_user_interaction_log_user_id ON user_interaction_log(user_id);
CREATE INDEX IF NOT EXISTS idx_user_interaction_log_movie_id ON user_interaction_log(movie_id);
CREATE INDEX IF NOT EXISTS idx_user_interaction_log_type ON user_interaction_log(interaction_type);
CREATE INDEX IF NOT EXISTS idx_user_interaction_log_created_at ON user_interaction_log(created_at);

-- Grant permissions on the new table
GRANT ALL PRIVILEGES ON TABLE user_interaction_log TO movie_user;
GRANT USAGE, SELECT ON SEQUENCE user_interaction_log_id_seq TO movie_user;

-- Create a view for popular movies (will be used for cold-start recommendations)
CREATE OR REPLACE VIEW popular_movies AS
SELECT 
    m.id,
    m.title,
    m.vote_average,
    m.vote_count,
    m.popularity,
    m.genres,
    m.release_date,
    ROW_NUMBER() OVER (ORDER BY m.popularity DESC, m.vote_average DESC) as popularity_rank
FROM movies m
WHERE m.vote_count >= 100  -- Only movies with sufficient votes
ORDER BY m.popularity DESC, m.vote_average DESC;

-- Create a view for recent movies
CREATE OR REPLACE VIEW recent_movies AS
SELECT 
    m.id,
    m.title,
    m.vote_average,
    m.popularity,
    m.genres,
    m.release_date,
    EXTRACT(YEAR FROM CURRENT_DATE) - EXTRACT(YEAR FROM m.release_date) as years_old
FROM movies m
WHERE m.release_date >= CURRENT_DATE - INTERVAL '2 years'
ORDER BY m.release_date DESC;

-- Grant permissions on views
GRANT SELECT ON popular_movies TO movie_user;
GRANT SELECT ON recent_movies TO movie_user;

-- Insert some sample data for testing (optional)
-- This can be removed in production as real data will be loaded via the application

-- Sample genres for testing
INSERT INTO movies (title, original_title, release_date, runtime, overview, genres, vote_average, vote_count, popularity, tmdb_id, imdb_id) VALUES
('The Shawshank Redemption', 'The Shawshank Redemption', '1994-09-22', 142, 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.', ARRAY['Drama'], 9.3, 2500000, 100.0, 278, 'tt0111161'),
('The Godfather', 'The Godfather', '1972-03-14', 175, 'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.', ARRAY['Crime', 'Drama'], 9.2, 1800000, 95.0, 238, 'tt0068646'),
('Pulp Fiction', 'Pulp Fiction', '1994-10-14', 154, 'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.', ARRAY['Crime', 'Drama'], 8.9, 2000000, 90.0, 680, 'tt0110912')
ON CONFLICT (tmdb_id) DO NOTHING;

-- Create a comment for documentation
COMMENT ON DATABASE movie_recommendation IS 'Database for Movie Recommendation System - stores movies, users, ratings, and interaction data'; 