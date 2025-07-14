"""
Knowledge Graph Integration for Movie Recommendations.
Provides contextual information from Neo4j for enhanced explanations.
"""

import logging
from typing import List, Dict, Optional, Any
from neo4j import GraphDatabase
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MovieKnowledgeGraph:
    """Knowledge graph integration for movie relationships and contextual information."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", password: str = "test_password"):
        """Initialize connection to Neo4j."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._test_connection()
    
    def _test_connection(self):
        """Test the connection to Neo4j."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
    
    def create_movie_schema(self):
        """Create the movie knowledge graph schema."""
        with self.driver.session() as session:
            # Create constraints and indexes
            session.run("CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE")
            session.run("CREATE INDEX movie_title IF NOT EXISTS FOR (m:Movie) ON (m.title)")
            session.run("CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)")
            session.run("CREATE INDEX person_type IF NOT EXISTS FOR (p:Person) ON (p.type)")
            
            logger.info("Movie knowledge graph schema created")
    
    def load_movie_data(self, enriched_movies: List[Dict]):
        """Load enriched movie data into Neo4j."""
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create movies
            for movie in enriched_movies:
                if not movie.get('enriched'):
                    continue
                
                # Create movie node
                movie_props = {
                    'movieId': movie['movieId'],
                    'title': movie['title'],
                    'year': movie.get('year'),
                    'overview': movie.get('overview', ''),
                    'tagline': movie.get('tagline', ''),
                    'runtime': movie.get('runtime'),
                    'budget': movie.get('budget'),
                    'revenue': movie.get('revenue'),
                    'vote_average': movie.get('vote_average'),
                    'vote_count': movie.get('vote_count'),
                    'popularity': movie.get('popularity'),
                    'release_date': movie.get('release_date'),
                    'status': movie.get('status'),
                    'original_language': movie.get('original_language')
                }
                
                session.run("""
                    CREATE (m:Movie $props)
                """, props=movie_props)
                
                # Create genre relationships
                if movie.get('genres'):
                    genres = movie['genres'].split('|')
                    for genre in genres:
                        if genre.strip():
                            session.run("""
                                MERGE (g:Genre {name: $genre})
                                WITH g
                                MATCH (m:Movie {movieId: $movieId})
                                MERGE (m)-[:HAS_GENRE]->(g)
                            """, genre=genre.strip(), movieId=movie['movieId'])
                
                # Create cast relationships
                if movie.get('cast'):
                    for i, actor in enumerate(movie['cast'][:10]):  # Top 10 actors
                        session.run("""
                            MERGE (p:Person {name: $name})
                            SET p.type = 'Actor'
                            WITH p
                            MATCH (m:Movie {movieId: $movieId})
                            MERGE (m)-[:HAS_ACTOR {order: $order}]->(p)
                        """, name=actor, movieId=movie['movieId'], order=i)
                
                # Create director relationships
                if movie.get('director'):
                    directors = movie['director'].split(', ')
                    for director in directors:
                        if director.strip():
                            session.run("""
                                MERGE (p:Person {name: $name})
                                SET p.type = 'Director'
                                WITH p
                                MATCH (m:Movie {movieId: $movieId})
                                MERGE (m)-[:HAS_DIRECTOR]->(p)
                            """, name=director.strip(), movieId=movie['movieId'])
                
                # Create keyword relationships
                if movie.get('keywords'):
                    for keyword in movie['keywords'][:10]:  # Top 10 keywords
                        session.run("""
                            MERGE (k:Keyword {name: $keyword})
                            WITH k
                            MATCH (m:Movie {movieId: $movieId})
                            MERGE (m)-[:HAS_KEYWORD]->(k)
                        """, keyword=keyword, movieId=movie['movieId'])
                
                # Create production company relationships
                if movie.get('production_companies'):
                    for company in movie['production_companies'][:5]:  # Top 5 companies
                        session.run("""
                            MERGE (c:Company {name: $company})
                            WITH c
                            MATCH (m:Movie {movieId: $movieId})
                            MERGE (m)-[:PRODUCED_BY]->(c)
                        """, company=company, movieId=movie['movieId'])
            
            logger.info(f"Loaded {len([m for m in enriched_movies if m.get('enriched')])} movies into knowledge graph")
    
    def get_movie_relationships(self, movie_id: int) -> Dict[str, Any]:
        """Get all relationships for a specific movie."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Movie {movieId: $movieId})
                OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
                OPTIONAL MATCH (m)-[:HAS_ACTOR]->(a:Person)
                OPTIONAL MATCH (m)-[:HAS_DIRECTOR]->(d:Person)
                OPTIONAL MATCH (m)-[:HAS_KEYWORD]->(k:Keyword)
                OPTIONAL MATCH (m)-[:PRODUCED_BY]->(c:Company)
                RETURN m, 
                       collect(DISTINCT g.name) as genres,
                       collect(DISTINCT a.name) as actors,
                       collect(DISTINCT d.name) as directors,
                       collect(DISTINCT k.name) as keywords,
                       collect(DISTINCT c.name) as companies
            """, movieId=movie_id)
            
            record = result.single()
            if record:
                movie = dict(record['m'])
                return {
                    'movie': movie,
                    'genres': record['genres'],
                    'actors': record['actors'],
                    'directors': record['directors'],
                    'keywords': record['keywords'],
                    'companies': record['companies']
                }
            return None
    
    def find_shared_connections(self, movie_id_1: int, movie_id_2: int) -> Dict[str, Any]:
        """Find shared connections between two movies."""
        with self.driver.session() as session:
            # Find shared actors
            shared_actors = session.run("""
                MATCH (m1:Movie {movieId: $movie1})-[:HAS_ACTOR]->(a:Person)
                MATCH (m2:Movie {movieId: $movie2})-[:HAS_ACTOR]->(a)
                RETURN collect(DISTINCT a.name) as actors
            """, movie1=movie_id_1, movie2=movie_id_2).single()
            
            # Find shared directors
            shared_directors = session.run("""
                MATCH (m1:Movie {movieId: $movie1})-[:HAS_DIRECTOR]->(d:Person)
                MATCH (m2:Movie {movieId: $movie2})-[:HAS_DIRECTOR]->(d)
                RETURN collect(DISTINCT d.name) as directors
            """, movie1=movie_id_1, movie2=movie_id_2).single()
            
            # Find shared genres
            shared_genres = session.run("""
                MATCH (m1:Movie {movieId: $movie1})-[:HAS_GENRE]->(g:Genre)
                MATCH (m2:Movie {movieId: $movie2})-[:HAS_GENRE]->(g)
                RETURN collect(DISTINCT g.name) as genres
            """, movie1=movie_id_1, movie2=movie_id_2).single()
            
            # Find shared keywords
            shared_keywords = session.run("""
                MATCH (m1:Movie {movieId: $movie1})-[:HAS_KEYWORD]->(k:Keyword)
                MATCH (m2:Movie {movieId: $movie2})-[:HAS_KEYWORD]->(k)
                RETURN collect(DISTINCT k.name) as keywords
            """, movie1=movie_id_1, movie2=movie_id_2).single()
            
            # Find shared production companies
            shared_companies = session.run("""
                MATCH (m1:Movie {movieId: $movie1})-[:PRODUCED_BY]->(c:Company)
                MATCH (m2:Movie {movieId: $movie2})-[:PRODUCED_BY]->(c)
                RETURN collect(DISTINCT c.name) as companies
            """, movie1=movie_id_1, movie2=movie_id_2).single()
            
            return {
                'shared_actors': shared_actors['actors'] if shared_actors else [],
                'shared_directors': shared_directors['directors'] if shared_directors else [],
                'shared_genres': shared_genres['genres'] if shared_genres else [],
                'shared_keywords': shared_keywords['keywords'] if shared_keywords else [],
                'shared_companies': shared_companies['companies'] if shared_companies else []
            }
    
    def get_actor_filmography(self, actor_name: str, limit: int = 10) -> List[Dict]:
        """Get filmography for an actor."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Person {name: $name, type: 'Actor'})<-[:HAS_ACTOR]-(m:Movie)
                RETURN m.title as title, m.movieId as movieId, m.year as year
                ORDER BY m.year DESC
                LIMIT $limit
            """, name=actor_name, limit=limit)
            
            return [dict(record) for record in result]
    
    def get_director_filmography(self, director_name: str, limit: int = 10) -> List[Dict]:
        """Get filmography for a director."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Person {name: $name, type: 'Director'})<-[:HAS_DIRECTOR]-(m:Movie)
                RETURN m.title as title, m.movieId as movieId, m.year as year
                ORDER BY m.year DESC
                LIMIT $limit
            """, name=director_name, limit=limit)
            
            return [dict(record) for record in result]
    
    def get_sequel_relationships(self, movie_id: int) -> List[Dict]:
        """Find sequel/prequel relationships for a movie."""
        with self.driver.session() as session:
            # Look for movies with similar titles that might be sequels
            movie = session.run("MATCH (m:Movie {movieId: $movieId}) RETURN m.title as title", 
                              movieId=movie_id).single()
            
            if not movie:
                return []
            
            title = movie['title']
            base_title = title.split(' (')[0]  # Remove year
            
            result = session.run("""
                MATCH (m:Movie)
                WHERE m.title CONTAINS $base_title AND m.movieId <> $movieId
                RETURN m.title as title, m.movieId as movieId, m.year as year
                ORDER BY m.year
            """, base_title=base_title, movieId=movie_id)
            
            return [dict(record) for record in result]
    
    def get_award_connections(self, movie_id: int) -> List[Dict]:
        """Get award-related information for a movie."""
        # This would require additional award data
        # For now, return empty list
        return []
    
    def get_contextual_facts(self, movie_id_1: int, movie_id_2: int) -> List[str]:
        """Get contextual facts that connect two movies."""
        facts = []
        
        # Get shared connections
        connections = self.find_shared_connections(movie_id_1, movie_id_2)
        
        # Add facts based on shared connections
        if connections['shared_actors']:
            actors = connections['shared_actors'][:2]  # Limit to 2 actors
            facts.append(f"both feature {', '.join(actors)}")
        
        if connections['shared_directors']:
            directors = connections['shared_directors'][:2]  # Limit to 2 directors
            facts.append(f"both directed by {', '.join(directors)}")
        
        if connections['shared_genres']:
            genres = connections['shared_genres'][:3]  # Limit to 3 genres
            facts.append(f"both are {', '.join(genres)} movies")
        
        if connections['shared_keywords']:
            keywords = connections['shared_keywords'][:2]  # Limit to 2 keywords
            facts.append(f"share themes like {', '.join(keywords)}")
        
        if connections['shared_companies']:
            companies = connections['shared_companies'][:1]  # Limit to 1 company
            facts.append(f"both produced by {', '.join(companies)}")
        
        # Check for sequel relationships
        sequels = self.get_sequel_relationships(movie_id_1)
        for sequel in sequels:
            if sequel['movieId'] == movie_id_2:
                facts.append("are part of the same franchise")
                break
        
        return facts


def load_knowledge_graph_from_enriched_data():
    """Load enriched movie data into the knowledge graph."""
    # Load enriched movies
    enriched_file = Path("data/enriched_movies.json")
    if not enriched_file.exists():
        logger.error("Enriched movies file not found. Run enrichment script first.")
        return None
    
    with open(enriched_file, 'r', encoding='utf-8') as f:
        enriched_movies = json.load(f)
    
    # Initialize knowledge graph
    kg = MovieKnowledgeGraph()
    
    try:
        # Create schema
        kg.create_movie_schema()
        
        # Load data
        kg.load_movie_data(enriched_movies)
        
        logger.info("Knowledge graph loaded successfully")
        return kg
        
    except Exception as e:
        logger.error(f"Failed to load knowledge graph: {e}")
        kg.close()
        return None


if __name__ == "__main__":
    # Test the knowledge graph
    kg = load_knowledge_graph_from_enriched_data()
    if kg:
        # Test some queries
        print("Testing knowledge graph...")
        
        # Test shared connections
        connections = kg.find_shared_connections(1, 2)  # Toy Story and Jumanji
        print(f"Shared connections: {connections}")
        
        # Test contextual facts
        facts = kg.get_contextual_facts(1, 2)
        print(f"Contextual facts: {facts}")
        
        kg.close() 