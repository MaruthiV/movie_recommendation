from neo4j import GraphDatabase, basic_auth
from src.config.graph_config import GraphConfig
import logging

logger = logging.getLogger(__name__)

class GraphDatabaseManager:
    """Neo4j connection and schema management."""
    def __init__(self):
        self.driver = GraphDatabase.driver(
            GraphConfig.NEO4J_URI,
            auth=basic_auth(GraphConfig.NEO4J_USER, GraphConfig.NEO4J_PASSWORD)
        )

    def close(self):
        if self.driver:
            self.driver.close()

    def test_connection(self):
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS result")
                return result.single()["result"] == 1
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False

    def create_constraints(self):
        """Create uniqueness constraints for Movie, Person, Genre nodes."""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.tmdb_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.tmdb_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")
            logger.info("Neo4j constraints created.") 