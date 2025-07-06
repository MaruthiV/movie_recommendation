#!/usr/bin/env python3
"""
Script to initialize Neo4j schema and test connection.
"""
import logging
from src.data.graph_database import GraphDatabaseManager

def main():
    logging.basicConfig(level=logging.INFO)
    graph_db = GraphDatabaseManager()
    print("Testing Neo4j connection...")
    if graph_db.test_connection():
        print("Neo4j connection successful.")
    else:
        print("Neo4j connection failed.")
        return
    print("Creating constraints...")
    graph_db.create_constraints()
    print("Neo4j schema initialized.")
    graph_db.close()

if __name__ == "__main__":
    main() 