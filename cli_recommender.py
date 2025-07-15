#!/usr/bin/env python3
"""
CLI Movie Recommendation System
- Option 1: Find movies similar to one you just watched
- Option 2: Discover something new (show popular/random movies)
- Uses LightGCN and RAG for recommendations and explanations
"""
import sys
import random
from pathlib import Path
from src.rag.enhanced_rag_system import create_enhanced_rag_system

# Load the enhanced RAG system (with LightGCN, RAG, KG, etc.)
print("Loading movie recommendation system... (this may take a few seconds)")
enhanced_rag = create_enhanced_rag_system()
movie_metadata = enhanced_rag.base_rag.movie_metadata

# Helper: Find movie by title (case-insensitive, exact match or close)
def find_movie_by_title(title):
    title = title.strip().lower()
    # Try exact match first
    for m in movie_metadata:
        if m['title'].lower() == title:
            return m
    # Try partial match
    for m in movie_metadata:
        if title in m['title'].lower():
            return m
    return None

# Helper: Show recommendations
def show_recommendations(source_movie, k=5):
    movie_id = int(source_movie['movieId'])
    recs = enhanced_rag.base_rag.get_similar_movies(movie_id, k=k)
    print(f"\nTop {k} recommendations similar to '{source_movie['title']}':\n" + "-"*60)
    for idx, rec in enumerate(recs, 1):
        rec_id = int(rec['movieId'])
        explanation = enhanced_rag.explain_recommendation(movie_id, rec_id)
        print(f"{idx}. {rec['title']} ({rec.get('year', 'N/A')})")
        print(f"   Genres: {rec.get('genres', 'N/A')}")
        print(f"   Similarity: {rec['similarity_score']:.2f}   Confidence: {min(rec['similarity_score']*2, 1.0):.2f}")
        print(f"   Why? {explanation}")
        print()

# Helper: Show popular/random movies
def show_popular_movies(n=8):
    # For now, just pick random movies (could sort by rating/popularity if available)
    print("\nHere are some movies you might like:")
    sample = random.sample(movie_metadata, min(n, len(movie_metadata)))
    for idx, m in enumerate(sample, 1):
        print(f"{idx}. {m['title']} ({m.get('year', 'N/A')}) - {m.get('genres', 'N/A')}")
    print()

# Main CLI loop
def main():
    print("\nWelcome to the Movie CLI Recommender!")
    while True:
        print("\nWhat would you like to do?")
        print("  1. Find movies similar to one I just watched")
        print("  2. I'm not sure, help me discover")
        print("  3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice == '1':
            while True:
                title = input("Enter the title of a movie you liked (or 'back' to return): ").strip()
                if title.lower() == 'back':
                    break
                movie = find_movie_by_title(title)
                if not movie:
                    print("Movie not found in the database. Please try another title.")
                    continue
                show_recommendations(movie, k=5)
                break
        elif choice == '2':
            show_popular_movies(n=8)
            print("You can enter the number of a movie above to get recommendations, or 'back' to return.")
            sub = input("Your choice: ").strip()
            if sub.isdigit():
                idx = int(sub) - 1
                sample = random.sample(movie_metadata, min(8, len(movie_metadata)))
                if 0 <= idx < len(sample):
                    show_recommendations(sample[idx], k=5)
            # else just return to main menu
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    main() 