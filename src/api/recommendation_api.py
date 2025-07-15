from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from pathlib import Path

app = FastAPI(title="Movie Recommendation API")

# Load movie data
def load_movie_data():
    """Load movie data from the enriched movies file."""
    try:
        with open("data/enriched_movies.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Load movie data at startup
movies_data = load_movie_data()
print(f"Loaded {len(movies_data)} movies")

class RecommendationRequest(BaseModel):
    movie_id: Optional[int] = None
    movie_title: Optional[str] = None
    top_k: int = 5

class RecommendationResponse(BaseModel):
    recommendations: List[dict]

@app.get("/health")
def health():
    return {"status": "ok", "movies_loaded": len(movies_data)}

@app.get("/movies")
def get_movies(limit: int = 10, offset: int = 0):
    """Get a list of movies."""
    return {
        "movies": movies_data[offset:offset + limit],
        "total": len(movies_data),
        "limit": limit,
        "offset": offset
    }

@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    """Get a specific movie by ID."""
    for movie in movies_data:
        if movie.get('movieId') == movie_id:
            return movie
    return {"error": "Movie not found"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest):
    """Get movie recommendations."""
    # Find movie_id if only title is provided
    movie_id = req.movie_id
    if not movie_id and req.movie_title:
        for movie in movies_data:
            if movie.get('title', '').lower() == req.movie_title.lower():
                movie_id = movie.get('movieId')
                break
    
    if not movie_id:
        return {"recommendations": []}
    
    # Simple recommendation logic: return movies with similar genres
    target_movie = None
    for movie in movies_data:
        if movie.get('movieId') == movie_id:
            target_movie = movie
            break
    
    if not target_movie:
        return {"recommendations": []}
    
    # Get target genres
    target_genres = set(target_movie.get('genres', '').split('|') if target_movie.get('genres') else [])
    
    # Find similar movies
    recommendations = []
    for movie in movies_data:
        if movie.get('movieId') == movie_id:
            continue
            
        movie_genres = set(movie.get('genres', '').split('|') if movie.get('genres') else [])
        
        # Calculate simple similarity based on genre overlap
        if target_genres and movie_genres:
            similarity = len(target_genres.intersection(movie_genres)) / len(target_genres.union(movie_genres))
        else:
            similarity = 0.0
        
        if similarity > 0:
            recommendations.append({
                "movieId": movie.get('movieId'),
                "title": movie.get('title', 'Unknown'),
                "genres": movie.get('genres', ''),
                "similarity_score": similarity,
                "explanation": f"Similar genres: {', '.join(target_genres.intersection(movie_genres))}",
                "confidence": min(similarity * 2, 1.0)  # Scale confidence
            })
    
    # Sort by similarity and return top_k
    recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
    return {"recommendations": recommendations[:req.top_k]}

@app.get("/search")
def search_movies(q: str, limit: int = 10):
    """Search movies by title."""
    query = q.lower()
    results = []
    
    for movie in movies_data:
        title = movie.get('title', '').lower()
        if query in title:
            results.append({
                "movieId": movie.get('movieId'),
                "title": movie.get('title'),
                "genres": movie.get('genres'),
                "year": movie.get('year')
            })
    
    return {"movies": results[:limit], "query": q}

# To run: uvicorn src.api.recommendation_api:app --reload --port 8000 