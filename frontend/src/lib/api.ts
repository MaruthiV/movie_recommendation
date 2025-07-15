import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Movie {
  movieId: number;
  title: string;
  genres: string;
  year: number;
  overview?: string;
  cast?: string[];
  director?: string;
  vote_average?: number;
  runtime?: number;
  tagline?: string;
  release_date?: string;
  production_companies?: string[];
  omdb_awards?: string;
  omdb_imdb_rating?: string;
  omdb_box_office?: string;
}

export interface Recommendation {
  movieId: number;
  title: string;
  genres: string;
  similarity_score: number;
  explanation: string;
  confidence: number;
}

export interface RecommendationRequest {
  movie_id?: number;
  movie_title?: string;
  top_k: number;
}

export interface RecommendationResponse {
  recommendations: Recommendation[];
}

export interface SearchResponse {
  movies: Movie[];
  query: string;
}

export interface MoviesResponse {
  movies: Movie[];
  total: number;
  limit: number;
  offset: number;
}

export interface HealthResponse {
  status: string;
  movies_loaded: number;
}

// API functions
export const apiService = {
  // Health check
  async getHealth(): Promise<HealthResponse> {
    const response = await api.get('/health');
    return response.data;
  },

  // Get movies with pagination
  async getMovies(limit: number = 10, offset: number = 0): Promise<MoviesResponse> {
    const response = await api.get(`/movies?limit=${limit}&offset=${offset}`);
    return response.data;
  },

  // Get individual movie
  async getMovie(movieId: number): Promise<Movie> {
    const response = await api.get(`/movies/${movieId}`);
    return response.data;
  },

  // Search movies
  async searchMovies(query: string, limit: number = 10): Promise<SearchResponse> {
    const response = await api.get(`/search?q=${encodeURIComponent(query)}&limit=${limit}`);
    return response.data;
  },

  // Get recommendations
  async getRecommendations(request: RecommendationRequest): Promise<RecommendationResponse> {
    const response = await api.post('/recommend', request);
    return response.data;
  },
};

export default apiService; 