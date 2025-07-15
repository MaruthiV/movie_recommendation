'use client';

import { useState, useEffect } from 'react';
import { Movie, Recommendation, apiService } from '@/lib/api';
import MovieSearch from '@/components/MovieSearch';
import MovieCard from '@/components/MovieCard';
import RecommendationCard from '@/components/RecommendationCard';
import { Film, TrendingUp, Sparkles, Loader2 } from 'lucide-react';

export default function Home() {
  const [selectedMovie, setSelectedMovie] = useState<Movie | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [popularMovies, setPopularMovies] = useState<Movie[]>([]);
  const [systemStatus, setSystemStatus] = useState<{ status: string; movies_loaded: number } | null>(null);

  // Load popular movies and system status on component mount
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        // Load system status
        const health = await apiService.getHealth();
        setSystemStatus(health);

        // Load popular movies
        const moviesResponse = await apiService.getMovies(8, 0);
        setPopularMovies(moviesResponse.movies);
      } catch (error) {
        console.error('Error loading initial data:', error);
      }
    };

    loadInitialData();
  }, []);

  const handleMovieSelect = async (movie: Movie) => {
    setSelectedMovie(movie);
    setIsLoading(true);
    
    try {
      const response = await apiService.getRecommendations({
        movie_id: movie.movieId,
        top_k: 6
      });
      setRecommendations(response.recommendations);
    } catch (error) {
      console.error('Error getting recommendations:', error);
      setRecommendations([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePopularMovieClick = (movie: Movie) => {
    handleMovieSelect(movie);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <Film className="w-8 h-8 text-blue-600" />
              <h1 className="text-2xl font-bold text-gray-900">MovieRec</h1>
            </div>
            
            {systemStatus && (
              <div className="text-sm text-gray-600">
                {systemStatus.movies_loaded} movies available
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Discover Your Next Favorite Movie
          </h2>
          <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            Get personalized movie recommendations based on your preferences. 
            Search for a movie you love and we'll find similar ones for you.
          </p>
          
          {/* Search Component */}
          <div className="flex justify-center mb-8">
            <MovieSearch 
              onMovieSelect={handleMovieSelect}
              placeholder="Search for a movie you love..."
            />
          </div>
        </div>

        {/* Selected Movie and Recommendations */}
        {selectedMovie && (
          <div className="mb-12">
            <div className="bg-white rounded-lg shadow-md p-6 mb-8">
              <h3 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                <Sparkles className="w-6 h-6 text-blue-600" />
                Based on your selection
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <MovieCard movie={selectedMovie} showDetails={true} />
                </div>
                
                <div className="flex flex-col justify-center">
                  <h4 className="text-lg font-semibold text-gray-900 mb-4">
                    Why you might like similar movies:
                  </h4>
                  <div className="space-y-3">
                    <div className="flex items-start gap-3">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                      <p className="text-gray-700">
                        Genre similarity and thematic elements
                      </p>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                      <p className="text-gray-700">
                        Shared cast, director, or production team
                      </p>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                      <p className="text-gray-700">
                        Similar audience ratings and critical reception
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                <TrendingUp className="w-6 h-6 text-green-600" />
                Recommended for you
              </h3>
              
              {isLoading ? (
                <div className="flex justify-center items-center py-12">
                  <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
                  <span className="ml-3 text-gray-600">Finding recommendations...</span>
                </div>
              ) : recommendations.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {recommendations.map((recommendation) => (
                    <RecommendationCard
                      key={recommendation.movieId}
                      recommendation={recommendation}
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <p className="text-gray-600">No recommendations found. Try a different movie.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Popular Movies Section */}
        {!selectedMovie && (
          <div>
            <h3 className="text-2xl font-bold text-gray-900 mb-6">
              Popular Movies
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {popularMovies.map((movie) => (
                <MovieCard
                  key={movie.movieId}
                  movie={movie}
                  onClick={() => handlePopularMovieClick(movie)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Call to Action */}
        {!selectedMovie && (
          <div className="mt-16 text-center">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-8 text-white">
              <h3 className="text-2xl font-bold mb-4">
                Ready to discover new movies?
              </h3>
              <p className="text-blue-100 mb-6">
                Search for a movie you love and get personalized recommendations instantly.
              </p>
              <div className="flex justify-center">
                <MovieSearch 
                  onMovieSelect={handleMovieSelect}
                  placeholder="Start by searching for a movie..."
                />
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-600">
            <p>&copy; 2024 MovieRec. Powered by AI-driven recommendations.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
