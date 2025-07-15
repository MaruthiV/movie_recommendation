'use client';

import { Movie } from '@/lib/api';
import { Star, Clock, Calendar, Users } from 'lucide-react';

interface MovieCardProps {
  movie: Movie;
  onClick?: () => void;
  showDetails?: boolean;
}

export default function MovieCard({ movie, onClick, showDetails = false }: MovieCardProps) {
  const genres = movie.genres?.split('|') || [];
  
  return (
    <div 
      className={`bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-200 ${
        onClick ? 'cursor-pointer' : ''
      }`}
      onClick={onClick}
    >
      {/* Movie Poster Placeholder */}
      <div className="h-48 bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
        <div className="text-white text-center">
          <div className="text-2xl font-bold mb-2">{movie.title.split(' ')[0]}</div>
          <div className="text-sm opacity-90">{movie.year}</div>
        </div>
      </div>

      {/* Movie Info */}
      <div className="p-4">
        <h3 className="font-semibold text-gray-900 mb-2 line-clamp-2">
          {movie.title}
        </h3>
        
        {/* Genres */}
        <div className="flex flex-wrap gap-1 mb-3">
          {genres.slice(0, 3).map((genre, index) => (
            <span
              key={index}
              className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded-full"
            >
              {genre}
            </span>
          ))}
          {genres.length > 3 && (
            <span className="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded-full">
              +{genres.length - 3}
            </span>
          )}
        </div>

        {/* Movie Stats */}
        <div className="flex items-center gap-4 text-sm text-gray-600">
          {movie.vote_average && (
            <div className="flex items-center gap-1">
              <Star className="w-4 h-4 text-yellow-500 fill-current" />
              <span>{movie.vote_average.toFixed(1)}</span>
            </div>
          )}
          
          {movie.runtime && (
            <div className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              <span>{movie.runtime}m</span>
            </div>
          )}
          
          {movie.release_date && (
            <div className="flex items-center gap-1">
              <Calendar className="w-4 h-4" />
              <span>{new Date(movie.release_date).getFullYear()}</span>
            </div>
          )}
        </div>

        {/* Additional Details */}
        {showDetails && movie.overview && (
          <div className="mt-3">
            <p className="text-sm text-gray-600 line-clamp-3">
              {movie.overview}
            </p>
          </div>
        )}

        {showDetails && movie.director && (
          <div className="mt-2 text-sm text-gray-600">
            <span className="font-medium">Director:</span> {movie.director}
          </div>
        )}

        {showDetails && movie.cast && movie.cast.length > 0 && (
          <div className="mt-2 text-sm text-gray-600">
            <span className="font-medium">Cast:</span> {movie.cast.slice(0, 3).join(', ')}
            {movie.cast.length > 3 && ` +${movie.cast.length - 3} more`}
          </div>
        )}
      </div>
    </div>
  );
} 