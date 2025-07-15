'use client';

import { Recommendation } from '@/lib/api';
import { Star, TrendingUp, Lightbulb } from 'lucide-react';

interface RecommendationCardProps {
  recommendation: Recommendation;
  onClick?: () => void;
}

export default function RecommendationCard({ recommendation, onClick }: RecommendationCardProps) {
  const genres = recommendation.genres?.split('|') || [];
  const confidenceColor = recommendation.confidence > 0.7 ? 'text-green-600' : 
                          recommendation.confidence > 0.4 ? 'text-yellow-600' : 'text-red-600';
  
  return (
    <div 
      className={`bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-all duration-200 border-l-4 border-blue-500 ${
        onClick ? 'cursor-pointer' : ''
      }`}
      onClick={onClick}
    >
      {/* Movie Poster Placeholder */}
      <div className="h-40 bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center relative">
        <div className="text-white text-center">
          <div className="text-xl font-bold mb-1">{recommendation.title.split(' ')[0]}</div>
          <div className="text-xs opacity-90">{recommendation.title.split('(')[1]?.replace(')', '')}</div>
        </div>
        
        {/* Confidence Badge */}
        <div className="absolute top-2 right-2">
          <div className={`px-2 py-1 text-xs font-medium bg-white rounded-full ${confidenceColor}`}>
            {Math.round(recommendation.confidence * 100)}% match
          </div>
        </div>
      </div>

      {/* Movie Info */}
      <div className="p-4">
        <h3 className="font-semibold text-gray-900 mb-2 line-clamp-1">
          {recommendation.title}
        </h3>
        
        {/* Genres */}
        <div className="flex flex-wrap gap-1 mb-3">
          {genres.slice(0, 2).map((genre, index) => (
            <span
              key={index}
              className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded-full"
            >
              {genre}
            </span>
          ))}
        </div>

        {/* Similarity Score */}
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-4 h-4 text-blue-500" />
          <span className="text-sm font-medium text-gray-700">
            Similarity: {(recommendation.similarity_score * 100).toFixed(0)}%
          </span>
        </div>

        {/* Explanation */}
        <div className="bg-blue-50 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <Lightbulb className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
            <p className="text-sm text-blue-800">
              {recommendation.explanation}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
} 