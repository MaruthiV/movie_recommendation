'use client';

import { useState, useEffect } from 'react';
import { apiService } from '@/lib/api';

export default function TestApi() {
  const [health, setHealth] = useState<any>(null);
  const [movies, setMovies] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const testHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.getHealth();
      setHealth(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const testMovies = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.getMovies(3, 0);
      setMovies(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">API Test Page</h1>
        
        <div className="space-y-6">
          {/* Health Test */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Health Check</h2>
            <button
              onClick={testHealth}
              disabled={loading}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
            >
              {loading ? 'Testing...' : 'Test Health'}
            </button>
            
            {health && (
              <div className="mt-4 p-4 bg-green-50 rounded">
                <h3 className="font-semibold text-green-800">Health Response:</h3>
                <pre className="text-sm text-green-700 mt-2">
                  {JSON.stringify(health, null, 2)}
                </pre>
              </div>
            )}
          </div>

          {/* Movies Test */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Movies Test</h2>
            <button
              onClick={testMovies}
              disabled={loading}
              className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:opacity-50"
            >
              {loading ? 'Loading...' : 'Load Movies'}
            </button>
            
            {movies && (
              <div className="mt-4 p-4 bg-green-50 rounded">
                <h3 className="font-semibold text-green-800">Movies Response:</h3>
                <pre className="text-sm text-green-700 mt-2">
                  {JSON.stringify(movies, null, 2)}
                </pre>
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 p-4 rounded">
              <h3 className="font-semibold text-red-800">Error:</h3>
              <p className="text-red-700 mt-2">{error}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 