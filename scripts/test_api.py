#!/usr/bin/env python3
"""
Test script for the FastAPI recommendation backend.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_get_movies():
    """Test the movies endpoint."""
    print("\nTesting movies endpoint...")
    response = requests.get(f"{BASE_URL}/movies?limit=3")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total movies: {data['total']}")
    print(f"Returned: {len(data['movies'])}")
    return response.status_code == 200 and len(data['movies']) > 0

def test_get_movie():
    """Test the individual movie endpoint."""
    print("\nTesting individual movie endpoint...")
    response = requests.get(f"{BASE_URL}/movies/1")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Movie: {data.get('title', 'Unknown')}")
    return response.status_code == 200 and data.get('title') == "Toy Story (1995)"

def test_search():
    """Test the search endpoint."""
    print("\nTesting search endpoint...")
    response = requests.get(f"{BASE_URL}/search?q=toy&limit=2")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Search results: {len(data['movies'])}")
    for movie in data['movies']:
        print(f"  - {movie['title']}")
    return response.status_code == 200 and len(data['movies']) > 0

def test_recommendations():
    """Test the recommendation endpoint."""
    print("\nTesting recommendation endpoint...")
    
    # Test with movie ID
    payload = {"movie_id": 1, "top_k": 3}
    response = requests.post(f"{BASE_URL}/recommend", json=payload)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Recommendations: {len(data['recommendations'])}")
    for rec in data['recommendations']:
        print(f"  - {rec['title']} (score: {rec['similarity_score']:.2f})")
    
    return response.status_code == 200 and len(data['recommendations']) > 0

def test_recommendations_by_title():
    """Test recommendations using movie title."""
    print("\nTesting recommendations by title...")
    
    payload = {"movie_title": "Toy Story", "top_k": 2}
    response = requests.post(f"{BASE_URL}/recommend", json=payload)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Recommendations: {len(data['recommendations'])}")
    for rec in data['recommendations']:
        print(f"  - {rec['title']} (score: {rec['similarity_score']:.2f})")
    
    return response.status_code == 200

def main():
    """Run all tests."""
    print("FastAPI Recommendation Backend Test")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Get Movies", test_get_movies),
        ("Get Individual Movie", test_get_movie),
        ("Search Movies", test_search),
        ("Recommendations by ID", test_recommendations),
        ("Recommendations by Title", test_recommendations_by_title),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the API.")

if __name__ == "__main__":
    main() 