#!/usr/bin/env python3
"""
Complete system test for the movie recommendation system.
Tests both backend API and frontend connectivity.
"""

import requests
import json
import time
import sys

# Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"

def test_backend_health():
    """Test backend health endpoint."""
    print("🔍 Testing Backend Health...")
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend is healthy! {data['movies_loaded']} movies loaded")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend health check error: {e}")
        return False

def test_backend_movies():
    """Test backend movies endpoint."""
    print("\n🎬 Testing Backend Movies...")
    try:
        response = requests.get(f"{BACKEND_URL}/movies?limit=3", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Movies loaded successfully! {len(data['movies'])} movies returned")
            for movie in data['movies']:
                print(f"   - {movie['title']} ({movie['year']})")
            return True
        else:
            print(f"❌ Movies endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Movies endpoint error: {e}")
        return False

def test_backend_search():
    """Test backend search functionality."""
    print("\n🔍 Testing Backend Search...")
    try:
        response = requests.get(f"{BACKEND_URL}/search?q=toy&limit=2", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Search working! Found {len(data['movies'])} movies for 'toy'")
            for movie in data['movies']:
                print(f"   - {movie['title']}")
            return True
        else:
            print(f"❌ Search failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

def test_backend_recommendations():
    """Test backend recommendation functionality."""
    print("\n🎯 Testing Backend Recommendations...")
    try:
        payload = {"movie_id": 1, "top_k": 3}
        response = requests.post(f"{BACKEND_URL}/recommend", json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Recommendations working! {len(data['recommendations'])} recommendations for Toy Story")
            for rec in data['recommendations']:
                print(f"   - {rec['title']} (similarity: {rec['similarity_score']:.2f})")
            return True
        else:
            print(f"❌ Recommendations failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Recommendations error: {e}")
        return False

def test_frontend_accessibility():
    """Test if frontend is accessible."""
    print("\n🌐 Testing Frontend Accessibility...")
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible!")
            return True
        else:
            print(f"❌ Frontend accessibility failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend accessibility error: {e}")
        return False

def test_frontend_api_page():
    """Test frontend API test page."""
    print("\n🧪 Testing Frontend API Test Page...")
    try:
        response = requests.get(f"{FRONTEND_URL}/test-api", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend API test page is accessible!")
            return True
        else:
            print(f"❌ Frontend API test page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend API test page error: {e}")
        return False

def main():
    """Run all tests."""
    print("🎬 Movie Recommendation System - Complete Test")
    print("=" * 50)
    
    tests = [
        ("Backend Health", test_backend_health),
        ("Backend Movies", test_backend_movies),
        ("Backend Search", test_backend_search),
        ("Backend Recommendations", test_backend_recommendations),
        ("Frontend Accessibility", test_frontend_accessibility),
        ("Frontend API Test Page", test_frontend_api_page),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is working perfectly!")
        print("\n🌐 You can now:")
        print("   - Visit the main app: http://localhost:3000")
        print("   - Test the API: http://localhost:3000/test-api")
        print("   - View API docs: http://localhost:8000/docs")
        print("\n🎯 To get recommendations:")
        print("   1. Go to http://localhost:3000")
        print("   2. Search for a movie (e.g., 'Toy Story')")
        print("   3. Click on a movie to get recommendations")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the system.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 