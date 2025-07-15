# Movie Recommendation System

A machine learning-powered movie recommendation system that combines collaborative filtering, content-based, and hybrid approaches, with explainable recommendations and support for cold-start scenarios. The system is designed for experimentation and research, and is operated via a command-line interface (CLI).

## Project Summary

This project implements a comprehensive movie recommendation pipeline using the MovieLens-100K dataset, enriched with external metadata. It supports similarity search, hybrid recommendations, knowledge graph integration, and generates explanations for recommendations. The system is intended for ML/RAG research and demo purposes, and does not require a web interface or user management.

## Tech Stack

- **Python 3.9+**
- **PyTorch** (LightGCN collaborative filtering)
- **FAISS** (vector similarity search)
- **Neo4j** (knowledge graph)
- **TMDB/OMDB APIs** (movie metadata enrichment)
- **Other**: NumPy, Pandas, scikit-learn, tqdm, etc.

## How to Use (CLI)

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Enrich Data and Build Indexes

If you want to re-run data enrichment or embedding/index setup:

```bash
python scripts/enrich_movie_data.py         # Enrich movie metadata (TMDB/OMDB)
python scripts/setup_faiss_index.py         # Build FAISS vector index
```

### 3. Run the CLI Recommender

```bash
python cli_recommender.py
```

You will be prompted to enter a movie title to get similar movies, or to discover new recommendations. The CLI will display recommended movies along with explanations and confidence scores.

## Notes

- All recommendations and explanations are generated locally using the pre-built models and data.
- No web server or user management is required.
- For advanced usage or retraining, see the scripts/ directory.
