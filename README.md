# Movie Recommendation System

A sophisticated movie recommendation system that leverages advanced machine learning techniques including graph neural networks, sequential transformers, and multi-modal embeddings to provide highly personalized movie recommendations.

## 🎯 Features

- **Personalized Recommendations**: AI-powered movie suggestions based on user preferences and viewing history
- **Natural Language Search**: Find movies using natural language queries like "feel-good movies like The Princess Bride"
- **Explainable AI**: Get explanations for why specific movies are recommended
- **Cold Start Handling**: Provide relevant recommendations for new users and new content
- **Diversity Promotion**: Avoid filter bubbles with intelligent diversity constraints
- **Real-time Performance**: Sub-100ms recommendation latency
- **Multi-modal Understanding**: Leverage text, posters, and trailers for better recommendations

## 🏗️ Architecture

The system is built with a modern, scalable architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   FastAPI API   │    │   ML Pipeline   │
│   (React/Next)  │◄──►│   (Python)      │◄──►│   (PyTorch)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Neo4j Graph   │    │   Milvus Vector │
│   (User Data)   │    │   (Knowledge)   │    │   (Embeddings)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- 8GB+ RAM (for ML models)
- GPU (optional, for faster inference)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd movie_recc
```

### 2. Start the Database

```bash
# Start PostgreSQL, Redis, and pgAdmin
docker-compose up -d

# Wait for services to be ready (check with docker-compose ps)
```

### 3. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Set Up the Database

```bash
# Initialize database tables and create sample data
python scripts/setup_database.py

# Or with sample data
CREATE_SAMPLE_DATA=true python scripts/setup_database.py
```

### 4b. Set Up Neo4j Knowledge Graph

```bash
# Start Neo4j (if not already running)
docker-compose up -d neo4j

# Initialize Neo4j schema and test connection
python scripts/setup_neo4j.py
```

### 4c. Set Up Milvus Vector Database

```bash
# Start Milvus and dependencies (if not already running)
docker-compose up -d milvus

# Wait for Milvus to be ready (may take a few minutes)
# Then initialize collections and test connection
python scripts/setup_milvus.py
```

### 4d. Set Up Kafka/Flink Streaming

```bash
# Start Kafka, Zookeeper, and Flink (if not already running)
docker-compose up -d kafka flink-jobmanager flink-taskmanager

# Wait for services to be ready (may take a few minutes)
# Then initialize Kafka topics and test streaming
python scripts/setup_streaming.py
```

### 4e. Set Up Feast Feature Store

```bash
# Start Feast services (Redis for online store, PostgreSQL for offline store and registry)
docker-compose up -d feast-online-store feast-offline-store feast-registry

# Wait for services to be ready (may take a few minutes)
# Then initialize feature store and test connection
python scripts/setup_feature_store.py
```

### 4f. Set Up Backup and Recovery System

```bash
# Set up comprehensive backup and recovery system
python scripts/setup_backup_system.py

# This creates:
# - backup_config.json (backup configuration)
# - backup directories and structure
# - Utility scripts for manual backup/recovery
# - Documentation in docs/backup_system.md

# Test backup functionality
python scripts/run_backup.py --type data_files

# Run all backups
python scripts/run_backup.py --type all

# Start automated backup scheduler (optional)
python -c "
from src.backup.backup_config import BackupConfig
from src.backup.backup_manager import BackupManager
from src.backup.backup_scheduler import BackupScheduler

config = BackupConfig.load_from_file('backup_config.json')
manager = BackupManager(config)
scheduler = BackupScheduler(config, manager)
scheduler.start()
"
```

### 5. Set Up TMDB API Key

```bash
# Get a free API key from https://www.themoviedb.org/settings/api
export TMDB_API_KEY=your_tmdb_api_key
```

### 6. Run Data Ingestion

```bash
# Run sample ingestion (100 movies) - recommended for testing
python scripts/run_ingestion.py --mode sample --sample-size 100

# Run full MovieLens-20M + TMDB ingestion (requires TMDB API key)
python scripts/run_ingestion.py --mode full

# Run MovieLens-only ingestion (no TMDB metadata)
python scripts/run_ingestion.py --mode movielens-only

# Run with custom sample size
python scripts/run_ingestion.py --mode sample --sample-size 500

# Skip specific storage systems
python scripts/run_ingestion.py --mode sample --skip-neo4j --skip-feature-store

# Set TMDB API key
export TMDB_API_KEY=your_tmdb_api_key
python scripts/run_ingestion.py --mode full
```

### 7. Run Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest src/data/test_movie_database.py

# Run ingestion pipeline tests
python -m pytest src/data/test_ingestion_pipeline.py
```

### 8. Start the Application

```bash
# Start the FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Database Schema

The system uses PostgreSQL with the following main tables:

- **movies**: Movie metadata, ratings, and content embeddings
- **persons**: Actors, directors, and crew members
- **cast_members**: Movie-actor relationships
- **crew_members**: Movie-crew relationships
- **users**: User profiles and preferences
- **user_ratings**: User movie ratings and reviews
- **user_watches**: User viewing history and behavior
- **search_history**: User search queries and interactions

## 🔧 Configuration

Environment variables can be set in a `.env` file:

```env
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=movie_recommendation
POSTGRES_USER=movie_user
POSTGRES_PASSWORD=movie_password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# ML Model Configuration
MODEL_CACHE_DIR=./models
GPU_ENABLED=false

# External APIs
TMDB_API_KEY=your_tmdb_api_key
OPENAI_API_KEY=your_openai_api_key
```

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/performance/
```

## 📈 Performance

The system is optimized for:

- **Latency**: <100ms end-to-end recommendation time
- **Throughput**: 10,000+ recommendations per second
- **Accuracy**: >70% movie completion rate
- **Diversity**: >30% discovery rate for new content

## 🔍 API Endpoints

### Core Endpoints

- `GET /api/v1/movies` - Get movie recommendations
- `POST /api/v1/movies/search` - Search movies
- `GET /api/v1/movies/{movie_id}` - Get movie details
- `POST /api/v1/movies/{movie_id}/rate` - Rate a movie
- `GET /api/v1/movies/{movie_id}/explain` - Get recommendation explanation

### User Management

- `POST /api/v1/users/register` - Register new user
- `POST /api/v1/users/login` - User login
- `GET /api/v1/users/profile` - Get user profile
- `PUT /api/v1/users/preferences` - Update user preferences

## 🛠️ Development

### Project Structure

```
movie_recc/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── data/             # Database models and operations
│   ├── models/           # ML models (LightGCN, Transformers, etc.)
│   ├── rag/              # RAG system for explanations
│   ├── ranking/          # RL-based ranking
│   ├── utils/            # Utility functions
│   └── config/           # Configuration management
├── scripts/              # Setup and utility scripts
├── tests/                # Test files
├── docker/               # Docker configurations
├── monitoring/           # Monitoring and alerting
└── docs/                 # Documentation
```

### Adding New Features

1. **Database Changes**: Update models in `src/data/movie_database.py`
2. **API Endpoints**: Add routes in `src/api/`
3. **ML Models**: Implement in `src/models/`
4. **Tests**: Add corresponding test files
5. **Documentation**: Update README and API docs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MovieLens](https://movielens.org/) for the dataset
- [TMDB](https://www.themoviedb.org/) for movie metadata
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for GNN support
- [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec) for sequential modeling

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the [documentation](docs/)
- Review the [API documentation](http://localhost:8000/docs) when running locally

---

**Note**: This is a development version. For production deployment, additional security, monitoring, and scaling considerations should be implemented. 