# Alexandria

**Document Research Platform for Investigative Journalism**

Alexandria is a RAG/GraphRAG-powered platform that helps investigative journalists process large volumes of documents, extract and link knowledge (entities, relationships, timelines), and conduct intelligent research through AI-powered assistants.

## Features

- **Multi-format Document Ingestion** - PDFs, images, audio, video, web pages, spreadsheets
- **Intelligent Extraction** - Entity recognition, relationship discovery, timeline construction
- **Knowledge Graph** - Neo4j-powered graph for exploring connections between people, organizations, and events
- **Hybrid Search** - Semantic (vector), keyword, and graph-based search with facets
- **AI Research Assistant** - Natural language interface with tool use for document research
- **Sandboxed Analysis** - Safe execution of data analysis code with gVisor isolation
- **Multi-tenant** - Organization and project-based isolation for team collaboration

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (package manager)
- Docker & Docker Compose
- [Tilt](https://tilt.dev/) (optional, for Kubernetes development)

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-org/alexandria.git
cd alexandria

# Install dependencies with uv
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Start infrastructure services
docker compose up -d

# Run database migrations
uv run alembic upgrade head

# Start the API server
uv run uvicorn alexandria_api:app --reload

# Start workers (in separate terminals)
uv run python -m ingestion_worker
uv run python -m agent_worker
uv run python -m query_worker
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test categories
uv run pytest -m unit        # Unit tests only
uv run pytest -m integration # Integration tests (requires services)
```

## Project Structure

```
alexandria/
├── api/                    # FastAPI backend
│   ├── src/alexandria_api/ # API source code
│   ├── tests/              # API tests
│   └── config/             # API configuration
├── workers/                # Temporal workers
│   ├── ingestion/          # Document processing worker
│   ├── agent/              # AI agent worker
│   └── query/              # Search/retrieval worker
├── libs/                   # Shared libraries
│   ├── core/               # Domain models, schemas, utilities
│   └── db/                 # Database clients, repositories
├── frontend/               # HTMX + Tailwind frontend
│   ├── templates/          # Jinja2 templates
│   └── static/             # CSS, JS, images
├── infra/                  # Infrastructure configuration
│   ├── docker/             # Docker configurations
│   ├── k8s/                # Kubernetes manifests
│   ├── tilt/               # Tilt configuration
│   └── scripts/            # Infrastructure scripts
├── docs/                   # Documentation
│   └── plans/              # Archived implementation plans
├── scripts/                # Development scripts
│   └── plan/               # Planning workflow scripts
├── .plan/                  # Active implementation plan
├── pyproject.toml          # Root project configuration
├── docker-compose.yml      # Local development services
└── Tiltfile                # Kubernetes development workflow
```

## Technology Stack

### Backend
- **Python 3.12** with type hints
- **FastAPI** for REST API
- **Temporal** for workflow orchestration
- **OpenAI Agent SDK** for AI agents
- **Docling** for document parsing/OCR
- **spaCy** for NLP

### Data Layer
- **PostgreSQL** - Application metadata
- **Neo4j** - Knowledge graph
- **Qdrant** - Vector embeddings
- **MeiliSearch** - Full-text search
- **MinIO** - Object storage

### LLM
- **vLLM** serving Llama 3.1 8B (chat) and BGE-large-en-v1.5 (embeddings)

### Frontend
- **HTMX** for dynamic interactions
- **Tailwind CSS** for styling
- **vis.js** for graph visualization

### Infrastructure
- **Docker Compose** for local development
- **Kubernetes** for production
- **Tilt** for K8s development workflow
- **OpenTelemetry** for observability

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                                │
│   HTMX + Tailwind UI  │  File Upload  │  Search + Graph Viz    │
└───────────────────────┴───────────────┴─────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER (FastAPI)                         │
│   Chat/Agent Endpoints  │  Ingestion API  │  Search API         │
└───────────────────────┴───────────────────┴─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TEMPORAL ORCHESTRATION                          │
│   Ingestion Workflows  │  Agent Workflows  │  Query Workflows   │
└───────────────────────┴───────────────────┴─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      WORKER LAYER                                │
│   Ingestion Worker     │  Agent Worker     │  Query Worker      │
│   (Docling, spaCy)     │  (OpenAI SDK)     │  (Search, Graph)   │
└───────────────────────┴───────────────────┴─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
│  MinIO  │  Qdrant  │  Neo4j  │  MeiliSearch  │  PostgreSQL      │
└─────────┴──────────┴─────────┴───────────────┴──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM SERVING (vLLM)                          │
│   Llama 3.1 8B (Chat)   │   BGE-large-en-v1.5 (Embeddings)      │
└─────────────────────────┴───────────────────────────────────────┘
```

## Configuration

Environment variables (see `.env.example`):

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/alexandria

# Object Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Vector Database
QDRANT_URL=http://localhost:6333

# Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Search
MEILISEARCH_URL=http://localhost:7700
MEILISEARCH_API_KEY=masterKey

# Temporal
TEMPORAL_ADDRESS=localhost:7233

# LLM
VLLM_BASE_URL=http://localhost:8000/v1

# Multi-tenancy
SINGLE_TENANT_MODE=true  # Set to false for multi-tenant
```

## Development

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check --fix .

# Type check
uv run mypy .

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

### Database Migrations

```bash
# Create a new migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head

# Rollback one migration
uv run alembic downgrade -1
```

## Documentation

- [Implementation Plan](.plan/00_overview.md) - Current development overview
- [User Stories](.plan/01_stories.md) - Feature requirements
- [Tasks](.plan/02_tasks.md) - Development task breakdown
- [Research](docs/plans/01_initial_implementation/) - Original research documents

## Contributing

1. Check the [active plan](.plan/) for current priorities
2. Pick a task from [02_tasks.md](.plan/02_tasks.md)
3. Create a feature branch
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.
