# Initial Implementation - Tasks

> Detailed task breakdown with hierarchical IDs

## Task Format

- **ID**: P{phase}-E{epic}-S{story}-T{task} (e.g., P0-E1-S1-T1)
- **Effort**: S (< 2h) / M (2-8h) / L (> 8h)
- **Status**: Pending / In Progress / Done / Blocked

## Progress Summary

| Phase | Total | Done | In Progress | Blocked |
|-------|-------|------|-------------|---------|
| Phase 0 | 89 | 48 | 0 | 0 |
| Phase 1 | 92 | 8 | 0 | 0 |
| Phase 2 | 88 | 0 | 0 | 0 |
| Phase 3 | 68 | 0 | 0 | 0 |
| Phase 4 | 56 | 0 | 0 | 0 |
| **Total** | **393** | 56 | 0 | 0 |

---

## Phase 0: Project Scaffolding & Infrastructure

### P0-E1-S1: Initialize Monorepo Structure

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E1-S1-T1 | Create root pyproject.toml with workspace configuration | S | Done | - |
| P0-E1-S1-T2 | Create .pre-commit-config.yaml with hooks | S | Done | - |
| P0-E1-S1-T3 | Create /api directory structure (src, tests, config) | S | Done | T1 |
| P0-E1-S1-T4 | Create /workers directory structure (ingestion, agent, query) | S | Done | T1 |
| P0-E1-S1-T5 | Create /frontend directory structure (templates, static) | S | Done | T1 |
| P0-E1-S1-T6 | Create /infra directory (docker, k8s, tilt) | S | Done | T1 |
| P0-E1-S1-T7 | Create /libs directory for shared packages | S | Done | T1 |
| P0-E1-S1-T8 | Write comprehensive README.md | M | Done | T3-T7 |
| P0-E1-S1-T9 | Configure .gitignore for Python, Node, IDE files | S | Done | - |

---

### P0-E1-S2: Configure Python Development Environment

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E1-S2-T1 | Configure pyproject.toml with uv workspace | M | Done | P0-E1-S1-T1 |
| P0-E1-S2-T2 | Set up ruff configuration (linting + formatting) | S | Done | T1 |
| P0-E1-S2-T3 | Configure mypy with strict mode | S | Done | T1 |
| P0-E1-S2-T4 | Configure pytest and coverage settings | S | Done | T1 |
| P0-E1-S2-T5 | Install pre-commit and set up hooks | S | Pending | P0-E1-S1-T2 |
| P0-E1-S2-T6 | Create VS Code workspace settings | S | Done | T2, T3 |
| P0-E1-S2-T7 | Generate initial uv.lock file | S | Done | T1 |

---

### P0-E1-S3: Create Docker Base Images

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E1-S3-T1 | Create base Python Dockerfile (python:3.12-slim) | M | Done | P0-E1-S2-T1 |
| P0-E1-S3-T2 | Create API Dockerfile with FastAPI/uvicorn | M | Done | T1 |
| P0-E1-S3-T3 | Create Worker Dockerfile with Temporal SDK | M | Done | T1 |
| P0-E1-S3-T4 | Implement multi-stage builds for production | M | Done | T2, T3 |
| P0-E1-S3-T5 | Add non-root user configuration | S | Done | T4 |
| P0-E1-S3-T6 | Add health check commands | S | Done | T4 |
| P0-E1-S3-T7 | Create .dockerignore files | S | Done | T1 |

---

### P0-E2-S0: Tenant and Project Schema

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E2-S0-T1 | Create Alembic migration for tenants table | M | Pending | P0-E1-S1 |
| P0-E2-S0-T2 | Create Alembic migration for projects table | M | Pending | T1 |
| P0-E2-S0-T3 | Create project_documents join table migration | S | Pending | T2 |
| P0-E2-S0-T4 | Create project_entities join table migration | S | Pending | T2 |
| P0-E2-S0-T5 | Add tenant_id to all existing tables | M | Pending | T1-T4 |
| P0-E2-S0-T6 | Create indexes for tenant_id columns | S | Pending | T5 |
| P0-E2-S0-T7 | Implement dev-mode default tenant/project creation | M | Pending | T1-T6 |
| P0-E2-S0-T8 | Add SINGLE_TENANT_MODE environment variable handling | S | Pending | T7 |

---

### P0-E2-S0b: Tenant Middleware and Request Context

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E2-S0b-T1 | Create TenantContext dataclass | S | Done | P0-E2-S0 |
| P0-E2-S0b-T2 | Create RequestContext dataclass | S | Done | T1 |
| P0-E2-S0b-T3 | Implement get_tenant dependency | M | Done | T1 |
| P0-E2-S0b-T4 | Implement get_request_context dependency | M | Done | T2, T3 |
| P0-E2-S0b-T5 | Add project validation (belongs to tenant) | S | Done | T4 |
| P0-E2-S0b-T6 | Implement single-tenant mode bypass | S | Done | T3 |
| P0-E2-S0b-T7 | Write unit tests for middleware | M | Pending | T1-T6 |

---

### P0-E2-S1: PostgreSQL Setup with Schema

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E2-S1-T1 | Add PostgreSQL 16 service to docker-compose | S | Done | P0-E1-S3 |
| P0-E2-S1-T2 | Configure volume mount for data persistence | S | Done | T1 |
| P0-E2-S1-T3 | Set up Alembic configuration | M | Pending | P0-E1-S2-T1 |
| P0-E2-S1-T4 | Create initial migrations (documents, ingestion_jobs) | M | Pending | T3 |
| P0-E2-S1-T5 | Configure connection pooling (asyncpg) | S | Pending | T1 |
| P0-E2-S1-T6 | Add health check endpoint | S | Done | T1 |

---

### P0-E2-S2: MinIO Object Storage Setup

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E2-S2-T1 | Add MinIO service to docker-compose | S | Done | P0-E1-S3 |
| P0-E2-S2-T2 | Configure buckets via init script | M | Done | T1 |
| P0-E2-S2-T3 | Set up IAM policies for service accounts | M | Pending | T2 |
| P0-E2-S2-T4 | Configure console access (port 9001) | S | Done | T1 |
| P0-E2-S2-T5 | Create pre-signed URL utility functions | M | Done | T3 |
| P0-E2-S2-T6 | Write MinIO client wrapper class | M | Done | T5 |

---

### P0-E2-S3: Qdrant Vector Database Setup

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E2-S3-T1 | Add Qdrant service to docker-compose | S | Done | P0-E1-S3 |
| P0-E2-S3-T2 | Configure collection initialization script | M | Pending | T1 |
| P0-E2-S3-T3 | Set up payload indexes (document_id, tenant_id) | M | Pending | T2 |
| P0-E2-S3-T4 | Configure quantization (INT8 scalar) | S | Pending | T2 |
| P0-E2-S3-T5 | Create Qdrant client wrapper class | M | Done | T1 |
| P0-E2-S3-T6 | Configure snapshot/backup settings | S | Pending | T1 |

---

### P0-E2-S4: Neo4j Graph Database Setup

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E2-S4-T1 | Add Neo4j 5.x service to docker-compose | S | Done | P0-E1-S3 |
| P0-E2-S4-T2 | Configure APOC plugin installation | M | Done | T1 |
| P0-E2-S4-T3 | Create constraints initialization script | M | Done | T2 |
| P0-E2-S4-T4 | Create indexes (Entity.name, Entity.type, Document.id) | S | Done | T3 |
| P0-E2-S4-T5 | Create Neo4j client wrapper class | M | Done | T1 |
| P0-E2-S4-T6 | Configure browser access (port 7474, 7687) | S | Done | T1 |

---

### P0-E2-S5: MeiliSearch Setup

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E2-S5-T1 | Add MeiliSearch service to docker-compose | S | Done | P0-E1-S3 |
| P0-E2-S5-T2 | Create index initialization script | M | Pending | T1 |
| P0-E2-S5-T3 | Configure searchable/filterable attributes | S | Pending | T2 |
| P0-E2-S5-T4 | Configure facets (document_type, source, date_range) | S | Pending | T2 |
| P0-E2-S5-T5 | Create MeiliSearch client wrapper class | M | Done | T1 |
| P0-E2-S5-T6 | Disable analytics for privacy | S | Done | T1 |

---

### P0-E2-S6: Temporal Server Setup

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E2-S6-T1 | Add Temporal server service to docker-compose | M | Done | P0-E2-S1 |
| P0-E2-S6-T2 | Add temporal-admin-tools service | S | Pending | T1 |
| P0-E2-S6-T3 | Add temporal-ui service | S | Done | T1 |
| P0-E2-S6-T4 | Configure PostgreSQL backend for Temporal | M | Done | T1, P0-E2-S1 |
| P0-E2-S6-T5 | Create namespace initialization script | M | Pending | T1-T3 |
| P0-E2-S6-T6 | Create Temporal client wrapper class | M | Done | T1 |

---

### P0-E3-S1: vLLM Chat Model Deployment

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E3-S1-T1 | Add vLLM chat service to docker-compose | M | Pending | P0-E1-S3 |
| P0-E3-S1-T2 | Configure NVIDIA runtime and GPU allocation | M | Pending | T1 |
| P0-E3-S1-T3 | Set up environment variables for model config | S | Pending | T1 |
| P0-E3-S1-T4 | Configure Prometheus metrics endpoint | S | Pending | T1 |
| P0-E3-S1-T5 | Add health check endpoint | S | Pending | T1 |
| P0-E3-S1-T6 | Document GPU requirements and setup | S | Pending | T2 |

---

### P0-E3-S2: vLLM Embedding Model Deployment

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E3-S2-T1 | Add vLLM embedding service to docker-compose | M | Pending | P0-E3-S1 |
| P0-E3-S2-T2 | Configure task mode for embeddings | S | Pending | T1 |
| P0-E3-S2-T3 | Set up separate port (8001) | S | Pending | T1 |
| P0-E3-S2-T4 | Optimize batch processing settings | S | Pending | T1 |
| P0-E3-S2-T5 | Add Prometheus metrics endpoint | S | Pending | T1 |

---

### P0-E3-S3: LLM Client Abstraction Layer

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E3-S3-T1 | Create LLMClient class wrapping OpenAI SDK | M | Pending | P0-E3-S1 |
| P0-E3-S3-T2 | Implement configuration via env vars | S | Pending | T1 |
| P0-E3-S3-T3 | Implement retry logic with exponential backoff | M | Pending | T1 |
| P0-E3-S3-T4 | Implement async chat_completion method | M | Pending | T1 |
| P0-E3-S3-T5 | Implement async embeddings method | M | Pending | T1, P0-E3-S2 |
| P0-E3-S3-T6 | Add structured output support (Pydantic parsing) | M | Pending | T4 |
| P0-E3-S3-T7 | Add token counting utility | S | Pending | T1 |
| P0-E3-S3-T8 | Write unit tests for LLMClient | M | Pending | T1-T7 |

---

### P0-E4-S1: OpenTelemetry Collector Setup

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E4-S1-T1 | Add OTEL Collector service to docker-compose | S | Pending | P0-E1-S3 |
| P0-E4-S1-T2 | Configure OTLP receivers (gRPC 4317, HTTP 4318) | S | Pending | T1 |
| P0-E4-S1-T3 | Configure Jaeger exporter for traces | S | Pending | T1 |
| P0-E4-S1-T4 | Configure Prometheus exporter for metrics | S | Pending | T1 |
| P0-E4-S1-T5 | Set up batch processor | S | Pending | T2-T4 |

---

### P0-E4-S2: Jaeger Tracing Setup

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E4-S2-T1 | Add Jaeger all-in-one service to docker-compose | S | Pending | P0-E4-S1 |
| P0-E4-S2-T2 | Configure UI access (port 16686) | S | Pending | T1 |
| P0-E4-S2-T3 | Enable OTLP ingestion | S | Pending | T1, P0-E4-S1 |
| P0-E4-S2-T4 | Configure retention policy | S | Pending | T1 |

---

### P0-E4-S3: Prometheus and Grafana Setup

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E4-S3-T1 | Add Prometheus service to docker-compose | S | Pending | P0-E4-S1 |
| P0-E4-S3-T2 | Configure scrape configs for all services | M | Pending | T1 |
| P0-E4-S3-T3 | Add Grafana service to docker-compose | S | Pending | T1 |
| P0-E4-S3-T4 | Create vLLM metrics dashboard | M | Pending | T3 |
| P0-E4-S3-T5 | Create Temporal metrics dashboard | M | Pending | T3 |
| P0-E4-S3-T6 | Create API latency dashboard | M | Pending | T3 |
| P0-E4-S3-T7 | Configure alert rules for critical failures | M | Pending | T2 |
| P0-E4-S3-T8 | Auto-provision Grafana datasources | S | Pending | T3 |

---

### P0-E4-S4: FastAPI OTEL Instrumentation

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E4-S4-T1 | Install opentelemetry-instrumentation-fastapi | S | Pending | P0-E4-S1 |
| P0-E4-S4-T2 | Configure automatic instrumentation | M | Pending | T1 |
| P0-E4-S4-T3 | Add custom span creation utilities | M | Pending | T2 |
| P0-E4-S4-T4 | Configure excluded paths (/health, /metrics) | S | Pending | T2 |
| P0-E4-S4-T5 | Set up trace context propagation | S | Pending | T2 |

---

### P0-E5-S1: Tiltfile for Local Development

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E5-S1-T1 | Create base Tiltfile | M | Pending | P0-E2-S6 |
| P0-E5-S1-T2 | Add docker_build for API service with live_update | M | Pending | T1 |
| P0-E5-S1-T3 | Add docker_build for worker services with live_update | M | Pending | T1 |
| P0-E5-S1-T4 | Configure resource dependencies | M | Pending | T1-T3 |
| P0-E5-S1-T5 | Set up port forwards for all services | S | Pending | T1 |
| P0-E5-S1-T6 | Configure labels for Tilt UI organization | S | Pending | T1 |

---

### P0-E5-S2: Custom Tilt Buttons/Actions

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E5-S2-T1 | Create "Clear All Data" button | M | Pending | P0-E5-S1 |
| P0-E5-S2-T2 | Create "Run Migrations" button | S | Pending | P0-E5-S1 |
| P0-E5-S2-T3 | Create "Seed Test Data" button | M | Pending | P0-E5-S1 |
| P0-E5-S2-T4 | Create "Reset Temporal" button | S | Pending | P0-E5-S1 |
| P0-E5-S2-T5 | Configure all actions as manual trigger | S | Pending | T1-T4 |

---

### P0-E5-S3: Docker Compose Fallback

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E5-S3-T1 | Create comprehensive docker-compose.yml | L | Pending | P0-E2-S6 |
| P0-E5-S3-T2 | Create docker-compose.override.yml for dev extras | M | Pending | T1 |
| P0-E5-S3-T3 | Create Makefile with common commands | M | Pending | T1 |
| P0-E5-S3-T4 | Document Tilt vs Docker Compose workflows | S | Pending | T1-T3 |

---

### P0-E6-S1: GitHub Actions for Testing

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E6-S1-T1 | Create CI workflow file (.github/workflows/ci.yml) | M | Pending | P0-E1-S2 |
| P0-E6-S1-T2 | Configure workflow triggers (PR, push to main) | S | Pending | T1 |
| P0-E6-S1-T3 | Add lint step (ruff) | S | Pending | T1 |
| P0-E6-S1-T4 | Add type check step (mypy) | S | Pending | T1 |
| P0-E6-S1-T5 | Add unit tests step (pytest) | S | Pending | T1 |
| P0-E6-S1-T6 | Configure service containers (PostgreSQL, Qdrant) | M | Pending | T5 |
| P0-E6-S1-T7 | Add test coverage reporting | S | Pending | T5 |
| P0-E6-S1-T8 | Configure parallel job execution | S | Pending | T1-T7 |

---

### P0-E6-S2: Docker Image Build Pipeline

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P0-E6-S2-T1 | Create build workflow file | M | Pending | P0-E6-S1 |
| P0-E6-S2-T2 | Configure tag push trigger (semver) | S | Pending | T1 |
| P0-E6-S2-T3 | Set up multi-platform builds (amd64, arm64) | M | Pending | T1 |
| P0-E6-S2-T4 | Configure registry push (ghcr.io) | S | Pending | T1 |
| P0-E6-S2-T5 | Add Trivy image scanning | M | Pending | T3 |
| P0-E6-S2-T6 | Configure build caching | S | Pending | T3 |

---

## Phase 1: Ingestion Pipeline

### P1-E1-S1: Document Model and Database Schema

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E1-S1-T1 | Create SQLAlchemy Document model | M | Done | P0-E2-S1 |
| P1-E1-S1-T2 | Create SQLAlchemy IngestionJob model | M | Done | T1 |
| P1-E1-S1-T3 | Create SQLAlchemy Chunk model | M | Done | T1 |
| P1-E1-S1-T4 | Create SQLAlchemy Entity model | M | Done | T1 |
| P1-E1-S1-T5 | Create Alembic migrations for all models | M | Done | T1-T4 |
| P1-E1-S1-T6 | Implement DocumentRepository | M | Done | T1 |
| P1-E1-S1-T7 | Implement IngestionJobRepository | M | Done | T2 |
| P1-E1-S1-T8 | Write unit tests for repositories | M | Pending | T6, T7 |

---

### P1-E1-S1b: Project Management API

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E1-S1b-T1 | Create Project Pydantic schemas (request/response) | S | Pending | P0-E2-S0b |
| P1-E1-S1b-T2 | Implement ProjectRepository | M | Pending | P0-E2-S0 |
| P1-E1-S1b-T3 | Create POST /api/v1/projects endpoint | M | Pending | T1, T2 |
| P1-E1-S1b-T4 | Create GET /api/v1/projects endpoint | M | Pending | T2 |
| P1-E1-S1b-T5 | Create GET /api/v1/projects/{id} endpoint | M | Pending | T2 |
| P1-E1-S1b-T6 | Create PATCH /api/v1/projects/{id} endpoint | M | Pending | T2 |
| P1-E1-S1b-T7 | Create archive/unarchive endpoints | S | Pending | T2 |
| P1-E1-S1b-T8 | Write integration tests for Project API | M | Pending | T3-T7 |

---

### P1-E1-S2: File Upload API Endpoint

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E1-S2-T1 | Create upload Pydantic schemas | S | Pending | P1-E1-S1 |
| P1-E1-S2-T2 | Implement file validation (size, types) | M | Pending | T1 |
| P1-E1-S2-T3 | Implement MinIO upload helper | M | Pending | P0-E2-S2 |
| P1-E1-S2-T4 | Create POST /api/v1/projects/{id}/documents/upload endpoint | M | Pending | T1-T3 |
| P1-E1-S2-T5 | Create Document and IngestionJob records | M | Pending | T4 |
| P1-E1-S2-T6 | Start ingestion workflow after upload | M | Pending | T5, P1-E2-S1 |
| P1-E1-S2-T7 | Support batch upload (multiple files) | M | Pending | T4 |
| P1-E1-S2-T8 | Write integration tests for upload | M | Pending | T4-T7 |

---

### P1-E1-S2b: Document-Project Linking API

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E1-S2b-T1 | Create linking Pydantic schemas | S | Pending | P1-E1-S1b |
| P1-E1-S2b-T2 | Implement link documents to project endpoint | M | Pending | T1 |
| P1-E1-S2b-T3 | Implement unlink document from project endpoint | M | Pending | T1 |
| P1-E1-S2b-T4 | Update Qdrant payload on link/unlink | M | Pending | T2, T3 |
| P1-E1-S2b-T5 | Update MeiliSearch document on link/unlink | M | Pending | T2, T3 |
| P1-E1-S2b-T6 | Update Neo4j relationships on link/unlink | M | Pending | T2, T3 |
| P1-E1-S2b-T7 | Create GET /documents/{id}/projects endpoint | S | Pending | T1 |
| P1-E1-S2b-T8 | Write integration tests | M | Pending | T2-T7 |

---

### P1-E1-S3: URL Ingestion API Endpoint

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E1-S3-T1 | Create URL ingestion Pydantic schemas | S | Pending | P1-E1-S1 |
| P1-E1-S3-T2 | Implement URL validation and sanitization | M | Pending | T1 |
| P1-E1-S3-T3 | Create POST /api/v1/documents/ingest-url endpoint | M | Pending | T1, T2 |
| P1-E1-S3-T4 | Handle web pages, direct links, YouTube URLs | M | Pending | T3 |
| P1-E1-S3-T5 | Start ingestion workflow for URL | M | Pending | T3, P1-E2-S1 |
| P1-E1-S3-T6 | Write integration tests | M | Pending | T3-T5 |

---

### P1-E1-S4: Ingestion Job Status API

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E1-S4-T1 | Create job status Pydantic schemas | S | Pending | P1-E1-S1 |
| P1-E1-S4-T2 | Create GET /api/v1/jobs/{job_id} endpoint | M | Pending | T1 |
| P1-E1-S4-T3 | Create GET /api/v1/jobs endpoint with pagination | M | Pending | T1 |
| P1-E1-S4-T4 | Implement SSE endpoint for real-time updates | M | Pending | T2 |
| P1-E1-S4-T5 | Write integration tests | M | Pending | T2-T4 |

---

### P1-E2-S1: Document Ingestion Workflow Definition

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E2-S1-T1 | Create IngestionWorkflowInput dataclass | S | Pending | P0-E2-S6 |
| P1-E2-S1-T2 | Define DocumentIngestionWorkflow class | M | Pending | T1 |
| P1-E2-S1-T3 | Implement workflow steps orchestration | L | Pending | T2 |
| P1-E2-S1-T4 | Add query methods (get_progress, get_status) | M | Pending | T3 |
| P1-E2-S1-T5 | Add signal methods (cancel, pause) | M | Pending | T3 |
| P1-E2-S1-T6 | Implement error handling and compensation | M | Pending | T3 |
| P1-E2-S1-T7 | Write workflow unit tests | M | Pending | T1-T6 |

---

### P1-E2-S2: Document Classification Activity

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E2-S2-T1 | Create DocumentType enum | S | Pending | P1-E2-S1 |
| P1-E2-S2-T2 | Implement classify_document activity | M | Pending | T1 |
| P1-E2-S2-T3 | Implement magic bytes detection | M | Pending | T2 |
| P1-E2-S2-T4 | Implement file extension detection | S | Pending | T2 |
| P1-E2-S2-T5 | Store classification in Document record | S | Pending | T2 |
| P1-E2-S2-T6 | Write activity unit tests | M | Pending | T2-T5 |

---

### P1-E2-S3: File Download Activity

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E2-S3-T1 | Implement download_file activity | M | Pending | P1-E2-S1 |
| P1-E2-S3-T2 | Add HTTP/HTTPS download with headers | M | Pending | T1 |
| P1-E2-S3-T3 | Integrate yt-dlp for YouTube downloads | M | Pending | T1 |
| P1-E2-S3-T4 | Implement activity heartbeat | S | Pending | T1 |
| P1-E2-S3-T5 | Store downloaded file in MinIO | M | Pending | T1 |
| P1-E2-S3-T6 | Write activity unit tests | M | Pending | T1-T5 |

---

### P1-E2-S4: Parallel Processing Orchestration

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E2-S4-T1 | Implement fan-out logic based on document type | M | Pending | P1-E2-S2 |
| P1-E2-S4-T2 | Use asyncio.gather for parallel activities | M | Pending | T1 |
| P1-E2-S4-T3 | Implement result aggregation | M | Pending | T2 |
| P1-E2-S4-T4 | Handle partial failures | M | Pending | T3 |
| P1-E2-S4-T5 | Configure parallelism limits | S | Pending | T2 |

---

### P1-E3-S1: Docling OCR Activity

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E3-S1-T1 | Add Docling to worker dependencies | S | Pending | P1-E2-S2 |
| P1-E3-S1-T2 | Implement process_with_docling activity | M | Pending | T1 |
| P1-E3-S1-T3 | Configure OCR and table structure settings | S | Pending | T2 |
| P1-E3-S1-T4 | Implement heartbeat during processing | S | Pending | T2 |
| P1-E3-S1-T5 | Export to Markdown format | M | Pending | T2 |
| P1-E3-S1-T6 | Export to JSON format | M | Pending | T2 |
| P1-E3-S1-T7 | Write activity unit tests | M | Pending | T2-T6 |

---

### P1-E3-S2: Audio Transcription Activity (Parakeet)

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E3-S2-T1 | Set up Parakeet model access (API or local) | M | Pending | P1-E2-S2 |
| P1-E3-S2-T2 | Implement transcribe_audio activity | M | Pending | T1 |
| P1-E3-S2-T3 | Add ffmpeg audio extraction from video | M | Pending | T2 |
| P1-E3-S2-T4 | Support multiple audio formats | S | Pending | T2 |
| P1-E3-S2-T5 | Output timestamps (word or segment level) | M | Pending | T2 |
| P1-E3-S2-T6 | Implement activity heartbeat | S | Pending | T2 |
| P1-E3-S2-T7 | Write activity unit tests | M | Pending | T2-T6 |

---

### P1-E3-S3: Object Detection Activity (YOLO)

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E3-S3-T1 | Add YOLOv8/v9 to worker dependencies | S | Pending | P1-E2-S2 |
| P1-E3-S3-T2 | Implement detect_objects activity | M | Pending | T1 |
| P1-E3-S3-T3 | Extract frames from video at intervals | M | Pending | T2 |
| P1-E3-S3-T4 | Configure detection classes | S | Pending | T2 |
| P1-E3-S3-T5 | Output bounding boxes and confidence | M | Pending | T2 |
| P1-E3-S3-T6 | Store detected frames in MinIO | M | Pending | T5 |
| P1-E3-S3-T7 | Write activity unit tests | M | Pending | T2-T6 |

---

### P1-E3-S4: Web Scraping Activity (Playwright)

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E3-S4-T1 | Add Playwright to worker dependencies | S | Pending | P1-E2-S3 |
| P1-E3-S4-T2 | Implement scrape_webpage activity | M | Pending | T1 |
| P1-E3-S4-T3 | Handle JavaScript-rendered pages | M | Pending | T2 |
| P1-E3-S4-T4 | Extract main content, metadata, links | M | Pending | T2 |
| P1-E3-S4-T5 | Handle pagination | M | Pending | T2 |
| P1-E3-S4-T6 | Implement robots.txt checking | S | Pending | T2 |
| P1-E3-S4-T7 | Screenshot capture option | S | Pending | T2 |
| P1-E3-S4-T8 | Write activity unit tests | M | Pending | T2-T7 |

---

### P1-E4-S1: Semantic Chunking Activity

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E4-S1-T1 | Create Chunk dataclass | S | Pending | P1-E3-S1 |
| P1-E4-S1-T2 | Implement chunk_document activity | M | Pending | T1 |
| P1-E4-S1-T3 | Implement semantic boundary detection | M | Pending | T2 |
| P1-E4-S1-T4 | Configure chunk size and overlap | S | Pending | T2 |
| P1-E4-S1-T5 | Preserve document structure metadata | M | Pending | T2 |
| P1-E4-S1-T6 | Write activity unit tests | M | Pending | T2-T5 |

---

### P1-E4-S2: Embedding Generation Activity

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E4-S2-T1 | Implement generate_embeddings activity | M | Pending | P0-E3-S2, P1-E4-S1 |
| P1-E4-S2-T2 | Implement batch processing | M | Pending | T1 |
| P1-E4-S2-T3 | Generate dense embeddings (BGE) | M | Pending | T1 |
| P1-E4-S2-T4 | Generate sparse vectors (BM25 tokens) | M | Pending | T1 |
| P1-E4-S2-T5 | Implement retry logic | S | Pending | T1 |
| P1-E4-S2-T6 | Write activity unit tests | M | Pending | T1-T5 |

---

### P1-E4-S3: Qdrant Indexing Activity

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E4-S3-T1 | Implement index_to_qdrant activity | M | Pending | P0-E2-S3, P1-E4-S2 |
| P1-E4-S3-T2 | Implement batch upsert | M | Pending | T1 |
| P1-E4-S3-T3 | Store dense and sparse vectors | M | Pending | T1 |
| P1-E4-S3-T4 | Configure payload (document_id, chunk_index, etc.) | S | Pending | T1 |
| P1-E4-S3-T5 | Use deterministic point IDs for idempotency | S | Pending | T1 |
| P1-E4-S3-T6 | Write activity unit tests | M | Pending | T1-T5 |

---

### P1-E4-S4: MeiliSearch Indexing Activity

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E4-S4-T1 | Implement index_to_meilisearch activity | M | Pending | P0-E2-S5, P1-E4-S1 |
| P1-E4-S4-T2 | Index document metadata and chunk text | M | Pending | T1 |
| P1-E4-S4-T3 | Configure searchable attributes | S | Pending | T1 |
| P1-E4-S4-T4 | Configure filterable attributes | S | Pending | T1 |
| P1-E4-S4-T5 | Wait for indexing task completion | S | Pending | T1 |
| P1-E4-S4-T6 | Write activity unit tests | M | Pending | T1-T5 |

---

### P1-E5-S1: Entity Extraction Activity (Hybrid NER)

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E5-S1-T1 | Add spaCy to worker dependencies | S | Pending | P1-E4-S1 |
| P1-E5-S1-T2 | Implement extract_entities activity | M | Pending | T1 |
| P1-E5-S1-T3 | Implement spaCy NER extraction | M | Pending | T2 |
| P1-E5-S1-T4 | Implement LLM refinement step | M | Pending | T3, P0-E3-S3 |
| P1-E5-S1-T5 | Add confidence scores | S | Pending | T4 |
| P1-E5-S1-T6 | Implement coreference resolution | M | Pending | T3 |
| P1-E5-S1-T7 | Write activity unit tests | M | Pending | T2-T6 |

---

### P1-E5-S2: Relationship Extraction Activity

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E5-S2-T1 | Define relationship types enum | S | Pending | P1-E5-S1 |
| P1-E5-S2-T2 | Create Pydantic models for relationships | S | Pending | T1 |
| P1-E5-S2-T3 | Implement extract_relationships activity | M | Pending | T2 |
| P1-E5-S2-T4 | LLM-based extraction with structured output | M | Pending | T3 |
| P1-E5-S2-T5 | Add confidence scoring | S | Pending | T4 |
| P1-E5-S2-T6 | Track provenance (source chunk) | S | Pending | T4 |
| P1-E5-S2-T7 | Write activity unit tests | M | Pending | T3-T6 |

---

### P1-E5-S3: Entity Resolution Activity

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E5-S3-T1 | Implement resolve_entities activity | M | Pending | P1-E5-S1 |
| P1-E5-S3-T2 | Implement embedding-based blocking | M | Pending | T1 |
| P1-E5-S3-T3 | Implement LLM-assisted matching | M | Pending | T2 |
| P1-E5-S3-T4 | Create SAME_AS relationships | S | Pending | T3 |
| P1-E5-S3-T5 | Merge entity attributes | M | Pending | T3 |
| P1-E5-S3-T6 | Write activity unit tests | M | Pending | T1-T5 |

---

### P1-E5-S4: Neo4j Graph Construction Activity

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E5-S4-T1 | Implement build_knowledge_graph activity | M | Pending | P0-E2-S4, P1-E5-S2 |
| P1-E5-S4-T2 | Create/merge Entity nodes | M | Pending | T1 |
| P1-E5-S4-T3 | Create Relationship edges | M | Pending | T1 |
| P1-E5-S4-T4 | Create Document and Chunk nodes | M | Pending | T1 |
| P1-E5-S4-T5 | Create MENTIONED_IN relationships | M | Pending | T2, T4 |
| P1-E5-S4-T6 | Use MERGE for idempotency | S | Pending | T2-T5 |
| P1-E5-S4-T7 | Write activity unit tests | M | Pending | T1-T6 |

---

### P1-E5-S5: Entity-Project Linking Activity

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E5-S5-T1 | Implement link_entities_to_projects activity | M | Pending | P1-E5-S4 |
| P1-E5-S5-T2 | Create project_entities records | M | Pending | T1 |
| P1-E5-S5-T3 | Create APPEARS_IN_PROJECT Neo4j relationships | M | Pending | T1 |
| P1-E5-S5-T4 | Make activity idempotent | S | Pending | T2, T3 |
| P1-E5-S5-T5 | Write activity unit tests | M | Pending | T1-T4 |

---

### P1-E6-S1: Ingestion Worker Implementation

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E6-S1-T1 | Create IngestionWorker class | M | Pending | P1-E5-S5 |
| P1-E6-S1-T2 | Register all ingestion activities | M | Pending | T1 |
| P1-E6-S1-T3 | Configure task queue | S | Pending | T1 |
| P1-E6-S1-T4 | Implement graceful shutdown | M | Pending | T1 |
| P1-E6-S1-T5 | Add Prometheus metrics interceptor | M | Pending | T1 |
| P1-E6-S1-T6 | Add OTEL tracing interceptor | M | Pending | T1 |
| P1-E6-S1-T7 | Create worker Dockerfile | M | Pending | T1 |

---

### P1-E6-S2: Integration Tests for Ingestion Pipeline

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E6-S2-T1 | Create pytest fixtures for test documents | M | Pending | P1-E6-S1 |
| P1-E6-S2-T2 | Test complete workflow execution | L | Pending | T1 |
| P1-E6-S2-T3 | Mock external services (vLLM, OCR) | M | Pending | T2 |
| P1-E6-S2-T4 | Verify data in all datastores | M | Pending | T2 |
| P1-E6-S2-T5 | Test error handling and retry | M | Pending | T2 |
| P1-E6-S2-T6 | Test idempotency | M | Pending | T2 |

---

### P1-E6-S3: Ingestion Monitoring Dashboard

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P1-E6-S3-T1 | Create Grafana dashboard JSON | M | Pending | P0-E4-S3, P1-E6-S1 |
| P1-E6-S3-T2 | Add jobs started/completed/failed metrics | S | Pending | T1 |
| P1-E6-S3-T3 | Add processing time by document type | S | Pending | T1 |
| P1-E6-S3-T4 | Add queue depth and wait times | S | Pending | T1 |
| P1-E6-S3-T5 | Add error rate by activity type | S | Pending | T1 |
| P1-E6-S3-T6 | Add LLM token usage tracking | S | Pending | T1 |

---

## Phase 2: Chat/Agent Interface

### P2-E1-S1: OpenAI Agent SDK Integration with Temporal

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E1-S1-T1 | Add OpenAI Agent SDK to dependencies | S | Pending | P0-E2-S6 |
| P2-E1-S1-T2 | Configure OpenAIAgentsPlugin on Temporal client | M | Pending | T1 |
| P2-E1-S1-T3 | Configure ModelActivityParameters | M | Pending | T2 |
| P2-E1-S1-T4 | Set up Pydantic data converter | M | Pending | T2 |
| P2-E1-S1-T5 | Add tracing interceptor | M | Pending | T2 |
| P2-E1-S1-T6 | Test agent survives worker restart | M | Pending | T2-T5 |

---

### P2-E1-S2: Agent Workflow Definition

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E1-S2-T1 | Create AgentWorkflowInput dataclass | S | Pending | P2-E1-S1 |
| P2-E1-S2-T2 | Define AgentExecutionWorkflow class | M | Pending | T1 |
| P2-E1-S2-T3 | Implement agent instantiation with tools | M | Pending | T2 |
| P2-E1-S2-T4 | Implement streaming output via signals | M | Pending | T2 |
| P2-E1-S2-T5 | Add query methods for status | M | Pending | T2 |
| P2-E1-S2-T6 | Add timeout handling | S | Pending | T2 |
| P2-E1-S2-T7 | Write workflow unit tests | M | Pending | T2-T6 |

---

### P2-E1-S3: Base Agent Tools Infrastructure

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E1-S3-T1 | Create BaseToolActivity class | M | Pending | P2-E1-S1 |
| P2-E1-S3-T2 | Define standard input/output Pydantic models | M | Pending | T1 |
| P2-E1-S3-T3 | Implement error handling utilities | M | Pending | T1 |
| P2-E1-S3-T4 | Configure tool execution timeouts | S | Pending | T1 |
| P2-E1-S3-T5 | Add logging and tracing decorators | M | Pending | T1 |

---

### P2-E2-S1: Vector Search Tool

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E2-S1-T1 | Create SearchInput Pydantic model | S | Pending | P2-E1-S3, P1-E4-S3 |
| P2-E2-S1-T2 | Implement search_documents activity | M | Pending | T1 |
| P2-E2-S1-T3 | Implement hybrid search (dense + sparse) | M | Pending | T2 |
| P2-E2-S1-T4 | Add filter support (date, type, entities) | M | Pending | T2 |
| P2-E2-S1-T5 | Format results for LLM consumption | M | Pending | T2 |
| P2-E2-S1-T6 | Write activity unit tests | M | Pending | T2-T5 |

---

### P2-E2-S2: Full-Text Search Tool

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E2-S2-T1 | Create FulltextSearchInput Pydantic model | S | Pending | P2-E1-S3, P1-E4-S4 |
| P2-E2-S2-T2 | Implement fulltext_search activity | M | Pending | T1 |
| P2-E2-S2-T3 | Add faceted results | M | Pending | T2 |
| P2-E2-S2-T4 | Add term highlighting | M | Pending | T2 |
| P2-E2-S2-T5 | Add pagination support | S | Pending | T2 |
| P2-E2-S2-T6 | Write activity unit tests | M | Pending | T2-T5 |

---

### P2-E2-S3: Graph Exploration Tool

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E2-S3-T1 | Create GraphExploreInput Pydantic model | S | Pending | P2-E1-S3, P1-E5-S4 |
| P2-E2-S3-T2 | Implement explore_graph activity | M | Pending | T1 |
| P2-E2-S3-T3 | Implement natural language to Cypher (LLM) | M | Pending | T2 |
| P2-E2-S3-T4 | Execute query against Neo4j | M | Pending | T2 |
| P2-E2-S3-T5 | Format output for visualization | M | Pending | T4 |
| P2-E2-S3-T6 | Write activity unit tests | M | Pending | T2-T5 |

---

### P2-E2-S4: Timeline Construction Tool

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E2-S4-T1 | Create TimelineInput Pydantic model | S | Pending | P2-E2-S3 |
| P2-E2-S4-T2 | Implement build_timeline activity | M | Pending | T1 |
| P2-E2-S4-T3 | Query temporal events | M | Pending | T2 |
| P2-E2-S4-T4 | Sort chronologically | S | Pending | T3 |
| P2-E2-S4-T5 | Include source documents | M | Pending | T3 |
| P2-E2-S4-T6 | Write activity unit tests | M | Pending | T2-T5 |

---

### P2-E2-S5: Document Retrieval Tool

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E2-S5-T1 | Create GetDocumentInput Pydantic model | S | Pending | P2-E1-S3 |
| P2-E2-S5-T2 | Implement get_document activity | M | Pending | T1 |
| P2-E2-S5-T3 | Retrieve metadata from PostgreSQL | M | Pending | T2 |
| P2-E2-S5-T4 | Retrieve content from MinIO | M | Pending | T2 |
| P2-E2-S5-T5 | Support pagination for large documents | M | Pending | T4 |
| P2-E2-S5-T6 | Write activity unit tests | M | Pending | T2-T5 |

---

### P2-E3-S1: CSV/Excel Analysis Tool

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E3-S1-T1 | Create AnalyzeTabularInput Pydantic model | S | Pending | P2-E4-S3 |
| P2-E3-S1-T2 | Implement analyze_tabular activity | M | Pending | T1 |
| P2-E3-S1-T3 | Load data into pandas DataFrame | M | Pending | T2 |
| P2-E3-S1-T4 | Execute analysis operations | M | Pending | T3 |
| P2-E3-S1-T5 | Integrate gVisor sandbox execution | M | Pending | T4 |
| P2-E3-S1-T6 | Format results as table | M | Pending | T4 |
| P2-E3-S1-T7 | Write activity unit tests | M | Pending | T2-T6 |

---

### P2-E3-S2: SQL Query Tool

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E3-S2-T1 | Create SQLQueryInput Pydantic model | S | Pending | P2-E4-S3 |
| P2-E3-S2-T2 | Implement execute_sql activity | M | Pending | T1 |
| P2-E3-S2-T3 | Load CSVs into SQLite in-memory | M | Pending | T2 |
| P2-E3-S2-T4 | Execute query with timeout | M | Pending | T3 |
| P2-E3-S2-T5 | Integrate gVisor sandbox execution | M | Pending | T4 |
| P2-E3-S2-T6 | Implement query validation | M | Pending | T2 |
| P2-E3-S2-T7 | Write activity unit tests | M | Pending | T2-T6 |

---

### P2-E3-S3: Data Visualization Tool

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E3-S3-T1 | Create VisualizationInput Pydantic model | S | Pending | P2-E4-S3 |
| P2-E3-S3-T2 | Implement create_visualization activity | M | Pending | T1 |
| P2-E3-S3-T3 | Generate charts using matplotlib/plotly | M | Pending | T2 |
| P2-E3-S3-T4 | Integrate gVisor sandbox execution | M | Pending | T3 |
| P2-E3-S3-T5 | Save chart image to MinIO | M | Pending | T3 |
| P2-E3-S3-T6 | Support multiple chart types | M | Pending | T3 |
| P2-E3-S3-T7 | Write activity unit tests | M | Pending | T2-T6 |

---

### P2-E4-S1: gVisor RuntimeClass Configuration

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E4-S1-T1 | Create RuntimeClass YAML manifest | S | Pending | P0-E5-S1 |
| P2-E4-S1-T2 | Configure node selector | S | Pending | T1 |
| P2-E4-S1-T3 | Document gVisor installation | M | Pending | T1 |
| P2-E4-S1-T4 | Test pod deployment with RuntimeClass | M | Pending | T1 |

---

### P2-E4-S2: Sandboxed Execution Pod Template

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E4-S2-T1 | Create pod template with runtimeClassName | M | Pending | P2-E4-S1 |
| P2-E4-S2-T2 | Configure read-only root filesystem | S | Pending | T1 |
| P2-E4-S2-T3 | Configure tmpfs mounts | S | Pending | T1 |
| P2-E4-S2-T4 | Set resource limits | S | Pending | T1 |
| P2-E4-S2-T5 | Configure network policy | M | Pending | T1 |
| P2-E4-S2-T6 | Configure non-root user | S | Pending | T1 |
| P2-E4-S2-T7 | Configure time limit | S | Pending | T1 |

---

### P2-E4-S3: Sandbox Executor Activity

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E4-S3-T1 | Create SandboxInput Pydantic model | S | Pending | P2-E4-S2 |
| P2-E4-S3-T2 | Implement execute_sandboxed activity | L | Pending | T1 |
| P2-E4-S3-T3 | Create ephemeral pod | M | Pending | T2 |
| P2-E4-S3-T4 | Copy input files to pod | M | Pending | T3 |
| P2-E4-S3-T5 | Execute code with timeout | M | Pending | T3 |
| P2-E4-S3-T6 | Capture stdout, stderr, output files | M | Pending | T5 |
| P2-E4-S3-T7 | Clean up pod after execution | M | Pending | T5 |
| P2-E4-S3-T8 | Write activity unit tests | M | Pending | T2-T7 |

---

### P2-E4-S4: Docker Compose Sandbox Fallback (nsjail)

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E4-S4-T1 | Create nsjail Docker image | M | Pending | P2-E4-S3 |
| P2-E4-S4-T2 | Configure nsjail for Python execution | M | Pending | T1 |
| P2-E4-S4-T3 | Create API endpoint for code execution | M | Pending | T2 |
| P2-E4-S4-T4 | Match resource limits with K8s config | S | Pending | T2 |
| P2-E4-S4-T5 | Document dev vs production differences | S | Pending | T1-T4 |

---

### P2-E5-S1: Chat Conversation API

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E5-S1-T1 | Create Conversation SQLAlchemy model | M | Pending | P2-E1-S2 |
| P2-E5-S1-T2 | Create Message SQLAlchemy model | M | Pending | T1 |
| P2-E5-S1-T3 | Create conversation Pydantic schemas | M | Pending | T1 |
| P2-E5-S1-T4 | Implement POST /conversations endpoint | M | Pending | T3 |
| P2-E5-S1-T5 | Implement GET /conversations endpoint | M | Pending | T3 |
| P2-E5-S1-T6 | Implement GET /conversations/{id} endpoint | M | Pending | T3 |
| P2-E5-S1-T7 | Implement POST /conversations/{id}/messages endpoint | M | Pending | T3 |
| P2-E5-S1-T8 | Implement DELETE /conversations/{id} endpoint | M | Pending | T3 |
| P2-E5-S1-T9 | Write integration tests | M | Pending | T4-T8 |

---

### P2-E5-S2: SSE Streaming Endpoint

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E5-S2-T1 | Create SSE event types | S | Pending | P2-E5-S1 |
| P2-E5-S2-T2 | Implement GET /conversations/{id}/stream endpoint | M | Pending | T1 |
| P2-E5-S2-T3 | Implement message_delta events | M | Pending | T2 |
| P2-E5-S2-T4 | Implement tool_call and tool_result events | M | Pending | T2 |
| P2-E5-S2-T5 | Add heartbeat events | S | Pending | T2 |
| P2-E5-S2-T6 | Handle client disconnection | M | Pending | T2 |
| P2-E5-S2-T7 | Write integration tests | M | Pending | T2-T6 |

---

### P2-E5-S3: Agent Response Formatting

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E5-S3-T1 | Create response formatting utilities | M | Pending | P2-E5-S2 |
| P2-E5-S3-T2 | Implement Markdown formatting | M | Pending | T1 |
| P2-E5-S3-T3 | Implement table result formatting | M | Pending | T1 |
| P2-E5-S3-T4 | Implement file attachment formatting | M | Pending | T1 |
| P2-E5-S3-T5 | Implement citation formatting | M | Pending | T1 |
| P2-E5-S3-T6 | Implement error message formatting | S | Pending | T1 |

---

### P2-E6-S1: Chat Page Layout

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E6-S1-T1 | Create base HTML layout template | M | Pending | P2-E5-S2 |
| P2-E6-S1-T2 | Create sidebar with conversation list | M | Pending | T1 |
| P2-E6-S1-T3 | Create message thread area | M | Pending | T1 |
| P2-E6-S1-T4 | Create input area with send button | M | Pending | T1 |
| P2-E6-S1-T5 | Style with Tailwind CSS | M | Pending | T1-T4 |
| P2-E6-S1-T6 | Make responsive (mobile-friendly) | M | Pending | T5 |

---

### P2-E6-S2: SSE Streaming Integration

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E6-S2-T1 | Set up HTMX SSE extension | S | Pending | P2-E6-S1 |
| P2-E6-S2-T2 | Implement streaming message append | M | Pending | T1 |
| P2-E6-S2-T3 | Add typing indicator | M | Pending | T2 |
| P2-E6-S2-T4 | Implement smooth scrolling | S | Pending | T2 |
| P2-E6-S2-T5 | Handle connection errors with retry | M | Pending | T1 |

---

### P2-E6-S3: Tool Execution Display

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E6-S3-T1 | Create collapsible tool call card component | M | Pending | P2-E6-S2 |
| P2-E6-S3-T2 | Display tool name, inputs, status | M | Pending | T1 |
| P2-E6-S3-T3 | Create expandable tool results | M | Pending | T1 |
| P2-E6-S3-T4 | Create table rendering component | M | Pending | T3 |
| P2-E6-S3-T5 | Create chart rendering component | M | Pending | T3 |
| P2-E6-S3-T6 | Create document preview component | M | Pending | T3 |
| P2-E6-S3-T7 | Create code block component with syntax highlighting | M | Pending | T3 |

---

### P2-E6-S4: File and Document Rendering

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E6-S4-T1 | Integrate pdf.js for PDF viewing | M | Pending | P2-E6-S3 |
| P2-E6-S4-T2 | Create Markdown rendering component | M | Pending | P2-E6-S1 |
| P2-E6-S4-T3 | Create image display with zoom | M | Pending | P2-E6-S1 |
| P2-E6-S4-T4 | Create table rendering with pagination | M | Pending | P2-E6-S1 |
| P2-E6-S4-T5 | Add download buttons | S | Pending | T1-T4 |

---

### P2-E6-S5: Conversation Management UI

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E6-S5-T1 | Create new conversation button | S | Pending | P2-E6-S1 |
| P2-E6-S5-T2 | Implement inline rename | M | Pending | T1 |
| P2-E6-S5-T3 | Implement delete with confirmation | M | Pending | T1 |
| P2-E6-S5-T4 | Add conversation search/filter | M | Pending | T1 |
| P2-E6-S5-T5 | Add export as Markdown | M | Pending | T1 |

---

### P2-E7-S1: Agent Worker Implementation

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E7-S1-T1 | Create AgentWorker class | M | Pending | P2-E2-S5, P2-E3-S1 |
| P2-E7-S1-T2 | Register all agent tools | M | Pending | T1 |
| P2-E7-S1-T3 | Configure task queue | S | Pending | T1 |
| P2-E7-S1-T4 | Configure OpenAIAgentsPlugin | M | Pending | T1 |
| P2-E7-S1-T5 | Set concurrency limits | S | Pending | T1 |
| P2-E7-S1-T6 | Add OTEL tracing | M | Pending | T1 |
| P2-E7-S1-T7 | Create worker Dockerfile | M | Pending | T1 |

---

### P2-E7-S2: Agent Integration Tests

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P2-E7-S2-T1 | Test agent workflow execution | M | Pending | P2-E7-S1 |
| P2-E7-S2-T2 | Mock LLM responses | M | Pending | T1 |
| P2-E7-S2-T3 | Test each tool individually | L | Pending | T1 |
| P2-E7-S2-T4 | Test multi-turn conversations | M | Pending | T1 |
| P2-E7-S2-T5 | Test error handling and recovery | M | Pending | T1 |

---

## Phase 3: Search Experience

### P3-E1-S1: Unified Search Endpoint

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E1-S1-T1 | Create unified SearchRequest Pydantic model | M | Pending | P2-E2-S1, P2-E2-S2 |
| P3-E1-S1-T2 | Create POST /api/v1/search endpoint | M | Pending | T1 |
| P3-E1-S1-T3 | Implement mode parameter handling | M | Pending | T2 |
| P3-E1-S1-T4 | Implement cursor-based pagination | M | Pending | T2 |
| P3-E1-S1-T5 | Add facet counts to response | M | Pending | T2 |
| P3-E1-S1-T6 | Add search query logging | S | Pending | T2 |
| P3-E1-S1-T7 | Write integration tests | M | Pending | T2-T6 |

---

### P3-E1-S2: Semantic Search Implementation

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E1-S2-T1 | Generate query embedding | M | Pending | P3-E1-S1 |
| P3-E1-S2-T2 | Search Qdrant with filters | M | Pending | T1 |
| P3-E1-S2-T3 | Implement hybrid dense + sparse option | M | Pending | T2 |
| P3-E1-S2-T4 | Add optional cross-encoder re-ranking | M | Pending | T3 |
| P3-E1-S2-T5 | Include document metadata | M | Pending | T2 |

---

### P3-E1-S3: Keyword Search Implementation

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E1-S3-T1 | Search MeiliSearch index | M | Pending | P3-E1-S1 |
| P3-E1-S3-T2 | Support exact phrase matching | M | Pending | T1 |
| P3-E1-S3-T3 | Add faceted results | M | Pending | T1 |
| P3-E1-S3-T4 | Highlight matching terms | M | Pending | T1 |
| P3-E1-S3-T5 | Configure typo tolerance | S | Pending | T1 |

---

### P3-E1-S4: Graph Search Implementation

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E1-S4-T1 | Implement entity search by name | M | Pending | P3-E1-S1 |
| P3-E1-S4-T2 | Implement relationship traversal | M | Pending | T1 |
| P3-E1-S4-T3 | Implement path finding | M | Pending | T1 |
| P3-E1-S4-T4 | Add community exploration | M | Pending | T1 |
| P3-E1-S4-T5 | Return graph structure for visualization | M | Pending | T1-T4 |

---

### P3-E1-S5: Search Result Aggregation

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E1-S5-T1 | Merge results from multiple sources | M | Pending | P3-E1-S2, P3-E1-S3, P3-E1-S4 |
| P3-E1-S5-T2 | De-duplicate overlapping results | M | Pending | T1 |
| P3-E1-S5-T3 | Implement consistent relevance scoring | M | Pending | T1 |
| P3-E1-S5-T4 | Add group by document option | M | Pending | T1 |
| P3-E1-S5-T5 | Generate display snippets | M | Pending | T1 |

---

### P3-E2-S0: Project Management UI

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E2-S0-T1 | Create projects list page | M | Pending | P2-E6-S1 |
| P3-E2-S0-T2 | Create project cards component | M | Pending | T1 |
| P3-E2-S0-T3 | Create project modal | M | Pending | T1 |
| P3-E2-S0-T4 | Create project settings page | M | Pending | T1 |
| P3-E2-S0-T5 | Create project selector in header | M | Pending | T1 |
| P3-E2-S0-T6 | Add global toggle in sidebar | M | Pending | T5 |
| P3-E2-S0-T7 | Implement HTMX project switching | M | Pending | T5 |
| P3-E2-S0-T8 | Set up URL structure | M | Pending | T7 |

---

### P3-E2-S1: Search Page Layout

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E2-S1-T1 | Create search page template | M | Pending | P3-E1-S1 |
| P3-E2-S1-T2 | Create search input with suggestions | M | Pending | T1 |
| P3-E2-S1-T3 | Create filter sidebar | M | Pending | T1 |
| P3-E2-S1-T4 | Create results area | M | Pending | T1 |
| P3-E2-S1-T5 | Add pagination controls | M | Pending | T4 |
| P3-E2-S1-T6 | Add sort options | S | Pending | T4 |

---

### P3-E2-S2: Faceted Filtering UI

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E2-S2-T1 | Create document type filter | M | Pending | P3-E2-S1 |
| P3-E2-S2-T2 | Create date range filter | M | Pending | T1 |
| P3-E2-S2-T3 | Create entity type filter | M | Pending | T1 |
| P3-E2-S2-T4 | Create source filter | M | Pending | T1 |
| P3-E2-S2-T5 | Create active filter pills | M | Pending | T1-T4 |
| P3-E2-S2-T6 | Implement HTMX dynamic updates | M | Pending | T5 |

---

### P3-E2-S3: Search Result Cards

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E2-S3-T1 | Create result card component | M | Pending | P3-E2-S1 |
| P3-E2-S3-T2 | Add search term highlighting | M | Pending | T1 |
| P3-E2-S3-T3 | Add entity badges | M | Pending | T1 |
| P3-E2-S3-T4 | Add quick actions | M | Pending | T1 |
| P3-E2-S3-T5 | Add relevance indicator | S | Pending | T1 |

---

### P3-E2-S4: Document Preview Panel

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E2-S4-T1 | Create slide-in panel | M | Pending | P3-E2-S3 |
| P3-E2-S4-T2 | Display document metadata | M | Pending | T1 |
| P3-E2-S4-T3 | Add full text preview | M | Pending | T1 |
| P3-E2-S4-T4 | Add entity list | M | Pending | T1 |
| P3-E2-S4-T5 | Add related documents | M | Pending | T1 |
| P3-E2-S4-T6 | Add "Ask about this" button | S | Pending | T1 |

---

### P3-E3-S1: Entity Search Interface

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E3-S1-T1 | Create entity search page | M | Pending | P3-E1-S4 |
| P3-E3-S1-T2 | Create entity type filter | M | Pending | T1 |
| P3-E3-S1-T3 | Create entity card component | M | Pending | T1 |
| P3-E3-S1-T4 | Add click to explore | M | Pending | T3 |
| P3-E3-S1-T5 | Add entity merge/link UI (admin) | M | Pending | T3 |

---

### P3-E3-S2: Graph Visualization Component

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E3-S2-T1 | Integrate vis.js (or similar) | M | Pending | P3-E3-S1 |
| P3-E3-S2-T2 | Implement force-directed layout | M | Pending | T1 |
| P3-E3-S2-T3 | Configure node colors by entity type | S | Pending | T2 |
| P3-E3-S2-T4 | Add edge labels | S | Pending | T2 |
| P3-E3-S2-T5 | Implement zoom and pan | M | Pending | T2 |
| P3-E3-S2-T6 | Add node click to expand | M | Pending | T2 |
| P3-E3-S2-T7 | Add node details on hover | M | Pending | T2 |

---

### P3-E3-S3: Relationship Exploration Panel

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E3-S3-T1 | Create entity details panel | M | Pending | P3-E3-S2 |
| P3-E3-S3-T2 | List relationships (incoming/outgoing) | M | Pending | T1 |
| P3-E3-S3-T3 | Add relationship type filter | M | Pending | T2 |
| P3-E3-S3-T4 | Add source documents | M | Pending | T2 |
| P3-E3-S3-T5 | Add timeline of mentions | M | Pending | T1 |

---

### P3-E3-S4: Path Finder Interface

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E3-S4-T1 | Create two entity selectors | M | Pending | P3-E3-S2 |
| P3-E3-S4-T2 | Implement find shortest path | M | Pending | T1 |
| P3-E3-S4-T3 | Display path as linear visualization | M | Pending | T2 |
| P3-E3-S4-T4 | Show intermediate entities | M | Pending | T3 |
| P3-E3-S4-T5 | List source documents along path | M | Pending | T3 |

---

### P3-E4-S1: Document Library Page

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E4-S1-T1 | Create document library page | M | Pending | P3-E2-S0 |
| P3-E4-S1-T2 | Implement list/grid view toggle | M | Pending | T1 |
| P3-E4-S1-T3 | Add sorting options | M | Pending | T1 |
| P3-E4-S1-T4 | Add filtering | M | Pending | T1 |
| P3-E4-S1-T5 | Add bulk selection | M | Pending | T1 |
| P3-E4-S1-T6 | Add upload button | S | Pending | T1 |

---

### P3-E4-S2: Document Details Page

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E4-S2-T1 | Create document details page | M | Pending | P3-E4-S1 |
| P3-E4-S2-T2 | Display metadata | M | Pending | T1 |
| P3-E4-S2-T3 | Display processing status | M | Pending | T1 |
| P3-E4-S2-T4 | Display entities list | M | Pending | T1 |
| P3-E4-S2-T5 | Display chunk preview | M | Pending | T1 |
| P3-E4-S2-T6 | Add re-process button | S | Pending | T1 |
| P3-E4-S2-T7 | Add delete button | S | Pending | T1 |

---

### P3-E4-S3: Upload Interface

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E4-S3-T1 | Create drag-and-drop zone | M | Pending | P3-E4-S1 |
| P3-E4-S3-T2 | Support multiple file selection | M | Pending | T1 |
| P3-E4-S3-T3 | Add upload progress bars | M | Pending | T1 |
| P3-E4-S3-T4 | Add file validation feedback | M | Pending | T1 |
| P3-E4-S3-T5 | Add URL input tab | M | Pending | T1 |
| P3-E4-S3-T6 | Auto-start ingestion | S | Pending | T1 |

---

### P3-E4-S4: Batch Operations

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E4-S4-T1 | Implement multi-select | M | Pending | P3-E4-S1 |
| P3-E4-S4-T2 | Add bulk delete | M | Pending | T1 |
| P3-E4-S4-T3 | Add bulk re-process | M | Pending | T1 |
| P3-E4-S4-T4 | Add export as ZIP | M | Pending | T1 |
| P3-E4-S4-T5 | Add batch tagging | M | Pending | T1 |

---

### P3-E4-S5: Cross-Project Document Linking UI

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P3-E4-S5-T1 | Add "Also in projects" display | M | Pending | P3-E4-S2 |
| P3-E4-S5-T2 | Create "Link to project" modal | M | Pending | T1 |
| P3-E4-S5-T3 | Add "Unlink from project" action | M | Pending | T1 |
| P3-E4-S5-T4 | Add bulk link action | M | Pending | P3-E4-S4 |

---

## Phase 4: Polish & Advanced Features

### P4-E1-S1: Authentication and Authorization

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E1-S1-T1 | Implement API key authentication | M | Pending | P3-E4-S1 |
| P4-E1-S1-T2 | Implement session-based auth | M | Pending | T1 |
| P4-E1-S1-T3 | Implement RBAC (admin, user, viewer) | M | Pending | T1 |
| P4-E1-S1-T4 | Create API key management UI | M | Pending | T1 |
| P4-E1-S1-T5 | Implement audit logging | M | Pending | T1 |

---

### P4-E1-S2: Rate Limiting and Quotas

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E1-S2-T1 | Implement request rate limiting | M | Pending | P4-E1-S1 |
| P4-E1-S2-T2 | Implement LLM token quota | M | Pending | T1 |
| P4-E1-S2-T3 | Implement storage quota | M | Pending | T1 |
| P4-E1-S2-T4 | Add quota warning notifications | M | Pending | T2, T3 |
| P4-E1-S2-T5 | Create admin quota management UI | M | Pending | T1-T4 |

---

### P4-E1-S3: Error Handling and Resilience

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E1-S3-T1 | Implement consistent error format | M | Pending | P4-E1-S1 |
| P4-E1-S3-T2 | Create user-friendly error messages | M | Pending | T1 |
| P4-E1-S3-T3 | Add optional Sentry integration | M | Pending | T1 |
| P4-E1-S3-T4 | Implement circuit breakers | M | Pending | T1 |
| P4-E1-S3-T5 | Implement graceful degradation | M | Pending | T4 |

---

### P4-E1-S4: Performance Optimization

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E1-S4-T1 | Optimize database queries | M | Pending | P4-E1-S3 |
| P4-E1-S4-T2 | Add Redis caching layer | M | Pending | T1 |
| P4-E1-S4-T3 | Tune connection pooling | S | Pending | T1 |
| P4-E1-S4-T4 | Add response compression | S | Pending | T1 |
| P4-E1-S4-T5 | Document performance benchmarks | M | Pending | T1-T4 |

---

### P4-E2-S1: Helm Chart Creation

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E2-S1-T1 | Create Helm chart structure | M | Pending | P4-E1-S3 |
| P4-E2-S1-T2 | Create values.yaml with defaults | M | Pending | T1 |
| P4-E2-S1-T3 | Configure resource limits | M | Pending | T2 |
| P4-E2-S1-T4 | Add secret management | M | Pending | T2 |
| P4-E2-S1-T5 | Add health checks and probes | M | Pending | T2 |
| P4-E2-S1-T6 | Configure HPA for workers | M | Pending | T2 |

---

### P4-E2-S2: gVisor Node Pool Configuration

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E2-S2-T1 | Configure node labels | S | Pending | P4-E2-S1 |
| P4-E2-S2-T2 | Configure RuntimeClass | S | Pending | T1 |
| P4-E2-S2-T3 | Document node pool sizing | S | Pending | T1 |
| P4-E2-S2-T4 | Document installation (GKE, EKS, self-managed) | M | Pending | T1 |
| P4-E2-S2-T5 | Create testing procedure | M | Pending | T2 |

---

### P4-E2-S3: Database Operators/StatefulSets

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E2-S3-T1 | Configure Neo4j StatefulSet | M | Pending | P4-E2-S1 |
| P4-E2-S3-T2 | Configure Qdrant StatefulSet | M | Pending | T1 |
| P4-E2-S3-T3 | Configure PostgreSQL (CloudNativePG or Helm) | M | Pending | T1 |
| P4-E2-S3-T4 | Configure MinIO operator | M | Pending | T1 |
| P4-E2-S3-T5 | Configure backup CronJobs | M | Pending | T1-T4 |

---

### P4-E2-S4: Ingress and TLS Configuration

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E2-S4-T1 | Configure Ingress resource | M | Pending | P4-E2-S1 |
| P4-E2-S4-T2 | Set up cert-manager | M | Pending | T1 |
| P4-E2-S4-T3 | Optional service mesh (mTLS) | M | Pending | T1 |
| P4-E2-S4-T4 | Configure network policies | M | Pending | T1 |

---

### P4-E3-S1: Incremental Re-indexing

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E3-S1-T1 | Implement document change detection | M | Pending | P4-E1-S3 |
| P4-E3-S1-T2 | Update only changed chunks | M | Pending | T1 |
| P4-E3-S1-T3 | Preserve valid entity links | M | Pending | T2 |
| P4-E3-S1-T4 | Re-run relationship extraction | M | Pending | T2 |
| P4-E3-S1-T5 | Update search indexes incrementally | M | Pending | T2 |

---

### P4-E3-S2: Scheduled Web Scraping

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E3-S2-T1 | Define URL pattern monitoring | M | Pending | P4-E3-S1 |
| P4-E3-S2-T2 | Implement scheduling (cron) | M | Pending | T1 |
| P4-E3-S2-T3 | Implement change detection | M | Pending | T2 |
| P4-E3-S2-T4 | Add new content notifications | M | Pending | T3 |
| P4-E3-S2-T5 | Create source management UI | M | Pending | T1-T4 |

---

### P4-E3-S3: Document Collections/Projects

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E3-S3-T1 | Create collection CRUD API | M | Pending | P4-E1-S1 |
| P4-E3-S3-T2 | Add documents to collections | M | Pending | T1 |
| P4-E3-S3-T3 | Implement collection-scoped search | M | Pending | T1 |
| P4-E3-S3-T4 | Add collection permissions | M | Pending | T1 |
| P4-E3-S3-T5 | Export collection as dataset | M | Pending | T1 |

---

### P4-E4-S1: Multi-Agent Research Workflows

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E4-S1-T1 | Implement agent handoff | M | Pending | P2-E7-S1 |
| P4-E4-S1-T2 | Create research planner agent | M | Pending | T1 |
| P4-E4-S1-T3 | Create entity specialist agent | M | Pending | T1 |
| P4-E4-S1-T4 | Create timeline analyst agent | M | Pending | T1 |
| P4-E4-S1-T5 | Create summary writer agent | M | Pending | T1 |
| P4-E4-S1-T6 | Create team composition config | M | Pending | T2-T5 |

---

### P4-E4-S2: Persistent Agent Memory

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E4-S2-T1 | Create memory storage schema | M | Pending | P4-E4-S1 |
| P4-E4-S2-T2 | Implement memory retrieval tool | M | Pending | T1 |
| P4-E4-S2-T3 | Create memory management UI | M | Pending | T1 |
| P4-E4-S2-T4 | Implement memory scoping | M | Pending | T1 |

---

### P4-E4-S3: Custom Tool Configuration

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E4-S3-T1 | Create admin tool configuration UI | M | Pending | P4-E4-S1 |
| P4-E4-S3-T2 | Enable/disable tools per conversation | M | Pending | T1 |
| P4-E4-S3-T3 | Add tool parameter customization | M | Pending | T1 |
| P4-E4-S3-T4 | Add custom prompt injection | M | Pending | T1 |

---

### P4-E5-S1: Research Report Generation

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E5-S1-T1 | Generate report from conversation | M | Pending | P2-E6-S5 |
| P4-E5-S1-T2 | Include citations and sources | M | Pending | T1 |
| P4-E5-S1-T3 | Create configurable templates | M | Pending | T1 |
| P4-E5-S1-T4 | Export as PDF/Markdown/DOCX | M | Pending | T1 |
| P4-E5-S1-T5 | Include visualizations | M | Pending | T1 |

---

### P4-E5-S2: Data Export

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E5-S2-T1 | Export all documents as ZIP | M | Pending | P4-E5-S1 |
| P4-E5-S2-T2 | Export entities as CSV/JSON | M | Pending | T1 |
| P4-E5-S2-T3 | Export knowledge graph | M | Pending | T1 |
| P4-E5-S2-T4 | Implement selective export | M | Pending | T1-T3 |
| P4-E5-S2-T5 | Create background job for large exports | M | Pending | T4 |

---

### P4-E5-S3: Audit Trail

| ID | Task | Effort | Status | Dependencies |
|----|------|--------|--------|--------------|
| P4-E5-S3-T1 | Log all data access/modifications | M | Pending | P4-E1-S1 |
| P4-E5-S3-T2 | Create searchable audit log UI | M | Pending | T1 |
| P4-E5-S3-T3 | Export audit logs | M | Pending | T2 |
| P4-E5-S3-T4 | Configure retention policy | S | Pending | T1 |
| P4-E5-S3-T5 | Generate compliance reports | M | Pending | T2 |

---

## Notes

### Effort Estimates
- **S (Small)**: < 2 hours
- **M (Medium)**: 2-8 hours  
- **L (Large)**: > 8 hours

### Status Legend
- **Pending**: Not started
- **In Progress**: Currently being worked on
- **Done**: Completed
- **Blocked**: Waiting on dependencies or external factors

### Dependency Notation
- `T1, T2` = Tasks T1 and T2 from the same story
- `P0-E1-S1` = Another story in the same phase
- `P0-E1-S1-T3` = Specific task from another story
