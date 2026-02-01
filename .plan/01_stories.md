# Initial Implementation - User Stories

> User stories organized by phase and epic

## Story Format

Each story follows this structure:
- **ID**: P{phase}-E{epic}-S{story} (e.g., P0-E1-S1)
- **Title**: Brief descriptive title
- **Acceptance Criteria**: Specific, testable requirements
- **Priority**: High / Medium / Low
- **Dependencies**: Other stories this depends on

---

## Phase 0: Project Scaffolding & Infrastructure

### Epic 0.1: Repository and Development Environment Setup

#### P0-E1-S1: Initialize Monorepo Structure

Create project repository with organized directory structure for all components.

**Acceptance Criteria**:
- [ ] Root directory with shared configuration (pyproject.toml, .pre-commit-config.yaml)
- [ ] `/api` directory for FastAPI application
- [ ] `/workers` directory for Temporal workers (ingestion, agent, query)
- [ ] `/frontend` directory for HTMX templates
- [ ] `/infra` directory for Docker, K8s manifests, Tilt configuration
- [ ] `/libs` directory for shared Python packages
- [ ] README.md with project overview and setup instructions
- [ ] .gitignore configured for Python, Node, IDE files

**Priority**: High  
**Dependencies**: None

---

#### P0-E1-S2: Configure Python Development Environment

Set up Python tooling with modern package management and code quality tools.

**Acceptance Criteria**:
- [ ] pyproject.toml with project metadata and dependencies
- [ ] uv.lock file for reproducible builds
- [ ] ruff configuration for linting and formatting
- [ ] mypy configuration with strict mode
- [ ] pre-commit hooks for code quality
- [ ] VS Code/PyCharm configuration files

**Priority**: High  
**Dependencies**: P0-E1-S1

---

#### P0-E1-S3: Create Docker Base Images

Build optimized Docker images for Python services.

**Acceptance Criteria**:
- [ ] Base Python image with common dependencies (python:3.12-slim)
- [ ] API Dockerfile with FastAPI, uvicorn
- [ ] Worker Dockerfile with Temporal SDK, ML dependencies
- [ ] Multi-stage builds for smaller production images
- [ ] Non-root user configuration
- [ ] Health check endpoints

**Priority**: High  
**Dependencies**: P0-E1-S2

---

### Epic 0.2: Infrastructure Services (Docker Compose)

#### P0-E2-S0: Tenant and Project Schema

Create database schema for multi-tenancy support from day one.

**Acceptance Criteria**:
- [ ] Alembic migration creating: tenants, projects, project_documents, project_entities tables
- [ ] All tables gain tenant_id column with NOT NULL + foreign key
- [ ] conversations table gains nullable project_id column
- [ ] Indexes on tenant_id for all tables, compound indexes where needed
- [ ] Dev-mode startup creates default tenant and default project
- [ ] Environment variable: SINGLE_TENANT_MODE=true for local development

**Priority**: High  
**Dependencies**: P0-E1-S1

---

#### P0-E2-S0b: Tenant Middleware and Request Context

Create FastAPI middleware for tenant resolution and request context.

**Acceptance Criteria**:
- [ ] TenantContext dataclass with tenant_id, user_id, roles
- [ ] RequestContext dataclass with tenant + optional project_id
- [ ] Dependency injection via Depends() for all route handlers
- [ ] Single-tenant mode bypasses auth, uses default tenant
- [ ] All repository methods require tenant_id parameter
- [ ] 403 returned if project doesn't belong to tenant

**Priority**: High  
**Dependencies**: P0-E2-S0

---

#### P0-E2-S1: PostgreSQL Setup with Schema

Configure PostgreSQL for application metadata.

**Acceptance Criteria**:
- [ ] Docker Compose service for PostgreSQL 16
- [ ] Volume mount for data persistence
- [ ] Initial schema migration with Alembic
- [ ] Tables: documents, ingestion_jobs, users, api_keys
- [ ] Connection pooling configuration
- [ ] Health check endpoint

**Priority**: High  
**Dependencies**: P0-E2-S0

---

#### P0-E2-S2: MinIO Object Storage Setup

Configure MinIO for document storage.

**Acceptance Criteria**:
- [ ] Docker Compose service for MinIO
- [ ] Buckets: raw-documents, processed-documents, exports
- [ ] IAM policies for service accounts
- [ ] Console accessible on port 9001
- [ ] Event notifications configured for new uploads
- [ ] Pre-signed URL generation utility functions

**Priority**: High  
**Dependencies**: P0-E1-S3

---

#### P0-E2-S3: Qdrant Vector Database Setup

Configure Qdrant for vector storage.

**Acceptance Criteria**:
- [ ] Docker Compose service for Qdrant
- [ ] Collections: document_chunks (dense + sparse vectors)
- [ ] Payload indexes for filtering (document_id, tenant_id, document_type)
- [ ] Quantization configuration (INT8 scalar)
- [ ] gRPC and REST endpoints exposed
- [ ] Snapshot configuration for backups

**Priority**: High  
**Dependencies**: P0-E1-S3

---

#### P0-E2-S4: Neo4j Graph Database Setup

Configure Neo4j Community Edition for knowledge graph.

**Acceptance Criteria**:
- [ ] Docker Compose service for Neo4j 5.x
- [ ] APOC plugin installed and configured
- [ ] Initial constraints: unique entity IDs, document IDs
- [ ] Indexes on: Entity.name, Entity.type, Document.id
- [ ] Browser accessible on port 7474
- [ ] Bolt protocol on port 7687

**Priority**: High  
**Dependencies**: P0-E1-S3

---

#### P0-E2-S5: MeiliSearch Setup

Configure MeiliSearch for full-text search.

**Acceptance Criteria**:
- [ ] Docker Compose service for MeiliSearch
- [ ] Index: documents with searchable, filterable attributes
- [ ] Facets configured: document_type, source, date_range
- [ ] API key management
- [ ] Analytics disabled for privacy

**Priority**: High  
**Dependencies**: P0-E1-S3

---

#### P0-E2-S6: Temporal Server Setup

Configure Temporal server for workflow orchestration.

**Acceptance Criteria**:
- [ ] Docker Compose services: temporal, temporal-admin-tools, temporal-ui
- [ ] PostgreSQL backend for Temporal persistence
- [ ] Namespaces: default, ingestion, agents
- [ ] UI accessible on port 8080
- [ ] CLI (tctl) available in admin-tools container

**Priority**: High  
**Dependencies**: P0-E2-S1

---

### Epic 0.3: vLLM Serving Infrastructure

#### P0-E3-S1: vLLM Chat Model Deployment

Deploy vLLM server for chat/extraction model.

**Acceptance Criteria**:
- [ ] Docker Compose service with NVIDIA runtime
- [ ] Model: meta-llama/Llama-3.1-8B-Instruct (or configurable)
- [ ] OpenAI-compatible API on port 8000
- [ ] Prometheus metrics exposed
- [ ] Environment variables for model configuration
- [ ] GPU memory allocation (tensor-parallel-size if multi-GPU)

**Priority**: High  
**Dependencies**: P0-E1-S3

---

#### P0-E3-S2: vLLM Embedding Model Deployment

Deploy vLLM server for embedding generation.

**Acceptance Criteria**:
- [ ] Separate Docker Compose service
- [ ] Model: BAAI/bge-large-en-v1.5
- [ ] Task mode: embed
- [ ] Port 8001 (separate from chat)
- [ ] Batch processing optimized

**Priority**: High  
**Dependencies**: P0-E3-S1

---

#### P0-E3-S3: LLM Client Abstraction Layer

Create Python client for LLM services.

**Acceptance Criteria**:
- [ ] LLMClient class wrapping OpenAI Python SDK
- [ ] Configuration via environment variables (base_url, api_key)
- [ ] Retry logic with exponential backoff
- [ ] Async methods for chat_completion, embeddings
- [ ] Structured output support (Pydantic response parsing)
- [ ] Token counting utility

**Priority**: High  
**Dependencies**: P0-E3-S1

---

### Epic 0.4: Observability Stack

#### P0-E4-S1: OpenTelemetry Collector Setup

Deploy OTEL Collector for traces, metrics, logs.

**Acceptance Criteria**:
- [ ] Docker Compose service for otel-collector
- [ ] Receivers: OTLP (gRPC 4317, HTTP 4318)
- [ ] Exporters: Jaeger (traces), Prometheus (metrics)
- [ ] Batch processor configured
- [ ] Service pipeline definitions

**Priority**: Medium  
**Dependencies**: P0-E1-S3

---

#### P0-E4-S2: Jaeger Tracing Setup

Deploy Jaeger for distributed tracing.

**Acceptance Criteria**:
- [ ] Docker Compose service for Jaeger all-in-one
- [ ] UI accessible on port 16686
- [ ] OTLP ingestion enabled
- [ ] Retention policy configured

**Priority**: Medium  
**Dependencies**: P0-E4-S1

---

#### P0-E4-S3: Prometheus and Grafana Setup

Deploy monitoring stack.

**Acceptance Criteria**:
- [ ] Prometheus with scrape configs for all services
- [ ] Grafana with pre-configured dashboards
- [ ] Dashboards: vLLM metrics, Temporal metrics, API latency
- [ ] Alert rules for critical failures
- [ ] Grafana datasources auto-provisioned

**Priority**: Medium  
**Dependencies**: P0-E4-S1

---

#### P0-E4-S4: FastAPI OTEL Instrumentation

Instrument FastAPI with OpenTelemetry.

**Acceptance Criteria**:
- [ ] opentelemetry-instrumentation-fastapi configured
- [ ] Custom spans for business logic
- [ ] Request/response hooks for user context
- [ ] Excluded paths: /health, /metrics
- [ ] Trace context propagation headers

**Priority**: Medium  
**Dependencies**: P0-E4-S1

---

### Epic 0.5: Tilt Development Environment

#### P0-E5-S1: Tiltfile for Local Development

Create Tilt configuration for local K8s development.

**Acceptance Criteria**:
- [ ] Tiltfile loading all K8s manifests
- [ ] docker_build with live_update for Python services
- [ ] Resource dependencies (DBs start before workers)
- [ ] Port forwards for all services
- [ ] Labels for UI organization

**Priority**: Medium  
**Dependencies**: P0-E2-S6

---

#### P0-E5-S2: Custom Tilt Buttons/Actions

Add custom actions for development workflows.

**Acceptance Criteria**:
- [ ] Button: "Clear All Data" - drops DB schemas, clears MinIO
- [ ] Button: "Run Migrations" - executes Alembic migrations
- [ ] Button: "Seed Test Data" - loads sample documents
- [ ] Button: "Reset Temporal" - clears workflow history
- [ ] All actions with auto_init=False, TRIGGER_MODE_MANUAL

**Priority**: Low  
**Dependencies**: P0-E5-S1

---

#### P0-E5-S3: Docker Compose Fallback

Ensure project runs without Tilt.

**Acceptance Criteria**:
- [ ] docker-compose.yml with all services
- [ ] docker-compose.override.yml for development extras
- [ ] Makefile with common commands (up, down, logs, test)
- [ ] README documenting both Tilt and Docker Compose workflows

**Priority**: High  
**Dependencies**: P0-E2-S6

---

### Epic 0.6: CI/CD Pipeline

#### P0-E6-S1: GitHub Actions for Testing

Create CI pipeline for testing and linting.

**Acceptance Criteria**:
- [ ] Workflow triggers on PR and push to main
- [ ] Steps: lint (ruff), type check (mypy), unit tests (pytest)
- [ ] Service containers for PostgreSQL, Qdrant
- [ ] Test coverage reporting
- [ ] Parallel job execution

**Priority**: High  
**Dependencies**: P0-E1-S2

---

#### P0-E6-S2: Docker Image Build Pipeline

Create pipeline for building and pushing images.

**Acceptance Criteria**:
- [ ] Build on tag push (semver)
- [ ] Multi-platform builds (amd64, arm64)
- [ ] Push to container registry (ghcr.io or configurable)
- [ ] Image scanning with Trivy
- [ ] Build caching for faster builds

**Priority**: Medium  
**Dependencies**: P0-E6-S1

---

## Phase 1: Ingestion Pipeline

### Epic 1.1: Core Ingestion Infrastructure

#### P1-E1-S1: Document Model and Database Schema

Define document data model and persistence layer.

**Acceptance Criteria**:
- [ ] SQLAlchemy models: Document, IngestionJob, Chunk, Entity
- [ ] Alembic migrations for all tables
- [ ] Document states: pending, processing, completed, failed
- [ ] IngestionJob with status, progress, error tracking
- [ ] Chunk with vector_id, text, metadata
- [ ] Repository pattern for database operations

**Priority**: High  
**Dependencies**: P0-E2-S1

---

#### P1-E1-S1b: Project Management API

Create CRUD API for projects.

**Acceptance Criteria**:
- [ ] POST /api/v1/projects — create project (name, description)
- [ ] GET /api/v1/projects — list projects for tenant
- [ ] GET /api/v1/projects/{id} — get project details with stats (document count, entity count, last activity)
- [ ] PATCH /api/v1/projects/{id} — update name/description/settings
- [ ] POST /api/v1/projects/{id}/archive — soft archive
- [ ] POST /api/v1/projects/{id}/unarchive — restore
- [ ] All routes enforce tenant_id from auth context

**Priority**: High  
**Dependencies**: P0-E2-S0b

---

#### P1-E1-S2: File Upload API Endpoint

Create FastAPI endpoint for file uploads.

**Acceptance Criteria**:
- [ ] POST /api/v1/projects/{project_id}/documents/upload accepting multipart/form-data
- [ ] File validation (size limits, allowed types)
- [ ] Upload to MinIO raw-documents bucket
- [ ] Create Document and IngestionJob records
- [ ] Return job_id for status tracking
- [ ] Support for batch upload (multiple files)

**Priority**: High  
**Dependencies**: P1-E1-S1

---

#### P1-E1-S2b: Document-Project Linking API

Create API for linking documents to projects.

**Acceptance Criteria**:
- [ ] POST /api/v1/projects/{id}/documents/link - Body: { document_ids: [...] }
- [ ] DELETE /api/v1/projects/{id}/documents/{doc_id}/unlink
- [ ] GET /api/v1/documents/{doc_id}/projects — list projects a doc belongs to
- [ ] Linking updates Qdrant payload (project_ids), MeiliSearch document, and Neo4j relationships
- [ ] Unlinking updates the same stores

**Priority**: High  
**Dependencies**: P1-E1-S1b

---

#### P1-E1-S3: URL Ingestion API Endpoint

Create endpoint for URL-based ingestion.

**Acceptance Criteria**:
- [ ] POST /api/v1/documents/ingest-url accepting URL
- [ ] URL validation and sanitization
- [ ] Support for: web pages, direct file links, YouTube URLs
- [ ] Create Document record with source_url
- [ ] Start ingestion workflow

**Priority**: High  
**Dependencies**: P1-E1-S2

---

#### P1-E1-S4: Ingestion Job Status API

Create endpoint for tracking ingestion progress.

**Acceptance Criteria**:
- [ ] GET /api/v1/jobs/{job_id} returning job status
- [ ] Include: status, progress percentage, current step, errors
- [ ] GET /api/v1/jobs for listing all jobs with pagination
- [ ] WebSocket/SSE endpoint for real-time updates

**Priority**: High  
**Dependencies**: P1-E1-S2

---

### Epic 1.2: Temporal Ingestion Workflow

#### P1-E2-S1: Document Ingestion Workflow Definition

Create main Temporal workflow for document ingestion.

**Acceptance Criteria**:
- [ ] @workflow.defn class DocumentIngestionWorkflow
- [ ] Input: document_id, source (file_path or url)
- [ ] Workflow steps: classify → download → process → extract → index
- [ ] Query methods: get_progress(), get_status()
- [ ] Signal methods: cancel(), pause()
- [ ] Error handling with compensation (cleanup on failure)

**Priority**: High  
**Dependencies**: P0-E2-S6, P1-E1-S1

---

#### P1-E2-S2: Document Classification Activity

Create activity to classify document type.

**Acceptance Criteria**:
- [ ] @activity.defn async def classify_document
- [ ] Input: document_id, file_path in MinIO
- [ ] Detect type: pdf, docx, xlsx, csv, image, audio, video, web_page
- [ ] Use magic bytes and file extension
- [ ] Return DocumentType enum
- [ ] Store classification in Document record

**Priority**: High  
**Dependencies**: P1-E2-S1

---

#### P1-E2-S3: File Download Activity

Create activity to download files from URLs.

**Acceptance Criteria**:
- [ ] @activity.defn async def download_file
- [ ] Support HTTP/HTTPS URLs with proper headers
- [ ] YouTube download via yt-dlp
- [ ] Heartbeat during large downloads
- [ ] Store in MinIO raw-documents bucket
- [ ] Return MinIO object path

**Priority**: High  
**Dependencies**: P1-E2-S1

---

#### P1-E2-S4: Parallel Processing Orchestration

Implement fan-out/fan-in for parallel processing.

**Acceptance Criteria**:
- [ ] Use asyncio.gather for parallel activity execution
- [ ] Parallel tasks: OCR, transcription, object detection (based on type)
- [ ] Aggregate results after all complete
- [ ] Handle partial failures (continue if some succeed)
- [ ] Configurable parallelism limits

**Priority**: Medium  
**Dependencies**: P1-E2-S1

---

### Epic 1.3: Document Processing Activities

#### P1-E3-S1: Docling OCR Activity

Create activity for document parsing with Docling.

**Acceptance Criteria**:
- [ ] @activity.defn async def process_with_docling
- [ ] Input: document bytes or MinIO path
- [ ] Configure: do_ocr=True, do_table_structure=True
- [ ] Support: PDF, DOCX, PPTX, images
- [ ] Output: extracted text, table data, document structure
- [ ] Export to both Markdown and JSON formats
- [ ] Heartbeat during processing

**Priority**: High  
**Dependencies**: P1-E2-S2

---

#### P1-E3-S2: Audio Transcription Activity (Parakeet)

Create activity for audio/video transcription.

**Acceptance Criteria**:
- [ ] @activity.defn async def transcribe_audio
- [ ] Use NVIDIA Parakeet model via API or local inference
- [ ] Extract audio from video files (ffmpeg)
- [ ] Support: WAV, MP3, MP4, WebM
- [ ] Output: transcribed text with timestamps
- [ ] Word-level or segment-level timestamps
- [ ] Heartbeat during long transcriptions

**Priority**: High  
**Dependencies**: P1-E2-S2

---

#### P1-E3-S3: Object Detection Activity (YOLO)

Create activity for image/video object detection.

**Acceptance Criteria**:
- [ ] @activity.defn async def detect_objects
- [ ] Use YOLOv8 or YOLOv9 model
- [ ] Extract frames from video at configurable intervals
- [ ] Detect: people, objects, text regions, faces (configurable)
- [ ] Output: detected objects with bounding boxes, confidence
- [ ] Store detected frame images in MinIO

**Priority**: Medium  
**Dependencies**: P1-E2-S2

---

#### P1-E3-S4: Web Scraping Activity (Playwright)

Create activity for web page extraction.

**Acceptance Criteria**:
- [ ] @activity.defn async def scrape_webpage
- [ ] Use Playwright for JavaScript-rendered pages
- [ ] Extract: main content, metadata, links, images
- [ ] Handle pagination for multi-page content
- [ ] Respect robots.txt (configurable)
- [ ] Screenshot capture option
- [ ] Store extracted content and assets in MinIO

**Priority**: Medium  
**Dependencies**: P1-E2-S3

---

### Epic 1.4: Chunking and Embedding

#### P1-E4-S1: Semantic Chunking Activity

Create activity for intelligent document chunking.

**Acceptance Criteria**:
- [ ] @activity.defn async def chunk_document
- [ ] Hierarchical chunking preserving document structure
- [ ] Semantic boundaries using sentence embeddings
- [ ] Configurable chunk size (default 500 tokens)
- [ ] 10-20% overlap between chunks
- [ ] Preserve metadata: section headers, page numbers
- [ ] Output: list of Chunk objects with text and metadata

**Priority**: High  
**Dependencies**: P1-E3-S1

---

#### P1-E4-S2: Embedding Generation Activity

Create activity for generating embeddings.

**Acceptance Criteria**:
- [ ] @activity.defn async def generate_embeddings
- [ ] Batch processing for efficiency
- [ ] Use vLLM embedding endpoint
- [ ] Generate both dense (BGE) and sparse (BM25 tokens) vectors
- [ ] Return list of embeddings aligned with chunks
- [ ] Retry logic for API failures

**Priority**: High  
**Dependencies**: P0-E3-S2, P1-E4-S1

---

#### P1-E4-S3: Qdrant Indexing Activity

Create activity to store vectors in Qdrant.

**Acceptance Criteria**:
- [ ] @activity.defn async def index_to_qdrant
- [ ] Batch upsert to Qdrant collection
- [ ] Store both dense and sparse vectors
- [ ] Payload: document_id, chunk_index, text preview, metadata
- [ ] Idempotent: use deterministic point IDs
- [ ] Return count of indexed vectors

**Priority**: High  
**Dependencies**: P0-E2-S3, P1-E4-S2

---

#### P1-E4-S4: MeiliSearch Indexing Activity

Create activity to index in MeiliSearch.

**Acceptance Criteria**:
- [ ] @activity.defn async def index_to_meilisearch
- [ ] Index document metadata and chunk text
- [ ] Configure searchable attributes: title, content, entities
- [ ] Filterable attributes: document_type, date, source
- [ ] Wait for indexing task completion
- [ ] Return indexed document count

**Priority**: High  
**Dependencies**: P0-E2-S5, P1-E4-S1

---

### Epic 1.5: Entity Extraction and Knowledge Graph

#### P1-E5-S1: Entity Extraction Activity (Hybrid NER)

Create activity for entity extraction.

**Acceptance Criteria**:
- [ ] @activity.defn async def extract_entities
- [ ] Hybrid approach: spaCy NER + LLM refinement
- [ ] Entity types: PERSON, ORGANIZATION, LOCATION, DATE, EVENT, MONEY
- [ ] Confidence scores for each entity
- [ ] Coreference resolution to link mentions
- [ ] Output: list of Entity objects with type, mentions, confidence

**Priority**: High  
**Dependencies**: P0-E3-S3, P1-E4-S1

---

#### P1-E5-S2: Relationship Extraction Activity

Create activity for relationship discovery.

**Acceptance Criteria**:
- [ ] @activity.defn async def extract_relationships
- [ ] LLM-based extraction with structured output (Pydantic)
- [ ] Relationship types: WORKS_AT, AFFILIATED_WITH, FUNDED_BY, PARTICIPATED_IN, etc.
- [ ] Source-target entity pairs with relationship type
- [ ] Confidence scoring
- [ ] Provenance tracking (source chunk)

**Priority**: High  
**Dependencies**: P1-E5-S1

---

#### P1-E5-S3: Entity Resolution Activity

Create activity for entity deduplication.

**Acceptance Criteria**:
- [ ] @activity.defn async def resolve_entities
- [ ] Embedding-based blocking (cluster similar entities)
- [ ] LLM-assisted matching for ambiguous cases
- [ ] Create SAME_AS relationships for duplicates
- [ ] Merge entity attributes from multiple sources
- [ ] Return canonical entity list

**Priority**: Medium  
**Dependencies**: P1-E5-S1

---

#### P1-E5-S4: Neo4j Graph Construction Activity

Create activity to build knowledge graph.

**Acceptance Criteria**:
- [ ] @activity.defn async def build_knowledge_graph
- [ ] Create/merge Entity nodes with properties
- [ ] Create Relationship edges between entities
- [ ] Create Document and Chunk nodes
- [ ] MENTIONED_IN relationships linking entities to chunks
- [ ] Idempotent using MERGE operations
- [ ] Return node/edge counts

**Priority**: High  
**Dependencies**: P0-E2-S4, P1-E5-S2

---

#### P1-E5-S5: Entity-Project Linking Activity

Create activity to link discovered entities to projects.

**Acceptance Criteria**:
- [ ] @activity.defn async def link_entities_to_projects
- [ ] Input: tenant_id, document_id, project_ids
- [ ] For each entity found in the document, create project_entities records
- [ ] Create APPEARS_IN_PROJECT relationships in Neo4j
- [ ] Idempotent (safe to re-run)

**Priority**: High  
**Dependencies**: P1-E5-S4

---

### Epic 1.6: Ingestion Worker and Testing

#### P1-E6-S1: Ingestion Worker Implementation

Create Temporal worker for ingestion activities.

**Acceptance Criteria**:
- [ ] Worker class registering all ingestion activities
- [ ] Connection to Temporal server with retry
- [ ] Task queue: ingestion-task-queue
- [ ] Graceful shutdown handling
- [ ] Prometheus metrics for activity execution
- [ ] OTEL tracing interceptor configured

**Priority**: High  
**Dependencies**: P1-E5-S4

---

#### P1-E6-S2: Integration Tests for Ingestion Pipeline

Create comprehensive tests for ingestion.

**Acceptance Criteria**:
- [ ] pytest fixtures for test documents (PDF, DOCX, image, audio)
- [ ] Test complete workflow execution with time-skipping
- [ ] Mock external services (vLLM, OCR)
- [ ] Verify data stored correctly in all datastores
- [ ] Test error handling and retry behavior
- [ ] Test idempotency (re-running same document)

**Priority**: High  
**Dependencies**: P1-E6-S1

---

#### P1-E6-S3: Ingestion Monitoring Dashboard

Create Grafana dashboard for ingestion metrics.

**Acceptance Criteria**:
- [ ] Metrics: jobs started, completed, failed per hour
- [ ] Processing time by document type
- [ ] Queue depth and wait times
- [ ] Error rate by activity type
- [ ] LLM token usage tracking

**Priority**: Medium  
**Dependencies**: P0-E4-S3, P1-E6-S1

---

## Phase 2: Chat/Agent Interface

### Epic 2.1: Agent Infrastructure

#### P2-E1-S1: OpenAI Agent SDK Integration with Temporal

Configure OpenAI Agent SDK with Temporal durability.

**Acceptance Criteria**:
- [ ] OpenAIAgentsPlugin configured on Temporal client
- [ ] ModelActivityParameters with appropriate timeouts
- [ ] Pydantic data converter for serialization
- [ ] Tracing interceptor for observability
- [ ] Test agent execution survives worker restart

**Priority**: High  
**Dependencies**: P0-E2-S6, P0-E3-S3

---

#### P2-E1-S2: Agent Workflow Definition

Create Temporal workflow for agent execution.

**Acceptance Criteria**:
- [ ] @workflow.defn class AgentExecutionWorkflow
- [ ] Input: conversation_id, user_message, context
- [ ] Agent instantiation with tools via activity_as_tool
- [ ] Streaming output via signals
- [ ] Query for current status
- [ ] Timeout handling (max execution time)

**Priority**: High  
**Dependencies**: P2-E1-S1

---

#### P2-E1-S3: Base Agent Tools Infrastructure

Create framework for agent tools as activities.

**Acceptance Criteria**:
- [ ] Base ToolActivity class with standard interface
- [ ] Input/output Pydantic models for each tool
- [ ] Error handling and user-friendly error messages
- [ ] Tool execution timeout configuration
- [ ] Logging and tracing for tool calls

**Priority**: High  
**Dependencies**: P2-E1-S1

---

### Epic 2.2: Research Agent Tools

#### P2-E2-S1: Vector Search Tool

Create tool for semantic document search.

**Acceptance Criteria**:
- [ ] @activity.defn async def search_documents
- [ ] Input: query string, optional filters (date, type, entities)
- [ ] Hybrid search: dense + sparse in Qdrant
- [ ] Return top-k chunks with relevance scores
- [ ] Include document metadata in results
- [ ] Format results for LLM consumption

**Priority**: High  
**Dependencies**: P2-E1-S3, P1-E4-S3

---

#### P2-E2-S2: Full-Text Search Tool

Create tool for keyword/phrase search.

**Acceptance Criteria**:
- [ ] @activity.defn async def fulltext_search
- [ ] Input: query string, filters
- [ ] Search MeiliSearch index
- [ ] Faceted results with counts
- [ ] Highlight matching terms
- [ ] Pagination support

**Priority**: High  
**Dependencies**: P2-E1-S3, P1-E4-S4

---

#### P2-E2-S3: Graph Exploration Tool

Create tool for knowledge graph queries.

**Acceptance Criteria**:
- [ ] @activity.defn async def explore_graph
- [ ] Input: entity name or query description
- [ ] Generate Cypher query from natural language (LLM)
- [ ] Execute query against Neo4j
- [ ] Return entities and relationships
- [ ] Visualizable output format

**Priority**: High  
**Dependencies**: P2-E1-S3, P1-E5-S4

---

#### P2-E2-S4: Timeline Construction Tool

Create tool for building event timelines.

**Acceptance Criteria**:
- [ ] @activity.defn async def build_timeline
- [ ] Input: entity name or topic
- [ ] Query events with temporal relationships
- [ ] Sort chronologically
- [ ] Include source documents for each event
- [ ] Output structured timeline data

**Priority**: Medium  
**Dependencies**: P2-E2-S3

---

#### P2-E2-S5: Document Retrieval Tool

Create tool for fetching full documents.

**Acceptance Criteria**:
- [ ] @activity.defn async def get_document
- [ ] Input: document_id
- [ ] Retrieve document metadata from PostgreSQL
- [ ] Retrieve full text from MinIO
- [ ] Return formatted document content
- [ ] Support pagination for large documents

**Priority**: High  
**Dependencies**: P2-E1-S3

---

### Epic 2.3: Tabular Data Tools

#### P2-E3-S1: CSV/Excel Analysis Tool

Create tool for analyzing tabular data.

**Acceptance Criteria**:
- [ ] @activity.defn async def analyze_tabular
- [ ] Input: document_id (CSV/Excel), analysis query
- [ ] Load data into pandas DataFrame
- [ ] Execute analysis (aggregations, filters, stats)
- [ ] **Execute in gVisor sandbox**
- [ ] Return results as formatted table
- [ ] Support natural language queries

**Priority**: High  
**Dependencies**: P2-E4-S3

---

#### P2-E3-S2: SQL Query Tool

Create tool for SQL queries on structured data.

**Acceptance Criteria**:
- [ ] @activity.defn async def execute_sql
- [ ] Input: SQL query, target database/table
- [ ] Load CSVs into SQLite in-memory database
- [ ] Execute query with timeout
- [ ] **Execute in gVisor sandbox**
- [ ] Return results as formatted table
- [ ] Query validation to prevent dangerous operations

**Priority**: Medium  
**Dependencies**: P2-E4-S3

---

#### P2-E3-S3: Data Visualization Tool

Create tool for generating charts.

**Acceptance Criteria**:
- [ ] @activity.defn async def create_visualization
- [ ] Input: data, chart type, configuration
- [ ] Generate charts using matplotlib/plotly
- [ ] **Execute in gVisor sandbox**
- [ ] Save chart image to MinIO
- [ ] Return image URL for display
- [ ] Support: bar, line, pie, scatter, timeline

**Priority**: Medium  
**Dependencies**: P2-E4-S3

---

### Epic 2.4: Sandbox Execution Environment

#### P2-E4-S1: gVisor RuntimeClass Configuration

Configure Kubernetes RuntimeClass for gVisor.

**Acceptance Criteria**:
- [ ] RuntimeClass definition: name=gvisor, handler=runsc
- [ ] Node selector for gVisor-capable nodes
- [ ] Documentation for gVisor installation on nodes
- [ ] Test pod deployment with RuntimeClass

**Priority**: High  
**Dependencies**: P0-E5-S1

---

#### P2-E4-S2: Sandboxed Execution Pod Template

Create pod specification for sandboxed code execution.

**Acceptance Criteria**:
- [ ] Pod template with runtimeClassName: gvisor
- [ ] Read-only root filesystem
- [ ] tmpfs mount for /tmp and working directory
- [ ] Resource limits: CPU, memory, ephemeral storage
- [ ] Network policy: deny all (or controlled egress)
- [ ] Non-root user execution
- [ ] Time limit via activeDeadlineSeconds

**Priority**: High  
**Dependencies**: P2-E4-S1

---

#### P2-E4-S3: Sandbox Executor Activity

Create activity for running code in sandbox.

**Acceptance Criteria**:
- [ ] @activity.defn async def execute_sandboxed
- [ ] Input: code string, language, input files
- [ ] Create ephemeral pod with gVisor runtime
- [ ] Copy input files to pod
- [ ] Execute code with timeout
- [ ] Capture stdout, stderr, output files
- [ ] Clean up pod after execution
- [ ] Return execution results

**Priority**: High  
**Dependencies**: P2-E4-S2

---

#### P2-E4-S4: Docker Compose Sandbox Fallback (nsjail)

Create fallback sandbox for local development.

**Acceptance Criteria**:
- [ ] Docker container with nsjail installed
- [ ] nsjail configuration for Python execution
- [ ] API endpoint for code execution
- [ ] Resource limits matching K8s configuration
- [ ] Document development vs production differences

**Priority**: Medium  
**Dependencies**: P2-E4-S3

---

### Epic 2.5: Chat API and Streaming

#### P2-E5-S1: Chat Conversation API

Create API endpoints for chat conversations.

**Acceptance Criteria**:
- [ ] POST /api/v1/projects/{project_id}/chat/conversations - create new conversation
- [ ] GET /api/v1/projects/{project_id}/chat/conversations - list for project
- [ ] GET /api/v1/chat/conversations/{id} - get conversation history
- [ ] POST /api/v1/chat/conversations/{id}/messages - send message
- [ ] DELETE /api/v1/chat/conversations/{id} - delete conversation
- [ ] Also support: POST /api/v1/chat/conversations (global, no project)
- [ ] Conversation storage in PostgreSQL
- [ ] Message history with tool calls and results

**Priority**: High  
**Dependencies**: P2-E1-S2

---

#### P2-E5-S2: SSE Streaming Endpoint

Create Server-Sent Events endpoint for streaming responses.

**Acceptance Criteria**:
- [ ] GET /api/v1/chat/conversations/{id}/stream - SSE connection
- [ ] Event types: message_delta, tool_call, tool_result, completed, error
- [ ] Heartbeat events to keep connection alive
- [ ] Graceful handling of client disconnection
- [ ] Base64 encoding for multi-line content

**Priority**: High  
**Dependencies**: P2-E5-S1

---

#### P2-E5-S3: Agent Response Formatting

Create utilities for formatting agent output.

**Acceptance Criteria**:
- [ ] Markdown formatting for text responses
- [ ] Structured output for tool results (tables, charts)
- [ ] File attachment formatting (links to MinIO)
- [ ] Citation formatting (source documents, chunks)
- [ ] Error message formatting

**Priority**: Medium  
**Dependencies**: P2-E5-S2

---

### Epic 2.6: Chat UI (HTMX)

#### P2-E6-S1: Chat Page Layout

Create main chat interface page.

**Acceptance Criteria**:
- [ ] Full-page layout with sidebar and main content
- [ ] Conversation list in sidebar
- [ ] Message thread in main area
- [ ] Input area with send button
- [ ] Tailwind CSS styling
- [ ] Responsive design (mobile-friendly)

**Priority**: High  
**Dependencies**: P2-E5-S2

---

#### P2-E6-S2: SSE Streaming Integration

Implement streaming message display.

**Acceptance Criteria**:
- [ ] HTMX SSE extension configured
- [ ] Messages append to thread as they stream
- [ ] Typing indicator during generation
- [ ] Smooth scrolling to new messages
- [ ] Handle connection errors with retry

**Priority**: High  
**Dependencies**: P2-E6-S1

---

#### P2-E6-S3: Tool Execution Display

Create UI components for tool execution.

**Acceptance Criteria**:
- [ ] Collapsible tool call cards
- [ ] Show tool name, inputs, status
- [ ] Expandable tool results
- [ ] Special rendering for: Tables (sortable, scrollable), Charts (embedded images), Documents (expandable previews), Code blocks (syntax highlighting)

**Priority**: High  
**Dependencies**: P2-E6-S2

---

#### P2-E6-S4: File and Document Rendering

Implement inline document viewing.

**Acceptance Criteria**:
- [ ] PDF viewer (pdf.js) in modal/panel
- [ ] Markdown rendering with syntax highlighting
- [ ] Image display with zoom
- [ ] Table rendering with pagination
- [ ] Download button for all files

**Priority**: Medium  
**Dependencies**: P2-E6-S3

---

#### P2-E6-S5: Conversation Management UI

Create UI for managing conversations.

**Acceptance Criteria**:
- [ ] Create new conversation button
- [ ] Rename conversation (inline edit)
- [ ] Delete conversation with confirmation
- [ ] Conversation search/filter
- [ ] Export conversation as Markdown

**Priority**: Medium  
**Dependencies**: P2-E6-S1

---

### Epic 2.7: Agent Worker

#### P2-E7-S1: Agent Worker Implementation

Create Temporal worker for agent activities.

**Acceptance Criteria**:
- [ ] Worker registering all agent tools
- [ ] Task queue: agent-task-queue
- [ ] OpenAIAgentsPlugin configured
- [ ] Concurrency limits for LLM calls
- [ ] OTEL tracing for all tool executions

**Priority**: High  
**Dependencies**: P2-E2-S5, P2-E3-S1

---

#### P2-E7-S2: Agent Integration Tests

Create tests for agent functionality.

**Acceptance Criteria**:
- [ ] Test agent workflow execution
- [ ] Mock LLM responses for deterministic tests
- [ ] Test each tool individually
- [ ] Test multi-turn conversations
- [ ] Test error handling and recovery

**Priority**: High  
**Dependencies**: P2-E7-S1

---

## Phase 3: Search Experience

### Epic 3.1: Search API

#### P3-E1-S1: Unified Search Endpoint

Create search API with multiple modes.

**Acceptance Criteria**:
- [ ] POST /api/v1/search with mode parameter
- [ ] Modes: semantic, keyword, hybrid, graph
- [ ] Common response format across modes
- [ ] Pagination with cursor-based navigation
- [ ] Facet counts in response
- [ ] Search query logging for analytics

**Priority**: High  
**Dependencies**: P2-E2-S1, P2-E2-S2

---

#### P3-E1-S2: Semantic Search Implementation

Implement vector-based semantic search.

**Acceptance Criteria**:
- [ ] Generate embedding for query
- [ ] Search Qdrant with filters
- [ ] Hybrid dense + sparse search option
- [ ] Re-ranking with cross-encoder (optional)
- [ ] Return chunks with relevance scores
- [ ] Include document metadata

**Priority**: High  
**Dependencies**: P3-E1-S1

---

#### P3-E1-S3: Keyword Search Implementation

Implement full-text keyword search.

**Acceptance Criteria**:
- [ ] Search MeiliSearch index
- [ ] Support exact phrase matching
- [ ] Faceted results (document type, date range)
- [ ] Highlight matching terms in results
- [ ] Typo tolerance configuration

**Priority**: High  
**Dependencies**: P3-E1-S1

---

#### P3-E1-S4: Graph Search Implementation

Implement knowledge graph exploration.

**Acceptance Criteria**:
- [ ] Entity search by name
- [ ] Relationship traversal queries
- [ ] Path finding between entities
- [ ] Community/cluster exploration
- [ ] Return graph structure for visualization

**Priority**: High  
**Dependencies**: P3-E1-S1

---

#### P3-E1-S5: Search Result Aggregation

Create unified result aggregation.

**Acceptance Criteria**:
- [ ] Merge results from multiple sources
- [ ] De-duplicate overlapping results
- [ ] Consistent relevance scoring
- [ ] Group by document option
- [ ] Snippet generation for display

**Priority**: Medium  
**Dependencies**: P3-E1-S2, P3-E1-S3, P3-E1-S4

---

### Epic 3.2: Search UI

#### P3-E2-S0: Project Management UI

Create project management interface.

**Acceptance Criteria**:
- [ ] Projects list page with cards (name, description, doc count, last activity)
- [ ] Create project modal (name, description)
- [ ] Project settings page (rename, archive)
- [ ] Project selector in header (dropdown, persisted in session/cookie)
- [ ] "All Projects" / Global toggle in sidebar
- [ ] HTMX-driven project switching (no full page reload)
- [ ] URL structure: /projects/{slug}/files, /projects/{slug}/chat, etc.

**Priority**: High  
**Dependencies**: P2-E6-S1

---

#### P3-E2-S1: Search Page Layout

Create main search interface.

**Acceptance Criteria**:
- [ ] Search input with type-ahead suggestions
- [ ] Filter sidebar (facets)
- [ ] Results area with list/card toggle
- [ ] Pagination controls
- [ ] Sort options (relevance, date, title)

**Priority**: High  
**Dependencies**: P3-E1-S1

---

#### P3-E2-S2: Faceted Filtering UI

Implement interactive facet filters.

**Acceptance Criteria**:
- [ ] Document type filter (checkboxes)
- [ ] Date range filter (date picker)
- [ ] Entity type filter
- [ ] Source filter
- [ ] Active filter pills with remove button
- [ ] HTMX for dynamic filter updates

**Priority**: High  
**Dependencies**: P3-E2-S1

---

#### P3-E2-S3: Search Result Cards

Create result display components.

**Acceptance Criteria**:
- [ ] Result card with title, snippet, metadata
- [ ] Highlighted search terms
- [ ] Entity badges on cards
- [ ] Quick actions: view, expand, download
- [ ] Relevance indicator

**Priority**: High  
**Dependencies**: P3-E2-S1

---

#### P3-E2-S4: Document Preview Panel

Create side panel for document preview.

**Acceptance Criteria**:
- [ ] Slide-in panel on result click
- [ ] Document metadata display
- [ ] Full text preview (paginated for long docs)
- [ ] Entity list from document
- [ ] Related documents section
- [ ] "Ask about this" button (opens chat)

**Priority**: Medium  
**Dependencies**: P3-E2-S3

---

### Epic 3.3: Graph Exploration UI

#### P3-E3-S1: Entity Search Interface

Create interface for entity exploration.

**Acceptance Criteria**:
- [ ] Entity search input
- [ ] Entity type filter
- [ ] Entity cards with details
- [ ] Click to explore relationships
- [ ] Entity merge/link UI (admin)

**Priority**: High  
**Dependencies**: P3-E1-S4

---

#### P3-E3-S2: Graph Visualization Component

Create interactive graph visualization.

**Acceptance Criteria**:
- [ ] Force-directed graph layout (vis.js or similar)
- [ ] Node colors by entity type
- [ ] Edge labels for relationship types
- [ ] Zoom and pan controls
- [ ] Node click to expand connections
- [ ] Node details on hover
- [ ] No JavaScript build step (use CDN or pre-built)

**Priority**: High  
**Dependencies**: P3-E3-S1

---

#### P3-E3-S3: Relationship Exploration Panel

Create panel for exploring entity relationships.

**Acceptance Criteria**:
- [ ] Selected entity details
- [ ] List of relationships (incoming/outgoing)
- [ ] Filter relationships by type
- [ ] Source documents for each relationship
- [ ] Timeline of entity mentions

**Priority**: Medium  
**Dependencies**: P3-E3-S2

---

#### P3-E3-S4: Path Finder Interface

Create interface for finding paths between entities.

**Acceptance Criteria**:
- [ ] Two entity selectors (start, end)
- [ ] Find shortest path button
- [ ] Display path as linear visualization
- [ ] Show intermediate entities and relationships
- [ ] List source documents along path

**Priority**: Medium  
**Dependencies**: P3-E3-S2

---

### Epic 3.4: File Management

#### P3-E4-S1: Document Library Page

Create document management interface.

**Acceptance Criteria**:
- [ ] List/grid view of all documents
- [ ] Sort by name, date, type, status
- [ ] Filter by status, type, date
- [ ] Bulk selection for batch operations
- [ ] Upload button (opens upload modal)

**Priority**: High  
**Dependencies**: P3-E2-S0

---

#### P3-E4-S2: Document Details Page

Create individual document view.

**Acceptance Criteria**:
- [ ] Document metadata (name, type, dates, size)
- [ ] Processing status and history
- [ ] Extracted entities list
- [ ] Chunk preview (paginated)
- [ ] Re-process button
- [ ] Delete button with confirmation

**Priority**: High  
**Dependencies**: P3-E4-S1

---

#### P3-E4-S3: Upload Interface

Create file upload component.

**Acceptance Criteria**:
- [ ] Drag-and-drop zone
- [ ] Multiple file selection
- [ ] Upload progress bars
- [ ] File validation feedback
- [ ] URL input tab for URL ingestion
- [ ] Auto-start ingestion on upload

**Priority**: High  
**Dependencies**: P3-E4-S1

---

#### P3-E4-S4: Batch Operations

Implement batch document operations.

**Acceptance Criteria**:
- [ ] Select multiple documents
- [ ] Bulk delete with confirmation
- [ ] Bulk re-process
- [ ] Export selected as ZIP
- [ ] Batch add tags/labels

**Priority**: Medium  
**Dependencies**: P3-E4-S1

---

#### P3-E4-S5: Cross-Project Document Linking UI

Create UI for linking documents across projects.

**Acceptance Criteria**:
- [ ] On document detail page: "Also in projects: [list]" with add button
- [ ] "Link to project" modal with project picker
- [ ] "Unlink from project" with confirmation
- [ ] Bulk link: select multiple docs → "Add to project" action

**Priority**: Medium  
**Dependencies**: P3-E4-S2

---

## Phase 4: Polish & Advanced Features

### Epic 4.1: Production Hardening

#### P4-E1-S1: Authentication and Authorization

Implement user authentication.

**Acceptance Criteria**:
- [ ] API key authentication for API access
- [ ] Session-based auth for UI
- [ ] Role-based access control (admin, user, viewer)
- [ ] API key management UI
- [ ] Audit logging for sensitive operations

**Priority**: High  
**Dependencies**: P3-E4-S1

---

#### P4-E1-S2: Rate Limiting and Quotas

Implement rate limiting.

**Acceptance Criteria**:
- [ ] Request rate limiting per API key
- [ ] LLM token quota tracking
- [ ] Storage quota per user/tenant
- [ ] Quota warning notifications
- [ ] Admin quota management UI

**Priority**: High  
**Dependencies**: P4-E1-S1

---

#### P4-E1-S3: Error Handling and Resilience

Improve error handling across system.

**Acceptance Criteria**:
- [ ] Consistent error response format
- [ ] User-friendly error messages
- [ ] Error tracking (Sentry integration optional)
- [ ] Circuit breaker for external services
- [ ] Graceful degradation when services unavailable

**Priority**: High  
**Dependencies**: P4-E1-S1

---

#### P4-E1-S4: Performance Optimization

Optimize system performance.

**Acceptance Criteria**:
- [ ] Database query optimization (indexes, query plans)
- [ ] Caching layer (Redis) for frequent queries
- [ ] Connection pooling tuning
- [ ] Response compression
- [ ] Performance benchmarks documented

**Priority**: Medium  
**Dependencies**: P4-E1-S3

---

### Epic 4.2: Kubernetes Deployment

#### P4-E2-S1: Helm Chart Creation

Create Helm chart for platform deployment.

**Acceptance Criteria**:
- [ ] Helm chart with all components
- [ ] values.yaml with sensible defaults
- [ ] Configurable resource limits
- [ ] Secret management (external-secrets optional)
- [ ] Health checks and readiness probes
- [ ] Horizontal Pod Autoscaler for workers

**Priority**: High  
**Dependencies**: P4-E1-S3

---

#### P4-E2-S2: gVisor Node Pool Configuration

Document and configure gVisor nodes.

**Acceptance Criteria**:
- [ ] Node label for gVisor capability
- [ ] RuntimeClass configuration
- [ ] Node pool sizing recommendations
- [ ] Installation instructions (GKE, EKS, self-managed)
- [ ] Testing procedure for gVisor functionality

**Priority**: High  
**Dependencies**: P4-E2-S1

---

#### P4-E2-S3: Database Operators/StatefulSets

Configure stateful services for K8s.

**Acceptance Criteria**:
- [ ] Neo4j StatefulSet or Helm chart
- [ ] Qdrant StatefulSet with persistence
- [ ] PostgreSQL via operator (CloudNativePG) or Helm
- [ ] MinIO operator or StatefulSet
- [ ] Backup CronJobs for all databases

**Priority**: High  
**Dependencies**: P4-E2-S1

---

#### P4-E2-S4: Ingress and TLS Configuration

Configure external access.

**Acceptance Criteria**:
- [ ] Ingress resource for API and UI
- [ ] TLS certificate management (cert-manager)
- [ ] Internal service mesh (optional, for mTLS)
- [ ] Network policies for isolation

**Priority**: High  
**Dependencies**: P4-E2-S1

---

### Epic 4.3: Advanced Ingestion Features

#### P4-E3-S1: Incremental Re-indexing

Support updating existing documents.

**Acceptance Criteria**:
- [ ] Detect document changes (hash comparison)
- [ ] Update only changed chunks
- [ ] Preserve existing entity links where valid
- [ ] Re-run relationship extraction
- [ ] Update search indexes incrementally

**Priority**: Medium  
**Dependencies**: P4-E1-S3

---

#### P4-E3-S2: Scheduled Web Scraping

Support recurring web source monitoring.

**Acceptance Criteria**:
- [ ] Define URL patterns for monitoring
- [ ] Schedule: daily, weekly, custom cron
- [ ] Change detection (don't re-process unchanged)
- [ ] New content notifications
- [ ] Source management UI

**Priority**: Low  
**Dependencies**: P4-E3-S1

---

#### P4-E3-S3: Document Collections/Projects

Support grouping documents into collections.

**Acceptance Criteria**:
- [ ] Create/edit/delete collections
- [ ] Add documents to collections
- [ ] Collection-scoped search
- [ ] Collection access permissions
- [ ] Export collection as dataset

**Priority**: Medium  
**Dependencies**: P4-E1-S1

---

### Epic 4.4: Advanced Agent Features

#### P4-E4-S1: Multi-Agent Research Workflows

Support multi-agent collaboration.

**Acceptance Criteria**:
- [ ] Agent handoff between specialized agents
- [ ] Research planner agent
- [ ] Entity specialist agent
- [ ] Timeline analyst agent
- [ ] Summary writer agent
- [ ] Configurable agent team composition

**Priority**: Low  
**Dependencies**: P2-E7-S1

---

#### P4-E4-S2: Persistent Agent Memory

Implement long-term agent memory.

**Acceptance Criteria**:
- [ ] Store important facts discovered during research
- [ ] Memory retrieval tool for agents
- [ ] Memory management UI (view, edit, delete)
- [ ] Memory scoped to conversation or global

**Priority**: Low  
**Dependencies**: P4-E4-S1

---

#### P4-E4-S3: Custom Tool Configuration

Allow users to configure/extend tools.

**Acceptance Criteria**:
- [ ] Admin UI for tool configuration
- [ ] Enable/disable tools per conversation
- [ ] Tool parameter customization
- [ ] Custom prompt injection for tools

**Priority**: Low  
**Dependencies**: P4-E4-S1

---

### Epic 4.5: Export and Reporting

#### P4-E5-S1: Research Report Generation

Create automated research reports.

**Acceptance Criteria**:
- [ ] Generate report from conversation
- [ ] Include citations and sources
- [ ] Configurable report templates
- [ ] Export as PDF/Markdown/DOCX
- [ ] Include visualizations (charts, graphs)

**Priority**: Medium  
**Dependencies**: P2-E6-S5

---

#### P4-E5-S2: Data Export

Support bulk data export.

**Acceptance Criteria**:
- [ ] Export all documents as ZIP
- [ ] Export entities as CSV/JSON
- [ ] Export knowledge graph as GraphML/JSON
- [ ] Selective export (by project, date range)
- [ ] Background job for large exports

**Priority**: Medium  
**Dependencies**: P4-E5-S1

---

#### P4-E5-S3: Audit Trail

Implement comprehensive audit logging.

**Acceptance Criteria**:
- [ ] Log all data access and modifications
- [ ] Searchable audit log UI
- [ ] Export audit logs
- [ ] Retention policy configuration
- [ ] Compliance report generation

**Priority**: Medium  
**Dependencies**: P4-E1-S1

---

## Summary

| Phase | Epics | Stories |
|-------|-------|---------|
| Phase 0 | 6 | 21 |
| Phase 1 | 6 | 23 |
| Phase 2 | 7 | 22 |
| Phase 3 | 4 | 17 |
| Phase 4 | 5 | 14 |
| **Total** | **28** | **97** |
