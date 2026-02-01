# Greenfield RAG/GraphRAG Document Research Platform: Implementation Plan

**Bottom Line Up Front**: This comprehensive implementation plan delivers a document research platform for investigative journalism in four phases over approximately 20-24 weeks. The recommended technology stack includes **Neo4j Community Edition** for graph storage, **vLLM** for LLM serving, and **gVisor** for secure agent code execution. The Temporal + OpenAI Agent SDK integration provides durable, observable AI workflows with automatic retry and state persistence.

---

## Architecture Overview

The platform follows a modular, event-driven architecture with clear separation between ingestion, retrieval, and agent execution layers. All components communicate through Temporal workflows for durability and observability.

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                       │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│   │   HTMX + TW     │   │   File Upload   │   │   Search UI     │          │
│   │   Chat UI       │   │   Interface     │   │   + Graph Viz   │          │
│   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘          │
└────────────┼──────────────────────┼──────────────────────┼──────────────────┘
             │                      │                      │
             ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API LAYER (FastAPI)                                  │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│   │  Chat/Agent     │   │  Ingestion      │   │  Search         │          │
│   │  Endpoints      │   │  Endpoints      │   │  Endpoints      │          │
│   │  (SSE Stream)   │   │  (Upload/URL)   │   │  (Vector+FTS)   │          │
│   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘          │
└────────────┼──────────────────────┼──────────────────────┼──────────────────┘
             │                      │                      │
             ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL ORCHESTRATION LAYER                              │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    Temporal Server                                │      │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │      │
│   │  │  Ingestion   │  │    Agent     │  │    Query     │           │      │
│   │  │  Workflows   │  │  Workflows   │  │  Workflows   │           │      │
│   │  └──────────────┘  └──────────────┘  └──────────────┘           │      │
│   └─────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
             │                      │                      │
             ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORKER LAYER                                         │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│   │  Ingestion      │   │  Agent          │   │  Query          │          │
│   │  Worker         │   │  Worker         │   │  Worker         │          │
│   │  ───────────    │   │  ───────────    │   │  ───────────    │          │
│   │  • Docling OCR  │   │  • OpenAI SDK   │   │  • Vector Search│          │
│   │  • Transcription│   │  • Tool Exec    │   │  • Graph Query  │          │
│   │  • Chunking     │   │  • Sandbox      │   │  • Reranking    │          │
│   │  • Embedding    │   │    (gVisor)     │   │                 │          │
│   │  • Entity Extr. │   │                 │   │                 │          │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
             │                      │                      │
             ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                            │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│   │  MinIO   │  │  Qdrant  │  │  Neo4j   │  │MeiliSearch│  │ Postgres │    │
│   │ (Objects)│  │ (Vectors)│  │ (Graph)  │  │  (FTS)   │  │(Metadata)│    │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM SERVING (vLLM)                                    │
│   ┌─────────────────────────────┐   ┌─────────────────────────────┐        │
│   │  Chat/Extraction Model      │   │  Embedding Model            │        │
│   │  (Llama 3.1 8B/70B)         │   │  (BGE-large-en-v1.5)        │        │
│   └─────────────────────────────┘   └─────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Document Ingestion

```
Upload/URL → MinIO (raw) → Classification Activity
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              OCR (Docling)   Transcription    Object Detection
                    │          (Parakeet)        (YOLO)
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
                         Text Extraction Result
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Chunking      Entity Extraction   Relationship
            (Semantic)      (spaCy + LLM)      Extraction (LLM)
                    │               │               │
                    ▼               ▼               ▼
              Qdrant          Entity            Neo4j
            (embeddings)    Resolution      (knowledge graph)
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
                              MeiliSearch
                            (full-text index)
```

---

## Key Architectural Decisions

### 1. Graph Database: Neo4j Community Edition

**Recommendation**: Neo4j Community Edition for primary deployment.

**Rationale**:
- **Best-in-class GraphRAG ecosystem**: Native `neo4j-graphrag` Python package, first-class LangChain/LlamaIndex integration, Microsoft GraphRAG support
- **Mature and proven**: Largest graph database community, extensive documentation, GraphAcademy training
- **Cypher proficiency**: Full Cypher query language enables complex investigative queries (connection tracing, timeline analysis)
- **Scale appropriate**: Handles millions of entities/relationships on single node, matching expected workload
- **Docker/K8s ready**: Official Helm charts, well-documented deployment patterns

**Trade-offs**:
- No clustering in Community Edition (acceptable for initial scale)
- Prometheus metrics require Enterprise (mitigate with application-level OTEL instrumentation)
- Offline-only backups (acceptable with proper scheduling)

**Alternative**: FalkorDB for ultra-low latency requirements or multi-tenant scenarios with 10K+ separate graphs.

### 2. LLM Serving: vLLM

**Recommendation**: vLLM for all LLM serving needs.

**Rationale**:
- **Complete OpenAI API compatibility**: Seamless integration with OpenAI Agent SDK, function calling, streaming
- **High-throughput batch processing**: Essential for document ingestion pipeline; PagedAttention enables 2-4x throughput improvement
- **Native embedding support**: Serve both chat and embedding models (BGE, E5) from unified infrastructure
- **Production observability**: Built-in Prometheus metrics, OpenTelemetry support, official Grafana dashboards
- **Kubernetes-native**: Official Helm charts and production stack with autoscaling

**Configuration Strategy**:
- **Chat model**: Llama 3.1 8B-Instruct (or 70B with tensor parallelism for higher quality)
- **Embedding model**: BAAI/bge-large-en-v1.5 (1024 dimensions, excellent retrieval performance)
- **Quantization**: AWQ for chat model (fast Marlin kernels), FP16 for embeddings

### 3. Sandboxing: gVisor (runsc)

**Recommendation**: gVisor for secure AI agent code execution.

**Rationale**:
- **Kubernetes native**: Full RuntimeClass support, works seamlessly with existing K8s tooling
- **Docker Compose compatible**: Simple `runtime: runsc` for local development
- **No hardware requirements**: Works on any Linux host (unlike Kata/Firecracker requiring KVM)
- **Proven at scale**: Powers Google Cloud Run, App Engine, Cloud Functions
- **Fast startup**: 50-100ms, suitable for on-demand code execution
- **Python compatibility**: Excellent support for Python and data science libraries

**Implementation**:
- RuntimeClass `gvisor` for sandboxed agent execution pods
- Read-only root filesystem with tmpfs for working directories
- Network policies for controlled/isolated network access
- cgroups resource limits (CPU, memory, time)

### 4. Temporal + OpenAI Agent SDK Integration

**Pattern**: Use `temporalio.contrib.openai_agents.OpenAIAgentsPlugin` for seamless integration.

**Key Benefits**:
- Every agent invocation automatically executes as a Temporal Activity
- Built-in retry logic for LLM API rate limits and failures
- Crash recovery picks up exactly where execution stopped
- Tools become Activities via `activity_as_tool()` helper

---

## Phase Breakdown

### Phase 0: Project Scaffolding and Infrastructure (Weeks 1-3)

**Goal**: Establish development environment, CI/CD, and core infrastructure services.

#### Epic 0.1: Repository and Development Environment Setup

**Story 0.1.1: Initialize monorepo structure**
```
Task: Create project repository with monorepo structure
Acceptance Criteria:
- Root directory with shared configuration (pyproject.toml, .pre-commit-config.yaml)
- /api directory for FastAPI application
- /workers directory for Temporal workers (ingestion, agent, query)
- /frontend directory for HTMX templates
- /infra directory for Docker, K8s manifests, Tilt configuration
- /libs directory for shared Python packages
- README.md with project overview and setup instructions
- .gitignore configured for Python, Node, IDE files
```

**Story 0.1.2: Configure Python development environment**
```
Task: Set up Python tooling with uv, ruff, mypy
Acceptance Criteria:
- pyproject.toml with project metadata and dependencies
- uv.lock file for reproducible builds
- ruff configuration for linting and formatting
- mypy configuration with strict mode
- pre-commit hooks for code quality
- VS Code/PyCharm configuration files
```

**Story 0.1.3: Create Docker base images**
```
Task: Build optimized Docker images for Python services
Acceptance Criteria:
- Base Python image with common dependencies (python:3.12-slim)
- API Dockerfile with FastAPI, uvicorn
- Worker Dockerfile with Temporal SDK, ML dependencies
- Multi-stage builds for smaller production images
- Non-root user configuration
- Health check endpoints
```

#### Epic 0.2: Infrastructure Services (Docker Compose)

**Story 0.2.1: PostgreSQL setup with schema**
```
Task: Configure PostgreSQL for application metadata
Acceptance Criteria:
- Docker Compose service for PostgreSQL 16
- Volume mount for data persistence
- Initial schema migration with Alembic
- Tables: documents, ingestion_jobs, users, api_keys
- Connection pooling configuration
- Health check endpoint
```

**Story 0.2.2: MinIO object storage setup**
```
Task: Configure MinIO for document storage
Acceptance Criteria:
- Docker Compose service for MinIO
- Buckets: raw-documents, processed-documents, exports
- IAM policies for service accounts
- Console accessible on port 9001
- Event notifications configured for new uploads
- Pre-signed URL generation utility functions
```

**Story 0.2.3: Qdrant vector database setup**
```
Task: Configure Qdrant for vector storage
Acceptance Criteria:
- Docker Compose service for Qdrant
- Collections: document_chunks (dense + sparse vectors)
- Payload indexes for filtering (document_id, tenant_id, document_type)
- Quantization configuration (INT8 scalar)
- gRPC and REST endpoints exposed
- Snapshot configuration for backups
```

**Story 0.2.4: Neo4j graph database setup**
```
Task: Configure Neo4j Community Edition
Acceptance Criteria:
- Docker Compose service for Neo4j 5.x
- APOC plugin installed and configured
- Initial constraints: unique entity IDs, document IDs
- Indexes on: Entity.name, Entity.type, Document.id
- Browser accessible on port 7474
- Bolt protocol on port 7687
```

**Story 0.2.5: MeiliSearch setup**
```
Task: Configure MeiliSearch for full-text search
Acceptance Criteria:
- Docker Compose service for MeiliSearch
- Index: documents with searchable, filterable attributes
- Facets configured: document_type, source, date_range
- API key management
- Analytics disabled for privacy
```

**Story 0.2.6: Temporal server setup**
```
Task: Configure Temporal server for workflow orchestration
Acceptance Criteria:
- Docker Compose services: temporal, temporal-admin-tools, temporal-ui
- PostgreSQL backend for Temporal persistence
- Namespaces: default, ingestion, agents
- UI accessible on port 8080
- CLI (tctl) available in admin-tools container
```

#### Epic 0.3: vLLM Serving Infrastructure

**Story 0.3.1: vLLM chat model deployment**
```
Task: Deploy vLLM server for chat/extraction model
Acceptance Criteria:
- Docker Compose service with NVIDIA runtime
- Model: meta-llama/Llama-3.1-8B-Instruct (or configurable)
- OpenAI-compatible API on port 8000
- Prometheus metrics exposed
- Environment variables for model configuration
- GPU memory allocation (tensor-parallel-size if multi-GPU)
```

**Story 0.3.2: vLLM embedding model deployment**
```
Task: Deploy vLLM server for embedding generation
Acceptance Criteria:
- Separate Docker Compose service
- Model: BAAI/bge-large-en-v1.5
- Task mode: embed
- Port 8001 (separate from chat)
- Batch processing optimized
```

**Story 0.3.3: LLM client abstraction layer**
```
Task: Create Python client for LLM services
Acceptance Criteria:
- LLMClient class wrapping OpenAI Python SDK
- Configuration via environment variables (base_url, api_key)
- Retry logic with exponential backoff
- Async methods for chat_completion, embeddings
- Structured output support (Pydantic response parsing)
- Token counting utility
```

#### Epic 0.4: Observability Stack

**Story 0.4.1: OpenTelemetry Collector setup**
```
Task: Deploy OTEL Collector for traces, metrics, logs
Acceptance Criteria:
- Docker Compose service for otel-collector
- Receivers: OTLP (gRPC 4317, HTTP 4318)
- Exporters: Jaeger (traces), Prometheus (metrics)
- Batch processor configured
- Service pipeline definitions
```

**Story 0.4.2: Jaeger tracing setup**
```
Task: Deploy Jaeger for distributed tracing
Acceptance Criteria:
- Docker Compose service for Jaeger all-in-one
- UI accessible on port 16686
- OTLP ingestion enabled
- Retention policy configured
```

**Story 0.4.3: Prometheus and Grafana setup**
```
Task: Deploy monitoring stack
Acceptance Criteria:
- Prometheus with scrape configs for all services
- Grafana with pre-configured dashboards
- Dashboards: vLLM metrics, Temporal metrics, API latency
- Alert rules for critical failures
- Grafana datasources auto-provisioned
```

**Story 0.4.4: FastAPI OTEL instrumentation**
```
Task: Instrument FastAPI with OpenTelemetry
Acceptance Criteria:
- opentelemetry-instrumentation-fastapi configured
- Custom spans for business logic
- Request/response hooks for user context
- Excluded paths: /health, /metrics
- Trace context propagation headers
```

#### Epic 0.5: Tilt Development Environment

**Story 0.5.1: Tiltfile for local development**
```
Task: Create Tilt configuration for local K8s development
Acceptance Criteria:
- Tiltfile loading all K8s manifests
- docker_build with live_update for Python services
- Resource dependencies (DBs start before workers)
- Port forwards for all services
- Labels for UI organization
```

**Story 0.5.2: Custom Tilt buttons/actions**
```
Task: Add custom actions for development workflows
Acceptance Criteria:
- Button: "Clear All Data" - drops DB schemas, clears MinIO
- Button: "Run Migrations" - executes Alembic migrations
- Button: "Seed Test Data" - loads sample documents
- Button: "Reset Temporal" - clears workflow history
- All actions with auto_init=False, TRIGGER_MODE_MANUAL
```

**Story 0.5.3: Docker Compose fallback**
```
Task: Ensure project runs without Tilt
Acceptance Criteria:
- docker-compose.yml with all services
- docker-compose.override.yml for development extras
- Makefile with common commands (up, down, logs, test)
- README documenting both Tilt and Docker Compose workflows
```

#### Epic 0.6: CI/CD Pipeline

**Story 0.6.1: GitHub Actions for testing**
```
Task: Create CI pipeline for testing and linting
Acceptance Criteria:
- Workflow triggers on PR and push to main
- Steps: lint (ruff), type check (mypy), unit tests (pytest)
- Service containers for PostgreSQL, Qdrant
- Test coverage reporting
- Parallel job execution
```

**Story 0.6.2: Docker image build pipeline**
```
Task: Create pipeline for building and pushing images
Acceptance Criteria:
- Build on tag push (semver)
- Multi-platform builds (amd64, arm64)
- Push to container registry (ghcr.io or configurable)
- Image scanning with Trivy
- Build caching for faster builds
```

---

### Phase 1: Ingestion Pipeline (Weeks 4-9)

**Goal**: Implement complete document ingestion workflow with OCR, transcription, entity extraction, and indexing.

#### Epic 1.1: Core Ingestion Infrastructure

**Story 1.1.1: Document model and database schema**
```
Task: Define document data model and persistence layer
Acceptance Criteria:
- SQLAlchemy models: Document, IngestionJob, Chunk, Entity
- Alembic migrations for all tables
- Document states: pending, processing, completed, failed
- IngestionJob with status, progress, error tracking
- Chunk with vector_id, text, metadata
- Repository pattern for database operations
```

**Story 1.1.2: File upload API endpoint**
```
Task: Create FastAPI endpoint for file uploads
Acceptance Criteria:
- POST /api/v1/documents/upload accepting multipart/form-data
- File validation (size limits, allowed types)
- Upload to MinIO raw-documents bucket
- Create Document and IngestionJob records
- Return job_id for status tracking
- Support for batch upload (multiple files)
```

**Story 1.1.3: URL ingestion API endpoint**
```
Task: Create endpoint for URL-based ingestion
Acceptance Criteria:
- POST /api/v1/documents/ingest-url accepting URL
- URL validation and sanitization
- Support for: web pages, direct file links, YouTube URLs
- Create Document record with source_url
- Start ingestion workflow
```

**Story 1.1.4: Ingestion job status API**
```
Task: Create endpoint for tracking ingestion progress
Acceptance Criteria:
- GET /api/v1/jobs/{job_id} returning job status
- Include: status, progress percentage, current step, errors
- GET /api/v1/jobs for listing all jobs with pagination
- WebSocket/SSE endpoint for real-time updates
```

#### Epic 1.2: Temporal Ingestion Workflow

**Story 1.2.1: Document ingestion workflow definition**
```
Task: Create main Temporal workflow for document ingestion
Acceptance Criteria:
- @workflow.defn class DocumentIngestionWorkflow
- Input: document_id, source (file_path or url)
- Workflow steps: classify → download → process → extract → index
- Query methods: get_progress(), get_status()
- Signal methods: cancel(), pause()
- Error handling with compensation (cleanup on failure)
```

**Story 1.2.2: Document classification activity**
```
Task: Create activity to classify document type
Acceptance Criteria:
- @activity.defn async def classify_document
- Input: document_id, file_path in MinIO
- Detect type: pdf, docx, xlsx, csv, image, audio, video, web_page
- Use magic bytes and file extension
- Return DocumentType enum
- Store classification in Document record
```

**Story 1.2.3: File download activity**
```
Task: Create activity to download files from URLs
Acceptance Criteria:
- @activity.defn async def download_file
- Support HTTP/HTTPS URLs with proper headers
- YouTube download via yt-dlp
- Heartbeat during large downloads
- Store in MinIO raw-documents bucket
- Return MinIO object path
```

**Story 1.2.4: Parallel processing orchestration**
```
Task: Implement fan-out/fan-in for parallel processing
Acceptance Criteria:
- Use asyncio.gather for parallel activity execution
- Parallel tasks: OCR, transcription, object detection (based on type)
- Aggregate results after all complete
- Handle partial failures (continue if some succeed)
- Configurable parallelism limits
```

#### Epic 1.3: Document Processing Activities

**Story 1.3.1: Docling OCR activity**
```
Task: Create activity for document parsing with Docling
Acceptance Criteria:
- @activity.defn async def process_with_docling
- Input: document bytes or MinIO path
- Configure: do_ocr=True, do_table_structure=True
- Support: PDF, DOCX, PPTX, images
- Output: extracted text, table data, document structure
- Export to both Markdown and JSON formats
- Heartbeat during processing
```

**Story 1.3.2: Audio transcription activity (Parakeet)**
```
Task: Create activity for audio/video transcription
Acceptance Criteria:
- @activity.defn async def transcribe_audio
- Use NVIDIA Parakeet model via API or local inference
- Extract audio from video files (ffmpeg)
- Support: WAV, MP3, MP4, WebM
- Output: transcribed text with timestamps
- Word-level or segment-level timestamps
- Heartbeat during long transcriptions
```

**Story 1.3.3: Object detection activity (YOLO)**
```
Task: Create activity for image/video object detection
Acceptance Criteria:
- @activity.defn async def detect_objects
- Use YOLOv8 or YOLOv9 model
- Extract frames from video at configurable intervals
- Detect: people, objects, text regions, faces (configurable)
- Output: detected objects with bounding boxes, confidence
- Store detected frame images in MinIO
```

**Story 1.3.4: Web scraping activity (Playwright)**
```
Task: Create activity for web page extraction
Acceptance Criteria:
- @activity.defn async def scrape_webpage
- Use Playwright for JavaScript-rendered pages
- Extract: main content, metadata, links, images
- Handle pagination for multi-page content
- Respect robots.txt (configurable)
- Screenshot capture option
- Store extracted content and assets in MinIO
```

#### Epic 1.4: Chunking and Embedding

**Story 1.4.1: Semantic chunking activity**
```
Task: Create activity for intelligent document chunking
Acceptance Criteria:
- @activity.defn async def chunk_document
- Hierarchical chunking preserving document structure
- Semantic boundaries using sentence embeddings
- Configurable chunk size (default 500 tokens)
- 10-20% overlap between chunks
- Preserve metadata: section headers, page numbers
- Output: list of Chunk objects with text and metadata
```

**Story 1.4.2: Embedding generation activity**
```
Task: Create activity for generating embeddings
Acceptance Criteria:
- @activity.defn async def generate_embeddings
- Batch processing for efficiency
- Use vLLM embedding endpoint
- Generate both dense (BGE) and sparse (BM25 tokens) vectors
- Return list of embeddings aligned with chunks
- Retry logic for API failures
```

**Story 1.4.3: Qdrant indexing activity**
```
Task: Create activity to store vectors in Qdrant
Acceptance Criteria:
- @activity.defn async def index_to_qdrant
- Batch upsert to Qdrant collection
- Store both dense and sparse vectors
- Payload: document_id, chunk_index, text preview, metadata
- Idempotent: use deterministic point IDs
- Return count of indexed vectors
```

**Story 1.4.4: MeiliSearch indexing activity**
```
Task: Create activity to index in MeiliSearch
Acceptance Criteria:
- @activity.defn async def index_to_meilisearch
- Index document metadata and chunk text
- Configure searchable attributes: title, content, entities
- Filterable attributes: document_type, date, source
- Wait for indexing task completion
- Return indexed document count
```

#### Epic 1.5: Entity Extraction and Knowledge Graph

**Story 1.5.1: Entity extraction activity (hybrid NER)**
```
Task: Create activity for entity extraction
Acceptance Criteria:
- @activity.defn async def extract_entities
- Hybrid approach: spaCy NER + LLM refinement
- Entity types: PERSON, ORGANIZATION, LOCATION, DATE, EVENT, MONEY
- Confidence scores for each entity
- Coreference resolution to link mentions
- Output: list of Entity objects with type, mentions, confidence
```

**Story 1.5.2: Relationship extraction activity**
```
Task: Create activity for relationship discovery
Acceptance Criteria:
- @activity.defn async def extract_relationships
- LLM-based extraction with structured output (Pydantic)
- Relationship types: WORKS_AT, AFFILIATED_WITH, FUNDED_BY, PARTICIPATED_IN, etc.
- Source-target entity pairs with relationship type
- Confidence scoring
- Provenance tracking (source chunk)
```

**Story 1.5.3: Entity resolution activity**
```
Task: Create activity for entity deduplication
Acceptance Criteria:
- @activity.defn async def resolve_entities
- Embedding-based blocking (cluster similar entities)
- LLM-assisted matching for ambiguous cases
- Create SAME_AS relationships for duplicates
- Merge entity attributes from multiple sources
- Return canonical entity list
```

**Story 1.5.4: Neo4j graph construction activity**
```
Task: Create activity to build knowledge graph
Acceptance Criteria:
- @activity.defn async def build_knowledge_graph
- Create/merge Entity nodes with properties
- Create Relationship edges between entities
- Create Document and Chunk nodes
- MENTIONED_IN relationships linking entities to chunks
- Idempotent using MERGE operations
- Return node/edge counts
```

#### Epic 1.6: Ingestion Worker and Testing

**Story 1.6.1: Ingestion worker implementation**
```
Task: Create Temporal worker for ingestion activities
Acceptance Criteria:
- Worker class registering all ingestion activities
- Connection to Temporal server with retry
- Task queue: ingestion-task-queue
- Graceful shutdown handling
- Prometheus metrics for activity execution
- OTEL tracing interceptor configured
```

**Story 1.6.2: Integration tests for ingestion pipeline**
```
Task: Create comprehensive tests for ingestion
Acceptance Criteria:
- pytest fixtures for test documents (PDF, DOCX, image, audio)
- Test complete workflow execution with time-skipping
- Mock external services (vLLM, OCR)
- Verify data stored correctly in all datastores
- Test error handling and retry behavior
- Test idempotency (re-running same document)
```

**Story 1.6.3: Ingestion monitoring dashboard**
```
Task: Create Grafana dashboard for ingestion metrics
Acceptance Criteria:
- Metrics: jobs started, completed, failed per hour
- Processing time by document type
- Queue depth and wait times
- Error rate by activity type
- LLM token usage tracking
```

---

### Phase 2: Chat/Agent Interface (Weeks 10-14)

**Goal**: Implement conversational AI interface with tool-using agents for document research.

#### Epic 2.1: Agent Infrastructure

**Story 2.1.1: OpenAI Agent SDK integration with Temporal**
```
Task: Configure OpenAI Agent SDK with Temporal durability
Acceptance Criteria:
- OpenAIAgentsPlugin configured on Temporal client
- ModelActivityParameters with appropriate timeouts
- Pydantic data converter for serialization
- Tracing interceptor for observability
- Test agent execution survives worker restart
```

**Story 2.1.2: Agent workflow definition**
```
Task: Create Temporal workflow for agent execution
Acceptance Criteria:
- @workflow.defn class AgentExecutionWorkflow
- Input: conversation_id, user_message, context
- Agent instantiation with tools via activity_as_tool
- Streaming output via signals
- Query for current status
- Timeout handling (max execution time)
```

**Story 2.1.3: Base agent tools infrastructure**
```
Task: Create framework for agent tools as activities
Acceptance Criteria:
- Base ToolActivity class with standard interface
- Input/output Pydantic models for each tool
- Error handling and user-friendly error messages
- Tool execution timeout configuration
- Logging and tracing for tool calls
```

#### Epic 2.2: Research Agent Tools

**Story 2.2.1: Vector search tool**
```
Task: Create tool for semantic document search
Acceptance Criteria:
- @activity.defn async def search_documents
- Input: query string, optional filters (date, type, entities)
- Hybrid search: dense + sparse in Qdrant
- Return top-k chunks with relevance scores
- Include document metadata in results
- Format results for LLM consumption
```

**Story 2.2.2: Full-text search tool**
```
Task: Create tool for keyword/phrase search
Acceptance Criteria:
- @activity.defn async def fulltext_search
- Input: query string, filters
- Search MeiliSearch index
- Faceted results with counts
- Highlight matching terms
- Pagination support
```

**Story 2.2.3: Graph exploration tool**
```
Task: Create tool for knowledge graph queries
Acceptance Criteria:
- @activity.defn async def explore_graph
- Input: entity name or query description
- Generate Cypher query from natural language (LLM)
- Execute query against Neo4j
- Return entities and relationships
- Visualizable output format
```

**Story 2.2.4: Timeline construction tool**
```
Task: Create tool for building event timelines
Acceptance Criteria:
- @activity.defn async def build_timeline
- Input: entity name or topic
- Query events with temporal relationships
- Sort chronologically
- Include source documents for each event
- Output structured timeline data
```

**Story 2.2.5: Document retrieval tool**
```
Task: Create tool for fetching full documents
Acceptance Criteria:
- @activity.defn async def get_document
- Input: document_id
- Retrieve document metadata from PostgreSQL
- Retrieve full text from MinIO
- Return formatted document content
- Support pagination for large documents
```

#### Epic 2.3: Tabular Data Tools

**Story 2.3.1: CSV/Excel analysis tool**
```
Task: Create tool for analyzing tabular data
Acceptance Criteria:
- @activity.defn async def analyze_tabular
- Input: document_id (CSV/Excel), analysis query
- Load data into pandas DataFrame
- Execute analysis (aggregations, filters, stats)
- **Execute in gVisor sandbox**
- Return results as formatted table
- Support natural language queries
```

**Story 2.3.2: SQL query tool**
```
Task: Create tool for SQL queries on structured data
Acceptance Criteria:
- @activity.defn async def execute_sql
- Input: SQL query, target database/table
- Load CSVs into SQLite in-memory database
- Execute query with timeout
- **Execute in gVisor sandbox**
- Return results as formatted table
- Query validation to prevent dangerous operations
```

**Story 2.3.3: Data visualization tool**
```
Task: Create tool for generating charts
Acceptance Criteria:
- @activity.defn async def create_visualization
- Input: data, chart type, configuration
- Generate charts using matplotlib/plotly
- **Execute in gVisor sandbox**
- Save chart image to MinIO
- Return image URL for display
- Support: bar, line, pie, scatter, timeline
```

#### Epic 2.4: Sandbox Execution Environment

**Story 2.4.1: gVisor RuntimeClass configuration**
```
Task: Configure Kubernetes RuntimeClass for gVisor
Acceptance Criteria:
- RuntimeClass definition: name=gvisor, handler=runsc
- Node selector for gVisor-capable nodes
- Documentation for gVisor installation on nodes
- Test pod deployment with RuntimeClass
```

**Story 2.4.2: Sandboxed execution pod template**
```
Task: Create pod specification for sandboxed code execution
Acceptance Criteria:
- Pod template with runtimeClassName: gvisor
- Read-only root filesystem
- tmpfs mount for /tmp and working directory
- Resource limits: CPU, memory, ephemeral storage
- Network policy: deny all (or controlled egress)
- Non-root user execution
- Time limit via activeDeadlineSeconds
```

**Story 2.4.3: Sandbox executor activity**
```
Task: Create activity for running code in sandbox
Acceptance Criteria:
- @activity.defn async def execute_sandboxed
- Input: code string, language, input files
- Create ephemeral pod with gVisor runtime
- Copy input files to pod
- Execute code with timeout
- Capture stdout, stderr, output files
- Clean up pod after execution
- Return execution results
```

**Story 2.4.4: Docker Compose sandbox fallback (nsjail)**
```
Task: Create fallback sandbox for local development
Acceptance Criteria:
- Docker container with nsjail installed
- nsjail configuration for Python execution
- API endpoint for code execution
- Resource limits matching K8s configuration
- Document development vs production differences
```

#### Epic 2.5: Chat API and Streaming

**Story 2.5.1: Chat conversation API**
```
Task: Create API endpoints for chat conversations
Acceptance Criteria:
- POST /api/v1/chat/conversations - create new conversation
- GET /api/v1/chat/conversations/{id} - get conversation history
- POST /api/v1/chat/conversations/{id}/messages - send message
- DELETE /api/v1/chat/conversations/{id} - delete conversation
- Conversation storage in PostgreSQL
- Message history with tool calls and results
```

**Story 2.5.2: SSE streaming endpoint**
```
Task: Create Server-Sent Events endpoint for streaming responses
Acceptance Criteria:
- GET /api/v1/chat/conversations/{id}/stream - SSE connection
- Event types: message_delta, tool_call, tool_result, completed, error
- Heartbeat events to keep connection alive
- Graceful handling of client disconnection
- Base64 encoding for multi-line content
```

**Story 2.5.3: Agent response formatting**
```
Task: Create utilities for formatting agent output
Acceptance Criteria:
- Markdown formatting for text responses
- Structured output for tool results (tables, charts)
- File attachment formatting (links to MinIO)
- Citation formatting (source documents, chunks)
- Error message formatting
```

#### Epic 2.6: Chat UI (HTMX)

**Story 2.6.1: Chat page layout**
```
Task: Create main chat interface page
Acceptance Criteria:
- Full-page layout with sidebar and main content
- Conversation list in sidebar
- Message thread in main area
- Input area with send button
- Tailwind CSS styling
- Responsive design (mobile-friendly)
```

**Story 2.6.2: SSE streaming integration**
```
Task: Implement streaming message display
Acceptance Criteria:
- HTMX SSE extension configured
- Messages append to thread as they stream
- Typing indicator during generation
- Smooth scrolling to new messages
- Handle connection errors with retry
```

**Story 2.6.3: Tool execution display**
```
Task: Create UI components for tool execution
Acceptance Criteria:
- Collapsible tool call cards
- Show tool name, inputs, status
- Expandable tool results
- Special rendering for:
  - Tables (sortable, scrollable)
  - Charts (embedded images)
  - Documents (expandable previews)
  - Code blocks (syntax highlighting)
```

**Story 2.6.4: File and document rendering**
```
Task: Implement inline document viewing
Acceptance Criteria:
- PDF viewer (pdf.js) in modal/panel
- Markdown rendering with syntax highlighting
- Image display with zoom
- Table rendering with pagination
- Download button for all files
```

**Story 2.6.5: Conversation management UI**
```
Task: Create UI for managing conversations
Acceptance Criteria:
- Create new conversation button
- Rename conversation (inline edit)
- Delete conversation with confirmation
- Conversation search/filter
- Export conversation as Markdown
```

#### Epic 2.7: Agent Worker

**Story 2.7.1: Agent worker implementation**
```
Task: Create Temporal worker for agent activities
Acceptance Criteria:
- Worker registering all agent tools
- Task queue: agent-task-queue
- OpenAIAgentsPlugin configured
- Concurrency limits for LLM calls
- OTEL tracing for all tool executions
```

**Story 2.7.2: Agent integration tests**
```
Task: Create tests for agent functionality
Acceptance Criteria:
- Test agent workflow execution
- Mock LLM responses for deterministic tests
- Test each tool individually
- Test multi-turn conversations
- Test error handling and recovery
```

---

### Phase 3: Search Experience (Weeks 15-18)

**Goal**: Implement comprehensive search interface with vector, full-text, and graph exploration.

#### Epic 3.1: Search API

**Story 3.1.1: Unified search endpoint**
```
Task: Create search API with multiple modes
Acceptance Criteria:
- POST /api/v1/search with mode parameter
- Modes: semantic, keyword, hybrid, graph
- Common response format across modes
- Pagination with cursor-based navigation
- Facet counts in response
- Search query logging for analytics
```

**Story 3.1.2: Semantic search implementation**
```
Task: Implement vector-based semantic search
Acceptance Criteria:
- Generate embedding for query
- Search Qdrant with filters
- Hybrid dense + sparse search option
- Re-ranking with cross-encoder (optional)
- Return chunks with relevance scores
- Include document metadata
```

**Story 3.1.3: Keyword search implementation**
```
Task: Implement full-text keyword search
Acceptance Criteria:
- Search MeiliSearch index
- Support exact phrase matching
- Faceted results (document type, date range)
- Highlight matching terms in results
- Typo tolerance configuration
```

**Story 3.1.4: Graph search implementation**
```
Task: Implement knowledge graph exploration
Acceptance Criteria:
- Entity search by name
- Relationship traversal queries
- Path finding between entities
- Community/cluster exploration
- Return graph structure for visualization
```

**Story 3.1.5: Search result aggregation**
```
Task: Create unified result aggregation
Acceptance Criteria:
- Merge results from multiple sources
- De-duplicate overlapping results
- Consistent relevance scoring
- Group by document option
- Snippet generation for display
```

#### Epic 3.2: Search UI

**Story 3.2.1: Search page layout**
```
Task: Create main search interface
Acceptance Criteria:
- Search input with type-ahead suggestions
- Filter sidebar (facets)
- Results area with list/card toggle
- Pagination controls
- Sort options (relevance, date, title)
```

**Story 3.2.2: Faceted filtering UI**
```
Task: Implement interactive facet filters
Acceptance Criteria:
- Document type filter (checkboxes)
- Date range filter (date picker)
- Entity type filter
- Source filter
- Active filter pills with remove button
- HTMX for dynamic filter updates
```

**Story 3.2.3: Search result cards**
```
Task: Create result display components
Acceptance Criteria:
- Result card with title, snippet, metadata
- Highlighted search terms
- Entity badges on cards
- Quick actions: view, expand, download
- Relevance indicator
```

**Story 3.2.4: Document preview panel**
```
Task: Create side panel for document preview
Acceptance Criteria:
- Slide-in panel on result click
- Document metadata display
- Full text preview (paginated for long docs)
- Entity list from document
- Related documents section
- "Ask about this" button (opens chat)
```

#### Epic 3.3: Graph Exploration UI

**Story 3.3.1: Entity search interface**
```
Task: Create interface for entity exploration
Acceptance Criteria:
- Entity search input
- Entity type filter
- Entity cards with details
- Click to explore relationships
- Entity merge/link UI (admin)
```

**Story 3.3.2: Graph visualization component**
```
Task: Create interactive graph visualization
Acceptance Criteria:
- Force-directed graph layout (vis.js or similar)
- Node colors by entity type
- Edge labels for relationship types
- Zoom and pan controls
- Node click to expand connections
- Node details on hover
- No JavaScript build step (use CDN or pre-built)
```

**Story 3.3.3: Relationship exploration panel**
```
Task: Create panel for exploring entity relationships
Acceptance Criteria:
- Selected entity details
- List of relationships (incoming/outgoing)
- Filter relationships by type
- Source documents for each relationship
- Timeline of entity mentions
```

**Story 3.3.4: Path finder interface**
```
Task: Create interface for finding paths between entities
Acceptance Criteria:
- Two entity selectors (start, end)
- Find shortest path button
- Display path as linear visualization
- Show intermediate entities and relationships
- List source documents along path
```

#### Epic 3.4: File Management

**Story 3.4.1: Document library page**
```
Task: Create document management interface
Acceptance Criteria:
- List/grid view of all documents
- Sort by name, date, type, status
- Filter by status, type, date
- Bulk selection for batch operations
- Upload button (opens upload modal)
```

**Story 3.4.2: Document details page**
```
Task: Create individual document view
Acceptance Criteria:
- Document metadata (name, type, dates, size)
- Processing status and history
- Extracted entities list
- Chunk preview (paginated)
- Re-process button
- Delete button with confirmation
```

**Story 3.4.3: Upload interface**
```
Task: Create file upload component
Acceptance Criteria:
- Drag-and-drop zone
- Multiple file selection
- Upload progress bars
- File validation feedback
- URL input tab for URL ingestion
- Auto-start ingestion on upload
```

**Story 3.4.4: Batch operations**
```
Task: Implement batch document operations
Acceptance Criteria:
- Select multiple documents
- Bulk delete with confirmation
- Bulk re-process
- Export selected as ZIP
- Batch add tags/labels
```

---

### Phase 4: Polish and Advanced Features (Weeks 19-24)

**Goal**: Production hardening, advanced features, and Kubernetes deployment.

#### Epic 4.1: Production Hardening

**Story 4.1.1: Authentication and authorization**
```
Task: Implement user authentication
Acceptance Criteria:
- API key authentication for API access
- Session-based auth for UI
- Role-based access control (admin, user, viewer)
- API key management UI
- Audit logging for sensitive operations
```

**Story 4.1.2: Rate limiting and quotas**
```
Task: Implement rate limiting
Acceptance Criteria:
- Request rate limiting per API key
- LLM token quota tracking
- Storage quota per user/tenant
- Quota warning notifications
- Admin quota management UI
```

**Story 4.1.3: Error handling and resilience**
```
Task: Improve error handling across system
Acceptance Criteria:
- Consistent error response format
- User-friendly error messages
- Error tracking (Sentry integration optional)
- Circuit breaker for external services
- Graceful degradation when services unavailable
```

**Story 4.1.4: Performance optimization**
```
Task: Optimize system performance
Acceptance Criteria:
- Database query optimization (indexes, query plans)
- Caching layer (Redis) for frequent queries
- Connection pooling tuning
- Response compression
- Performance benchmarks documented
```

#### Epic 4.2: Kubernetes Deployment

**Story 4.2.1: Helm chart creation**
```
Task: Create Helm chart for platform deployment
Acceptance Criteria:
- Helm chart with all components
- values.yaml with sensible defaults
- Configurable resource limits
- Secret management (external-secrets optional)
- Health checks and readiness probes
- Horizontal Pod Autoscaler for workers
```

**Story 4.2.2: gVisor node pool configuration**
```
Task: Document and configure gVisor nodes
Acceptance Criteria:
- Node label for gVisor capability
- RuntimeClass configuration
- Node pool sizing recommendations
- Installation instructions (GKE, EKS, self-managed)
- Testing procedure for gVisor functionality
```

**Story 4.2.3: Database operators/StatefulSets**
```
Task: Configure stateful services for K8s
Acceptance Criteria:
- Neo4j StatefulSet or Helm chart
- Qdrant StatefulSet with persistence
- PostgreSQL via operator (CloudNativePG) or Helm
- MinIO operator or StatefulSet
- Backup CronJobs for all databases
```

**Story 4.2.4: Ingress and TLS configuration**
```
Task: Configure external access
Acceptance Criteria:
- Ingress resource for API and UI
- TLS certificate management (cert-manager)
- Internal service mesh (optional, for mTLS)
- Network policies for isolation
```

#### Epic 4.3: Advanced Ingestion Features

**Story 4.3.1: Incremental re-indexing**
```
Task: Support updating existing documents
Acceptance Criteria:
- Detect document changes (hash comparison)
- Update only changed chunks
- Preserve existing entity links where valid
- Re-run relationship extraction
- Update search indexes incrementally
```

**Story 4.3.2: Scheduled web scraping**
```
Task: Support recurring web source monitoring
Acceptance Criteria:
- Define URL patterns for monitoring
- Schedule: daily, weekly, custom cron
- Change detection (don't re-process unchanged)
- New content notifications
- Source management UI
```

**Story 4.3.3: Document collections/projects**
```
Task: Support grouping documents into collections
Acceptance Criteria:
- Create/edit/delete collections
- Add documents to collections
- Collection-scoped search
- Collection access permissions
- Export collection as dataset
```

#### Epic 4.4: Advanced Agent Features

**Story 4.4.1: Multi-agent research workflows**
```
Task: Support multi-agent collaboration
Acceptance Criteria:
- Agent handoff between specialized agents
- Research planner agent
- Entity specialist agent
- Timeline analyst agent
- Summary writer agent
- Configurable agent team composition
```

**Story 4.4.2: Persistent agent memory**
```
Task: Implement long-term agent memory
Acceptance Criteria:
- Store important facts discovered during research
- Memory retrieval tool for agents
- Memory management UI (view, edit, delete)
- Memory scoped to conversation or global
```

**Story 4.4.3: Custom tool configuration**
```
Task: Allow users to configure/extend tools
Acceptance Criteria:
- Admin UI for tool configuration
- Enable/disable tools per conversation
- Tool parameter customization
- Custom prompt injection for tools
```

#### Epic 4.5: Export and Reporting

**Story 4.5.1: Research report generation**
```
Task: Create automated research reports
Acceptance Criteria:
- Generate report from conversation
- Include citations and sources
- Entity timeline visualization
- Relationship diagram
- Export formats: PDF, Markdown, HTML
```

**Story 4.5.2: Data export functionality**
```
Task: Support bulk data export
Acceptance Criteria:
- Export entities as CSV/JSON
- Export relationships as CSV/JSON
- Export knowledge graph as GraphML
- Export embeddings as NumPy/Parquet
- Background export jobs for large datasets
```

---

## Risk Areas and Mitigation Strategies

### High-Risk Areas

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **LLM API instability** | High | Medium | Retry logic with exponential backoff; circuit breakers; fallback to local models; Temporal automatic retries |
| **Entity extraction quality** | High | Medium | Hybrid NER approach; human-in-the-loop validation UI; confidence thresholds; iterative prompt refinement |
| **Graph database scaling** | Medium | Low | Start with Neo4j Community; monitor performance; plan upgrade path to Enterprise or FalkorDB |
| **gVisor compatibility issues** | Medium | Medium | Comprehensive testing of Python libraries in sandbox; fallback to nsjail for local dev; document known limitations |
| **Ingestion pipeline complexity** | High | Medium | Comprehensive integration tests; workflow versioning; monitoring dashboards; idempotent activities |

### Technical Debt Risks

- **Chunking strategy tuning**: Initial fixed-size chunking may need refinement; design for configurability
- **Embedding model selection**: Performance varies by domain; plan for embedding re-generation
- **Graph schema evolution**: Entity/relationship types may evolve; design flexible schema

### Operational Risks

- **GPU resource management**: vLLM requires dedicated GPU; plan capacity for concurrent users
- **Storage growth**: Document storage and vector indexes grow linearly; implement retention policies early
- **Temporal history limits**: Long-running workflows hit history limits; use Continue-As-New pattern

---

## Technology Recommendations Summary

| Component | Recommendation | Alternative | Rationale |
|-----------|---------------|-------------|-----------|
| **Graph Database** | Neo4j Community | FalkorDB | Best GraphRAG ecosystem, mature tooling, extensive documentation |
| **LLM Serving** | vLLM | llama.cpp | OpenAI API compatibility, high throughput, native embeddings, production observability |
| **Sandboxing** | gVisor (runsc) | Kata Containers (high security), nsjail (local dev) | K8s native, no hardware requirements, proven at scale |
| **Vector DB** | Qdrant | Milvus | Excellent Python client, hybrid search, quantization support |
| **Full-Text Search** | MeiliSearch | Elasticsearch | Simple setup, fast, faceted search, typo tolerance |
| **Orchestration** | Temporal | Celery | Durable execution, OpenAI SDK integration, native observability |
| **Document Processing** | Docling | Unstructured.io | Comprehensive format support, table extraction, active development |

---

## Implementation Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 0: Infrastructure** | 3 weeks | Dev environment, all databases, vLLM, OTEL stack, Tilt setup |
| **Phase 1: Ingestion** | 6 weeks | Complete ingestion pipeline, OCR, transcription, entity extraction, graph construction |
| **Phase 2: Chat/Agent** | 5 weeks | Research agent with tools, sandboxed execution, streaming chat UI |
| **Phase 3: Search** | 4 weeks | Unified search, graph exploration, file management UI |
| **Phase 4: Polish** | 6 weeks | Auth, K8s deployment, advanced features, production hardening |

**Total: 24 weeks (6 months)**

---

## Getting Started Checklist

For the AI coding agent to begin implementation:

1. **Phase 0.1.1**: Initialize repository with monorepo structure
2. **Phase 0.1.2**: Set up Python tooling (uv, ruff, mypy)
3. **Phase 0.2.1-0.2.6**: Create docker-compose.yml with all infrastructure services
4. **Phase 0.3.1**: Add vLLM service to docker-compose.yml
5. **Phase 0.4.1-0.4.3**: Add observability stack
6. **Phase 0.5.1**: Create Tiltfile for development

Each story is designed to be self-contained with clear acceptance criteria, enabling an AI coding agent to implement, test, and verify completion independently.
