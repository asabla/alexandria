# Initial Implementation - Overview

> Bird's eye view of the Alexandria document research platform implementation

## Goals

Build a **document research platform for investigative journalism** that enables:

1. **Ingest diverse document types** - PDFs, images, audio, video, web pages, spreadsheets
2. **Extract and link knowledge** - Entities, relationships, timelines from documents
3. **Intelligent research assistance** - AI agents that can search, analyze, and synthesize findings
4. **Collaborative investigation** - Multi-tenant with project-based organization

## Context

Investigative journalists need to process large volumes of documents (leaked records, public filings, interview transcripts) and identify connections between people, organizations, and events. Current tools are either too generic (standard search) or too complex (full data science pipelines).

This platform bridges the gap with:
- **RAG (Retrieval-Augmented Generation)** for semantic document search
- **GraphRAG** for relationship discovery and exploration
- **AI Agents** for natural language research queries
- **Multi-tenancy** for team collaboration on investigations

## Phases

### Phase 0: Project Scaffolding & Infrastructure (Weeks 1-3)
Establish development environment, CI/CD, and core infrastructure services.
- Monorepo structure with Python tooling (uv, ruff, mypy)
- Docker Compose for local development
- Infrastructure services: PostgreSQL, MinIO, Qdrant, Neo4j, MeiliSearch, Temporal
- vLLM for LLM serving (chat + embeddings)
- Observability stack: OpenTelemetry, Jaeger, Prometheus, Grafana
- Tilt for Kubernetes development
- GitHub Actions CI/CD

### Phase 1: Ingestion Pipeline (Weeks 4-9)
Implement complete document processing workflow.
- File upload and URL ingestion APIs
- Temporal workflows for durable processing
- Document parsing: Docling OCR, audio transcription (Parakeet), object detection (YOLO)
- Semantic chunking and embedding generation
- Entity extraction (spaCy + LLM) and relationship discovery
- Knowledge graph construction in Neo4j
- Vector indexing in Qdrant, full-text in MeiliSearch

### Phase 2: Chat/Agent Interface (Weeks 10-14)
Implement conversational AI for document research.
- OpenAI Agent SDK integration with Temporal durability
- Research tools: vector search, full-text search, graph exploration, timeline construction
- Tabular data analysis (pandas, SQL) in gVisor sandbox
- Streaming chat API with SSE
- HTMX-based chat UI with tool execution display

### Phase 3: Search Experience (Weeks 15-18)
Comprehensive search and exploration interface.
- Unified search API (semantic, keyword, hybrid, graph modes)
- Faceted search UI with filters
- Interactive graph visualization
- Document library management
- Bulk operations

### Phase 4: Polish & Advanced Features (Weeks 19-24)
Production hardening and advanced capabilities.
- Authentication, authorization, rate limiting
- Kubernetes deployment with Helm charts
- Incremental re-indexing and scheduled scraping
- Multi-agent research workflows
- Report generation and export

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Graph Database | **Neo4j Community Edition** | Best GraphRAG ecosystem, mature tooling, Cypher queries |
| LLM Serving | **vLLM** | OpenAI-compatible API, high throughput, native embeddings |
| Sandboxing | **gVisor (runsc)** | K8s native, Docker Compose compatible, fast startup |
| Workflow Orchestration | **Temporal** | Durable execution, crash recovery, observability |
| Frontend | **HTMX + Tailwind** | No JS build step, server-rendered, fast iteration |
| Multi-tenancy | **Built-in from day one** | Tenant isolation, project scoping, shared entities |

## Technology Stack

### Backend
- **Python 3.12** with FastAPI
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
- **HTMX** for interactivity
- **Tailwind CSS** for styling
- **vis.js** for graph visualization

### Infrastructure
- **Docker Compose** for local development
- **Kubernetes** for production
- **Tilt** for K8s development workflow
- **OpenTelemetry** for observability

## Success Criteria

- [ ] **Ingestion**: Process PDF, DOCX, images, audio, video, web pages, CSV/Excel
- [ ] **Entity Extraction**: Automatically identify PERSON, ORGANIZATION, LOCATION, DATE, EVENT entities
- [ ] **Knowledge Graph**: Build and query entity relationship graph
- [ ] **Search**: Semantic, keyword, and graph-based search with facets
- [ ] **Chat**: Natural language research assistant with tool use
- [ ] **Sandboxed Execution**: Safe code execution for data analysis
- [ ] **Multi-tenant**: Organization and project isolation
- [ ] **Observable**: Full tracing, metrics, and logging

## Timeline

| Phase | Duration | Focus |
|-------|----------|-------|
| Phase 0 | 3 weeks | Infrastructure & Scaffolding |
| Phase 1 | 6 weeks | Document Ingestion Pipeline |
| Phase 2 | 5 weeks | Chat & Agent Interface |
| Phase 3 | 4 weeks | Search Experience |
| Phase 4 | 6 weeks | Polish & Production |
| **Total** | **20-24 weeks** | |

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM quality/latency | High | Start with 8B model, upgrade to 70B if needed; batch processing for ingestion |
| Graph query performance | Medium | Proper indexing, query optimization, consider FalkorDB if Neo4j bottlenecks |
| OCR accuracy on poor quality docs | Medium | Docling + manual review UI for critical documents |
| Sandbox escape | Critical | gVisor provides strong isolation; network policies; no persistent storage |
| Scope creep | High | Strict phase boundaries; MVP first, polish later |

## Architecture Diagram

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
