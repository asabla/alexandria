# Amendment: Multi-Tenancy, Projects, and Scoped Access

**Applies to**: All phases of the RAG/GraphRAG Document Research Platform Implementation Plan

---

## Overview

This amendment introduces three hierarchical scoping layers that affect nearly every component in the system:

```
Tenant (organization boundary)
  └── Project (investigation/research scope)
       └── Documents, Entities, Relationships, Conversations
```

**Design principles**:

1. **Tenant isolation is a hard boundary** — no data leaks between tenants, ever
2. **Project scoping is the default UX** — journalists work on one investigation at a time
3. **Global (tenant-wide) view is opt-in** — explicit toggle, never the default
4. **Documents can belong to multiple projects** — a leaked financial record may be relevant to two investigations
5. **Entities are tenant-scoped, project-linked** — "John Smith" discovered in Project A is the same person if found in Project B

---

## 1. Data Model

### 1.1 Core Entities

```
┌──────────────────────────────────────────────────────────────────┐
│                          TENANT                                   │
│  id, name, slug, settings (json), created_at                      │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                       PROJECT                               │   │
│  │  id, tenant_id, name, slug, description, status,            │   │
│  │  settings (json), created_at, archived_at                   │   │
│  │                                                              │   │
│  │  ┌─────────────────────┐   ┌──────────────────────────┐    │   │
│  │  │  project_documents  │   │    conversations          │    │   │
│  │  │  (join table)       │   │    project_id (FK)       │    │   │
│  │  │  project_id         │   │    tenant_id (FK)        │    │   │
│  │  │  document_id        │   └──────────────────────────┘    │   │
│  │  │  added_at           │                                    │   │
│  │  │  added_by           │                                    │   │
│  │  └─────────────────────┘                                    │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                      DOCUMENT                               │   │
│  │  id, tenant_id, title, source_url, document_type,           │   │
│  │  storage_path, status, file_hash, metadata (json),          │   │
│  │  created_at, processed_at                                   │   │
│  │                                                              │   │
│  │  → chunks (1:N)                                             │   │
│  │  → entities (N:M via document_entities)                     │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                       ENTITY                                │   │
│  │  id, tenant_id, canonical_name, entity_type,                │   │
│  │  aliases (json), metadata (json), merged_into_id,           │   │
│  │  created_at                                                 │   │
│  │                                                              │   │
│  │  → project_entities (N:M join — which projects reference    │   │
│  │    this entity)                                             │   │
│  │  → relationships (as source or target)                      │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    RELATIONSHIP                             │   │
│  │  id, tenant_id, source_entity_id, target_entity_id,         │   │
│  │  relationship_type, confidence, properties (json),          │   │
│  │  source_chunk_id, created_at                                │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Design Decisions

**Documents are owned by the tenant, linked to projects:**

A `document` has a `tenant_id` (hard ownership) and is linked to projects through a `project_documents` join table. This means:
- A document uploaded in Project A can be linked to Project B without duplication
- "Unassigned" documents (not in any project) are valid — they appear in the global view
- Deleting a project doesn't delete documents — it removes the links

**Entities are tenant-scoped, project-linked:**

An `entity` has a `tenant_id` and is linked to projects through `project_entities`. The same real-world person/org appears once per tenant, referenced from multiple projects. This enables cross-project intelligence: "This person appeared in 3 of your investigations."

**Conversations are project-scoped:**

A `conversation` belongs to exactly one project (or null for global context). The agent's tool calls respect the project scope by default, with explicit opt-in to search broader.

### 1.3 PostgreSQL Schema (New/Modified Tables)

```sql
-- Tenant table
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Project table
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'active',  -- active, archived
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    archived_at TIMESTAMPTZ,
    UNIQUE(tenant_id, slug)
);
CREATE INDEX idx_projects_tenant ON projects(tenant_id);

-- Document table (modified)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    title TEXT,
    source_url TEXT,
    document_type TEXT NOT NULL,  -- pdf, docx, image, audio, video, web_page, csv, xlsx
    storage_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    file_hash TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at TIMESTAMPTZ
);
CREATE INDEX idx_documents_tenant ON documents(tenant_id);
CREATE INDEX idx_documents_status ON documents(tenant_id, status);

-- Many-to-many: documents ↔ projects
CREATE TABLE project_documents (
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    added_by UUID,  -- user who linked it
    PRIMARY KEY (project_id, document_id)
);
CREATE INDEX idx_project_documents_doc ON project_documents(document_id);

-- Entity table (modified)
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- PERSON, ORGANIZATION, LOCATION, etc.
    aliases JSONB NOT NULL DEFAULT '[]',
    metadata JSONB NOT NULL DEFAULT '{}',
    merged_into_id UUID REFERENCES entities(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_entities_tenant_type ON entities(tenant_id, entity_type);
CREATE INDEX idx_entities_name ON entities(tenant_id, canonical_name);

-- Many-to-many: entities ↔ projects
CREATE TABLE project_entities (
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project_id, entity_id)
);
CREATE INDEX idx_project_entities_entity ON project_entities(entity_id);

-- Conversations (modified)
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    project_id UUID REFERENCES projects(id),  -- nullable = global context
    title TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_conversations_project ON conversations(tenant_id, project_id);

-- Chunks (modified)
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    vector_id TEXT,  -- reference to Qdrant point ID
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_tenant ON chunks(tenant_id);
```

---

## 2. Impact on Each Data Store

### 2.1 Qdrant (Vector Database)

**Strategy**: Single collection per tenant with payload filtering for project scope.

Do NOT create a collection per project — it doesn't scale (thousands of small collections degrade performance) and prevents cross-project search.

```
Collection: documents_{tenant_id}
  Point payload:
    - tenant_id: string (redundant but useful for safety)
    - document_id: string
    - project_ids: string[]    ← array of project IDs this chunk's document belongs to
    - chunk_index: int
    - document_type: string
    - created_at: string (ISO)
    - entity_names: string[]   ← extracted entity names for filtering
```

**Query patterns**:

```python
# Project-scoped search (default)
client.query_points(
    collection_name=f"documents_{tenant_id}",
    query=query_vector,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="project_ids",
                match=models.MatchAny(any=[project_id])
            )
        ]
    ),
    limit=20
)

# Global (tenant-wide) search
client.query_points(
    collection_name=f"documents_{tenant_id}",
    query=query_vector,
    limit=20
    # no project filter — searches everything in the tenant
)
```

**When a document is linked to a new project**, update the `project_ids` payload on all its points:

```python
# When linking document to new project
chunk_point_ids = get_point_ids_for_document(document_id)
for point_id in chunk_point_ids:
    client.set_payload(
        collection_name=f"documents_{tenant_id}",
        payload={"project_ids": updated_project_ids},
        points=[point_id]
    )
```

**Indexes to create**:

```python
# Create payload indexes for efficient filtering
client.create_payload_index(
    collection_name=f"documents_{tenant_id}",
    field_name="project_ids",
    field_schema=models.PayloadSchemaType.KEYWORD
)
client.create_payload_index(
    collection_name=f"documents_{tenant_id}",
    field_name="document_type",
    field_schema=models.PayloadSchemaType.KEYWORD
)
```

### 2.2 Neo4j (Graph Database)

**Strategy**: Property-based scoping with `tenant_id` on all nodes. Project membership tracked via relationships.

```cypher
// Constraints
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE;

// Indexes for scoped queries
CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id);
CREATE INDEX document_tenant IF NOT EXISTS FOR (d:Document) ON (d.tenant_id);

// Node structure
(:Project {id, tenant_id, name})
(:Document {id, tenant_id, title, document_type, created_at})
(:Entity {id, tenant_id, name, type, aliases})

// Relationships
(:Document)-[:BELONGS_TO]->(:Project)
(:Entity)-[:FOUND_IN]->(:Document)
(:Entity)-[:APPEARS_IN_PROJECT]->(:Project)
(:Entity)-[:RELATED_TO {type, confidence, source_chunk_id}]->(:Entity)
```

**Project-scoped graph query** (default):

```cypher
// Find all entities and relationships within a project
MATCH (p:Project {id: $project_id, tenant_id: $tenant_id})
MATCH (e:Entity)-[:APPEARS_IN_PROJECT]->(p)
OPTIONAL MATCH (e)-[r:RELATED_TO]->(e2:Entity)-[:APPEARS_IN_PROJECT]->(p)
RETURN e, r, e2
```

**Global (tenant-wide) graph query**:

```cypher
// Find entity across all projects in tenant
MATCH (e:Entity {tenant_id: $tenant_id, name: $entity_name})
OPTIONAL MATCH (e)-[:APPEARS_IN_PROJECT]->(p:Project)
OPTIONAL MATCH (e)-[r:RELATED_TO]->(e2:Entity {tenant_id: $tenant_id})
RETURN e, collect(DISTINCT p) as projects, collect({rel: r, target: e2}) as relationships
```

**Cross-project intelligence query**:

```cypher
// "Show me entities that appear in multiple projects"
MATCH (e:Entity {tenant_id: $tenant_id})-[:APPEARS_IN_PROJECT]->(p:Project)
WITH e, collect(p) as projects, count(p) as project_count
WHERE project_count > 1
RETURN e.name, e.type, [p IN projects | p.name] as project_names
ORDER BY project_count DESC
```

### 2.3 MeiliSearch (Full-Text Search)

**Strategy**: Single index per tenant with `project_ids` as a filterable attribute.

MeiliSearch supports [multi-tenant token search](https://www.meilisearch.com/docs/learn/security/multitenancy_tenant_tokens) for API-level tenant isolation, plus standard attribute filtering for project scope.

```python
# Index configuration
index = client.index(f"documents_{tenant_id}")

index.update_filterable_attributes([
    "project_ids",
    "document_type",
    "status",
    "created_at",
    "entity_names"
])

index.update_searchable_attributes([
    "title",
    "content",
    "entity_names"
])

# Indexing a document
index.add_documents([{
    "id": document_id,
    "tenant_id": tenant_id,
    "project_ids": [project_id_1, project_id_2],
    "title": "Financial Report Q3",
    "content": "...",
    "document_type": "pdf",
    "entity_names": ["John Smith", "Acme Corp"],
    "created_at": "2025-01-15T00:00:00Z"
}])
```

**Query patterns**:

```python
# Project-scoped search (default)
index.search("corruption allegations", {
    "filter": f"project_ids = '{project_id}'"
})

# Global (tenant-wide) search
index.search("corruption allegations")
# No project filter
```

### 2.4 MinIO (Object Storage)

**Strategy**: Bucket-per-tenant with project-prefixed paths.

```
Bucket: tenant-{tenant_id}
  Prefix structure:
    /raw/{document_id}/original.pdf
    /processed/{document_id}/text.md
    /processed/{document_id}/tables/
    /processed/{document_id}/images/
    /exports/{export_id}/report.pdf
    /agent-outputs/{conversation_id}/{filename}
```

Why not prefix by project? Because documents can belong to multiple projects. The document's `storage_path` in PostgreSQL points to the canonical location. Project membership is resolved through the relational database, not the file path.

**Tenant isolation** is enforced through:
- IAM policies per tenant (one service account per tenant)
- Application-level bucket name resolution (never trust user input for bucket names)

---

## 3. Impact on API Design

### 3.1 URL Structure

All API routes are scoped under the tenant (resolved from auth context) and optionally a project:

```
# Project-scoped routes (primary)
POST   /api/v1/projects/{project_id}/documents/upload
GET    /api/v1/projects/{project_id}/documents
POST   /api/v1/projects/{project_id}/search
GET    /api/v1/projects/{project_id}/entities
GET    /api/v1/projects/{project_id}/graph
POST   /api/v1/projects/{project_id}/chat/conversations
GET    /api/v1/projects/{project_id}/chat/conversations/{conv_id}

# Global (tenant-wide) routes
GET    /api/v1/documents                    # all documents in tenant
POST   /api/v1/search                       # search across all projects
GET    /api/v1/entities                      # all entities in tenant
GET    /api/v1/graph                         # full tenant graph

# Project management
GET    /api/v1/projects
POST   /api/v1/projects
GET    /api/v1/projects/{project_id}
PATCH  /api/v1/projects/{project_id}
DELETE /api/v1/projects/{project_id}

# Link/unlink documents to projects
POST   /api/v1/projects/{project_id}/documents/link
DELETE /api/v1/projects/{project_id}/documents/{document_id}/unlink

# Tenant management (admin)
GET    /api/v1/tenant
PATCH  /api/v1/tenant/settings
```

### 3.2 Tenant Resolution Middleware

```python
from fastapi import Request, Depends

class TenantContext:
    """Resolved from auth token/API key. Injected into all request handlers."""
    tenant_id: str
    user_id: str
    roles: list[str]

class RequestContext:
    """Full context for a request, including optional project scope."""
    tenant: TenantContext
    project_id: str | None = None

async def get_tenant(request: Request) -> TenantContext:
    """Extract tenant from auth token. Single tenant in dev mode."""
    if settings.SINGLE_TENANT_MODE:
        return TenantContext(
            tenant_id=settings.DEFAULT_TENANT_ID,
            user_id="dev-user",
            roles=["admin"]
        )
    # Production: resolve from JWT / API key
    ...

async def get_request_context(
    tenant: TenantContext = Depends(get_tenant),
    project_id: str | None = None
) -> RequestContext:
    """Build full request context. Validates project belongs to tenant."""
    if project_id:
        # Verify project exists and belongs to tenant
        project = await project_repo.get(project_id, tenant.tenant_id)
        if not project:
            raise HTTPException(404, "Project not found")
    return RequestContext(tenant=tenant, project_id=project_id)
```

### 3.3 Dev Mode: Single Tenant

For local development, a single default tenant and a default project are auto-created on startup:

```python
# In application startup
async def ensure_dev_defaults():
    """Create default tenant and project for local development."""
    if not settings.SINGLE_TENANT_MODE:
        return

    tenant = await tenant_repo.get_or_create(
        id=settings.DEFAULT_TENANT_ID,
        name="Local Development",
        slug="local"
    )

    await project_repo.get_or_create(
        id=settings.DEFAULT_PROJECT_ID,
        tenant_id=tenant.id,
        name="Default Project",
        slug="default"
    )
```

---

## 4. Impact on Temporal Workflows

### 4.1 Workflow Input Always Carries Scope

Every workflow input model includes tenant and project context:

```python
@dataclass
class IngestionWorkflowInput:
    tenant_id: str
    project_id: str          # the project this upload was initiated from
    document_id: str
    source: str              # file path or URL
    link_to_projects: list[str] | None = None  # additional projects to link

@dataclass
class AgentWorkflowInput:
    tenant_id: str
    project_id: str | None   # null = global context
    conversation_id: str
    user_message: str
```

### 4.2 Activities Receive Scope

All activities that touch data stores receive the tenant/project context and enforce it:

```python
@activity.defn
async def search_documents(input: SearchInput) -> SearchResult:
    """Always filters by tenant_id. Filters by project_id if provided."""
    filters = [
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=input.tenant_id))
    ]
    if input.project_id:
        filters.append(
            models.FieldCondition(key="project_ids", match=models.MatchAny(any=[input.project_id]))
        )

    results = await qdrant_client.query_points(
        collection_name=f"documents_{input.tenant_id}",
        query=query_vector,
        query_filter=models.Filter(must=filters),
        limit=input.limit
    )
    return SearchResult(chunks=results)
```

### 4.3 Ingestion Workflow: Project Linking Step

The ingestion workflow gains a new step at the end — linking the processed document and its entities to the appropriate project(s):

```python
@workflow.defn
class DocumentIngestionWorkflow:
    @workflow.run
    async def run(self, input: IngestionWorkflowInput):
        # ... existing steps (classify, OCR, chunk, embed, extract entities) ...

        # NEW: Link document to project(s)
        project_ids = [input.project_id]
        if input.link_to_projects:
            project_ids.extend(input.link_to_projects)

        await workflow.execute_activity(
            link_document_to_projects,
            LinkInput(
                tenant_id=input.tenant_id,
                document_id=input.document_id,
                project_ids=project_ids,
            ),
            start_to_close_timeout=timedelta(seconds=30)
        )

        # NEW: Link discovered entities to project(s)
        await workflow.execute_activity(
            link_entities_to_projects,
            LinkEntitiesInput(
                tenant_id=input.tenant_id,
                document_id=input.document_id,
                project_ids=project_ids,
            ),
            start_to_close_timeout=timedelta(seconds=30)
        )
```

---

## 5. Impact on UI

### 5.1 Navigation Structure

The UI uses a project-first navigation model. The project selector is always visible, and switching projects re-scopes everything.

```
┌─────────────────────────────────────────────────────────┐
│  [Logo]  Project: ▼ Operation Cleanwater      [⚙ Admin]│   ← project selector in header
├──────────┬──────────────────────────────────────────────┤
│          │                                              │
│  📁 Files │                                             │
│  💬 Chat  │            Main Content Area                │
│  🔍 Search│                                             │
│  🕸 Graph │                                             │
│          │                                              │
│──────────│                                              │
│          │                                              │
│  ──────  │                                              │
│  🌐 Global│  ← explicit toggle to tenant-wide view     │
│          │                                              │
└──────────┴──────────────────────────────────────────────┘
```

### 5.2 Project Selector Behavior

```
Dropdown shows:
┌──────────────────────────────┐
│  🔍 Search projects...       │
│  ─────────────────────────── │
│  ★ Operation Cleanwater      │  ← current (starred = pinned)
│    Corporate Fraud Case      │
│    City Council Investigation│
│  ─────────────────────────── │
│  📦 Archived (3)             │  ← collapsed section
│  ─────────────────────────── │
│  + New Project               │
└──────────────────────────────┘
```

When switching projects:
- All views (files, chat, search, graph) re-scope to the new project
- Chat conversations from the previous project remain in the sidebar but dimmed
- URL updates to include project slug: `/projects/operation-cleanwater/chat`

### 5.3 Global View Toggle

The "Global" option in the sidebar switches to a tenant-wide view. Visual indicators make the scope change obvious:

- Header bar changes color or shows a banner: "Viewing all projects"
- Search results show which project each result belongs to (as a badge)
- Graph visualization colors nodes by project
- Files view shows a "Project" column

### 5.4 HTMX Implementation Pattern

Project switching uses HTMX's `hx-push-url` for clean URL updates without full page reloads:

```html
<!-- Project selector dropdown -->
<select name="project_id"
        hx-get="/partials/main-content"
        hx-target="#main-content"
        hx-push-url="true"
        hx-vals='{"scope": "project"}'>
    <option value="proj_123">Operation Cleanwater</option>
    <option value="proj_456">Corporate Fraud Case</option>
    <option value="__global__">🌐 All Projects</option>
</select>

<!-- Main content area that swaps on project change -->
<div id="main-content">
    <!-- Replaced by HTMX on project switch -->
</div>
```

Server-side, the partials endpoint reads the project context and renders accordingly:

```python
@app.get("/partials/main-content")
async def main_content_partial(
    request: Request,
    project_id: str | None = None,
    scope: str = "project"
):
    ctx = await get_request_context(request, project_id if scope == "project" else None)
    # Render the appropriate partial based on current page + scope
    ...
```

---

## 6. Impact on Agent Tools

### 6.1 Scoped Tool Behavior

By default, agent tools respect the conversation's project scope. The agent can explicitly request broader scope:

```python
class SearchToolInput(BaseModel):
    query: str
    scope: Literal["project", "global"] = "project"  # default to project
    project_ids: list[str] | None = None              # search specific projects

class GraphExploreInput(BaseModel):
    entity_name: str
    scope: Literal["project", "global"] = "project"
    include_cross_project_links: bool = False
```

### 6.2 Agent System Prompt Awareness

The agent's system prompt is augmented with scope context:

```python
def build_agent_system_prompt(project: Project | None, tenant: Tenant) -> str:
    if project:
        return f"""You are a research assistant working within the project "{project.name}".
Description: {project.description}

By default, your tools search only within this project's documents and entities.
If the user asks about something outside this project, you can set scope="global"
on your tool calls to search across all projects in the organization.

When returning results from outside the current project, always note which project
they belong to."""
    else:
        return f"""You are a research assistant with access to all documents and entities
across the organization. When presenting results, always indicate which project
each piece of information belongs to."""
```

### 6.3 Cross-Project Intelligence

The agent has a dedicated tool for cross-project analysis:

```python
@activity.defn
async def find_cross_project_connections(input: CrossProjectInput) -> CrossProjectResult:
    """Find entities or patterns that appear across multiple projects."""
    query = """
    MATCH (e:Entity {tenant_id: $tenant_id})-[:APPEARS_IN_PROJECT]->(p:Project)
    WITH e, collect(p) as projects, count(p) as cnt
    WHERE cnt > 1
    AND ($entity_filter IS NULL OR e.name CONTAINS $entity_filter)
    RETURN e.name, e.type,
           [p IN projects | {id: p.id, name: p.name}] as projects,
           cnt as project_count
    ORDER BY cnt DESC
    LIMIT $limit
    """
    ...
```

---

## 7. Revised User Stories

These stories should be inserted into the appropriate phases of the main implementation plan.

### Phase 0 Additions

**Story 0.2.0: Tenant and Project schema (insert before 0.2.1)**
```
Task: Create database schema for tenants, projects, and join tables
Acceptance Criteria:
- Alembic migration creating: tenants, projects, project_documents,
  project_entities tables
- All existing tables (documents, chunks, entities, conversations)
  gain tenant_id column with NOT NULL + foreign key
- conversations table gains nullable project_id column
- Indexes on tenant_id for all tables, compound indexes where needed
- Dev-mode startup creates default tenant and default project
- Environment variable: SINGLE_TENANT_MODE=true for local development
```

**Story 0.2.0b: Tenant middleware and request context**
```
Task: Create FastAPI middleware for tenant resolution
Acceptance Criteria:
- TenantContext dataclass with tenant_id, user_id, roles
- RequestContext dataclass with tenant + optional project_id
- Dependency injection via Depends() for all route handlers
- Single-tenant mode bypasses auth, uses default tenant
- All repository methods require tenant_id parameter
- 403 returned if project doesn't belong to tenant
```

### Phase 1 Additions

**Story 1.1.1b: Project management API**
```
Task: Create CRUD API for projects
Acceptance Criteria:
- POST /api/v1/projects — create project (name, description)
- GET /api/v1/projects — list projects for tenant
- GET /api/v1/projects/{id} — get project details with stats
  (document count, entity count, last activity)
- PATCH /api/v1/projects/{id} — update name/description/settings
- POST /api/v1/projects/{id}/archive — soft archive
- POST /api/v1/projects/{id}/unarchive — restore
- All routes enforce tenant_id from auth context
```

**Story 1.1.2b: Document-project linking API**
```
Task: Create API for linking documents to projects
Acceptance Criteria:
- POST /api/v1/projects/{id}/documents/link
  Body: { document_ids: [...] }
- DELETE /api/v1/projects/{id}/documents/{doc_id}/unlink
- GET /api/v1/documents/{doc_id}/projects — list projects a doc belongs to
- Linking updates Qdrant payload (project_ids), MeiliSearch document,
  and Neo4j relationships
- Unlinking updates the same stores
```

**Modify Story 1.1.2: File upload now takes project_id**
```
Modified: POST /api/v1/projects/{project_id}/documents/upload
- Document is created with tenant_id from auth context
- Document is automatically linked to the specified project
- If project_id is omitted (global upload), document exists unassigned
```

**Story 1.5.5: Entity-project linking activity**
```
Task: Create activity to link discovered entities to projects
Acceptance Criteria:
- @activity.defn async def link_entities_to_projects
- Input: tenant_id, document_id, project_ids
- For each entity found in the document, create project_entities records
- Create APPEARS_IN_PROJECT relationships in Neo4j
- Idempotent (safe to re-run)
```

### Phase 2 Additions

**Modify Story 2.5.1: Chat conversations are project-scoped**
```
Modified acceptance criteria:
- POST /api/v1/projects/{project_id}/chat/conversations
- GET /api/v1/projects/{project_id}/chat/conversations — list for project
- Also support: POST /api/v1/chat/conversations (global, no project)
- Agent system prompt includes project context
- Tools default to project scope
```

### Phase 3 Additions

**Story 3.2.0: Project management UI**
```
Task: Create project management interface
Acceptance Criteria:
- Projects list page with cards (name, description, doc count, last activity)
- Create project modal (name, description)
- Project settings page (rename, archive)
- Project selector in header (dropdown, persisted in session/cookie)
- "All Projects" / Global toggle in sidebar
- HTMX-driven project switching (no full page reload)
- URL structure: /projects/{slug}/files, /projects/{slug}/chat, etc.
```

**Story 3.4.5: Cross-project document linking UI**
```
Task: Create UI for linking documents across projects
Acceptance Criteria:
- On document detail page: "Also in projects: [list]" with add button
- "Link to project" modal with project picker
- "Unlink from project" with confirmation
- Bulk link: select multiple docs → "Add to project" action
```

---

## 8. Collection/Index Initialization

### Startup Sequence

When a new tenant is created, the system must initialize their data stores:

```python
async def provision_tenant(tenant_id: str):
    """Called when a new tenant is created."""

    # 1. Qdrant collection
    await qdrant_client.create_collection(
        collection_name=f"documents_{tenant_id}",
        vectors_config=models.VectorParams(
            size=1024,  # BGE-large-en-v1.5
            distance=models.Distance.COSINE
        ),
        sparse_vectors_config={
            "bm25": models.SparseVectorParams()
        }
    )
    # Create payload indexes
    for field, schema in [
        ("project_ids", models.PayloadSchemaType.KEYWORD),
        ("document_type", models.PayloadSchemaType.KEYWORD),
        ("document_id", models.PayloadSchemaType.KEYWORD),
    ]:
        await qdrant_client.create_payload_index(
            collection_name=f"documents_{tenant_id}",
            field_name=field,
            field_schema=schema
        )

    # 2. MeiliSearch index
    meili_client.create_index(f"documents_{tenant_id}", {"primaryKey": "id"})
    index = meili_client.index(f"documents_{tenant_id}")
    index.update_filterable_attributes(["project_ids", "document_type", "entity_names", "created_at"])
    index.update_searchable_attributes(["title", "content", "entity_names"])

    # 3. MinIO bucket
    minio_client.make_bucket(f"tenant-{tenant_id}")

    # 4. Neo4j constraints (tenant-scoped via properties, not separate DBs)
    # Constraints are global but data is filtered by tenant_id property
```

### Dev Mode Auto-Provisioning

```python
async def startup():
    if settings.SINGLE_TENANT_MODE:
        tenant = await ensure_dev_defaults()
        await provision_tenant(tenant.id)
```

---

## 9. Migration Path

Since this is greenfield, multi-tenancy is built in from day one. However, the **single-tenant dev mode** ensures that the added complexity doesn't slow down initial development:

| Concern | Dev Mode | Production |
|---------|----------|------------|
| Tenant resolution | Hardcoded default tenant | JWT / API key lookup |
| Project requirement | Default project auto-selected | User must select/create |
| Auth middleware | Pass-through (no auth) | Full RBAC enforcement |
| Collection naming | `documents_{default_id}` | `documents_{tenant_id}` |
| MinIO bucket | `tenant-{default_id}` | `tenant-{tenant_id}` |

The key insight: the code paths are identical in both modes. Single-tenant mode just pre-populates the context that would normally come from authentication.

---

## 10. Summary of Changes to Main Plan

| Phase | What Changes |
|-------|-------------|
| **Phase 0** | Add tenant/project tables, middleware, dev-mode provisioning. All Docker services unchanged. |
| **Phase 1** | Every activity/workflow carries tenant_id + project_id. New activities for project linking. Upload endpoint is project-scoped. |
| **Phase 2** | Conversations are project-scoped. Agent tools default to project scope with global opt-in. System prompt includes project context. |
| **Phase 3** | Project management UI is the first thing built. Project selector in header. All search/graph/file views are project-scoped by default with global toggle. |
| **Phase 4** | RBAC per project (viewer, editor, admin). Tenant admin panel. Cross-project analytics dashboard. |

The total additional effort is approximately **1.5-2 weeks** spread across phases, primarily because the scoping is woven into existing stories rather than being a separate feature.
