// =============================================================================
// Alexandria Neo4j Initialization Script
// =============================================================================
// Run with: CALL apoc.cypher.runFile('/var/lib/neo4j/import/init.cypher')
// Or execute via Neo4j Browser
// =============================================================================

// -----------------------------------------------------------------------------
// Constraints (ensure uniqueness)
// -----------------------------------------------------------------------------

// Tenant constraint
CREATE CONSTRAINT tenant_id IF NOT EXISTS
FOR (t:Tenant) REQUIRE t.id IS UNIQUE;

// Project constraint
CREATE CONSTRAINT project_id IF NOT EXISTS
FOR (p:Project) REQUIRE p.id IS UNIQUE;

// Document constraint
CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

// Entity constraint (composite: tenant + entity type + name for deduplication)
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Chunk constraint
CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

// -----------------------------------------------------------------------------
// Indexes for common queries
// -----------------------------------------------------------------------------

// Entity indexes
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id);

// Document indexes
CREATE INDEX document_tenant IF NOT EXISTS FOR (d:Document) ON (d.tenant_id);
CREATE INDEX document_type IF NOT EXISTS FOR (d:Document) ON (d.document_type);
CREATE INDEX document_created IF NOT EXISTS FOR (d:Document) ON (d.created_at);

// Chunk indexes
CREATE INDEX chunk_document IF NOT EXISTS FOR (c:Chunk) ON (c.document_id);
CREATE INDEX chunk_tenant IF NOT EXISTS FOR (c:Chunk) ON (c.tenant_id);

// Project indexes
CREATE INDEX project_tenant IF NOT EXISTS FOR (p:Project) ON (p.tenant_id);

// -----------------------------------------------------------------------------
// Full-text indexes for search
// -----------------------------------------------------------------------------

// Entity name full-text search
CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.aliases];

// Document content full-text search (for graph-based search)
CREATE FULLTEXT INDEX document_fulltext IF NOT EXISTS
FOR (d:Document) ON EACH [d.title, d.summary];

// -----------------------------------------------------------------------------
// Log initialization
// -----------------------------------------------------------------------------
RETURN 'Alexandria Neo4j initialized successfully' AS status;
