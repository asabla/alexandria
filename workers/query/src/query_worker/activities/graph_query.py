"""
Graph query activities for knowledge graph exploration.

These activities provide tools for agents and APIs to query the
Neo4j knowledge graph, including:
- Executing Cypher queries (with safety restrictions)
- Finding paths between entities
- Exploring entity neighborhoods
- Finding related entities by type
- Searching entities by name
- Getting documents where entities are mentioned
"""

from __future__ import annotations

from temporalio import activity

import os
import re
from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

NEO4J_URI_ENV = "NEO4J_URI"
NEO4J_USER_ENV = "NEO4J_USER"
NEO4J_PASSWORD_ENV = "NEO4J_PASSWORD"  # pragma: allowlist secret
NEO4J_DATABASE_ENV = "NEO4J_DATABASE"

DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "password"  # pragma: allowlist secret
DEFAULT_NEO4J_DATABASE = "neo4j"

# Safety: Forbidden Cypher keywords to prevent destructive operations
FORBIDDEN_CYPHER_KEYWORDS = [
    "DELETE",
    "DETACH",
    "REMOVE",
    "DROP",
    "CREATE INDEX",
    "DROP INDEX",
    "CREATE CONSTRAINT",
    "DROP CONSTRAINT",
    "SET",  # Prevent modifications
    "MERGE",  # Prevent modifications
    "CREATE",  # Prevent creating nodes/relationships
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""

    id: str
    labels: list[str]
    properties: dict[str, Any]

    @property
    def name(self) -> str:
        """Get the node's name property."""
        return self.properties.get("name", "")

    @property
    def entity_type(self) -> str:
        """Get the entity type from labels."""
        # Labels are like ["Entity", "Person"] - return the specific one
        for label in self.labels:
            if label not in ("Entity", "Document"):
                return label.lower()
        return "unknown"


@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph."""

    id: str
    type: str
    source_id: str
    target_id: str
    properties: dict[str, Any]


@dataclass
class GraphPath:
    """Represents a path through the knowledge graph."""

    nodes: list[GraphNode]
    relationships: list[GraphRelationship]
    length: int


@dataclass
class ExecuteCypherInput:
    """Input for executing a Cypher query."""

    query: str
    tenant_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    limit: int = 100  # Safety limit


@dataclass
class ExecuteCypherOutput:
    """Output from executing a Cypher query."""

    results: list[dict[str, Any]]
    row_count: int
    query_executed: str


@dataclass
class FindPathsInput:
    """Input for finding paths between entities."""

    source_entity_name: str
    target_entity_name: str
    tenant_id: str
    max_depth: int = 4
    relationship_types: list[str] | None = None  # Filter by relationship types


@dataclass
class FindPathsOutput:
    """Output from finding paths."""

    paths: list[GraphPath]
    shortest_path_length: int | None
    total_paths_found: int


@dataclass
class GetNeighborhoodInput:
    """Input for getting entity neighborhood."""

    entity_name: str
    tenant_id: str
    depth: int = 1  # How many hops from the entity
    relationship_types: list[str] | None = None
    limit: int = 50


@dataclass
class GetNeighborhoodOutput:
    """Output from getting entity neighborhood."""

    center_entity: GraphNode | None
    nodes: list[GraphNode]
    relationships: list[GraphRelationship]
    total_nodes: int
    total_relationships: int


@dataclass
class FindRelatedEntitiesInput:
    """Input for finding related entities."""

    entity_name: str
    tenant_id: str
    entity_type: str | None = None  # Filter results by type
    relationship_type: str | None = None  # Filter by relationship
    direction: str = "both"  # "in", "out", or "both"
    limit: int = 50


@dataclass
class FindRelatedEntitiesOutput:
    """Output from finding related entities."""

    source_entity: GraphNode | None
    related_entities: list[GraphNode]
    relationships: list[GraphRelationship]
    total_found: int


@dataclass
class SearchEntitiesInput:
    """Input for searching entities by name."""

    search_term: str
    tenant_id: str
    entity_type: str | None = None
    limit: int = 20


@dataclass
class SearchEntitiesOutput:
    """Output from searching entities."""

    entities: list[GraphNode]
    total_found: int


@dataclass
class GetEntityDocumentsInput:
    """Input for getting documents where entity is mentioned."""

    entity_name: str
    tenant_id: str
    limit: int = 20


@dataclass
class DocumentMention:
    """A document where an entity is mentioned."""

    document_id: str
    document_title: str | None
    mention_count: int
    properties: dict[str, Any]


@dataclass
class GetEntityDocumentsOutput:
    """Output from getting entity documents."""

    entity: GraphNode | None
    documents: list[DocumentMention]
    total_documents: int


# =============================================================================
# Helper Functions
# =============================================================================


def _is_safe_cypher_query(query: str) -> tuple[bool, str | None]:
    """
    Check if a Cypher query is safe to execute (read-only).

    Args:
        query: The Cypher query to check

    Returns:
        Tuple of (is_safe, error_message)
    """
    query_upper = query.upper()

    for keyword in FORBIDDEN_CYPHER_KEYWORDS:
        # Use word boundary matching to avoid false positives
        pattern = r"\b" + keyword.replace(" ", r"\s+") + r"\b"
        if re.search(pattern, query_upper):
            return False, f"Query contains forbidden keyword: {keyword}"

    return True, None


def _neo4j_node_to_graph_node(node: Any) -> GraphNode:
    """Convert a Neo4j node to a GraphNode."""
    return GraphNode(
        id=node.get("id", str(node.element_id) if hasattr(node, "element_id") else ""),
        labels=list(node.labels) if hasattr(node, "labels") else [],
        properties=dict(node) if hasattr(node, "__iter__") else {},
    )


def _neo4j_rel_to_graph_rel(rel: Any, source_id: str, target_id: str) -> GraphRelationship:
    """Convert a Neo4j relationship to a GraphRelationship."""
    return GraphRelationship(
        id=str(rel.element_id) if hasattr(rel, "element_id") else "",
        type=rel.type if hasattr(rel, "type") else str(rel),
        source_id=source_id,
        target_id=target_id,
        properties=dict(rel) if hasattr(rel, "__iter__") else {},
    )


async def _get_neo4j_client() -> Any:
    """Get a configured Neo4j client."""
    from alexandria_db.clients import Neo4jClient

    return Neo4jClient(
        uri=os.environ.get(NEO4J_URI_ENV, DEFAULT_NEO4J_URI),
        user=os.environ.get(NEO4J_USER_ENV, DEFAULT_NEO4J_USER),
        password=os.environ.get(NEO4J_PASSWORD_ENV, DEFAULT_NEO4J_PASSWORD),
        database=os.environ.get(NEO4J_DATABASE_ENV, DEFAULT_NEO4J_DATABASE),
    )


async def _find_entity_by_name(
    client: Any,
    name: str,
    tenant_id: str,
) -> GraphNode | None:
    """Find an entity by name (case-insensitive)."""
    query = """
    MATCH (e:Entity {tenant_id: $tenant_id})
    WHERE toLower(e.name) = toLower($name)
    RETURN e
    LIMIT 1
    """
    results = await client.execute_query(query, {"name": name, "tenant_id": tenant_id})

    if results:
        node = results[0]["e"]
        return GraphNode(
            id=node.get("id", ""),
            labels=list(node.labels),
            properties=dict(node),
        )
    return None


# =============================================================================
# Activities
# =============================================================================


@activity.defn
async def execute_cypher_query(input: ExecuteCypherInput) -> ExecuteCypherOutput:
    """
    Execute a read-only Cypher query against the knowledge graph.

    This activity allows agents to run custom Cypher queries for flexible
    graph exploration. For safety, destructive operations (DELETE, CREATE,
    SET, MERGE, etc.) are blocked.

    Args:
        input: Query parameters including the Cypher query and tenant_id

    Returns:
        Query results as a list of dictionaries

    Raises:
        ValueError: If the query contains forbidden keywords
    """
    activity.logger.info(f"Executing Cypher query for tenant {input.tenant_id}")

    # Safety check
    is_safe, error = _is_safe_cypher_query(input.query)
    if not is_safe:
        raise ValueError(f"Unsafe query rejected: {error}")

    # Add tenant filter if not present
    query = input.query
    if "$tenant_id" not in query and "tenant_id" not in query.lower():
        activity.logger.warning("Query does not include tenant_id filter")

    # Add LIMIT if not present
    if "LIMIT" not in query.upper():
        query = f"{query} LIMIT {input.limit}"

    client = await _get_neo4j_client()
    try:
        # Ensure tenant_id is in parameters
        parameters = {**input.parameters, "tenant_id": input.tenant_id}
        results = await client.execute_query(query, parameters)

        activity.logger.info(f"Query returned {len(results)} rows")

        return ExecuteCypherOutput(
            results=results,
            row_count=len(results),
            query_executed=query,
        )
    finally:
        await client.close()


@activity.defn
async def find_entity_paths(input: FindPathsInput) -> FindPathsOutput:
    """
    Find paths between two entities in the knowledge graph.

    Useful for discovering connections between people, organizations,
    or other entities. Returns the shortest paths up to max_depth.

    Args:
        input: Path finding parameters

    Returns:
        Paths connecting the two entities
    """
    activity.logger.info(
        f"Finding paths between '{input.source_entity_name}' and "
        f"'{input.target_entity_name}' for tenant {input.tenant_id}"
    )

    client = await _get_neo4j_client()
    try:
        # First, find the source and target entities
        source = await _find_entity_by_name(client, input.source_entity_name, input.tenant_id)
        target = await _find_entity_by_name(client, input.target_entity_name, input.tenant_id)

        if not source:
            activity.logger.warning(f"Source entity not found: {input.source_entity_name}")
            return FindPathsOutput(paths=[], shortest_path_length=None, total_paths_found=0)

        if not target:
            activity.logger.warning(f"Target entity not found: {input.target_entity_name}")
            return FindPathsOutput(paths=[], shortest_path_length=None, total_paths_found=0)

        # Build relationship filter
        rel_filter = ""
        if input.relationship_types:
            rel_types = "|".join(input.relationship_types)
            rel_filter = f":{rel_types}"

        # Find all shortest paths
        query = f"""
        MATCH path = allShortestPaths(
            (source:Entity {{id: $source_id, tenant_id: $tenant_id}})-[r{rel_filter}*1..{input.max_depth}]-
            (target:Entity {{id: $target_id, tenant_id: $tenant_id}})
        )
        RETURN path
        LIMIT 10
        """

        results = await client.execute_query(
            query,
            {
                "source_id": source.id,
                "target_id": target.id,
                "tenant_id": input.tenant_id,
            },
        )

        paths: list[GraphPath] = []
        shortest_length: int | None = None

        for r in results:
            path = r["path"]
            nodes = [
                GraphNode(
                    id=node.get("id", ""),
                    labels=list(node.labels),
                    properties=dict(node),
                )
                for node in path.nodes
            ]
            relationships = []
            for i, rel in enumerate(path.relationships):
                relationships.append(
                    GraphRelationship(
                        id=str(rel.element_id),
                        type=rel.type,
                        source_id=nodes[i].id,
                        target_id=nodes[i + 1].id,
                        properties=dict(rel),
                    )
                )

            path_length = len(relationships)
            paths.append(GraphPath(nodes=nodes, relationships=relationships, length=path_length))

            if shortest_length is None or path_length < shortest_length:
                shortest_length = path_length

        activity.logger.info(f"Found {len(paths)} paths, shortest length: {shortest_length}")

        return FindPathsOutput(
            paths=paths,
            shortest_path_length=shortest_length,
            total_paths_found=len(paths),
        )
    finally:
        await client.close()


@activity.defn
async def get_entity_neighborhood(input: GetNeighborhoodInput) -> GetNeighborhoodOutput:
    """
    Get the neighborhood of an entity (connected nodes within N hops).

    Returns all entities and relationships within the specified depth,
    useful for exploring the local graph structure around an entity.

    Args:
        input: Neighborhood exploration parameters

    Returns:
        Nodes and relationships in the entity's neighborhood
    """
    activity.logger.info(
        f"Getting neighborhood for '{input.entity_name}' "
        f"(depth={input.depth}) for tenant {input.tenant_id}"
    )

    client = await _get_neo4j_client()
    try:
        # Find the center entity
        center = await _find_entity_by_name(client, input.entity_name, input.tenant_id)

        if not center:
            activity.logger.warning(f"Entity not found: {input.entity_name}")
            return GetNeighborhoodOutput(
                center_entity=None,
                nodes=[],
                relationships=[],
                total_nodes=0,
                total_relationships=0,
            )

        # Build relationship filter
        rel_filter = ""
        if input.relationship_types:
            rel_types = "|".join(input.relationship_types)
            rel_filter = f":{rel_types}"

        # Get neighborhood
        query = f"""
        MATCH path = (center:Entity {{id: $entity_id, tenant_id: $tenant_id}})-[r{rel_filter}*1..{input.depth}]-(neighbor)
        WHERE neighbor:Entity OR neighbor:Document
        WITH center, neighbor, relationships(path) as rels
        RETURN DISTINCT neighbor, rels
        LIMIT $limit
        """

        results = await client.execute_query(
            query,
            {
                "entity_id": center.id,
                "tenant_id": input.tenant_id,
                "limit": input.limit,
            },
        )

        nodes: list[GraphNode] = []
        relationships: list[GraphRelationship] = []
        seen_node_ids: set[str] = {center.id}
        seen_rel_ids: set[str] = set()

        for r in results:
            neighbor = r["neighbor"]
            node_id = neighbor.get("id", "")

            if node_id not in seen_node_ids:
                nodes.append(
                    GraphNode(
                        id=node_id,
                        labels=list(neighbor.labels),
                        properties=dict(neighbor),
                    )
                )
                seen_node_ids.add(node_id)

            # Process relationships
            for rel in r["rels"]:
                rel_id = str(rel.element_id)
                if rel_id not in seen_rel_ids:
                    relationships.append(
                        GraphRelationship(
                            id=rel_id,
                            type=rel.type,
                            source_id=str(rel.start_node.element_id),
                            target_id=str(rel.end_node.element_id),
                            properties=dict(rel),
                        )
                    )
                    seen_rel_ids.add(rel_id)

        activity.logger.info(f"Found {len(nodes)} neighbors, {len(relationships)} relationships")

        return GetNeighborhoodOutput(
            center_entity=center,
            nodes=nodes,
            relationships=relationships,
            total_nodes=len(nodes),
            total_relationships=len(relationships),
        )
    finally:
        await client.close()


@activity.defn
async def find_related_entities(input: FindRelatedEntitiesInput) -> FindRelatedEntitiesOutput:
    """
    Find entities related to a given entity.

    Allows filtering by entity type (e.g., find all Organizations related to
    a Person) and relationship type (e.g., find all entities with WORKS_FOR
    relationship).

    Args:
        input: Related entity search parameters

    Returns:
        Related entities and their relationships
    """
    activity.logger.info(
        f"Finding entities related to '{input.entity_name}' "
        f"(type={input.entity_type}, rel={input.relationship_type}) "
        f"for tenant {input.tenant_id}"
    )

    client = await _get_neo4j_client()
    try:
        # Find the source entity
        source = await _find_entity_by_name(client, input.entity_name, input.tenant_id)

        if not source:
            activity.logger.warning(f"Entity not found: {input.entity_name}")
            return FindRelatedEntitiesOutput(
                source_entity=None,
                related_entities=[],
                relationships=[],
                total_found=0,
            )

        # Build relationship and direction filter
        rel_filter = ""
        if input.relationship_type:
            rel_filter = f":{input.relationship_type}"

        if input.direction == "out":
            pattern = f"(source)-[r{rel_filter}]->(related)"
        elif input.direction == "in":
            pattern = f"(source)<-[r{rel_filter}]-(related)"
        else:
            pattern = f"(source)-[r{rel_filter}]-(related)"

        # Build entity type filter
        type_filter = ""
        if input.entity_type:
            type_filter = f" AND related:{input.entity_type.title()}"

        query = f"""
        MATCH {pattern}
        WHERE source.id = $entity_id AND source.tenant_id = $tenant_id
              AND related:Entity{type_filter}
        RETURN related, r,
               CASE WHEN startNode(r) = source THEN 'out' ELSE 'in' END as direction
        LIMIT $limit
        """

        results = await client.execute_query(
            query,
            {
                "entity_id": source.id,
                "tenant_id": input.tenant_id,
                "limit": input.limit,
            },
        )

        related_entities: list[GraphNode] = []
        relationships: list[GraphRelationship] = []

        for r in results:
            related = r["related"]
            rel = r["r"]
            direction = r["direction"]

            related_entities.append(
                GraphNode(
                    id=related.get("id", ""),
                    labels=list(related.labels),
                    properties=dict(related),
                )
            )

            # Set source/target based on direction
            if direction == "out":
                src_id, tgt_id = source.id, related.get("id", "")
            else:
                src_id, tgt_id = related.get("id", ""), source.id

            relationships.append(
                GraphRelationship(
                    id=str(rel.element_id),
                    type=rel.type,
                    source_id=src_id,
                    target_id=tgt_id,
                    properties=dict(rel),
                )
            )

        activity.logger.info(f"Found {len(related_entities)} related entities")

        return FindRelatedEntitiesOutput(
            source_entity=source,
            related_entities=related_entities,
            relationships=relationships,
            total_found=len(related_entities),
        )
    finally:
        await client.close()


@activity.defn
async def search_entities_by_name(input: SearchEntitiesInput) -> SearchEntitiesOutput:
    """
    Search for entities by name (partial, case-insensitive match).

    Useful for finding entities when the exact name is unknown.

    Args:
        input: Search parameters

    Returns:
        Matching entities
    """
    activity.logger.info(
        f"Searching entities for '{input.search_term}' "
        f"(type={input.entity_type}) for tenant {input.tenant_id}"
    )

    client = await _get_neo4j_client()
    try:
        # Build type filter
        if input.entity_type:
            label = f":Entity:{input.entity_type.title()}"
        else:
            label = ":Entity"

        query = f"""
        MATCH (e{label} {{tenant_id: $tenant_id}})
        WHERE toLower(e.name) CONTAINS toLower($search_term)
        RETURN e
        ORDER BY e.name
        LIMIT $limit
        """

        results = await client.execute_query(
            query,
            {
                "search_term": input.search_term,
                "tenant_id": input.tenant_id,
                "limit": input.limit,
            },
        )

        entities: list[GraphNode] = []
        for r in results:
            node = r["e"]
            entities.append(
                GraphNode(
                    id=node.get("id", ""),
                    labels=list(node.labels),
                    properties=dict(node),
                )
            )

        activity.logger.info(f"Found {len(entities)} matching entities")

        return SearchEntitiesOutput(
            entities=entities,
            total_found=len(entities),
        )
    finally:
        await client.close()


@activity.defn
async def get_entity_documents(input: GetEntityDocumentsInput) -> GetEntityDocumentsOutput:
    """
    Get documents where an entity is mentioned.

    Returns documents linked via MENTIONED_IN relationships, useful for
    finding source documents for an entity.

    Args:
        input: Document search parameters

    Returns:
        Documents mentioning the entity
    """
    activity.logger.info(
        f"Getting documents for entity '{input.entity_name}' for tenant {input.tenant_id}"
    )

    client = await _get_neo4j_client()
    try:
        # Find the entity
        entity = await _find_entity_by_name(client, input.entity_name, input.tenant_id)

        if not entity:
            activity.logger.warning(f"Entity not found: {input.entity_name}")
            return GetEntityDocumentsOutput(
                entity=None,
                documents=[],
                total_documents=0,
            )

        # Find documents
        query = """
        MATCH (e:Entity {id: $entity_id, tenant_id: $tenant_id})-[r:MENTIONED_IN]->(d:Document)
        RETURN d, count(r) as mention_count
        ORDER BY mention_count DESC
        LIMIT $limit
        """

        results = await client.execute_query(
            query,
            {
                "entity_id": entity.id,
                "tenant_id": input.tenant_id,
                "limit": input.limit,
            },
        )

        documents: list[DocumentMention] = []
        for r in results:
            doc = r["d"]
            documents.append(
                DocumentMention(
                    document_id=doc.get("id", ""),
                    document_title=doc.get("title"),
                    mention_count=r["mention_count"],
                    properties=dict(doc),
                )
            )

        activity.logger.info(f"Found {len(documents)} documents")

        return GetEntityDocumentsOutput(
            entity=entity,
            documents=documents,
            total_documents=len(documents),
        )
    finally:
        await client.close()


# =============================================================================
# Mock Activities for Testing
# =============================================================================


@activity.defn
async def execute_cypher_query_mock(input: ExecuteCypherInput) -> ExecuteCypherOutput:
    """Mock Cypher query execution for testing."""
    activity.logger.info(f"Mock executing Cypher query for tenant {input.tenant_id}")

    # Safety check still applies
    is_safe, error = _is_safe_cypher_query(input.query)
    if not is_safe:
        raise ValueError(f"Unsafe query rejected: {error}")

    activity.heartbeat()

    return ExecuteCypherOutput(
        results=[],
        row_count=0,
        query_executed=input.query,
    )


@activity.defn
async def find_entity_paths_mock(input: FindPathsInput) -> FindPathsOutput:
    """Mock path finding for testing."""
    activity.logger.info(f"Mock finding paths for tenant {input.tenant_id}")
    activity.heartbeat()

    return FindPathsOutput(
        paths=[],
        shortest_path_length=None,
        total_paths_found=0,
    )


@activity.defn
async def get_entity_neighborhood_mock(input: GetNeighborhoodInput) -> GetNeighborhoodOutput:
    """Mock neighborhood exploration for testing."""
    activity.logger.info(f"Mock getting neighborhood for tenant {input.tenant_id}")
    activity.heartbeat()

    return GetNeighborhoodOutput(
        center_entity=None,
        nodes=[],
        relationships=[],
        total_nodes=0,
        total_relationships=0,
    )
