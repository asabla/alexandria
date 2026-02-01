"""Neo4j graph database client wrapper."""

from dataclasses import dataclass
from typing import Any

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import Neo4jError

logger = structlog.get_logger(__name__)


@dataclass
class NodeResult:
    """Result containing a Neo4j node."""

    id: str
    labels: list[str]
    properties: dict[str, Any]


@dataclass
class RelationshipResult:
    """Result containing a Neo4j relationship."""

    id: str
    type: str
    start_node_id: str
    end_node_id: str
    properties: dict[str, Any]


@dataclass
class PathResult:
    """Result containing a path through the graph."""

    nodes: list[NodeResult]
    relationships: list[RelationshipResult]


class Neo4jClient:
    """
    Wrapper for Neo4j graph database operations.

    Provides a simplified interface for knowledge graph operations
    including entity and relationship management.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ):
        """
        Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Username for authentication
            password: Password for authentication
            database: Database name to use
        """
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
        )
        self._database = database
        self._log = logger.bind(service="neo4j", uri=uri, database=database)

    async def close(self) -> None:
        """Close the driver connection."""
        await self._driver.close()

    async def _get_session(self) -> AsyncSession:
        """Get an async session."""
        return self._driver.session(database=self._database)

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dictionaries
        """
        async with await self._get_session() as session:
            try:
                result = await session.run(query, parameters or {})
                records = await result.data()
                self._log.debug(
                    "query_executed",
                    query=query[:100],
                    params=parameters,
                    result_count=len(records),
                )
                return records
            except Neo4jError as e:
                self._log.error(
                    "query_failed",
                    query=query[:100],
                    params=parameters,
                    error=str(e),
                )
                raise

    async def create_entity_node(
        self,
        entity_id: str,
        entity_type: str,
        tenant_id: str,
        properties: dict[str, Any],
    ) -> NodeResult:
        """
        Create or merge an entity node.

        Args:
            entity_id: Unique entity ID
            entity_type: Entity type (e.g., "Person", "Organization")
            tenant_id: Tenant ID for multi-tenancy
            properties: Node properties

        Returns:
            NodeResult with the created/updated node
        """
        query = """
        MERGE (e:Entity {id: $entity_id, tenant_id: $tenant_id})
        SET e += $properties
        SET e:$label
        RETURN e
        """
        # Neo4j doesn't support dynamic labels in SET, so we use a workaround
        query = f"""
        MERGE (e:Entity:{entity_type} {{id: $entity_id, tenant_id: $tenant_id}})
        SET e += $properties
        RETURN e
        """

        properties = {**properties, "entity_type": entity_type}
        result = await self.execute_query(
            query,
            {
                "entity_id": entity_id,
                "tenant_id": tenant_id,
                "properties": properties,
            },
        )

        if result:
            node = result[0]["e"]
            return NodeResult(
                id=str(node.element_id),
                labels=list(node.labels),
                properties=dict(node),
            )
        raise ValueError("Failed to create entity node")

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        tenant_id: str,
        properties: dict[str, Any] | None = None,
    ) -> RelationshipResult:
        """
        Create or merge a relationship between entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Type of relationship (e.g., "WORKS_FOR")
            tenant_id: Tenant ID for multi-tenancy
            properties: Relationship properties

        Returns:
            RelationshipResult with the created/updated relationship
        """
        query = f"""
        MATCH (source:Entity {{id: $source_id, tenant_id: $tenant_id}})
        MATCH (target:Entity {{id: $target_id, tenant_id: $tenant_id}})
        MERGE (source)-[r:{relationship_type}]->(target)
        SET r += $properties
        RETURN r, elementId(source) as source_element_id, elementId(target) as target_element_id
        """

        result = await self.execute_query(
            query,
            {
                "source_id": source_id,
                "target_id": target_id,
                "tenant_id": tenant_id,
                "properties": properties or {},
            },
        )

        if result:
            rel = result[0]["r"]
            return RelationshipResult(
                id=str(rel.element_id),
                type=rel.type,
                start_node_id=result[0]["source_element_id"],
                end_node_id=result[0]["target_element_id"],
                properties=dict(rel),
            )
        raise ValueError("Failed to create relationship")

    async def create_document_node(
        self,
        document_id: str,
        tenant_id: str,
        properties: dict[str, Any],
    ) -> NodeResult:
        """
        Create or merge a document node.

        Args:
            document_id: Unique document ID
            tenant_id: Tenant ID
            properties: Document properties

        Returns:
            NodeResult with the created/updated node
        """
        query = """
        MERGE (d:Document {id: $document_id, tenant_id: $tenant_id})
        SET d += $properties
        RETURN d
        """

        result = await self.execute_query(
            query,
            {
                "document_id": document_id,
                "tenant_id": tenant_id,
                "properties": properties,
            },
        )

        if result:
            node = result[0]["d"]
            return NodeResult(
                id=str(node.element_id),
                labels=list(node.labels),
                properties=dict(node),
            )
        raise ValueError("Failed to create document node")

    async def create_mentioned_in_relationship(
        self,
        entity_id: str,
        document_id: str,
        tenant_id: str,
        properties: dict[str, Any] | None = None,
    ) -> RelationshipResult:
        """
        Create MENTIONED_IN relationship between entity and document.

        Args:
            entity_id: Entity ID
            document_id: Document ID
            tenant_id: Tenant ID
            properties: Optional relationship properties

        Returns:
            RelationshipResult
        """
        query = """
        MATCH (e:Entity {id: $entity_id, tenant_id: $tenant_id})
        MATCH (d:Document {id: $document_id, tenant_id: $tenant_id})
        MERGE (e)-[r:MENTIONED_IN]->(d)
        SET r += $properties
        RETURN r, elementId(e) as entity_element_id, elementId(d) as doc_element_id
        """

        result = await self.execute_query(
            query,
            {
                "entity_id": entity_id,
                "document_id": document_id,
                "tenant_id": tenant_id,
                "properties": properties or {},
            },
        )

        if result:
            rel = result[0]["r"]
            return RelationshipResult(
                id=str(rel.element_id),
                type=rel.type,
                start_node_id=result[0]["entity_element_id"],
                end_node_id=result[0]["doc_element_id"],
                properties=dict(rel),
            )
        raise ValueError("Failed to create MENTIONED_IN relationship")

    async def find_entity_by_name(
        self,
        name: str,
        tenant_id: str,
        entity_type: str | None = None,
    ) -> list[NodeResult]:
        """
        Find entities by name (case-insensitive).

        Args:
            name: Entity name to search
            tenant_id: Tenant ID
            entity_type: Optional type filter

        Returns:
            List of matching NodeResults
        """
        if entity_type:
            query = f"""
            MATCH (e:Entity:{entity_type} {{tenant_id: $tenant_id}})
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN e
            LIMIT 100
            """
        else:
            query = """
            MATCH (e:Entity {tenant_id: $tenant_id})
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN e
            LIMIT 100
            """

        result = await self.execute_query(
            query,
            {"name": name, "tenant_id": tenant_id},
        )

        return [
            NodeResult(
                id=str(r["e"].element_id),
                labels=list(r["e"].labels),
                properties=dict(r["e"]),
            )
            for r in result
        ]

    async def get_entity_relationships(
        self,
        entity_id: str,
        tenant_id: str,
        direction: str = "both",
        relationship_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get relationships for an entity.

        Args:
            entity_id: Entity ID
            tenant_id: Tenant ID
            direction: "in", "out", or "both"
            relationship_types: Optional list of relationship types to filter

        Returns:
            List of relationships with connected entities
        """
        type_filter = ""
        if relationship_types:
            types = "|".join(relationship_types)
            type_filter = f":{types}"

        if direction == "out":
            query = f"""
            MATCH (e:Entity {{id: $entity_id, tenant_id: $tenant_id}})-[r{type_filter}]->(other)
            RETURN r, other, 'out' as direction
            LIMIT 100
            """
        elif direction == "in":
            query = f"""
            MATCH (e:Entity {{id: $entity_id, tenant_id: $tenant_id}})<-[r{type_filter}]-(other)
            RETURN r, other, 'in' as direction
            LIMIT 100
            """
        else:
            query = f"""
            MATCH (e:Entity {{id: $entity_id, tenant_id: $tenant_id}})-[r{type_filter}]-(other)
            RETURN r, other, 
                   CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END as direction
            LIMIT 100
            """

        return await self.execute_query(
            query,
            {"entity_id": entity_id, "tenant_id": tenant_id},
        )

    async def find_paths(
        self,
        source_id: str,
        target_id: str,
        tenant_id: str,
        max_depth: int = 3,
    ) -> list[PathResult]:
        """
        Find paths between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            tenant_id: Tenant ID
            max_depth: Maximum path length

        Returns:
            List of PathResults
        """
        query = f"""
        MATCH path = shortestPath(
            (source:Entity {{id: $source_id, tenant_id: $tenant_id}})-[*1..{max_depth}]-
            (target:Entity {{id: $target_id, tenant_id: $tenant_id}})
        )
        RETURN path
        LIMIT 10
        """

        result = await self.execute_query(
            query,
            {"source_id": source_id, "target_id": target_id, "tenant_id": tenant_id},
        )

        paths = []
        for r in result:
            path = r["path"]
            nodes = [
                NodeResult(
                    id=str(node.element_id),
                    labels=list(node.labels),
                    properties=dict(node),
                )
                for node in path.nodes
            ]
            relationships = [
                RelationshipResult(
                    id=str(rel.element_id),
                    type=rel.type,
                    start_node_id=str(rel.start_node.element_id),
                    end_node_id=str(rel.end_node.element_id),
                    properties=dict(rel),
                )
                for rel in path.relationships
            ]
            paths.append(PathResult(nodes=nodes, relationships=relationships))

        return paths

    async def delete_entity(
        self,
        entity_id: str,
        tenant_id: str,
    ) -> bool:
        """
        Delete an entity and all its relationships.

        Args:
            entity_id: Entity ID
            tenant_id: Tenant ID

        Returns:
            True if entity was deleted
        """
        query = """
        MATCH (e:Entity {id: $entity_id, tenant_id: $tenant_id})
        DETACH DELETE e
        RETURN count(e) as deleted
        """

        result = await self.execute_query(
            query,
            {"entity_id": entity_id, "tenant_id": tenant_id},
        )

        deleted = result[0]["deleted"] if result else 0
        self._log.info(
            "entity_deleted",
            entity_id=entity_id,
            tenant_id=tenant_id,
            deleted=deleted > 0,
        )
        return deleted > 0

    async def health_check(self) -> bool:
        """
        Check if Neo4j is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            async with await self._get_session() as session:
                await session.run("RETURN 1")
            return True
        except Exception as e:
            self._log.warning("health_check_failed", error=str(e))
            return False
