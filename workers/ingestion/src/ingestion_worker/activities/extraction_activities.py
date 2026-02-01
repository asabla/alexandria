"""
Entity and relationship extraction activities.

Activities for extracting entities, relationships, and building
the knowledge graph.
"""

from temporalio import activity

from ingestion_worker.workflows.document_ingestion import (
    ExtractEntitiesInput,
    ExtractEntitiesOutput,
    EntityInfo,
    ExtractRelationshipsInput,
    ExtractRelationshipsOutput,
    RelationshipInfo,
    BuildGraphInput,
    BuildGraphOutput,
)


@activity.defn
async def extract_entities(input: ExtractEntitiesInput) -> ExtractEntitiesOutput:
    """
    Extract named entities from document content.

    Uses a hybrid approach:
    1. spaCy NER for initial extraction
    2. LLM refinement for disambiguation and typing

    Args:
        input: Entity extraction input with document content

    Returns:
        List of extracted entities with mentions
    """
    activity.logger.info(f"Extracting entities from document {input.document_id}")

    entities: list[EntityInfo] = []

    # TODO: Implement actual entity extraction
    # In real implementation:
    # 1. Run spaCy NER on content
    # 2. Group mentions by normalized name
    # 3. Use LLM to:
    #    - Refine entity types
    #    - Disambiguate similar entities
    #    - Assign confidence scores
    # 4. Perform coreference resolution

    # Heartbeat to indicate progress
    activity.heartbeat()

    # Placeholder: extract simple patterns
    # In production, use proper NER pipeline

    activity.logger.info(f"Extracted {len(entities)} entities")

    return ExtractEntitiesOutput(entities=entities)


@activity.defn
async def extract_relationships(input: ExtractRelationshipsInput) -> ExtractRelationshipsOutput:
    """
    Extract relationships between entities.

    Uses LLM-based extraction with structured output to identify
    relationships mentioned in the document.

    Args:
        input: Relationship extraction input with content and entities

    Returns:
        List of extracted relationships
    """
    activity.logger.info(f"Extracting relationships from document {input.document_id}")

    relationships: list[RelationshipInfo] = []

    # TODO: Implement actual relationship extraction
    # In real implementation:
    # 1. For each pair of entities, check if relationship exists
    # 2. Use LLM with structured output to extract:
    #    - Relationship type
    #    - Evidence text
    #    - Confidence score
    # 3. Filter low-confidence relationships

    # Heartbeat to indicate progress
    activity.heartbeat()

    activity.logger.info(f"Extracted {len(relationships)} relationships")

    return ExtractRelationshipsOutput(relationships=relationships)


@activity.defn
async def build_graph(input: BuildGraphInput) -> BuildGraphOutput:
    """
    Build knowledge graph in Neo4j.

    Creates or updates entity nodes and relationship edges
    in the knowledge graph.

    Args:
        input: Graph building input with entities and relationships

    Returns:
        Count of created nodes and relationships
    """
    activity.logger.info(f"Building knowledge graph for document {input.document_id}")

    nodes_created = 0
    relationships_created = 0

    # TODO: Implement actual Neo4j graph building
    # In real implementation:
    # 1. Get Neo4j client
    # 2. For each entity:
    #    - MERGE entity node by (tenant_id, canonical_name, type)
    #    - Set/update properties
    #    - Create MENTIONED_IN relationship to document
    # 3. For each relationship:
    #    - MERGE relationship between entity nodes
    #    - Set properties including evidence and confidence
    # 4. Link entities to project if project_id is set

    # Use MERGE for idempotency

    activity.logger.info(
        f"Graph built: {nodes_created} nodes, {relationships_created} relationships"
    )

    return BuildGraphOutput(
        nodes_created=nodes_created,
        relationships_created=relationships_created,
    )
