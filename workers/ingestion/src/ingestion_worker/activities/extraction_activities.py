"""
Entity and relationship extraction activities.

Activities for extracting entities, relationships, and building
the knowledge graph.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import TYPE_CHECKING, Any

from temporalio import activity

from ingestion_worker.workflows.document_ingestion import (
    BuildGraphInput,
    BuildGraphOutput,
    ChunkInfo,
    EntityInfo,
    ExtractEntitiesInput,
    ExtractEntitiesOutput,
    ExtractRelationshipsInput,
    ExtractRelationshipsOutput,
    RelationshipInfo,
)

if TYPE_CHECKING:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc

# =============================================================================
# Constants and Configuration
# =============================================================================

# Environment variable for spaCy model selection
SPACY_MODEL_ENV = "SPACY_MODEL"
DEFAULT_SPACY_MODEL = "en_core_web_lg"

# Mapping from spaCy NER labels to our EntityType values
# See: https://spacy.io/models/en#en_core_web_lg-labels
SPACY_LABEL_TO_ENTITY_TYPE: dict[str, str] = {
    # Person
    "PERSON": "person",
    # Organizations
    "ORG": "organization",
    "NORP": "organization",  # Nationalities, religious/political groups
    # Locations
    "GPE": "location",  # Geopolitical entities (countries, cities, states)
    "LOC": "location",  # Non-GPE locations (mountains, water bodies)
    "FAC": "location",  # Facilities (buildings, airports, highways)
    # Date/Time
    "DATE": "date",
    "TIME": "date",
    # Events
    "EVENT": "event",
    # Money
    "MONEY": "money",
    # Legal
    "LAW": "law",
    # Products
    "PRODUCT": "product",
    # Work of art (could be document or product)
    "WORK_OF_ART": "document",
    # Quantities (map to unknown - not directly useful for knowledge graph)
    "QUANTITY": "unknown",
    "PERCENT": "unknown",
    "CARDINAL": "unknown",
    "ORDINAL": "unknown",
    # Languages
    "LANGUAGE": "unknown",
}

# Entity types we want to extract (skip low-value types)
EXTRACTED_ENTITY_TYPES = {
    "person",
    "organization",
    "location",
    "date",
    "event",
    "money",
    "law",
    "product",
    "document",
}

# Minimum entity name length (skip very short entities)
MIN_ENTITY_LENGTH = 2

# Maximum entity name length (skip overly long entities - likely extraction errors)
MAX_ENTITY_LENGTH = 200

# Context window size (characters before/after mention for context)
CONTEXT_WINDOW = 50

# Singleton for lazy-loaded spaCy model
_nlp_model: Language | None = None


# =============================================================================
# Helper Functions
# =============================================================================


def _get_spacy_model() -> Language:
    """
    Get the spaCy NLP model, loading it lazily on first use.

    This lazy loading pattern is important for Temporal activities to avoid
    loading heavy models at import time, which can cause sandbox issues.

    Returns:
        The loaded spaCy Language model.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    global _nlp_model

    if _nlp_model is not None:
        return _nlp_model

    # Lazy import spacy to avoid Temporal sandbox issues
    import spacy

    model_name = os.environ.get(SPACY_MODEL_ENV, DEFAULT_SPACY_MODEL)

    try:
        _nlp_model = spacy.load(model_name)
        activity.logger.info(f"Loaded spaCy model: {model_name}")
        return _nlp_model
    except OSError as e:
        error_msg = (
            f"Failed to load spaCy model '{model_name}'. "
            f"Please install it with: python -m spacy download {model_name}"
        )
        activity.logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def _normalize_entity_name(name: str) -> str:
    """
    Normalize an entity name for grouping/deduplication.

    This function applies several normalization steps:
    1. Unicode normalization (NFKC)
    2. Strip whitespace
    3. Collapse multiple spaces
    4. Convert to lowercase
    5. Remove possessive suffixes ('s, ')

    Args:
        name: The raw entity name from NER extraction.

    Returns:
        Normalized entity name suitable for grouping mentions.

    Examples:
        >>> _normalize_entity_name("  John   Doe  ")
        "john doe"
        >>> _normalize_entity_name("Acme Corp's")
        "acme corp"
        >>> _normalize_entity_name("UNITED STATES")
        "united states"
    """
    # Unicode normalization
    normalized = unicodedata.normalize("NFKC", name)

    # Strip and collapse whitespace
    normalized = " ".join(normalized.split())

    # Lowercase
    normalized = normalized.lower()

    # Remove possessive suffixes
    normalized = re.sub(r"['']s$", "", normalized)
    normalized = re.sub(r"s['']$", "s", normalized)

    return normalized.strip()


def _map_spacy_label(label: str) -> str:
    """
    Map a spaCy NER label to our EntityType value.

    Args:
        label: The spaCy NER label (e.g., "PERSON", "ORG", "GPE").

    Returns:
        The corresponding EntityType value string.
    """
    return SPACY_LABEL_TO_ENTITY_TYPE.get(label, "unknown")


def _is_valid_entity(name: str, entity_type: str) -> bool:
    """
    Check if an extracted entity is valid and should be kept.

    Filters out:
    - Empty or whitespace-only names
    - Names that are too short or too long
    - Entity types we don't care about
    - Purely numeric entities (except for money)
    - Single-character entities

    Args:
        name: The entity name.
        entity_type: The mapped entity type.

    Returns:
        True if the entity should be kept, False otherwise.
    """
    # Check basic validity
    if not name or not name.strip():
        return False

    stripped = name.strip()

    # Check length
    if len(stripped) < MIN_ENTITY_LENGTH or len(stripped) > MAX_ENTITY_LENGTH:
        return False

    # Check entity type is one we extract
    if entity_type not in EXTRACTED_ENTITY_TYPES:
        return False

    # Skip purely numeric entities (except money)
    if entity_type != "money" and stripped.replace(" ", "").isdigit():
        return False

    return True


def _extract_context(text: str, start: int, end: int, window: int = CONTEXT_WINDOW) -> str:
    """
    Extract surrounding context for an entity mention.

    Args:
        text: The full text containing the mention.
        start: Start character position of the mention.
        end: End character position of the mention.
        window: Number of characters to include on each side.

    Returns:
        The context string with the mention surrounded by context.
    """
    context_start = max(0, start - window)
    context_end = min(len(text), end + window)

    prefix = text[context_start:start]
    mention = text[start:end]
    suffix = text[end:context_end]

    # Add ellipsis if truncated
    if context_start > 0:
        prefix = "..." + prefix
    if context_end < len(text):
        suffix = suffix + "..."

    return prefix + mention + suffix


def _extract_entities_from_chunk(
    nlp: Language,
    chunk: ChunkInfo,
    chunk_index: int,
) -> list[dict[str, Any]]:
    """
    Extract entities from a single chunk using spaCy NER.

    Args:
        nlp: The spaCy Language model.
        chunk: The chunk to process.
        chunk_index: The index of the chunk (for mention tracking).

    Returns:
        List of mention dictionaries with entity info.
    """
    mentions: list[dict[str, Any]] = []

    # Process the chunk content
    doc: Doc = nlp(chunk.content)

    for ent in doc.ents:
        entity_type = _map_spacy_label(ent.label_)

        if not _is_valid_entity(ent.text, entity_type):
            continue

        # Extract context around the mention
        context = _extract_context(chunk.content, ent.start_char, ent.end_char)

        mention = {
            "name": ent.text,
            "entity_type": entity_type,
            "spacy_label": ent.label_,
            "chunk_index": chunk_index,
            "start": ent.start_char,
            "end": ent.end_char,
            "context": context,
            # Track global position if chunk has position info
            "global_start": chunk.start_char + ent.start_char if chunk.start_char else None,
            "global_end": chunk.start_char + ent.end_char if chunk.start_char else None,
        }

        mentions.append(mention)

    return mentions


def _group_entity_mentions(all_mentions: list[dict[str, Any]]) -> list[EntityInfo]:
    """
    Group entity mentions by normalized name and type.

    Entities with the same normalized name and type are merged into
    a single EntityInfo with multiple mentions.

    Confidence is calculated based on:
    - Number of mentions (more mentions = higher confidence)
    - Consistency of entity type across mentions

    Args:
        all_mentions: List of all extracted mentions across chunks.

    Returns:
        List of EntityInfo objects with grouped mentions.
    """
    # Group by (normalized_name, entity_type)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for mention in all_mentions:
        normalized_name = _normalize_entity_name(mention["name"])
        entity_type = mention["entity_type"]
        key = (normalized_name, entity_type)

        if key not in grouped:
            grouped[key] = {
                "canonical_name": mention["name"],  # Use first occurrence as canonical
                "entity_type": entity_type,
                "mentions": [],
                "name_variants": set(),
            }

        # Track name variants for analysis
        grouped[key]["name_variants"].add(mention["name"])

        # Store mention info (without redundant name/type)
        mention_info = {
            "chunk_index": mention["chunk_index"],
            "start": mention["start"],
            "end": mention["end"],
            "context": mention["context"],
            "text": mention["name"],  # Original text of this mention
        }

        # Add global positions if available
        if mention.get("global_start") is not None:
            mention_info["global_start"] = mention["global_start"]
            mention_info["global_end"] = mention["global_end"]

        grouped[key]["mentions"].append(mention_info)

    # Convert to EntityInfo objects
    entities: list[EntityInfo] = []

    for (normalized_name, entity_type), data in grouped.items():
        # Calculate confidence based on mention count
        # More mentions = higher confidence (up to a cap)
        mention_count = len(data["mentions"])
        base_confidence = min(0.5 + (mention_count * 0.1), 0.95)

        # Boost confidence if name variants are consistent
        variant_count = len(data["name_variants"])
        if variant_count == 1:
            # All mentions use the exact same form
            confidence = min(base_confidence + 0.05, 0.98)
        else:
            # Multiple variants - slightly reduce confidence
            confidence = base_confidence

        # Use the most common name variant as the canonical name
        name_counts: dict[str, int] = {}
        for mention in data["mentions"]:
            name = mention["text"]
            name_counts[name] = name_counts.get(name, 0) + 1

        canonical_name = max(name_counts.items(), key=lambda x: x[1])[0]

        entity = EntityInfo(
            name=canonical_name,
            entity_type=entity_type,
            mentions=data["mentions"],
            confidence=round(confidence, 3),
        )

        entities.append(entity)

    # Sort by confidence (descending) then by mention count (descending)
    entities.sort(key=lambda e: (-e.confidence, -len(e.mentions)))

    return entities


# =============================================================================
# Activities
# =============================================================================


@activity.defn
async def extract_entities(input: ExtractEntitiesInput) -> ExtractEntitiesOutput:
    """
    Extract named entities from document content using spaCy NER.

    This activity processes each chunk independently to:
    1. Track mention positions within chunks
    2. Enable heartbeating for long documents
    3. Provide better context for each mention

    Entity types extracted (mapped from spaCy):
    - PERSON -> person
    - ORG/NORP -> organization
    - GPE/LOC/FAC -> location
    - DATE/TIME -> date
    - EVENT -> event
    - MONEY -> money
    - LAW -> law
    - PRODUCT -> product

    Args:
        input: Entity extraction input with document content and chunks.

    Returns:
        List of extracted entities with mentions and confidence scores.
    """
    activity.logger.info(
        f"Extracting entities from document {input.document_id} ({len(input.chunks)} chunks)"
    )

    # Load spaCy model (lazy loading)
    nlp = _get_spacy_model()

    # Extract entities from each chunk
    all_mentions: list[dict[str, Any]] = []
    chunks_processed = 0

    for chunk_index, chunk in enumerate(input.chunks):
        # Handle both dict and ChunkInfo objects
        if isinstance(chunk, dict):
            chunk_obj = ChunkInfo(**chunk)
        else:
            chunk_obj = chunk

        # Extract entities from this chunk
        chunk_mentions = _extract_entities_from_chunk(nlp, chunk_obj, chunk_index)
        all_mentions.extend(chunk_mentions)

        chunks_processed += 1

        # Heartbeat after each chunk for long documents
        if chunks_processed % 10 == 0:
            activity.heartbeat(f"Processed {chunks_processed}/{len(input.chunks)} chunks")

    activity.logger.info(f"Found {len(all_mentions)} raw entity mentions")

    # Group mentions into entities
    entities = _group_entity_mentions(all_mentions)

    activity.logger.info(
        f"Extracted {len(entities)} unique entities from document {input.document_id}"
    )

    # Final heartbeat
    activity.heartbeat()

    return ExtractEntitiesOutput(entities=entities)


@activity.defn
async def extract_entities_mock(input: ExtractEntitiesInput) -> ExtractEntitiesOutput:
    """
    Mock entity extraction for testing without spaCy.

    Returns configurable test entities based on content patterns.
    Useful for unit testing workflows without loading NLP models.

    Args:
        input: Entity extraction input.

    Returns:
        Mock extracted entities.
    """
    activity.logger.info(f"Mock extracting entities from document {input.document_id}")

    entities: list[EntityInfo] = []

    # Simple pattern-based extraction for testing
    content = input.content.lower()

    # Check for common test patterns
    if "john" in content or "doe" in content:
        entities.append(
            EntityInfo(
                name="John Doe",
                entity_type="person",
                mentions=[{"chunk_index": 0, "start": 0, "end": 8, "context": "John Doe..."}],
                confidence=0.9,
            )
        )

    if "acme" in content or "corp" in content:
        entities.append(
            EntityInfo(
                name="Acme Corp",
                entity_type="organization",
                mentions=[{"chunk_index": 0, "start": 10, "end": 19, "context": "...Acme Corp..."}],
                confidence=0.85,
            )
        )

    if "new york" in content or "nyc" in content:
        entities.append(
            EntityInfo(
                name="New York",
                entity_type="location",
                mentions=[{"chunk_index": 0, "start": 20, "end": 28, "context": "...New York..."}],
                confidence=0.88,
            )
        )

    activity.heartbeat()
    activity.logger.info(f"Mock extracted {len(entities)} entities")

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
