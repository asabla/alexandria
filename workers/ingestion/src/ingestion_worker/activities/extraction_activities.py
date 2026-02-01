"""
Entity and relationship extraction activities.

Activities for extracting entities, relationships, and building
the knowledge graph.
"""

from __future__ import annotations

from temporalio import activity

from ingestion_worker.workflows.document_ingestion import (
    BuildGraphInput,
    BuildGraphOutput,
    ChunkInfo,
    EntityInfo,
    EntityMatch,
    ExtractEntitiesInput,
    ExtractEntitiesOutput,
    ExtractRelationshipsInput,
    ExtractRelationshipsOutput,
    RelationshipInfo,
    ResolveEntitiesInput,
    ResolveEntitiesOutput,
)

import os
import re
import unicodedata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
# Relationship Extraction Constants
# =============================================================================

# Verb patterns that indicate relationships between entities
# Maps dependency labels to relationship types based on entity type combinations
# Format: (subject_type, object_type, verb_lemmas) -> relationship_type

# Verbs indicating employment/work relationships
WORK_VERBS = {"work", "employ", "hire", "join", "lead", "head", "manage", "direct", "run"}
OWNERSHIP_VERBS = {"own", "acquire", "purchase", "buy", "control", "hold"}
LOCATION_VERBS = {"locate", "base", "headquarter", "situate", "establish", "found", "operate"}
MEMBERSHIP_VERBS = {"join", "belong", "member", "part", "affiliate"}
PARTNERSHIP_VERBS = {"partner", "collaborate", "cooperate", "ally", "associate"}
FAMILY_VERBS = {"marry", "divorce", "parent", "child", "sibling", "relate"}

# Prepositions that indicate relationships
WORK_PREPOSITIONS = {"at", "for", "with"}
LOCATION_PREPOSITIONS = {"in", "at", "from", "near"}
OWNERSHIP_PREPOSITIONS = {"of"}

# Relationship type mapping based on entity types and verb patterns
# (source_type, target_type, pattern_type) -> relationship_type
RELATIONSHIP_TYPE_MAP: dict[tuple[str, str, str], str] = {
    # Person -> Organization
    ("person", "organization", "work"): "works_for",
    ("person", "organization", "own"): "owns",
    ("person", "organization", "lead"): "works_for",  # CEO of X = works_for
    ("person", "organization", "member"): "member_of",
    # Organization -> Organization
    ("organization", "organization", "own"): "subsidiary_of",
    ("organization", "organization", "partner"): "partner_of",
    ("organization", "organization", "compete"): "competitor_of",
    # Entity -> Location
    ("person", "location", "locate"): "located_in",
    ("organization", "location", "locate"): "headquarters_in",
    ("organization", "location", "operate"): "located_in",
    ("event", "location", "locate"): "occurred_at",
    # Person -> Person
    ("person", "person", "work"): "works_with",
    ("person", "person", "family"): "family",
    ("person", "person", "report"): "reports_to",
    # Event relationships
    ("person", "event", "participate"): "participated_in",
    ("organization", "event", "participate"): "participated_in",
    ("event", "date", "occur"): "occurred_on",
}

# Minimum confidence threshold for extracted relationships
MIN_RELATIONSHIP_CONFIDENCE = 0.3

# Maximum sentence length to process (very long sentences are often parsing errors)
MAX_SENTENCE_LENGTH = 500


# =============================================================================
# Helper Functions - Entity Extraction
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

    for (_normalized_name, entity_type), data in grouped.items():
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
# Helper Functions - Relationship Extraction
# =============================================================================


def _build_entity_lookup(
    entities: list[EntityInfo] | list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Build a lookup dictionary for entities by normalized name.

    Args:
        entities: List of EntityInfo objects or dicts.

    Returns:
        Dictionary mapping normalized names to entity info.
    """
    lookup: dict[str, dict[str, Any]] = {}

    for entity in entities:
        if isinstance(entity, dict):
            name = entity.get("name", "")
            entity_type = entity.get("entity_type", "unknown")
        else:
            name = entity.name
            entity_type = entity.entity_type

        normalized = _normalize_entity_name(name)
        lookup[normalized] = {
            "name": name,
            "entity_type": entity_type,
        }

    return lookup


def _find_entity_in_span(
    span_text: str,
    entity_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Find if any known entity is mentioned in a text span.

    Args:
        span_text: The text span to search.
        entity_lookup: Lookup dictionary of known entities.

    Returns:
        Entity info dict if found, None otherwise.
    """
    normalized_span = _normalize_entity_name(span_text)

    # Direct match
    if normalized_span in entity_lookup:
        return entity_lookup[normalized_span]

    # Check if any entity name is contained in the span
    for normalized_name, entity_info in entity_lookup.items():
        if normalized_name in normalized_span or normalized_span in normalized_name:
            return entity_info

    return None


def _classify_verb_pattern(verb_lemma: str) -> str | None:
    """
    Classify a verb lemma into a relationship pattern type.

    Args:
        verb_lemma: The lemmatized form of the verb.

    Returns:
        Pattern type string or None if not recognized.
    """
    verb = verb_lemma.lower()

    if verb in WORK_VERBS:
        return "work"
    if verb in OWNERSHIP_VERBS:
        return "own"
    if verb in LOCATION_VERBS:
        return "locate"
    if verb in MEMBERSHIP_VERBS:
        return "member"
    if verb in PARTNERSHIP_VERBS:
        return "partner"
    if verb in FAMILY_VERBS:
        return "family"

    # Check for specific patterns
    if verb in {"report", "answer"}:
        return "report"
    if verb in {"compete", "rival"}:
        return "compete"
    if verb in {"attend", "participate", "join", "speak"}:
        return "participate"
    if verb in {"happen", "occur", "take"}:
        return "occur"

    return None


def _determine_relationship_type(
    source_type: str,
    target_type: str,
    pattern_type: str,
) -> str:
    """
    Determine the relationship type based on entity types and verb pattern.

    Args:
        source_type: Entity type of the source entity.
        target_type: Entity type of the target entity.
        pattern_type: Classified verb pattern type.

    Returns:
        Relationship type string.
    """
    key = (source_type, target_type, pattern_type)

    if key in RELATIONSHIP_TYPE_MAP:
        return RELATIONSHIP_TYPE_MAP[key]

    # Fallback to generic association
    return "associated_with"


def _extract_relationships_from_sentence(
    sent,
    entity_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Extract relationships from a single sentence using dependency parsing.

    This function analyzes the dependency tree to find:
    1. Subject-verb-object patterns where subject and object are entities
    2. Prepositional phrases linking entities
    3. Appositive constructions (X, the CEO of Y)

    Args:
        sent: A spaCy Span representing a sentence.
        entity_lookup: Lookup dictionary of known entities.

    Returns:
        List of relationship dictionaries.
    """
    relationships: list[dict[str, Any]] = []

    # Skip very long sentences (often parsing errors)
    if len(sent.text) > MAX_SENTENCE_LENGTH:
        return relationships

    # Find entities in this sentence
    sentence_entities: list[tuple[Any, dict[str, Any]]] = []
    for ent in sent.ents:
        entity_info = _find_entity_in_span(ent.text, entity_lookup)
        if entity_info:
            sentence_entities.append((ent, entity_info))

    # Need at least 2 entities for a relationship
    if len(sentence_entities) < 2:
        return relationships

    # Analyze dependency structure
    for token in sent:
        # Look for verbs that might indicate relationships
        if token.pos_ == "VERB":
            pattern_type = _classify_verb_pattern(token.lemma_)

            if pattern_type:
                # Find subject and object of the verb
                subject = None
                obj = None

                for child in token.children:
                    if child.dep_ in {"nsubj", "nsubjpass"}:
                        # Check if subject contains an entity
                        subject_span = child.text
                        if child.subtree:
                            subject_tokens = list(child.subtree)
                            subject_span = sent.doc[
                                subject_tokens[0].i : subject_tokens[-1].i + 1
                            ].text
                        subject = _find_entity_in_span(subject_span, entity_lookup)

                    elif child.dep_ in {"dobj", "pobj", "attr"}:
                        # Check if object contains an entity
                        obj_span = child.text
                        if child.subtree:
                            obj_tokens = list(child.subtree)
                            obj_span = sent.doc[obj_tokens[0].i : obj_tokens[-1].i + 1].text
                        obj = _find_entity_in_span(obj_span, entity_lookup)

                    elif child.dep_ == "prep":
                        # Check prepositional phrases
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                pobj_span = pobj.text
                                if pobj.subtree:
                                    pobj_tokens = list(pobj.subtree)
                                    pobj_span = sent.doc[
                                        pobj_tokens[0].i : pobj_tokens[-1].i + 1
                                    ].text
                                prep_entity = _find_entity_in_span(pobj_span, entity_lookup)
                                if prep_entity and not obj:
                                    obj = prep_entity

                if subject and obj and subject["name"] != obj["name"]:
                    rel_type = _determine_relationship_type(
                        subject["entity_type"],
                        obj["entity_type"],
                        pattern_type,
                    )

                    relationships.append(
                        {
                            "source_entity": subject["name"],
                            "source_type": subject["entity_type"],
                            "target_entity": obj["name"],
                            "target_type": obj["entity_type"],
                            "relationship_type": rel_type,
                            "evidence": sent.text.strip(),
                            "verb": token.lemma_,
                            "pattern_type": pattern_type,
                        }
                    )

    # Also check for co-occurrence based relationships (entities in same sentence)
    # This is a fallback for cases where dependency parsing doesn't capture the relationship
    if not relationships and len(sentence_entities) >= 2:
        # Check for appositive patterns (X, the Y of Z)
        for i, (ent1, info1) in enumerate(sentence_entities):
            for ent2, info2 in sentence_entities[i + 1 :]:
                # Skip if same entity
                if info1["name"] == info2["name"]:
                    continue

                # Check if entities are close together (potential appositive)
                distance = abs(ent1.start - ent2.start)
                if distance <= 5:  # Within 5 tokens
                    # Infer relationship type from entity types
                    if info1["entity_type"] == "person" and info2["entity_type"] == "organization":
                        relationships.append(
                            {
                                "source_entity": info1["name"],
                                "source_type": info1["entity_type"],
                                "target_entity": info2["name"],
                                "target_type": info2["entity_type"],
                                "relationship_type": "works_for",
                                "evidence": sent.text.strip(),
                                "verb": None,
                                "pattern_type": "cooccurrence",
                            }
                        )
                    elif (
                        info1["entity_type"] == "organization"
                        and info2["entity_type"] == "location"
                    ):
                        relationships.append(
                            {
                                "source_entity": info1["name"],
                                "source_type": info1["entity_type"],
                                "target_entity": info2["name"],
                                "target_type": info2["entity_type"],
                                "relationship_type": "located_in",
                                "evidence": sent.text.strip(),
                                "verb": None,
                                "pattern_type": "cooccurrence",
                            }
                        )

    return relationships


def _calculate_relationship_confidence(
    relationship: dict[str, Any],
    occurrence_count: int,
) -> float:
    """
    Calculate confidence score for a relationship.

    Factors considered:
    - Pattern type (verb-based is higher than co-occurrence)
    - Number of occurrences in the document
    - Relationship type specificity

    Args:
        relationship: The relationship dictionary.
        occurrence_count: Number of times this relationship was found.

    Returns:
        Confidence score between 0 and 1.
    """
    base_confidence = 0.4

    # Boost for verb-based extraction
    if relationship.get("verb"):
        base_confidence += 0.2

    # Boost for multiple occurrences
    if occurrence_count >= 2:
        base_confidence += min(occurrence_count * 0.05, 0.2)

    # Boost for specific relationship types (vs generic associated_with)
    if relationship["relationship_type"] != "associated_with":
        base_confidence += 0.1

    # Cap at 0.95
    return min(round(base_confidence, 3), 0.95)


def _deduplicate_relationships(
    all_relationships: list[dict[str, Any]],
) -> list[RelationshipInfo]:
    """
    Deduplicate relationships and convert to RelationshipInfo objects.

    Relationships are considered duplicates if they have the same
    source entity, target entity, and relationship type.

    Args:
        all_relationships: List of raw relationship dictionaries.

    Returns:
        List of deduplicated RelationshipInfo objects.
    """
    # Group by (source, target, type)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for rel in all_relationships:
        # Normalize entity names for grouping
        source_norm = _normalize_entity_name(rel["source_entity"])
        target_norm = _normalize_entity_name(rel["target_entity"])
        key = (source_norm, target_norm, rel["relationship_type"])

        if key not in grouped:
            grouped[key] = []
        grouped[key].append(rel)

    # Convert to RelationshipInfo objects
    results: list[RelationshipInfo] = []

    for (_source_norm, _target_norm, rel_type), occurrences in grouped.items():
        # Use the first occurrence for canonical names and evidence
        first = occurrences[0]

        # Collect all evidence texts (use first one)
        evidence_texts = list({r["evidence"] for r in occurrences})
        evidence = evidence_texts[0]

        confidence = _calculate_relationship_confidence(first, len(occurrences))

        # Filter by minimum confidence
        if confidence >= MIN_RELATIONSHIP_CONFIDENCE:
            results.append(
                RelationshipInfo(
                    source_entity=first["source_entity"],
                    target_entity=first["target_entity"],
                    relationship_type=rel_type,
                    evidence=evidence,
                    confidence=confidence,
                )
            )

    # Sort by confidence (descending)
    results.sort(key=lambda r: -r.confidence)

    return results


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
    Extract relationships between entities using spaCy dependency parsing.

    This activity analyzes the document content to find relationships
    between previously extracted entities. It uses:
    1. Dependency parsing to identify subject-verb-object patterns
    2. Prepositional phrase analysis
    3. Co-occurrence heuristics as fallback

    Relationship types extracted include:
    - works_for, works_with (person-organization)
    - owns, controls (ownership)
    - located_in, headquarters_in (location)
    - member_of, subsidiary_of (membership)
    - participated_in, occurred_at (events)
    - associated_with (generic fallback)

    Args:
        input: Relationship extraction input with content and entities.

    Returns:
        List of extracted relationships with confidence scores.
    """
    activity.logger.info(
        f"Extracting relationships from document {input.document_id} "
        f"({len(input.entities)} entities)"
    )

    # Handle case with no or single entity
    if len(input.entities) < 2:
        activity.logger.info("Not enough entities for relationship extraction")
        activity.heartbeat()
        return ExtractRelationshipsOutput(relationships=[])

    # Build entity lookup for efficient matching
    entity_lookup = _build_entity_lookup(input.entities)

    # Load spaCy model
    nlp = _get_spacy_model()

    # Process content sentence by sentence
    doc = nlp(input.content)

    all_relationships: list[dict[str, Any]] = []
    sentences_processed = 0

    for sent in doc.sents:
        # Extract relationships from this sentence
        sent_relationships = _extract_relationships_from_sentence(sent, entity_lookup)
        all_relationships.extend(sent_relationships)

        sentences_processed += 1

        # Heartbeat periodically
        if sentences_processed % 50 == 0:
            activity.heartbeat(f"Processed {sentences_processed} sentences")

    activity.logger.info(f"Found {len(all_relationships)} raw relationships")

    # Deduplicate and convert to RelationshipInfo
    relationships = _deduplicate_relationships(all_relationships)

    activity.logger.info(
        f"Extracted {len(relationships)} unique relationships from document {input.document_id}"
    )

    # Final heartbeat
    activity.heartbeat()

    return ExtractRelationshipsOutput(relationships=relationships)


@activity.defn
async def extract_relationships_mock(
    input: ExtractRelationshipsInput,
) -> ExtractRelationshipsOutput:
    """
    Mock relationship extraction for testing without spaCy.

    Returns configurable test relationships based on entity patterns.
    Useful for unit testing workflows without loading NLP models.

    Args:
        input: Relationship extraction input.

    Returns:
        Mock extracted relationships.
    """
    activity.logger.info(f"Mock extracting relationships from document {input.document_id}")

    relationships: list[RelationshipInfo] = []

    # Build simple entity name set for pattern matching
    entity_names = set()
    entity_types: dict[str, str] = {}

    for entity in input.entities:
        if isinstance(entity, dict):
            name = entity.get("name", "").lower()
            entity_type = entity.get("entity_type", "unknown")
        else:
            name = entity.name.lower()
            entity_type = entity.entity_type

        entity_names.add(name)
        entity_types[name] = entity_type

    # Check for common test patterns
    content_lower = input.content.lower()

    # Person works for Organization
    if any("john" in n or "doe" in n for n in entity_names):
        if any("acme" in n or "corp" in n for n in entity_names):
            if "works" in content_lower or "employee" in content_lower or "ceo" in content_lower:
                relationships.append(
                    RelationshipInfo(
                        source_entity="John Doe",
                        target_entity="Acme Corp",
                        relationship_type="works_for",
                        evidence="John Doe works at Acme Corp.",
                        confidence=0.8,
                    )
                )

    # Organization located in Location
    has_acme = any("acme" in n for n in entity_names)
    has_location = any("new york" in n or "york" in n or "nyc" in n for n in entity_names)
    if has_acme and has_location:
        if (
            "located" in content_lower
            or "headquarter" in content_lower  # matches headquarters, headquartered
            or "based" in content_lower
        ):
            relationships.append(
                RelationshipInfo(
                    source_entity="Acme Corp",
                    target_entity="New York",
                    relationship_type="headquarters_in",
                    evidence="Acme Corp is headquartered in New York.",
                    confidence=0.75,
                )
            )

    activity.heartbeat()
    activity.logger.info(f"Mock extracted {len(relationships)} relationships")

    return ExtractRelationshipsOutput(relationships=relationships)


# =============================================================================
# Graph Building Constants
# =============================================================================

# Environment variables for Neo4j configuration
NEO4J_URI_ENV = "NEO4J_URI"
NEO4J_USER_ENV = "NEO4J_USER"
NEO4J_PASSWORD_ENV = "NEO4J_PASSWORD"  # pragma: allowlist secret
NEO4J_DATABASE_ENV = "NEO4J_DATABASE"

# Default values
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "password"  # pragma: allowlist secret
DEFAULT_NEO4J_DATABASE = "neo4j"


def _generate_entity_id(tenant_id: str, name: str, entity_type: str) -> str:
    """
    Generate a deterministic entity ID for idempotent node creation.

    Uses normalized name and type to ensure the same entity always
    gets the same ID, enabling proper MERGE behavior.

    Args:
        tenant_id: Tenant identifier
        name: Entity name
        entity_type: Entity type

    Returns:
        Deterministic entity ID string
    """
    import hashlib

    # Normalize name for consistent IDs
    normalized_name = _normalize_entity_name(name)
    id_string = f"{tenant_id}:{entity_type}:{normalized_name}"
    hash_val = hashlib.sha256(id_string.encode()).hexdigest()
    return f"ent_{hash_val[:24]}"


def _entity_type_to_label(entity_type: str) -> str:
    """
    Convert entity type to Neo4j node label.

    Args:
        entity_type: Entity type string (e.g., "person", "organization")

    Returns:
        Neo4j label string (e.g., "Person", "Organization")
    """
    # Map to proper capitalized labels
    label_map = {
        "person": "Person",
        "organization": "Organization",
        "location": "Location",
        "date": "Date",
        "event": "Event",
        "money": "Money",
        "law": "Law",
        "product": "Product",
        "document": "Document",
        "unknown": "Entity",
    }
    return label_map.get(entity_type.lower(), "Entity")


def _relationship_type_to_label(relationship_type: str) -> str:
    """
    Convert relationship type to Neo4j relationship label.

    Args:
        relationship_type: Relationship type string (e.g., "works_for")

    Returns:
        Neo4j relationship label (e.g., "WORKS_FOR")
    """
    # Convert to uppercase with underscores (Neo4j convention)
    return relationship_type.upper().replace(" ", "_")


async def _create_entity_nodes(
    client: Any,
    entities: list[EntityInfo] | list[dict[str, Any]],
    _document_id: str,
    tenant_id: str,
    _project_id: str | None,
) -> tuple[int, dict[str, str]]:
    """
    Create or merge entity nodes in Neo4j.

    Args:
        client: Neo4j client instance
        entities: List of entities to create
        _document_id: Document ID (reserved for future use)
        tenant_id: Tenant ID
        _project_id: Optional project ID (reserved for future use)

    Returns:
        Tuple of (nodes_created, entity_id_map)
    """
    nodes_created = 0
    entity_id_map: dict[str, str] = {}  # Maps entity name -> entity_id

    for entity in entities:
        # Handle both EntityInfo and dict
        if isinstance(entity, dict):
            name = entity.get("name", "")
            entity_type = entity.get("entity_type", "unknown")
            confidence = entity.get("confidence", 0.5)
            mentions = entity.get("mentions", [])
        else:
            name = entity.name
            entity_type = entity.entity_type
            confidence = entity.confidence
            mentions = entity.mentions

        if not name:
            continue

        # Generate deterministic entity ID
        entity_id = _generate_entity_id(tenant_id, name, entity_type)
        entity_id_map[name] = entity_id

        # Get Neo4j label
        label = _entity_type_to_label(entity_type)

        # Build properties
        properties = {
            "name": name,
            "canonical_name": _normalize_entity_name(name),
            "entity_type": entity_type,
            "confidence": confidence,
            "mention_count": len(mentions),
        }

        # Create/merge entity node using custom query for dynamic label
        query = f"""
        MERGE (e:Entity:{label} {{id: $entity_id, tenant_id: $tenant_id}})
        ON CREATE SET e += $properties, e.created_at = datetime()
        ON MATCH SET e += $properties, e.updated_at = datetime()
        RETURN e
        """

        await client.execute_query(
            query,
            {
                "entity_id": entity_id,
                "tenant_id": tenant_id,
                "properties": properties,
            },
        )
        nodes_created += 1

    return nodes_created, entity_id_map


async def _create_document_node_and_mentions(
    client: Any,
    document_id: str,
    tenant_id: str,
    entity_id_map: dict[str, str],
    entities: list[EntityInfo] | list[dict[str, Any]],
) -> int:
    """
    Create document node and MENTIONED_IN relationships.

    Args:
        client: Neo4j client instance
        document_id: Document ID
        tenant_id: Tenant ID
        entity_id_map: Map of entity name -> entity_id
        entities: List of entities

    Returns:
        Number of MENTIONED_IN relationships created
    """
    # Create document node
    await client.create_document_node(
        document_id=document_id,
        tenant_id=tenant_id,
        properties={"processed_at": "datetime()"},
    )

    mentions_created = 0

    for entity in entities:
        if isinstance(entity, dict):
            name = entity.get("name", "")
            mentions = entity.get("mentions", [])
        else:
            name = entity.name
            mentions = entity.mentions

        entity_id = entity_id_map.get(name)
        if not entity_id:
            continue

        # Create MENTIONED_IN relationship with mention details
        mention_properties = {
            "mention_count": len(mentions),
            "first_mention_chunk": mentions[0].get("chunk_index", 0) if mentions else 0,
        }

        try:
            await client.create_mentioned_in_relationship(
                entity_id=entity_id,
                document_id=document_id,
                tenant_id=tenant_id,
                properties=mention_properties,
            )
            mentions_created += 1
        except Exception as e:
            activity.logger.warning(f"Failed to create MENTIONED_IN for {name}: {e}")

    return mentions_created


async def _create_entity_relationships(
    client: Any,
    relationships: list[RelationshipInfo] | list[dict[str, Any]],
    tenant_id: str,
    entity_id_map: dict[str, str],
) -> int:
    """
    Create relationships between entity nodes.

    Args:
        client: Neo4j client instance
        relationships: List of relationships to create
        tenant_id: Tenant ID
        entity_id_map: Map of entity name -> entity_id

    Returns:
        Number of relationships created
    """
    relationships_created = 0

    for rel in relationships:
        if isinstance(rel, dict):
            source_name = rel.get("source_entity", "")
            target_name = rel.get("target_entity", "")
            rel_type = rel.get("relationship_type", "ASSOCIATED_WITH")
            evidence = rel.get("evidence", "")
            confidence = rel.get("confidence", 0.5)
        else:
            source_name = rel.source_entity
            target_name = rel.target_entity
            rel_type = rel.relationship_type
            evidence = rel.evidence
            confidence = rel.confidence

        # Get entity IDs
        source_id = entity_id_map.get(source_name)
        target_id = entity_id_map.get(target_name)

        if not source_id or not target_id:
            activity.logger.debug(
                f"Skipping relationship {source_name} -> {target_name}: entity not found"
            )
            continue

        # Convert relationship type to Neo4j label
        rel_label = _relationship_type_to_label(rel_type)

        # Build properties
        properties = {
            "relationship_type": rel_type,
            "evidence": evidence[:500] if evidence else "",  # Truncate long evidence
            "confidence": confidence,
        }

        try:
            await client.create_relationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=rel_label,
                tenant_id=tenant_id,
                properties=properties,
            )
            relationships_created += 1
        except Exception as e:
            activity.logger.warning(
                f"Failed to create relationship {source_name} -> {target_name}: {e}"
            )

    return relationships_created


@activity.defn
async def build_graph(input: BuildGraphInput) -> BuildGraphOutput:
    """
    Build knowledge graph in Neo4j.

    Creates or updates entity nodes, document nodes, and relationship edges
    in the knowledge graph. Uses MERGE for idempotent operations.

    Operations performed:
    1. Create/update entity nodes with type-specific labels
    2. Create document node
    3. Create MENTIONED_IN relationships between entities and document
    4. Create inter-entity relationships

    Args:
        input: Graph building input with entities and relationships

    Returns:
        Count of created nodes and relationships
    """
    # Lazy import to avoid Temporal sandbox issues
    from alexandria_db.clients import Neo4jClient

    activity.logger.info(
        f"Building knowledge graph for document {input.document_id}",
        extra={
            "document_id": input.document_id,
            "tenant_id": input.tenant_id,
            "entity_count": len(input.entities),
            "relationship_count": len(input.relationships),
        },
    )

    # Handle case with no entities
    if not input.entities:
        activity.logger.info("No entities to add to graph")
        activity.heartbeat()
        return BuildGraphOutput(nodes_created=0, relationships_created=0)

    # Get configuration from environment
    neo4j_uri = os.getenv(NEO4J_URI_ENV, DEFAULT_NEO4J_URI)
    neo4j_user = os.getenv(NEO4J_USER_ENV, DEFAULT_NEO4J_USER)
    neo4j_password = os.getenv(NEO4J_PASSWORD_ENV, DEFAULT_NEO4J_PASSWORD)
    neo4j_database = os.getenv(NEO4J_DATABASE_ENV, DEFAULT_NEO4J_DATABASE)

    # Create client
    client = Neo4jClient(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
    )

    try:
        # Step 1: Create entity nodes
        activity.heartbeat("Creating entity nodes")
        nodes_created, entity_id_map = await _create_entity_nodes(
            client=client,
            entities=input.entities,
            _document_id=input.document_id,
            tenant_id=input.tenant_id,
            _project_id=input.project_id,
        )
        activity.logger.info(f"Created {nodes_created} entity nodes")

        # Step 2: Create document node and MENTIONED_IN relationships
        activity.heartbeat("Creating document mentions")
        mentions_created = await _create_document_node_and_mentions(
            client=client,
            document_id=input.document_id,
            tenant_id=input.tenant_id,
            entity_id_map=entity_id_map,
            entities=input.entities,
        )
        activity.logger.info(f"Created {mentions_created} MENTIONED_IN relationships")

        # Step 3: Create inter-entity relationships
        activity.heartbeat("Creating entity relationships")
        relationships_created = await _create_entity_relationships(
            client=client,
            relationships=input.relationships,
            tenant_id=input.tenant_id,
            entity_id_map=entity_id_map,
        )
        activity.logger.info(f"Created {relationships_created} entity relationships")

        # Total relationships = MENTIONED_IN + inter-entity
        total_relationships = mentions_created + relationships_created

        activity.logger.info(
            f"Graph built: {nodes_created} nodes, {total_relationships} relationships",
            extra={
                "document_id": input.document_id,
                "nodes_created": nodes_created,
                "mentions_created": mentions_created,
                "relationships_created": relationships_created,
            },
        )

        return BuildGraphOutput(
            nodes_created=nodes_created,
            relationships_created=total_relationships,
        )

    finally:
        await client.close()


@activity.defn
async def build_graph_mock(input: BuildGraphInput) -> BuildGraphOutput:
    """
    Mock graph building for testing without Neo4j.

    Simulates graph building by counting entities and relationships.
    Useful for unit testing workflows without a running Neo4j instance.

    Args:
        input: Graph building input

    Returns:
        Mock counts based on input
    """
    activity.logger.info(f"Mock building graph for document {input.document_id}")

    nodes_created = len(input.entities)
    # Count MENTIONED_IN (one per entity) + inter-entity relationships
    relationships_created = len(input.entities) + len(input.relationships)

    activity.heartbeat()

    activity.logger.info(
        f"Mock graph built: {nodes_created} nodes, {relationships_created} relationships"
    )

    return BuildGraphOutput(
        nodes_created=nodes_created,
        relationships_created=relationships_created,
    )


# =============================================================================
# Entity Resolution Activity
# =============================================================================

# Environment variables for LLM configuration
LLM_BASE_URL_ENV = "VLLM_BASE_URL"
LLM_API_KEY_ENV = "OPENAI_API_KEY"  # pragma: allowlist secret
LLM_MODEL_ENV = "LLM_MODEL"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Prompt template for entity verification
ENTITY_VERIFICATION_PROMPT = """You are an expert at entity resolution. Your task is to determine if two entity names refer to the same real-world entity.

Entity 1: {entity1_name} (type: {entity1_type})
Entity 2: {entity2_name} (type: {entity2_type})

Consider:
- Are these likely the same person/organization/location?
- Could one be a nickname, abbreviation, or variant of the other?
- Account for common name variations (John vs Jonathan, IBM vs International Business Machines)

Respond with ONLY one of these options:
- SAME: if you are confident they refer to the same entity
- DIFFERENT: if you are confident they are different entities
- UNCERTAIN: if you cannot determine with confidence

Your response:"""


async def _call_llm_api(
    prompt: str,
    base_url: str,
    api_key: str | None,
    model: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
) -> str:
    """
    Call the LLM API for chat completion.

    Args:
        prompt: The prompt to send
        base_url: API base URL
        api_key: Optional API key
        model: Model name
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        The model's response text
    """
    import httpx

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=httpx.Timeout(30.0),
    ) as client:
        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


async def _verify_match_with_llm(
    entity1_name: str,
    entity1_type: str,
    entity2_name: str,
    entity2_type: str,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[bool, float, str]:
    """
    Use LLM to verify if two entities are the same.

    Args:
        entity1_name: First entity name
        entity1_type: First entity type
        entity2_name: Second entity name
        entity2_type: Second entity type
        base_url: Optional LLM API base URL
        api_key: Optional API key
        model: Optional model name

    Returns:
        Tuple of (is_match, confidence, reason)
        - is_match: True if LLM says they're the same
        - confidence: 0.0-1.0 confidence score
        - reason: "llm_verified", "llm_rejected", or "llm_uncertain"
    """
    # Use environment variables if not provided
    base_url = base_url or os.environ.get(LLM_BASE_URL_ENV, DEFAULT_LLM_BASE_URL)
    api_key = api_key or os.environ.get(LLM_API_KEY_ENV)
    model = model or os.environ.get(LLM_MODEL_ENV, DEFAULT_LLM_MODEL)

    prompt = ENTITY_VERIFICATION_PROMPT.format(
        entity1_name=entity1_name,
        entity1_type=entity1_type,
        entity2_name=entity2_name,
        entity2_type=entity2_type,
    )

    try:
        response = await _call_llm_api(
            prompt=prompt,
            base_url=base_url,
            api_key=api_key,
            model=model,
        )

        # Parse response
        response_upper = response.upper().strip()

        if "SAME" in response_upper and "DIFFERENT" not in response_upper:
            return (True, 0.95, "llm_verified")
        elif "DIFFERENT" in response_upper:
            return (False, 0.95, "llm_rejected")
        else:
            # UNCERTAIN or unclear response
            return (False, 0.5, "llm_uncertain")

    except Exception as e:
        # Log error but don't fail - fall back to name similarity only
        activity.logger.warning(f"LLM verification failed: {e}")
        return (False, 0.5, "llm_error")


def _normalize_for_comparison(name: str) -> str:
    """
    Normalize a name for comparison.

    - Lowercase
    - Remove extra whitespace
    - Remove common suffixes/prefixes (Mr., Mrs., Dr., Inc., Corp., etc.)
    - Remove punctuation

    Args:
        name: Entity name to normalize

    Returns:
        Normalized name for comparison
    """
    import re

    normalized = name.lower().strip()

    # Remove common titles/prefixes
    prefixes = [
        r"^mr\.?\s+",
        r"^mrs\.?\s+",
        r"^ms\.?\s+",
        r"^dr\.?\s+",
        r"^prof\.?\s+",
        r"^the\s+",
    ]
    for prefix in prefixes:
        normalized = re.sub(prefix, "", normalized, flags=re.IGNORECASE)

    # Remove common suffixes for organizations
    suffixes = [
        r"\s+inc\.?$",
        r"\s+corp\.?$",
        r"\s+corporation$",
        r"\s+llc\.?$",
        r"\s+ltd\.?$",
        r"\s+limited$",
        r"\s+co\.?$",
        r"\s+company$",
        r"\s+plc\.?$",
        r"\s+gmbh$",
        r",?\s+jr\.?$",
        r",?\s+sr\.?$",
        r",?\s+ii$",
        r",?\s+iii$",
        r",?\s+iv$",
    ]
    for suffix in suffixes:
        normalized = re.sub(suffix, "", normalized, flags=re.IGNORECASE)

    # Remove punctuation except spaces
    normalized = re.sub(r"[^\w\s]", "", normalized)

    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def _calculate_name_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two names using multiple methods.

    Combines:
    - Exact match (normalized)
    - Token overlap (Jaccard similarity)
    - Edit distance (Levenshtein ratio)

    Args:
        name1: First name
        name2: Second name

    Returns:
        Similarity score between 0 and 1
    """
    norm1 = _normalize_for_comparison(name1)
    norm2 = _normalize_for_comparison(name2)

    # Empty check first - empty strings are not similar
    if not norm1 or not norm2:
        return 0.0

    # Exact match after normalization
    if norm1 == norm2:
        return 1.0

    # Token-based Jaccard similarity
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())

    if tokens1 and tokens2:
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        jaccard = intersection / union if union > 0 else 0.0
    else:
        jaccard = 0.0

    # Character-level Levenshtein ratio
    # Simple implementation for short strings
    len1, len2 = len(norm1), len(norm2)
    max_len = max(len1, len2)

    if max_len == 0:
        return 0.0

    # Use difflib for sequence matching (simpler than full Levenshtein)
    from difflib import SequenceMatcher

    seq_ratio = SequenceMatcher(None, norm1, norm2).ratio()

    # Combine scores with weights
    # Jaccard is good for partial matches, seq_ratio for character similarity
    combined = (jaccard * 0.4) + (seq_ratio * 0.6)

    return combined


def _is_substring_match(name1: str, name2: str) -> bool:
    """
    Check if one name is a substring/abbreviation of the other.

    Examples:
    - "John Smith" contains "John"
    - "IBM" is abbreviation of "International Business Machines"

    Args:
        name1: First name
        name2: Second name

    Returns:
        True if one is substring/abbreviation of the other
    """
    norm1 = _normalize_for_comparison(name1)
    norm2 = _normalize_for_comparison(name2)

    # One is substring of other
    if norm1 in norm2 or norm2 in norm1:
        return True

    # Check for initials/abbreviation
    tokens1 = norm1.split()
    tokens2 = norm2.split()

    # Check if shorter is initials of longer
    if len(tokens1) == 1 and len(tokens2) > 1:
        # e.g., "ibm" vs "international business machines"
        initials = "".join(t[0] for t in tokens2 if t)
        if tokens1[0] == initials:
            return True
    elif len(tokens2) == 1 and len(tokens1) > 1:
        initials = "".join(t[0] for t in tokens1 if t)
        if tokens2[0] == initials:
            return True

    return False


def _find_candidate_matches(
    entities: list[dict[str, Any]],
    similarity_threshold: float,
) -> list[tuple[dict[str, Any], dict[str, Any], float, str]]:
    """
    Find candidate entity pairs that might be the same.

    Uses blocking by entity type and first character to reduce comparisons.

    Args:
        entities: List of entity dicts with id, name, entity_type
        similarity_threshold: Minimum similarity score

    Returns:
        List of (entity1, entity2, score, reason) tuples
    """
    from collections import defaultdict

    candidates: list[tuple[dict[str, Any], dict[str, Any], float, str]] = []

    # Block by entity type and first character of normalized name
    blocks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for entity in entities:
        entity_type = entity.get("entity_type", "unknown")
        normalized = _normalize_for_comparison(entity.get("name", ""))
        if normalized:
            first_char = normalized[0]
            blocks[(entity_type, first_char)].append(entity)

    # Compare within blocks
    for _block_key, block_entities in blocks.items():
        n = len(block_entities)
        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = block_entities[i], block_entities[j]

                # Skip if same entity ID
                if e1.get("id") == e2.get("id"):
                    continue

                name1 = e1.get("name", "")
                name2 = e2.get("name", "")

                # Calculate similarity
                similarity = _calculate_name_similarity(name1, name2)

                if similarity >= similarity_threshold:
                    candidates.append((e1, e2, similarity, "name_similarity"))
                elif _is_substring_match(name1, name2):
                    # Substring matches get a boosted score
                    candidates.append((e1, e2, 0.75, "substring_match"))

    # Also check across similar first characters (for typos)
    # e.g., "john" vs "jon" would be in different first-char blocks
    type_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        entity_type = entity.get("entity_type", "unknown")
        type_groups[entity_type].append(entity)

    for _entity_type, type_entities in type_groups.items():
        # For small groups, do pairwise comparison
        if len(type_entities) <= 100:
            for i in range(len(type_entities)):
                for j in range(i + 1, len(type_entities)):
                    e1, e2 = type_entities[i], type_entities[j]

                    # Skip if same ID or already compared
                    if e1.get("id") == e2.get("id"):
                        continue

                    name1 = e1.get("name", "")
                    name2 = e2.get("name", "")

                    # Only check if not already in candidates
                    pair_key = tuple(sorted([e1.get("id", ""), e2.get("id", "")]))
                    existing_keys = {
                        tuple(sorted([c[0].get("id", ""), c[1].get("id", "")])) for c in candidates
                    }

                    if pair_key not in existing_keys:
                        similarity = _calculate_name_similarity(name1, name2)
                        if similarity >= similarity_threshold:
                            candidates.append((e1, e2, similarity, "name_similarity"))

    return candidates


async def _fetch_entities_from_neo4j(
    client: Any,
    tenant_id: str,
    entity_type: str | None,
    max_entities: int,
) -> list[dict[str, Any]]:
    """
    Fetch entities from Neo4j for resolution.

    Args:
        client: Neo4j client
        tenant_id: Tenant ID
        entity_type: Optional entity type filter
        max_entities: Maximum entities to fetch

    Returns:
        List of entity dicts
    """
    if entity_type:
        query = f"""
        MATCH (e:Entity:{entity_type} {{tenant_id: $tenant_id}})
        WHERE NOT (e)-[:SAME_AS]-()
        RETURN e.id as id, e.name as name, e.entity_type as entity_type,
               e.confidence as confidence
        LIMIT $limit
        """
    else:
        query = """
        MATCH (e:Entity {tenant_id: $tenant_id})
        WHERE NOT (e)-[:SAME_AS]-()
        RETURN e.id as id, e.name as name, e.entity_type as entity_type,
               e.confidence as confidence
        LIMIT $limit
        """

    results = await client.execute_query(
        query,
        {"tenant_id": tenant_id, "limit": max_entities},
    )

    return [
        {
            "id": r["id"],
            "name": r["name"],
            "entity_type": r["entity_type"],
            "confidence": r.get("confidence", 0.5),
        }
        for r in results
    ]


async def _create_same_as_relationship(
    client: Any,
    entity1_id: str,
    entity2_id: str,
    tenant_id: str,
    similarity_score: float,
    match_reason: str,
) -> bool:
    """
    Create a SAME_AS relationship between two entities.

    SAME_AS is bidirectional, so we create it in one direction.
    The entity with more mentions or higher confidence becomes the "canonical" one.

    Args:
        client: Neo4j client
        entity1_id: First entity ID
        entity2_id: Second entity ID
        tenant_id: Tenant ID
        similarity_score: How similar the entities are
        match_reason: Why they were matched

    Returns:
        True if relationship was created
    """
    query = """
    MATCH (e1:Entity {id: $entity1_id, tenant_id: $tenant_id})
    MATCH (e2:Entity {id: $entity2_id, tenant_id: $tenant_id})
    WHERE NOT (e1)-[:SAME_AS]-(e2)
    MERGE (e1)-[r:SAME_AS]->(e2)
    SET r.similarity_score = $similarity_score,
        r.match_reason = $match_reason,
        r.created_at = datetime()
    RETURN r
    """

    results = await client.execute_query(
        query,
        {
            "entity1_id": entity1_id,
            "entity2_id": entity2_id,
            "tenant_id": tenant_id,
            "similarity_score": similarity_score,
            "match_reason": match_reason,
        },
    )

    return len(results) > 0


@activity.defn
async def resolve_entities(input: ResolveEntitiesInput) -> ResolveEntitiesOutput:
    """
    Resolve and deduplicate entities within a tenant.

    This activity:
    1. Fetches entities from Neo4j that don't already have SAME_AS relationships
    2. Finds candidate matches using name similarity
    3. Creates SAME_AS relationships for high-confidence matches
    4. Optionally uses LLM verification for ambiguous cases

    The SAME_AS relationship connects entities that represent the same
    real-world entity (e.g., "John Smith" and "J. Smith").

    Args:
        input: Resolution parameters including tenant_id and thresholds

    Returns:
        Resolution statistics and match details
    """
    from alexandria_db.clients import Neo4jClient

    activity.logger.info(
        f"Resolving entities for tenant {input.tenant_id}, "
        f"type={input.entity_type}, threshold={input.similarity_threshold}"
    )

    # Get Neo4j connection from environment
    neo4j_uri = os.environ.get(NEO4J_URI_ENV, DEFAULT_NEO4J_URI)
    neo4j_user = os.environ.get(NEO4J_USER_ENV, DEFAULT_NEO4J_USER)
    neo4j_password = os.environ.get(NEO4J_PASSWORD_ENV, DEFAULT_NEO4J_PASSWORD)
    neo4j_database = os.environ.get(NEO4J_DATABASE_ENV, DEFAULT_NEO4J_DATABASE)

    client = Neo4jClient(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
    )

    try:
        # Step 1: Fetch entities
        activity.heartbeat("Fetching entities")
        entities = await _fetch_entities_from_neo4j(
            client,
            input.tenant_id,
            input.entity_type,
            input.max_entities,
        )
        activity.logger.info(f"Fetched {len(entities)} entities for resolution")

        if len(entities) < 2:
            activity.logger.info("Not enough entities to resolve")
            return ResolveEntitiesOutput(
                matches_found=0,
                same_as_relationships_created=0,
                entities_processed=len(entities),
                match_details=[],
            )

        # Step 2: Find candidate matches
        activity.heartbeat("Finding candidate matches")
        candidates = _find_candidate_matches(entities, input.similarity_threshold)
        activity.logger.info(f"Found {len(candidates)} candidate matches")

        # Step 3: Process candidates and create SAME_AS relationships
        activity.heartbeat("Processing candidate matches")
        matches_created = 0
        llm_verified = 0
        llm_rejected = 0
        match_details: list[EntityMatch] = []

        # Threshold for automatic acceptance vs LLM verification
        high_confidence_threshold = 0.95

        for i, (e1, e2, score, reason) in enumerate(candidates):
            final_score = score
            final_reason = reason

            # For ambiguous cases, use LLM verification if enabled
            if input.use_llm_verification and score < high_confidence_threshold:
                activity.heartbeat(f"LLM verifying match {i + 1}/{len(candidates)}")

                is_match, llm_confidence, llm_reason = await _verify_match_with_llm(
                    entity1_name=e1["name"],
                    entity1_type=e1.get("entity_type", "unknown"),
                    entity2_name=e2["name"],
                    entity2_type=e2.get("entity_type", "unknown"),
                )

                if llm_reason == "llm_verified":
                    # LLM confirmed the match
                    final_score = max(score, llm_confidence)
                    final_reason = "llm_verified"
                    llm_verified += 1
                elif llm_reason == "llm_rejected":
                    # LLM rejected the match - skip this candidate
                    llm_rejected += 1
                    activity.logger.debug(f"LLM rejected match: {e1['name']} vs {e2['name']}")
                    continue
                # For uncertain/error, fall through and use original score

            # Create the SAME_AS relationship
            created = await _create_same_as_relationship(
                client,
                e1["id"],
                e2["id"],
                input.tenant_id,
                final_score,
                final_reason,
            )

            if created:
                matches_created += 1
                match_details.append(
                    EntityMatch(
                        entity1_id=e1["id"],
                        entity1_name=e1["name"],
                        entity2_id=e2["id"],
                        entity2_name=e2["name"],
                        similarity_score=final_score,
                        match_reason=final_reason,
                    )
                )

            # Heartbeat periodically
            if (i + 1) % 10 == 0:
                activity.heartbeat(
                    f"Processed {i + 1}/{len(candidates)} candidates, "
                    f"{matches_created} matches created"
                )

        if input.use_llm_verification:
            activity.logger.info(
                f"Entity resolution complete: {matches_created} SAME_AS relationships created "
                f"(LLM verified: {llm_verified}, LLM rejected: {llm_rejected})"
            )
        else:
            activity.logger.info(
                f"Entity resolution complete: {matches_created} SAME_AS relationships created"
            )

        return ResolveEntitiesOutput(
            matches_found=len(candidates),
            same_as_relationships_created=matches_created,
            entities_processed=len(entities),
            match_details=match_details,
        )

    finally:
        await client.close()


@activity.defn
async def resolve_entities_mock(input: ResolveEntitiesInput) -> ResolveEntitiesOutput:
    """
    Mock entity resolution for testing without Neo4j.

    Simulates finding and resolving duplicate entities.

    Args:
        input: Resolution parameters

    Returns:
        Mock resolution results
    """
    activity.logger.info(f"Mock resolving entities for tenant {input.tenant_id}")

    activity.heartbeat()

    # Return mock results
    return ResolveEntitiesOutput(
        matches_found=0,
        same_as_relationships_created=0,
        entities_processed=0,
        match_details=[],
    )
