"""Entity and Relationship domain models."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from alexandria_core.models.base import (
    AuditMixin,
    IdentifiableMixin,
    MetadataMixin,
    SoftDeleteMixin,
    TenantScopedMixin,
)


class EntityType(StrEnum):
    """Standard entity types for knowledge extraction."""

    # Core types
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    EVENT = "event"

    # Extended types
    MONEY = "money"
    PHONE = "phone"
    EMAIL = "email"
    URL = "url"
    DOCUMENT = "document"  # Reference to another document
    PRODUCT = "product"
    LAW = "law"
    CASE = "case"  # Legal case

    # Custom/unknown
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class RelationshipType(StrEnum):
    """Standard relationship types between entities."""

    # Person relationships
    WORKS_FOR = "works_for"
    WORKS_WITH = "works_with"
    REPORTS_TO = "reports_to"
    OWNS = "owns"
    CONTROLS = "controls"
    RELATED_TO = "related_to"
    FAMILY = "family"

    # Organization relationships
    SUBSIDIARY_OF = "subsidiary_of"
    PARTNER_OF = "partner_of"
    COMPETITOR_OF = "competitor_of"
    MEMBER_OF = "member_of"

    # Location relationships
    LOCATED_IN = "located_in"
    HEADQUARTERS_IN = "headquarters_in"

    # Event relationships
    PARTICIPATED_IN = "participated_in"
    OCCURRED_AT = "occurred_at"
    OCCURRED_ON = "occurred_on"

    # Document relationships
    MENTIONED_IN = "mentioned_in"
    AUTHORED = "authored"
    RECEIVED = "received"
    SENT = "sent"

    # Generic
    ASSOCIATED_WITH = "associated_with"
    CUSTOM = "custom"


class Entity(IdentifiableMixin, TenantScopedMixin, AuditMixin, SoftDeleteMixin, MetadataMixin):
    """
    Entity represents a named entity extracted from documents.

    Entities are stored in both PostgreSQL (metadata) and Neo4j (graph).
    They can be linked to multiple documents and projects.
    """

    # Identity
    name: str = Field(..., min_length=1, max_length=500)
    canonical_name: str | None = None  # Normalized form for deduplication
    aliases: list[str] = Field(default_factory=list)
    entity_type: EntityType = EntityType.UNKNOWN

    # Description
    description: str | None = None
    summary: str | None = None  # AI-generated summary

    # External references
    external_ids: dict[str, str] = Field(default_factory=dict)  # e.g., {"wikidata": "Q123"}

    # Confidence
    confidence_score: float = 1.0  # 0.0 to 1.0
    is_verified: bool = False  # Human verified

    # Graph info (denormalized from Neo4j)
    neo4j_node_id: str | None = None
    mention_count: int = 0  # Number of document mentions


class EntityMention(IdentifiableMixin, TenantScopedMixin):
    """
    EntityMention links an entity to a specific location in a document.

    Tracks where entities appear in documents for provenance.
    """

    entity_id: UUID
    document_id: UUID
    chunk_id: UUID | None = None

    # Position in source
    start_char: int | None = None
    end_char: int | None = None
    page_number: int | None = None

    # Context
    context_text: str | None = None  # Surrounding text
    mention_text: str  # Exact text that was identified as this entity

    # Extraction info
    extraction_method: str = "spacy"  # spacy, llm, manual
    confidence_score: float = 1.0


class Relationship(IdentifiableMixin, TenantScopedMixin, AuditMixin, MetadataMixin):
    """
    Relationship represents a connection between two entities.

    Relationships are stored in Neo4j as edges between entity nodes.
    """

    # Source and target entities
    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: RelationshipType = RelationshipType.ASSOCIATED_WITH

    # Description
    description: str | None = None
    label: str | None = None  # Human-readable label

    # Provenance
    document_id: UUID | None = None  # Document where relationship was found
    evidence_text: str | None = None  # Text supporting the relationship

    # Confidence
    confidence_score: float = 1.0
    is_verified: bool = False

    # Temporal
    start_date: str | None = None  # ISO date or partial date
    end_date: str | None = None
    is_current: bool | None = None

    # Graph info
    neo4j_relationship_id: str | None = None
