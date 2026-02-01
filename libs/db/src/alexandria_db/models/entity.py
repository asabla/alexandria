"""Entity and Relationship SQLAlchemy models."""

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from alexandria_db.models.base import AuditMixin, Base, SoftDeleteMixin, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from alexandria_db.models.tenant import ProjectModel


class EntityModel(Base, UUIDMixin, AuditMixin, SoftDeleteMixin):
    """Entity represents a named entity extracted from documents."""

    __tablename__ = "entities"

    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Identity
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    canonical_name: Mapped[str | None] = mapped_column(String(500), nullable=True, index=True)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String), default=list, nullable=False)
    entity_type: Mapped[str] = mapped_column(
        String(50), default="unknown", nullable=False, index=True
    )

    # Description
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # External references
    external_ids: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # Confidence
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Graph info
    neo4j_node_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    mention_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Metadata
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    # Relationships
    mentions: Mapped[list["EntityMentionModel"]] = relationship(
        "EntityMentionModel", back_populates="entity", cascade="all, delete-orphan"
    )
    projects: Mapped[list["ProjectModel"]] = relationship(
        "ProjectModel",
        secondary="project_entities",
        back_populates="entities",
    )
    outgoing_relationships: Mapped[list["RelationshipModel"]] = relationship(
        "RelationshipModel",
        foreign_keys="RelationshipModel.source_entity_id",
        back_populates="source_entity",
        cascade="all, delete-orphan",
    )
    incoming_relationships: Mapped[list["RelationshipModel"]] = relationship(
        "RelationshipModel",
        foreign_keys="RelationshipModel.target_entity_id",
        back_populates="target_entity",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_entities_tenant_type", "tenant_id", "entity_type"),
        Index("ix_entities_tenant_name", "tenant_id", "name"),
    )


class EntityMentionModel(Base, UUIDMixin, TimestampMixin):
    """EntityMention links an entity to a specific location in a document."""

    __tablename__ = "entity_mentions"

    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Position in source
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Context
    context_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    mention_text: Mapped[str] = mapped_column(String(500), nullable=False)

    # Extraction info
    extraction_method: Mapped[str] = mapped_column(String(50), default="spacy", nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    # Relationships
    entity: Mapped["EntityModel"] = relationship("EntityModel", back_populates="mentions")

    __table_args__ = (
        Index("ix_entity_mentions_document", "document_id"),
        Index("ix_entity_mentions_entity_document", "entity_id", "document_id"),
    )


class RelationshipModel(Base, UUIDMixin, AuditMixin):
    """Relationship represents a connection between two entities."""

    __tablename__ = "relationships"

    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Source and target
    source_entity_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_entity_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relationship_type: Mapped[str] = mapped_column(
        String(50), default="associated_with", nullable=False, index=True
    )

    # Description
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Provenance
    document_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    evidence_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Confidence
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Temporal
    start_date: Mapped[str | None] = mapped_column(String(20), nullable=True)
    end_date: Mapped[str | None] = mapped_column(String(20), nullable=True)
    is_current: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Graph info
    neo4j_relationship_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Metadata
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    # Relationships
    source_entity: Mapped["EntityModel"] = relationship(
        "EntityModel",
        foreign_keys=[source_entity_id],
        back_populates="outgoing_relationships",
    )
    target_entity: Mapped["EntityModel"] = relationship(
        "EntityModel",
        foreign_keys=[target_entity_id],
        back_populates="incoming_relationships",
    )

    __table_args__ = (
        Index("ix_relationships_source_target", "source_entity_id", "target_entity_id"),
        Index("ix_relationships_type", "relationship_type"),
    )
