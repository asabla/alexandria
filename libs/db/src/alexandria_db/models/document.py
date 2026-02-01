"""Document and Chunk SQLAlchemy models."""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from alexandria_db.models.base import AuditMixin, Base, SoftDeleteMixin, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from alexandria_db.models.tenant import ProjectModel


class DocumentModel(Base, UUIDMixin, AuditMixin, SoftDeleteMixin):
    """Document represents a source document in the system."""

    __tablename__ = "documents"

    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Basic info
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_type: Mapped[str] = mapped_column(String(50), default="unknown", nullable=False)

    # Source info
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_filename: Mapped[str | None] = mapped_column(String(500), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Storage
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    storage_bucket: Mapped[str] = mapped_column(String(63), default="documents", nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # Processing status
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Extracted content
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    word_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Indexing status
    is_indexed_vector: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_indexed_fulltext: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_indexed_graph: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Metadata
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    # Relationships
    chunks: Mapped[list["ChunkModel"]] = relationship(
        "ChunkModel", back_populates="document", cascade="all, delete-orphan"
    )
    ingestion_jobs: Mapped[list["IngestionJobModel"]] = relationship(
        "IngestionJobModel", back_populates="document", cascade="all, delete-orphan"
    )
    projects: Mapped[list["ProjectModel"]] = relationship(
        "ProjectModel",
        secondary="project_documents",
        back_populates="documents",
    )

    __table_args__ = (
        Index("ix_documents_tenant_status", "tenant_id", "status"),
        Index("ix_documents_tenant_type", "tenant_id", "document_type"),
        Index("ix_documents_created", "created_at"),
    )


class ChunkModel(Base, UUIDMixin, TimestampMixin):
    """Chunk represents a segment of a document."""

    __tablename__ = "chunks"

    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Position in source
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Chunking metadata
    chunk_type: Mapped[str] = mapped_column(String(50), default="text", nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Embedding
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    embedding_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Metadata
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    # Relationships
    document: Mapped["DocumentModel"] = relationship("DocumentModel", back_populates="chunks")

    __table_args__ = (Index("ix_chunks_document_sequence", "document_id", "sequence_number"),)


class IngestionJobModel(Base, UUIDMixin, AuditMixin):
    """IngestionJob tracks the processing of a document."""

    __tablename__ = "ingestion_jobs"

    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Workflow tracking
    workflow_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    workflow_run_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False, index=True)
    current_step: Mapped[str | None] = mapped_column(String(50), nullable=True)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Results
    chunks_created: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    entities_extracted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel", back_populates="ingestion_jobs"
    )

    __table_args__ = (Index("ix_ingestion_jobs_tenant_status", "tenant_id", "status"),)
