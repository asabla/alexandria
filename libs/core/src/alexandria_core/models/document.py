"""Document and Chunk domain models."""

from datetime import datetime
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


class DocumentType(StrEnum):
    """Supported document types."""

    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"
    IMAGE = "image"  # jpg, png, etc.
    AUDIO = "audio"  # mp3, wav, etc.
    VIDEO = "video"  # mp4, etc.
    SPREADSHEET = "spreadsheet"  # xlsx, csv
    EMAIL = "email"  # eml, msg
    WEBPAGE = "webpage"  # scraped web pages
    UNKNOWN = "unknown"


class DocumentStatus(StrEnum):
    """Document processing status."""

    PENDING = "pending"  # Uploaded, awaiting processing
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"  # Processing complete
    FAILED = "failed"  # Processing failed
    PARTIAL = "partial"  # Partially processed (some steps failed)


class Document(IdentifiableMixin, TenantScopedMixin, AuditMixin, SoftDeleteMixin, MetadataMixin):
    """
    Document represents a source document in the system.

    Documents are the primary unit of content. They are processed through
    the ingestion pipeline to extract text, entities, and embeddings.
    """

    # Basic info
    title: str = Field(..., min_length=1, max_length=500)
    description: str | None = None
    document_type: DocumentType = DocumentType.UNKNOWN

    # Source info
    source_url: str | None = None  # Original URL if web-scraped
    source_filename: str | None = None  # Original filename if uploaded
    mime_type: str | None = None

    # Storage
    storage_key: str  # MinIO object key
    storage_bucket: str = "documents"
    file_size_bytes: int = 0
    file_hash: str | None = None  # SHA-256 hash for deduplication

    # Processing status
    status: DocumentStatus = DocumentStatus.PENDING
    error_message: str | None = None
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None

    # Extracted content (summary)
    page_count: int | None = None
    word_count: int | None = None
    language: str | None = None  # ISO 639-1 code
    summary: str | None = None  # AI-generated summary

    # Indexing status
    is_indexed_vector: bool = False
    is_indexed_fulltext: bool = False
    is_indexed_graph: bool = False


class Chunk(IdentifiableMixin, TenantScopedMixin, MetadataMixin):
    """
    Chunk represents a segment of a document.

    Documents are split into chunks for embedding and retrieval.
    Each chunk has its own embedding vector stored in Qdrant.
    """

    document_id: UUID
    sequence_number: int  # Order within document

    # Content
    content: str
    content_hash: str  # For deduplication

    # Position in source
    page_number: int | None = None
    start_char: int | None = None
    end_char: int | None = None

    # Chunking metadata
    chunk_type: str = "text"  # text, table, image_caption, etc.
    token_count: int = 0

    # Embedding
    embedding_model: str | None = None
    embedding_id: str | None = None  # Qdrant point ID


class IngestionJob(IdentifiableMixin, TenantScopedMixin, AuditMixin):
    """
    IngestionJob tracks the processing of a document.

    Linked to Temporal workflow execution for observability.
    """

    document_id: UUID

    # Workflow tracking
    workflow_id: str
    workflow_run_id: str | None = None

    # Status
    status: DocumentStatus = DocumentStatus.PENDING
    current_step: str | None = None  # e.g., "parsing", "chunking", "embedding"
    progress_percent: int = 0

    # Results
    chunks_created: int = 0
    entities_extracted: int = 0
    error_message: str | None = None

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
