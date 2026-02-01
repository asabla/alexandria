"""
Document Ingestion Workflow.

This module defines the main Temporal workflow for document ingestion.
The workflow orchestrates all activities needed to process a document:
1. Classify the document type
2. Download/retrieve the document
3. Parse and extract text (via Docling, OCR, transcription, etc.)
4. Chunk the document into segments
5. Generate embeddings for each chunk
6. Index in vector database (Qdrant)
7. Index in full-text search (MeiliSearch)
8. Extract entities and relationships
9. Build knowledge graph (Neo4j)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID

from temporalio import workflow
from temporalio.common import RetryPolicy


class DictObject:
    """Helper to allow attribute-style access to dictionaries returned by activities."""

    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, name: str):
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            value = self._d[name]
            # Recursively wrap nested dicts
            if isinstance(value, dict):
                return DictObject(value)
            # Don't wrap lists - keep them as raw data for passing to activities
            return value
        except KeyError:
            raise AttributeError(f"No attribute '{name}'")

    def __len__(self):
        return len(self._d)

    def raw(self) -> dict:
        """Get the raw dictionary."""
        return self._d


# Import activity stubs - these will be defined later
with workflow.unsafe.imports_passed_through():
    import structlog

# Default retry policy for activities
DEFAULT_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
)

# Longer retry for potentially slow activities
SLOW_ACTIVITY_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=10),
    maximum_attempts=3,
)


class IngestionStep(StrEnum):
    """Steps in the ingestion workflow."""

    INITIALIZING = "initializing"
    CLASSIFYING = "classifying"
    DOWNLOADING = "downloading"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING_VECTOR = "indexing_vector"
    INDEXING_FULLTEXT = "indexing_fulltext"
    EXTRACTING_ENTITIES = "extracting_entities"
    EXTRACTING_RELATIONSHIPS = "extracting_relationships"
    BUILDING_GRAPH = "building_graph"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class IngestionWorkflowInput:
    """Input parameters for the document ingestion workflow."""

    # Required fields
    document_id: str  # UUID as string for serialization
    tenant_id: str  # UUID as string

    # Storage location
    storage_bucket: str
    storage_key: str

    # Optional source info
    source_url: str | None = None
    source_filename: str | None = None
    mime_type: str | None = None

    # Processing options
    skip_ocr: bool = False
    skip_entity_extraction: bool = False
    skip_graph_building: bool = False
    force_reprocess: bool = False

    # Chunking options
    chunk_size: int = 1000  # Target tokens per chunk
    chunk_overlap: int = 200  # Overlap between chunks

    # Project association (optional)
    project_id: str | None = None

    # Metadata to pass through
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionWorkflowStatus:
    """Current status of the ingestion workflow."""

    current_step: IngestionStep
    progress_percent: int
    message: str

    # Counts
    chunks_created: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0

    # Error info (if failed)
    error_message: str | None = None
    error_step: IngestionStep | None = None

    # Timing
    started_at: str | None = None  # ISO format
    step_started_at: str | None = None


@dataclass
class IngestionWorkflowOutput:
    """Output of the document ingestion workflow."""

    document_id: str
    success: bool

    # Final status
    final_status: IngestionWorkflowStatus

    # Results
    document_type: str | None = None
    page_count: int | None = None
    word_count: int | None = None
    language: str | None = None

    # Processing results
    chunks_created: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0

    # Indexing status
    indexed_vector: bool = False
    indexed_fulltext: bool = False
    indexed_graph: bool = False

    # Error info
    error_message: str | None = None
    error_step: str | None = None


# Activity input/output dataclasses
@dataclass
class ClassifyDocumentInput:
    """Input for document classification activity."""

    storage_bucket: str
    storage_key: str
    mime_type: str | None = None
    source_filename: str | None = None


@dataclass
class ClassifyDocumentOutput:
    """Output from document classification activity."""

    document_type: str
    mime_type: str
    confidence: float = 1.0


@dataclass
class ParseDocumentInput:
    """Input for document parsing activity."""

    document_id: str
    storage_bucket: str
    storage_key: str
    document_type: str
    skip_ocr: bool = False


@dataclass
class ParseDocumentOutput:
    """Output from document parsing activity."""

    content: str
    page_count: int | None = None
    word_count: int | None = None
    language: str | None = None
    tables: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ChunkDocumentInput:
    """Input for document chunking activity."""

    document_id: str
    tenant_id: str
    content: str
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class ChunkInfo:
    """Information about a single chunk."""

    sequence_number: int
    content: str
    content_hash: str
    start_char: int
    end_char: int
    token_count: int
    page_number: int | None = None


@dataclass
class ChunkDocumentOutput:
    """Output from document chunking activity."""

    chunks: list[ChunkInfo]
    total_chunks: int


@dataclass
class GenerateEmbeddingsInput:
    """Input for embedding generation activity."""

    document_id: str
    tenant_id: str
    chunks: list[ChunkInfo]


@dataclass
class EmbeddingInfo:
    """Embedding information for a chunk."""

    chunk_sequence: int
    embedding_id: str
    model: str


@dataclass
class GenerateEmbeddingsOutput:
    """Output from embedding generation activity."""

    embeddings: list[EmbeddingInfo]
    model: str


@dataclass
class IndexVectorInput:
    """Input for vector indexing activity."""

    document_id: str
    tenant_id: str
    project_id: str | None
    chunks: list[ChunkInfo]
    embeddings: list[EmbeddingInfo]


@dataclass
class IndexVectorOutput:
    """Output from vector indexing activity."""

    indexed_count: int
    collection_name: str


@dataclass
class IndexFulltextInput:
    """Input for fulltext indexing activity."""

    document_id: str
    tenant_id: str
    project_id: str | None
    title: str
    content: str
    document_type: str
    metadata: dict[str, Any]


@dataclass
class IndexFulltextOutput:
    """Output from fulltext indexing activity."""

    indexed: bool
    index_name: str


@dataclass
class ExtractEntitiesInput:
    """Input for entity extraction activity."""

    document_id: str
    tenant_id: str
    content: str
    chunks: list[ChunkInfo]


@dataclass
class EntityInfo:
    """Extracted entity information."""

    name: str
    entity_type: str
    mentions: list[dict[str, Any]]
    confidence: float


@dataclass
class ExtractEntitiesOutput:
    """Output from entity extraction activity."""

    entities: list[EntityInfo]


@dataclass
class ExtractRelationshipsInput:
    """Input for relationship extraction activity."""

    document_id: str
    tenant_id: str
    content: str
    entities: list[EntityInfo]


@dataclass
class RelationshipInfo:
    """Extracted relationship information."""

    source_entity: str
    target_entity: str
    relationship_type: str
    evidence: str
    confidence: float


@dataclass
class ExtractRelationshipsOutput:
    """Output from relationship extraction activity."""

    relationships: list[RelationshipInfo]


@dataclass
class BuildGraphInput:
    """Input for graph building activity."""

    document_id: str
    tenant_id: str
    project_id: str | None
    entities: list[EntityInfo]
    relationships: list[RelationshipInfo]


@dataclass
class BuildGraphOutput:
    """Output from graph building activity."""

    nodes_created: int
    relationships_created: int


@dataclass
class UpdateDocumentStatusInput:
    """Input for updating document status."""

    document_id: str
    tenant_id: str
    status: str
    error_message: str | None = None
    document_type: str | None = None
    mime_type: str | None = None
    page_count: int | None = None
    word_count: int | None = None
    language: str | None = None
    is_indexed_vector: bool = False
    is_indexed_fulltext: bool = False
    is_indexed_graph: bool = False


@workflow.defn
class DocumentIngestionWorkflow:
    """
    Workflow for ingesting and processing documents.

    This workflow orchestrates the complete document ingestion pipeline:
    1. Document classification
    2. Content parsing (OCR, transcription, etc.)
    3. Text chunking
    4. Embedding generation
    5. Vector indexing
    6. Fulltext indexing
    7. Entity extraction
    8. Relationship extraction
    9. Knowledge graph construction

    The workflow supports:
    - Querying current status and progress
    - Signals for cancellation and pause/resume
    - Compensation on failure (cleanup partial results)
    - Retry policies for transient failures
    """

    def __init__(self) -> None:
        """Initialize workflow state."""
        self._status = IngestionWorkflowStatus(
            current_step=IngestionStep.INITIALIZING,
            progress_percent=0,
            message="Initializing workflow",
        )
        self._is_cancelled = False
        self._is_paused = False
        self._input: IngestionWorkflowInput | None = None

        # Intermediate results for compensation
        self._chunks: list[ChunkInfo] = []
        self._embeddings: list[EmbeddingInfo] = []
        self._entities: list[EntityInfo] = []
        self._relationships: list[RelationshipInfo] = []

    @workflow.run
    async def run(self, input: IngestionWorkflowInput) -> IngestionWorkflowOutput:
        """Execute the document ingestion workflow."""
        self._input = input
        self._update_status(IngestionStep.INITIALIZING, 0, "Starting document ingestion")

        log = workflow.logger
        log.info(f"Starting ingestion workflow for document {input.document_id}")

        try:
            # Step 1: Classify document (5%)
            await self._check_cancelled_or_paused()
            self._update_status(IngestionStep.CLASSIFYING, 5, "Classifying document type")

            classification = DictObject(
                await workflow.execute_activity(
                    "classify_document",
                    ClassifyDocumentInput(
                        storage_bucket=input.storage_bucket,
                        storage_key=input.storage_key,
                        mime_type=input.mime_type,
                        source_filename=input.source_filename,
                    ),
                    start_to_close_timeout=timedelta(minutes=2),
                    retry_policy=DEFAULT_RETRY_POLICY,
                )
            )
            log.info(f"Document classified as {classification.document_type}")

            # Step 2: Parse document (10-30%)
            await self._check_cancelled_or_paused()
            self._update_status(
                IngestionStep.PARSING, 10, f"Parsing {classification.document_type} document"
            )

            parsed = DictObject(
                await workflow.execute_activity(
                    "parse_document",
                    ParseDocumentInput(
                        document_id=input.document_id,
                        storage_bucket=input.storage_bucket,
                        storage_key=input.storage_key,
                        document_type=classification.document_type,
                        skip_ocr=input.skip_ocr,
                    ),
                    start_to_close_timeout=timedelta(minutes=30),  # OCR can be slow
                    heartbeat_timeout=timedelta(minutes=2),
                    retry_policy=SLOW_ACTIVITY_RETRY_POLICY,
                )
            )
            log.info(f"Document parsed: {parsed.word_count} words, {parsed.page_count} pages")

            # Step 3: Chunk document (35%)
            await self._check_cancelled_or_paused()
            self._update_status(IngestionStep.CHUNKING, 35, "Splitting document into chunks")

            chunked = DictObject(
                await workflow.execute_activity(
                    "chunk_document",
                    ChunkDocumentInput(
                        document_id=input.document_id,
                        tenant_id=input.tenant_id,
                        content=parsed.content,
                        chunk_size=input.chunk_size,
                        chunk_overlap=input.chunk_overlap,
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=DEFAULT_RETRY_POLICY,
                )
            )
            self._chunks = chunked.chunks
            self._status.chunks_created = chunked.total_chunks
            log.info(f"Document chunked into {chunked.total_chunks} chunks")

            # Step 4: Generate embeddings (40-55%)
            await self._check_cancelled_or_paused()
            self._update_status(
                IngestionStep.EMBEDDING,
                40,
                f"Generating embeddings for {chunked.total_chunks} chunks",
            )

            embeddings_result = DictObject(
                await workflow.execute_activity(
                    "generate_embeddings",
                    GenerateEmbeddingsInput(
                        document_id=input.document_id,
                        tenant_id=input.tenant_id,
                        chunks=chunked.chunks,
                    ),
                    start_to_close_timeout=timedelta(minutes=15),
                    heartbeat_timeout=timedelta(minutes=2),
                    retry_policy=SLOW_ACTIVITY_RETRY_POLICY,
                )
            )
            self._embeddings = embeddings_result.embeddings
            log.info(
                f"Generated {len(embeddings_result.embeddings)} embeddings with model {embeddings_result.model}"
            )

            # Step 5: Index in vector database (60%)
            await self._check_cancelled_or_paused()
            self._update_status(IngestionStep.INDEXING_VECTOR, 60, "Indexing in vector database")

            vector_indexed = DictObject(
                await workflow.execute_activity(
                    "index_vector",
                    IndexVectorInput(
                        document_id=input.document_id,
                        tenant_id=input.tenant_id,
                        project_id=input.project_id,
                        chunks=chunked.chunks,
                        embeddings=embeddings_result.embeddings,
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=DEFAULT_RETRY_POLICY,
                )
            )
            log.info(
                f"Indexed {vector_indexed.indexed_count} vectors in {vector_indexed.collection_name}"
            )

            # Step 6: Index in fulltext search (65%)
            await self._check_cancelled_or_paused()
            self._update_status(IngestionStep.INDEXING_FULLTEXT, 65, "Indexing in fulltext search")

            fulltext_indexed = DictObject(
                await workflow.execute_activity(
                    "index_fulltext",
                    IndexFulltextInput(
                        document_id=input.document_id,
                        tenant_id=input.tenant_id,
                        project_id=input.project_id,
                        title=input.source_filename or f"Document {input.document_id}",
                        content=parsed.content,
                        document_type=classification.document_type,
                        metadata=input.metadata,
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=DEFAULT_RETRY_POLICY,
                )
            )
            log.info(f"Fulltext indexed: {fulltext_indexed.indexed}")

            # Entity extraction (if not skipped)
            entities_output = DictObject({"entities": []})
            relationships_output = DictObject({"relationships": []})
            graph_output = DictObject({"nodes_created": 0, "relationships_created": 0})

            if not input.skip_entity_extraction:
                # Step 7: Extract entities (70-80%)
                await self._check_cancelled_or_paused()
                self._update_status(IngestionStep.EXTRACTING_ENTITIES, 70, "Extracting entities")

                entities_output = DictObject(
                    await workflow.execute_activity(
                        "extract_entities",
                        ExtractEntitiesInput(
                            document_id=input.document_id,
                            tenant_id=input.tenant_id,
                            content=parsed.content,
                            chunks=chunked.chunks,
                        ),
                        start_to_close_timeout=timedelta(minutes=15),
                        heartbeat_timeout=timedelta(minutes=2),
                        retry_policy=SLOW_ACTIVITY_RETRY_POLICY,
                    )
                )
                self._entities = entities_output.entities
                self._status.entities_extracted = len(entities_output.entities)
                log.info(f"Extracted {len(entities_output.entities)} entities")

                # Step 8: Extract relationships (85%)
                if entities_output.entities:
                    await self._check_cancelled_or_paused()
                    self._update_status(
                        IngestionStep.EXTRACTING_RELATIONSHIPS, 85, "Extracting relationships"
                    )

                    relationships_output = DictObject(
                        await workflow.execute_activity(
                            "extract_relationships",
                            ExtractRelationshipsInput(
                                document_id=input.document_id,
                                tenant_id=input.tenant_id,
                                content=parsed.content,
                                entities=entities_output.entities,
                            ),
                            start_to_close_timeout=timedelta(minutes=15),
                            heartbeat_timeout=timedelta(minutes=2),
                            retry_policy=SLOW_ACTIVITY_RETRY_POLICY,
                        )
                    )
                    self._relationships = relationships_output.relationships
                    self._status.relationships_extracted = len(relationships_output.relationships)
                    log.info(f"Extracted {len(relationships_output.relationships)} relationships")

            # Graph building (if not skipped)
            if not input.skip_graph_building and (
                entities_output.entities or relationships_output.relationships
            ):
                # Step 9: Build knowledge graph (90%)
                await self._check_cancelled_or_paused()
                self._update_status(IngestionStep.BUILDING_GRAPH, 90, "Building knowledge graph")

                graph_output = DictObject(
                    await workflow.execute_activity(
                        "build_graph",
                        BuildGraphInput(
                            document_id=input.document_id,
                            tenant_id=input.tenant_id,
                            project_id=input.project_id,
                            entities=entities_output.entities,
                            relationships=relationships_output.relationships,
                        ),
                        start_to_close_timeout=timedelta(minutes=10),
                        retry_policy=DEFAULT_RETRY_POLICY,
                    )
                )
                log.info(
                    f"Graph built: {graph_output.nodes_created} nodes, {graph_output.relationships_created} relationships"
                )

            # Step 10: Update document status (95%)
            await self._check_cancelled_or_paused()
            self._update_status(IngestionStep.FINALIZING, 95, "Updating document status")

            await workflow.execute_activity(
                "update_document_status",
                UpdateDocumentStatusInput(
                    document_id=input.document_id,
                    tenant_id=input.tenant_id,
                    status="completed",
                    document_type=classification.document_type,
                    mime_type=classification.mime_type,
                    page_count=parsed.page_count,
                    word_count=parsed.word_count,
                    language=parsed.language,
                    is_indexed_vector=True,
                    is_indexed_fulltext=fulltext_indexed.indexed,
                    is_indexed_graph=graph_output.nodes_created > 0,
                ),
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=DEFAULT_RETRY_POLICY,
            )

            # Complete!
            self._update_status(IngestionStep.COMPLETED, 100, "Document ingestion completed")
            log.info(f"Ingestion workflow completed for document {input.document_id}")

            return IngestionWorkflowOutput(
                document_id=input.document_id,
                success=True,
                final_status=self._status,
                document_type=classification.document_type,
                page_count=parsed.page_count,
                word_count=parsed.word_count,
                language=parsed.language,
                chunks_created=chunked.total_chunks,
                entities_extracted=len(entities_output.entities),
                relationships_extracted=len(relationships_output.relationships),
                indexed_vector=True,
                indexed_fulltext=fulltext_indexed.indexed,
                indexed_graph=graph_output.nodes_created > 0,
            )

        except asyncio.CancelledError:
            log.warning(f"Workflow cancelled for document {input.document_id}")
            self._update_status(
                IngestionStep.CANCELLED, self._status.progress_percent, "Workflow cancelled"
            )
            await self._compensate()
            return self._create_error_output("Workflow was cancelled")

        except Exception as e:
            log.error(f"Workflow failed for document {input.document_id}: {e}")
            error_msg = str(e)
            self._status.error_message = error_msg
            self._status.error_step = self._status.current_step
            self._update_status(
                IngestionStep.FAILED, self._status.progress_percent, f"Failed: {error_msg}"
            )

            # Update document status to failed
            try:
                await workflow.execute_activity(
                    "update_document_status",
                    UpdateDocumentStatusInput(
                        document_id=input.document_id,
                        tenant_id=input.tenant_id,
                        status="failed",
                        error_message=error_msg,
                    ),
                    start_to_close_timeout=timedelta(minutes=1),
                    retry_policy=DEFAULT_RETRY_POLICY,
                )
            except Exception:
                pass  # Best effort

            await self._compensate()
            return self._create_error_output(error_msg)

    @workflow.query
    def get_status(self) -> IngestionWorkflowStatus:
        """Query the current workflow status."""
        return self._status

    @workflow.query
    def get_progress(self) -> int:
        """Query the current progress percentage."""
        return self._status.progress_percent

    @workflow.signal
    def cancel(self) -> None:
        """Signal to cancel the workflow."""
        self._is_cancelled = True

    @workflow.signal
    def pause(self) -> None:
        """Signal to pause the workflow."""
        self._is_paused = True

    @workflow.signal
    def resume(self) -> None:
        """Signal to resume a paused workflow."""
        self._is_paused = False

    def _update_status(self, step: IngestionStep, progress: int, message: str) -> None:
        """Update the workflow status."""
        self._status.current_step = step
        self._status.progress_percent = progress
        self._status.message = message
        # Use workflow.now() for deterministic time in workflows
        self._status.step_started_at = workflow.now().isoformat()

        if self._status.started_at is None:
            self._status.started_at = self._status.step_started_at

    async def _check_cancelled_or_paused(self) -> None:
        """Check if workflow should stop or pause."""
        if self._is_cancelled:
            raise asyncio.CancelledError("Workflow cancelled by signal")

        # Wait while paused
        while self._is_paused:
            await workflow.wait_condition(
                lambda: not self._is_paused, timeout=timedelta(seconds=10)
            )

    async def _compensate(self) -> None:
        """
        Compensate for partial work on failure/cancellation.

        This cleans up any partial results that were created before
        the workflow failed or was cancelled.
        """
        if not self._input:
            return

        log = workflow.logger
        log.info(f"Running compensation for document {self._input.document_id}")

        # Note: In a real implementation, we would:
        # 1. Delete chunks from PostgreSQL
        # 2. Delete vectors from Qdrant
        # 3. Delete from MeiliSearch index
        # 4. Delete entities and relationships from Neo4j
        #
        # For now, we just log the intent. The actual cleanup activities
        # would be implemented as separate compensation activities.

        log.info(f"Compensation completed for document {self._input.document_id}")

    def _create_error_output(self, error_message: str) -> IngestionWorkflowOutput:
        """Create an error output."""
        return IngestionWorkflowOutput(
            document_id=self._input.document_id if self._input else "",
            success=False,
            final_status=self._status,
            error_message=error_message,
            error_step=self._status.error_step.value if self._status.error_step else None,
            chunks_created=self._status.chunks_created,
            entities_extracted=self._status.entities_extracted,
            relationships_extracted=self._status.relationships_extracted,
        )
