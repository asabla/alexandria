"""
Pytest fixtures for ingestion worker tests.

Provides fixtures for:
- Temporal test environment
- Mock activities
- Sample workflow inputs
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest
from temporalio import activity
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from ingestion_worker.workflows.document_ingestion import (
    BuildGraphInput,
    BuildGraphOutput,
    ChunkDocumentInput,
    ChunkDocumentOutput,
    ChunkInfo,
    ClassifyDocumentInput,
    ClassifyDocumentOutput,
    DocumentIngestionWorkflow,
    EmbeddingInfo,
    EntityInfo,
    ExtractEntitiesInput,
    ExtractEntitiesOutput,
    ExtractRelationshipsInput,
    ExtractRelationshipsOutput,
    GenerateEmbeddingsInput,
    GenerateEmbeddingsOutput,
    IndexFulltextInput,
    IndexFulltextOutput,
    IndexVectorInput,
    IndexVectorOutput,
    IngestionStep,
    IngestionWorkflowInput,
    ParseDocumentInput,
    ParseDocumentOutput,
    RelationshipInfo,
    UpdateDocumentStatusInput,
)


# =============================================================================
# Mock Activity Results
# =============================================================================


@dataclass
class MockActivityResults:
    """Container for mock activity return values."""

    classification: dict[str, Any] | None = None
    parsed: dict[str, Any] | None = None
    chunked: dict[str, Any] | None = None
    embeddings: dict[str, Any] | None = None
    vector_indexed: dict[str, Any] | None = None
    fulltext_indexed: dict[str, Any] | None = None
    entities: dict[str, Any] | None = None
    relationships: dict[str, Any] | None = None
    graph: dict[str, Any] | None = None

    # Error simulation
    fail_at_activity: str | None = None
    fail_message: str = "Simulated activity failure"


def get_default_mock_results() -> MockActivityResults:
    """Get default mock results for all activities."""
    return MockActivityResults(
        classification={
            "document_type": "pdf",
            "mime_type": "application/pdf",
            "confidence": 0.95,
        },
        parsed={
            "content": "This is the parsed content of the document. It contains multiple sentences and paragraphs for testing purposes.",
            "page_count": 5,
            "word_count": 150,
            "language": "en",
            "tables": [],
            "images": [],
        },
        chunked={
            "chunks": [
                {
                    "sequence_number": 0,
                    "content": "This is chunk 1 content.",
                    "content_hash": "hash1",
                    "start_char": 0,
                    "end_char": 50,
                    "token_count": 10,
                    "page_number": 1,
                },
                {
                    "sequence_number": 1,
                    "content": "This is chunk 2 content.",
                    "content_hash": "hash2",
                    "start_char": 51,
                    "end_char": 100,
                    "token_count": 10,
                    "page_number": 1,
                },
            ],
            "total_chunks": 2,
        },
        embeddings={
            "embeddings": [
                {"chunk_sequence": 0, "embedding_id": "emb-1", "model": "bge-base"},
                {"chunk_sequence": 1, "embedding_id": "emb-2", "model": "bge-base"},
            ],
            "model": "bge-base",
        },
        vector_indexed={
            "indexed_count": 2,
            "collection_name": "documents",
        },
        fulltext_indexed={
            "indexed": True,
            "index_name": "documents",
        },
        entities={
            "entities": [
                {
                    "name": "John Doe",
                    "entity_type": "PERSON",
                    "mentions": [{"chunk_index": 0, "start": 10, "end": 18}],
                    "confidence": 0.9,
                },
                {
                    "name": "Acme Corp",
                    "entity_type": "ORGANIZATION",
                    "mentions": [{"chunk_index": 1, "start": 5, "end": 14}],
                    "confidence": 0.85,
                },
            ]
        },
        relationships={
            "relationships": [
                {
                    "source_entity": "John Doe",
                    "target_entity": "Acme Corp",
                    "relationship_type": "WORKS_FOR",
                    "evidence": "John Doe is employed by Acme Corp",
                    "confidence": 0.8,
                }
            ]
        },
        graph={
            "nodes_created": 2,
            "relationships_created": 1,
        },
    )


# =============================================================================
# Mock Activities Factory
# =============================================================================


def create_mock_activities(
    results: MockActivityResults,
) -> list[Callable[..., Any]]:
    """Create mock activity functions with configurable results."""

    @activity.defn(name="classify_document")
    async def mock_classify_document(input: ClassifyDocumentInput) -> dict[str, Any]:
        if results.fail_at_activity == "classify_document":
            raise RuntimeError(results.fail_message)
        return results.classification or get_default_mock_results().classification

    @activity.defn(name="parse_document")
    async def mock_parse_document(input: ParseDocumentInput) -> dict[str, Any]:
        if results.fail_at_activity == "parse_document":
            raise RuntimeError(results.fail_message)
        return results.parsed or get_default_mock_results().parsed

    @activity.defn(name="chunk_document")
    async def mock_chunk_document(input: ChunkDocumentInput) -> dict[str, Any]:
        if results.fail_at_activity == "chunk_document":
            raise RuntimeError(results.fail_message)
        return results.chunked or get_default_mock_results().chunked

    @activity.defn(name="generate_embeddings")
    async def mock_generate_embeddings(input: GenerateEmbeddingsInput) -> dict[str, Any]:
        if results.fail_at_activity == "generate_embeddings":
            raise RuntimeError(results.fail_message)
        return results.embeddings or get_default_mock_results().embeddings

    @activity.defn(name="index_vector")
    async def mock_index_vector(input: IndexVectorInput) -> dict[str, Any]:
        if results.fail_at_activity == "index_vector":
            raise RuntimeError(results.fail_message)
        return results.vector_indexed or get_default_mock_results().vector_indexed

    @activity.defn(name="index_fulltext")
    async def mock_index_fulltext(input: IndexFulltextInput) -> dict[str, Any]:
        if results.fail_at_activity == "index_fulltext":
            raise RuntimeError(results.fail_message)
        return results.fulltext_indexed or get_default_mock_results().fulltext_indexed

    @activity.defn(name="extract_entities")
    async def mock_extract_entities(input: ExtractEntitiesInput) -> dict[str, Any]:
        if results.fail_at_activity == "extract_entities":
            raise RuntimeError(results.fail_message)
        return results.entities or get_default_mock_results().entities

    @activity.defn(name="extract_relationships")
    async def mock_extract_relationships(input: ExtractRelationshipsInput) -> dict[str, Any]:
        if results.fail_at_activity == "extract_relationships":
            raise RuntimeError(results.fail_message)
        return results.relationships or get_default_mock_results().relationships

    @activity.defn(name="build_graph")
    async def mock_build_graph(input: BuildGraphInput) -> dict[str, Any]:
        if results.fail_at_activity == "build_graph":
            raise RuntimeError(results.fail_message)
        return results.graph or get_default_mock_results().graph

    @activity.defn(name="update_document_status")
    async def mock_update_document_status(input: UpdateDocumentStatusInput) -> None:
        if results.fail_at_activity == "update_document_status":
            raise RuntimeError(results.fail_message)
        return None

    return [
        mock_classify_document,
        mock_parse_document,
        mock_chunk_document,
        mock_generate_embeddings,
        mock_index_vector,
        mock_index_fulltext,
        mock_extract_entities,
        mock_extract_relationships,
        mock_build_graph,
        mock_update_document_status,
    ]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_document_id() -> str:
    """Generate a sample document ID."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_tenant_id() -> str:
    """Generate a sample tenant ID."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_workflow_input(sample_document_id: str, sample_tenant_id: str) -> IngestionWorkflowInput:
    """Create a sample workflow input."""
    return IngestionWorkflowInput(
        document_id=sample_document_id,
        tenant_id=sample_tenant_id,
        storage_bucket="documents",
        storage_key=f"documents/{sample_tenant_id}/{sample_document_id}.pdf",
        source_filename="test-document.pdf",
        mime_type="application/pdf",
    )


@pytest.fixture
def mock_results() -> MockActivityResults:
    """Get default mock activity results."""
    return get_default_mock_results()


@pytest.fixture
async def workflow_environment():
    """Create a time-skipping workflow test environment."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


@pytest.fixture
def task_queue() -> str:
    """Get the task queue name for tests."""
    return "test-ingestion"


async def run_workflow_with_mocks(
    env: WorkflowEnvironment,
    workflow_input: IngestionWorkflowInput,
    mock_results: MockActivityResults,
    task_queue: str = "test-ingestion",
) -> Any:
    """Helper to run workflow with mock activities."""
    activities = create_mock_activities(mock_results)

    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[DocumentIngestionWorkflow],
        activities=activities,
    ):
        result = await env.client.execute_workflow(
            DocumentIngestionWorkflow.run,
            workflow_input,
            id=f"test-workflow-{uuid.uuid4()}",
            task_queue=task_queue,
        )
        return result


async def start_workflow_with_mocks(
    env: WorkflowEnvironment,
    workflow_input: IngestionWorkflowInput,
    mock_results: MockActivityResults,
    task_queue: str = "test-ingestion",
    workflow_id: str | None = None,
):
    """Helper to start workflow (without waiting) with mock activities."""
    activities = create_mock_activities(mock_results)
    wf_id = workflow_id or f"test-workflow-{uuid.uuid4()}"

    worker = Worker(
        env.client,
        task_queue=task_queue,
        workflows=[DocumentIngestionWorkflow],
        activities=activities,
    )

    return worker, wf_id, activities
