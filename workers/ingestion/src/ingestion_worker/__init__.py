"""Alexandria Ingestion Worker - Document processing with Temporal."""

from ingestion_worker.workflows import (
    DocumentIngestionWorkflow,
    IngestionWorkflowInput,
    IngestionWorkflowOutput,
    IngestionWorkflowStatus,
    IngestionStep,
)
from ingestion_worker.activities import (
    classify_document,
    parse_document,
    chunk_document,
    update_document_status,
    generate_embeddings,
    index_vector,
    index_fulltext,
    extract_entities,
    extract_relationships,
    build_graph,
)
from ingestion_worker.worker import create_worker, run_worker

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Workflows
    "DocumentIngestionWorkflow",
    "IngestionWorkflowInput",
    "IngestionWorkflowOutput",
    "IngestionWorkflowStatus",
    "IngestionStep",
    # Activities
    "classify_document",
    "parse_document",
    "chunk_document",
    "update_document_status",
    "generate_embeddings",
    "index_vector",
    "index_fulltext",
    "extract_entities",
    "extract_relationships",
    "build_graph",
    # Worker
    "create_worker",
    "run_worker",
]
