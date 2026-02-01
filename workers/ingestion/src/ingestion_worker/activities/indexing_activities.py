"""
Indexing activities.

Activities for indexing documents in vector and fulltext search databases.
"""

from temporalio import activity

from ingestion_worker.workflows.document_ingestion import (
    IndexVectorInput,
    IndexVectorOutput,
    IndexFulltextInput,
    IndexFulltextOutput,
)


@activity.defn
async def index_vector(input: IndexVectorInput) -> IndexVectorOutput:
    """
    Index document chunks in the vector database (Qdrant).

    Stores embedding vectors along with chunk metadata for hybrid search.

    Args:
        input: Vector indexing input with chunks and embeddings

    Returns:
        Indexing result with count and collection name
    """
    activity.logger.info(f"Indexing {len(input.chunks)} chunks in Qdrant")

    collection_name = "documents"  # Default collection

    # TODO: Implement actual Qdrant indexing
    # In real implementation:
    # 1. Get Qdrant client
    # 2. Create points with vectors and payloads
    # 3. Upsert points to collection
    # 4. Wait for indexing to complete

    # Payload should include:
    # - document_id
    # - tenant_id
    # - project_id (if set)
    # - chunk_sequence
    # - content (for retrieval)
    # - page_number
    # - any additional metadata

    activity.logger.info(f"Indexed {len(input.chunks)} vectors in {collection_name}")

    return IndexVectorOutput(
        indexed_count=len(input.chunks),
        collection_name=collection_name,
    )


@activity.defn
async def index_fulltext(input: IndexFulltextInput) -> IndexFulltextOutput:
    """
    Index document in fulltext search (MeiliSearch).

    Stores document content and metadata for keyword search with
    faceted filtering.

    Args:
        input: Fulltext indexing input with document content

    Returns:
        Indexing result with success status
    """
    activity.logger.info(f"Indexing document {input.document_id} in MeiliSearch")

    index_name = "documents"  # Default index

    # TODO: Implement actual MeiliSearch indexing
    # In real implementation:
    # 1. Get MeiliSearch client
    # 2. Create document object with searchable fields
    # 3. Add document to index
    # 4. Wait for task completion

    # Document should include:
    # - id (document_id)
    # - tenant_id (for filtering)
    # - project_id (for filtering)
    # - title
    # - content (truncated if very long)
    # - document_type (for facets)
    # - metadata fields

    activity.logger.info(f"Indexed document in {index_name}")

    return IndexFulltextOutput(
        indexed=True,
        index_name=index_name,
    )
