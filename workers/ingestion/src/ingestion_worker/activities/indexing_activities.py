"""
Indexing activities.

Activities for indexing documents in vector and fulltext search databases.
"""

import hashlib
import os
from typing import Any

from temporalio import activity

from ingestion_worker.workflows.document_ingestion import (
    IndexVectorInput,
    IndexVectorOutput,
    IndexFulltextInput,
    IndexFulltextOutput,
    ChunkInfo,
    EmbeddingInfo,
)


def _generate_point_id(document_id: str, chunk_sequence: int) -> str:
    """Generate a deterministic point ID for idempotent upserts.

    Uses a hash of document_id and chunk_sequence to ensure the same
    chunk always gets the same point ID, allowing re-indexing without
    creating duplicates.

    Args:
        document_id: The document's unique identifier
        chunk_sequence: The chunk's sequence number within the document

    Returns:
        A deterministic point ID string
    """
    # Create deterministic ID from document and chunk
    id_string = f"{document_id}:chunk:{chunk_sequence}"
    hash_val = hashlib.sha256(id_string.encode()).hexdigest()
    # Use first 32 chars as UUID-compatible string
    return f"{hash_val[:8]}-{hash_val[8:12]}-{hash_val[12:16]}-{hash_val[16:20]}-{hash_val[20:32]}"


def _build_point_payload(
    chunk: ChunkInfo,
    embedding: EmbeddingInfo,
    document_id: str,
    tenant_id: str,
    project_id: str | None,
) -> dict[str, Any]:
    """Build the payload for a Qdrant point.

    Args:
        chunk: Chunk information
        embedding: Embedding information
        document_id: Document ID
        tenant_id: Tenant ID
        project_id: Optional project ID

    Returns:
        Payload dictionary for the point
    """
    payload = {
        "document_id": document_id,
        "tenant_id": tenant_id,
        "chunk_sequence": chunk.sequence_number,
        "content": chunk.content,
        "content_hash": chunk.content_hash,
        "embedding_id": embedding.embedding_id,
        "embedding_model": embedding.model,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "token_count": chunk.token_count,
    }

    # Add optional fields
    if project_id:
        payload["project_ids"] = [project_id]

    if chunk.page_number is not None:
        payload["page_number"] = chunk.page_number

    if chunk.heading_context:
        payload["heading_context"] = chunk.heading_context

    if chunk.chunk_type:
        payload["chunk_type"] = chunk.chunk_type

    if chunk.metadata:
        # Flatten metadata into payload (prefixed to avoid conflicts)
        for key, value in chunk.metadata.items():
            payload[f"meta_{key}"] = value

    return payload


@activity.defn
async def index_vector(input: IndexVectorInput) -> IndexVectorOutput:
    """
    Index document chunks in the vector database (Qdrant).

    Stores embedding vectors along with chunk metadata for hybrid search.
    Uses deterministic point IDs for idempotent upserts - re-indexing the
    same document will update existing points rather than create duplicates.

    Features:
    - Dense vectors for semantic search
    - Sparse vectors for keyword matching (hybrid search)
    - Batch processing with heartbeat support
    - Deterministic point IDs for idempotency

    Args:
        input: Vector indexing input with chunks and embeddings

    Returns:
        Indexing result with count and collection name
    """
    # Lazy import to avoid Temporal sandbox issues
    from alexandria_db.clients import QdrantClient, HybridPoint, SparseVector

    activity.logger.info(
        f"Indexing {len(input.chunks)} chunks in Qdrant",
        extra={
            "document_id": input.document_id,
            "tenant_id": input.tenant_id,
            "chunk_count": len(input.chunks),
        },
    )

    # Get configuration from environment
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "documents")
    batch_size = int(os.getenv("QDRANT_BATCH_SIZE", "100"))

    # Create client
    client = QdrantClient(url=qdrant_url, collection=collection_name)

    # Build a map of chunk_sequence -> embedding for quick lookup
    embedding_map: dict[int, EmbeddingInfo] = {e.chunk_sequence: e for e in input.embeddings}

    # Validate we have embeddings for all chunks
    missing_embeddings = []
    for chunk in input.chunks:
        if chunk.sequence_number not in embedding_map:
            missing_embeddings.append(chunk.sequence_number)

    if missing_embeddings:
        activity.logger.warning(
            f"Missing embeddings for chunks: {missing_embeddings}",
            extra={"document_id": input.document_id},
        )

    # Get vector dimensions from first embedding (for collection setup)
    vector_size = 768  # Default
    if input.embeddings:
        vector_size = input.embeddings[0].dimensions or len(input.embeddings[0].dense_vector)

    try:
        # Ensure collection exists with hybrid search support
        created = await client.ensure_collection_hybrid(
            collection=collection_name,
            vector_size=vector_size,
        )
        if created:
            activity.logger.info(
                f"Created Qdrant collection: {collection_name}",
                extra={"vector_size": vector_size},
            )

        # Build points
        points: list[HybridPoint] = []

        for chunk in input.chunks:
            embedding = embedding_map.get(chunk.sequence_number)
            if not embedding:
                continue  # Skip chunks without embeddings

            # Generate deterministic point ID
            point_id = _generate_point_id(input.document_id, chunk.sequence_number)

            # Build payload
            payload = _build_point_payload(
                chunk=chunk,
                embedding=embedding,
                document_id=input.document_id,
                tenant_id=input.tenant_id,
                project_id=input.project_id,
            )

            # Convert sparse vector if present
            sparse = None
            if embedding.sparse_vector:
                sparse = SparseVector(
                    indices=embedding.sparse_vector.indices,
                    values=embedding.sparse_vector.values,
                )

            points.append(
                HybridPoint(
                    id=point_id,
                    dense_vector=embedding.dense_vector,
                    sparse_vector=sparse,
                    payload=payload,
                )
            )

        # Upsert points in batches with heartbeat
        indexed_count = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            count = await client.upsert_hybrid(batch, collection=collection_name)
            indexed_count += count

            # Heartbeat after each batch
            activity.heartbeat(f"Indexed {indexed_count}/{len(points)} points")

        activity.logger.info(
            f"Indexed {indexed_count} vectors in {collection_name}",
            extra={
                "document_id": input.document_id,
                "indexed_count": indexed_count,
                "collection": collection_name,
            },
        )

        return IndexVectorOutput(
            indexed_count=indexed_count,
            collection_name=collection_name,
        )

    except Exception as e:
        activity.logger.error(
            f"Vector indexing failed: {e}",
            extra={
                "document_id": input.document_id,
                "error": str(e),
            },
        )
        raise


@activity.defn
async def index_vector_mock(input: IndexVectorInput) -> IndexVectorOutput:
    """
    Mock vector indexing for testing.

    Simulates vector indexing without requiring a running Qdrant instance.

    Args:
        input: Vector indexing input with chunks and embeddings

    Returns:
        Mock indexing result
    """
    activity.logger.info(
        f"Mock indexing {len(input.chunks)} chunks",
        extra={"document_id": input.document_id},
    )

    # Simulate processing with heartbeat
    for i, chunk in enumerate(input.chunks):
        if i % 10 == 0:
            activity.heartbeat(f"Processing chunk {i}/{len(input.chunks)}")

    collection_name = os.getenv("QDRANT_COLLECTION", "documents")

    return IndexVectorOutput(
        indexed_count=len(input.chunks),
        collection_name=collection_name,
    )


def _generate_meili_doc_id(document_id: str, chunk_sequence: int) -> str:
    """Generate a deterministic MeiliSearch document ID.

    Creates a unique document ID by combining the document ID and chunk
    sequence number. This ensures the same chunk always gets the same
    document ID, allowing re-indexing without creating duplicates.

    Args:
        document_id: The document's unique identifier
        chunk_sequence: The chunk's sequence number within the document

    Returns:
        A deterministic document ID string
    """
    return f"{document_id}_chunk_{chunk_sequence}"


def _build_meili_document(
    chunk: ChunkInfo,
    document_id: str,
    tenant_id: str,
    project_id: str | None,
    title: str,
    document_type: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a MeiliSearch document from a chunk.

    Creates a document dictionary suitable for indexing in MeiliSearch,
    including all searchable and filterable fields.

    Args:
        chunk: The chunk information
        document_id: The parent document's unique identifier
        tenant_id: Tenant ID for multi-tenancy filtering
        project_id: Optional project association
        title: Document title (indexed with each chunk for context)
        document_type: Type of document (pdf, docx, etc.)
        metadata: Additional metadata dictionary

    Returns:
        Document dictionary ready for MeiliSearch indexing
    """
    doc: dict[str, Any] = {
        # Primary key (required)
        "id": _generate_meili_doc_id(document_id, chunk.sequence_number),
        # Searchable fields
        "title": title,
        "content": chunk.content,
        # Filterable fields (for multi-tenancy and faceted search)
        "tenant_id": tenant_id,
        "document_type": document_type,
        "document_id": document_id,  # For filtering/grouping chunks by document
        "chunk_sequence": chunk.sequence_number,
        "chunk_type": chunk.chunk_type,
    }

    # Add optional project_id as list (for multi-project support)
    if project_id:
        doc["project_ids"] = [project_id]
    else:
        doc["project_ids"] = []

    # Add heading context as searchable string
    if chunk.heading_context:
        doc["heading_context"] = " > ".join(chunk.heading_context)

    # Add optional metadata fields
    if metadata.get("language"):
        doc["language"] = metadata["language"]

    if metadata.get("created_at"):
        doc["created_at"] = metadata["created_at"]

    if metadata.get("updated_at"):
        doc["updated_at"] = metadata["updated_at"]

    if metadata.get("description"):
        doc["description"] = metadata["description"]

    if metadata.get("summary"):
        doc["summary"] = metadata["summary"]

    if metadata.get("entities"):
        doc["entities"] = metadata["entities"]

    if metadata.get("entity_types"):
        doc["entity_types"] = metadata["entity_types"]

    if metadata.get("status"):
        doc["status"] = metadata["status"]

    return doc


@activity.defn
async def index_fulltext(input: IndexFulltextInput) -> IndexFulltextOutput:
    """
    Index document chunks in MeiliSearch for full-text search.

    Indexes each chunk as a separate MeiliSearch document (~1KB each)
    following MeiliSearch best practices for optimal search performance.

    Features:
    - Chunk-level indexing for fine-grained search results
    - Batch processing with heartbeat support for long-running indexing
    - Automatic upsert (re-indexing updates existing documents)
    - Multi-tenant filtering support via tenant_id
    - Faceted search on document_type, language, entity_types

    Args:
        input: Fulltext indexing input with document chunks

    Returns:
        Indexing result with success status and count
    """
    # Lazy import to avoid Temporal sandbox issues
    from alexandria_db.clients import MeiliSearchClient

    activity.logger.info(
        f"Indexing {len(input.chunks)} chunks in MeiliSearch",
        extra={
            "document_id": input.document_id,
            "tenant_id": input.tenant_id,
            "chunk_count": len(input.chunks),
        },
    )

    # Get configuration from environment
    meilisearch_url = os.getenv("MEILISEARCH_URL", "http://localhost:7700")
    meilisearch_api_key = os.getenv("MEILISEARCH_API_KEY", "masterKey")
    index_name = os.getenv("MEILISEARCH_INDEX", "documents")
    batch_size = int(os.getenv("MEILISEARCH_BATCH_SIZE", "100"))

    # Create client
    client = MeiliSearchClient(
        url=meilisearch_url,
        api_key=meilisearch_api_key,
        index=index_name,
    )

    try:
        # Ensure index exists with proper configuration
        created = await client.ensure_index(index=index_name)
        if created:
            activity.logger.info(
                f"Created MeiliSearch index: {index_name}",
                extra={"index_name": index_name},
            )

        # Build documents from chunks
        documents = [
            _build_meili_document(
                chunk=chunk,
                document_id=input.document_id,
                tenant_id=input.tenant_id,
                project_id=input.project_id,
                title=input.title,
                document_type=input.document_type,
                metadata=input.metadata,
            )
            for chunk in input.chunks
        ]

        # Index in batches with heartbeat
        indexed_count = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            count = await client.index_documents(batch, index=index_name, wait=True)
            indexed_count += count

            # Heartbeat after each batch
            activity.heartbeat(f"Indexed {indexed_count}/{len(documents)} chunks")

        activity.logger.info(
            f"Indexed {indexed_count} chunks in {index_name}",
            extra={
                "document_id": input.document_id,
                "indexed_count": indexed_count,
                "index_name": index_name,
            },
        )

        return IndexFulltextOutput(
            indexed=True,
            indexed_count=indexed_count,
            index_name=index_name,
        )

    except Exception as e:
        activity.logger.error(
            f"Fulltext indexing failed: {e}",
            extra={
                "document_id": input.document_id,
                "error": str(e),
            },
        )
        raise


@activity.defn
async def index_fulltext_mock(input: IndexFulltextInput) -> IndexFulltextOutput:
    """
    Mock fulltext indexing for testing.

    Simulates MeiliSearch indexing without requiring a running instance.

    Args:
        input: Fulltext indexing input with document chunks

    Returns:
        Mock indexing result
    """
    activity.logger.info(
        f"Mock indexing {len(input.chunks)} chunks",
        extra={"document_id": input.document_id},
    )

    # Simulate processing with heartbeat
    for i, chunk in enumerate(input.chunks):
        if i % 10 == 0:
            activity.heartbeat(f"Processing chunk {i}/{len(input.chunks)}")

    index_name = os.getenv("MEILISEARCH_INDEX", "documents")

    return IndexFulltextOutput(
        indexed=True,
        indexed_count=len(input.chunks),
        index_name=index_name,
    )
