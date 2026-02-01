"""
Embedding generation activities.

Activities for generating vector embeddings from document chunks.
Uses OpenAI-compatible APIs (vLLM, OpenAI, etc.) for embedding generation.
"""

import os
import uuid

from temporalio import activity

from ingestion_worker.workflows.document_ingestion import (
    GenerateEmbeddingsInput,
    GenerateEmbeddingsOutput,
    EmbeddingInfo,
    SparseVector,
)


@activity.defn
async def generate_embeddings(input: GenerateEmbeddingsInput) -> GenerateEmbeddingsOutput:
    """
    Generate embeddings for document chunks.

    Uses the configured embedding model (e.g., BGE via vLLM) to generate
    dense vector embeddings for each chunk. Also generates sparse vectors
    for hybrid search (BM25-style).

    Features:
    - Batch processing for efficiency
    - Dense embeddings via OpenAI-compatible API
    - Sparse vectors for hybrid search
    - Retry logic with exponential backoff
    - Heartbeat support for long-running operations

    Args:
        input: Embedding input with chunks to embed

    Returns:
        Embedding IDs, vectors, and model information
    """
    from ingestion_worker.activities.embeddings import (
        EmbeddingConfig,
        EmbeddingClient,
        EmbeddingModel,
    )

    activity.logger.info(
        f"Generating embeddings for {len(input.chunks)} chunks",
        extra={
            "document_id": input.document_id,
            "tenant_id": input.tenant_id,
            "chunk_count": len(input.chunks),
        },
    )

    # Configure embedding client from environment
    config = EmbeddingConfig(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", EmbeddingModel.BGE_BASE_EN),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        generate_sparse=os.getenv("GENERATE_SPARSE_VECTORS", "true").lower() == "true",
    )

    # Extract text from chunks
    texts = [chunk.content for chunk in input.chunks]

    embeddings: list[EmbeddingInfo] = []

    # Heartbeat callback for batch progress
    batch_count = 0

    def on_batch_complete(current_batch: int, total_texts: int, processed: int):
        nonlocal batch_count
        batch_count = current_batch
        activity.heartbeat(f"Batch {current_batch}: {processed}/{total_texts} texts")

    try:
        async with EmbeddingClient(config) as client:
            result = await client.embed_texts(texts, on_batch_complete=on_batch_complete)

            # Map embedding results to chunks
            for i, (chunk, emb_result) in enumerate(zip(input.chunks, result.embeddings)):
                # Generate unique embedding ID
                embedding_id = str(uuid.uuid4())

                # Convert sparse vector if present
                sparse_vector = None
                if emb_result.sparse:
                    sparse_vector = SparseVector(
                        indices=emb_result.sparse.indices,
                        values=emb_result.sparse.values,
                    )

                embeddings.append(
                    EmbeddingInfo(
                        chunk_sequence=chunk.sequence_number,
                        embedding_id=embedding_id,
                        model=result.model,
                        dense_vector=emb_result.dense.vector,
                        sparse_vector=sparse_vector,
                        dimensions=emb_result.dense.dimensions,
                    )
                )

        activity.logger.info(
            f"Generated {len(embeddings)} embeddings",
            extra={
                "document_id": input.document_id,
                "model": result.model,
                "total_tokens": result.total_tokens,
                "batch_count": result.batch_count,
                "dimensions": embeddings[0].dimensions if embeddings else 0,
            },
        )

        return GenerateEmbeddingsOutput(
            embeddings=embeddings,
            model=result.model,
        )

    except Exception as e:
        activity.logger.error(
            f"Embedding generation failed: {e}",
            extra={
                "document_id": input.document_id,
                "error": str(e),
            },
        )
        raise


@activity.defn
async def generate_embeddings_mock(input: GenerateEmbeddingsInput) -> GenerateEmbeddingsOutput:
    """
    Mock embedding generation for testing.

    Generates deterministic fake embeddings without calling an external API.
    Useful for testing the workflow without a running embedding service.

    Args:
        input: Embedding input with chunks to embed

    Returns:
        Mock embedding results
    """
    import hashlib

    activity.logger.info(f"Generating mock embeddings for {len(input.chunks)} chunks")

    model = "mock-embedding-model"
    dimensions = 768  # Standard dimension for testing

    embeddings: list[EmbeddingInfo] = []

    for i, chunk in enumerate(input.chunks):
        # Heartbeat every 10 chunks
        if i % 10 == 0:
            activity.heartbeat()

        # Generate deterministic embedding ID from chunk content
        content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
        embedding_id = f"mock-{content_hash[:8]}"

        # Generate deterministic fake vector from content hash
        # This ensures same content always produces same vector
        seed = int(content_hash[:8], 16)
        import random

        rng = random.Random(seed)
        fake_vector = [rng.gauss(0, 1) for _ in range(dimensions)]

        # Normalize the vector
        norm = sum(x * x for x in fake_vector) ** 0.5
        fake_vector = [x / norm for x in fake_vector]

        # Generate mock sparse vector
        words = chunk.content.lower().split()[:20]  # First 20 words
        sparse_indices = [hash(w) % 30000 for w in set(words)]
        sparse_values = [1.0] * len(sparse_indices)

        embeddings.append(
            EmbeddingInfo(
                chunk_sequence=chunk.sequence_number,
                embedding_id=embedding_id,
                model=model,
                dense_vector=fake_vector,
                sparse_vector=SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                dimensions=dimensions,
            )
        )

    activity.logger.info(f"Generated {len(embeddings)} mock embeddings")

    return GenerateEmbeddingsOutput(
        embeddings=embeddings,
        model=model,
    )
