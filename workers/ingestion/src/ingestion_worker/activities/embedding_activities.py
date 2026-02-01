"""
Embedding generation activities.

Activities for generating vector embeddings from document chunks.
"""

import uuid
from temporalio import activity

from ingestion_worker.workflows.document_ingestion import (
    GenerateEmbeddingsInput,
    GenerateEmbeddingsOutput,
    EmbeddingInfo,
)


@activity.defn
async def generate_embeddings(input: GenerateEmbeddingsInput) -> GenerateEmbeddingsOutput:
    """
    Generate embeddings for document chunks.

    Uses the configured embedding model (e.g., BGE, OpenAI) to generate
    dense vector embeddings for each chunk.

    Args:
        input: Embedding input with chunks to embed

    Returns:
        Embedding IDs and model information
    """
    activity.logger.info(f"Generating embeddings for {len(input.chunks)} chunks")

    embeddings: list[EmbeddingInfo] = []
    model = "bge-base-en-v1.5"  # Default model

    # TODO: Implement actual embedding generation
    # In real implementation:
    # 1. Get embedding client (vLLM or OpenAI compatible)
    # 2. Batch chunks for efficient processing
    # 3. Generate embeddings for each batch
    # 4. Return embedding vectors

    for i, chunk in enumerate(input.chunks):
        # Heartbeat every 10 chunks
        if i % 10 == 0:
            activity.heartbeat()

        # Generate a placeholder embedding ID
        embedding_id = str(uuid.uuid4())

        embeddings.append(
            EmbeddingInfo(
                chunk_sequence=chunk.sequence_number,
                embedding_id=embedding_id,
                model=model,
            )
        )

    activity.logger.info(f"Generated {len(embeddings)} embeddings with model {model}")

    return GenerateEmbeddingsOutput(
        embeddings=embeddings,
        model=model,
    )
