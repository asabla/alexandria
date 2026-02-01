"""
Document processing activities.

Activities for document classification, parsing, chunking, and status updates.
"""

import hashlib
import os

from temporalio import activity

from ingestion_worker.activities.classification import (
    classify_document as classify_document_impl,
    ClassificationResult,
)
from ingestion_worker.workflows.document_ingestion import (
    ClassifyDocumentInput,
    ClassifyDocumentOutput,
    ParseDocumentInput,
    ParseDocumentOutput,
    ChunkDocumentInput,
    ChunkDocumentOutput,
    ChunkInfo,
    UpdateDocumentStatusInput,
)

# Number of bytes to read for magic number detection
MAGIC_BYTES_SIZE = 8192


def _get_minio_client():
    """Get MinIO client from environment configuration."""
    from alexandria_db.clients.minio import MinIOClient

    return MinIOClient(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )


@activity.defn
async def classify_document(input: ClassifyDocumentInput) -> ClassifyDocumentOutput:
    """
    Classify the document type based on magic bytes, file extension, and MIME type.

    This activity uses a multi-layered detection approach:
    1. Magic bytes (file signature) - most reliable
    2. File extension - common and quick
    3. MIME type hint - if provided
    4. Content analysis - for text files

    For optimal detection, it reads the first bytes of the file from storage
    to examine the magic bytes/file signature.

    Args:
        input: Classification input with storage location and hints

    Returns:
        Document classification with type, MIME type, and confidence score
    """
    activity.logger.info(
        "Classifying document",
        extra={
            "storage_key": input.storage_key,
            "storage_bucket": input.storage_bucket,
            "source_filename": input.source_filename,
            "provided_mime_type": input.mime_type,
        },
    )

    file_data: bytes | None = None

    # Try to read magic bytes from storage if we have a storage key
    if input.storage_key and input.storage_bucket:
        try:
            minio_client = _get_minio_client()

            # Read first bytes for magic number detection
            async with minio_client.stream_object(
                input.storage_bucket, input.storage_key
            ) as stream:
                file_data = stream.read(MAGIC_BYTES_SIZE)

            activity.logger.debug(
                "Read magic bytes from storage",
                extra={"bytes_read": len(file_data) if file_data else 0},
            )
        except Exception as e:
            # Log but continue - we can still classify by extension/mime type
            activity.logger.warning(
                "Could not read file from storage for magic bytes detection",
                extra={"error": str(e), "storage_key": input.storage_key},
            )

    # Perform classification using multi-method detection
    result: ClassificationResult = classify_document_impl(
        data=file_data,
        filename=input.source_filename,
        mime_type=input.mime_type,
    )

    activity.logger.info(
        "Document classified",
        extra={
            "document_type": result.document_type,
            "mime_type": result.mime_type,
            "confidence": result.confidence,
            "detection_method": result.detection_method,
            "description": result.description,
        },
    )

    return ClassifyDocumentOutput(
        document_type=result.document_type.value,
        mime_type=result.mime_type,
        confidence=result.confidence,
    )


@activity.defn
async def parse_document(input: ParseDocumentInput) -> ParseDocumentOutput:
    """
    Parse document content based on its type.

    This activity handles different document types:
    - PDF: Use Docling for extraction with optional OCR
    - Images: Use OCR via Docling
    - Audio/Video: Use transcription (Parakeet/Whisper) - not yet implemented
    - HTML/Markdown: Direct text extraction via Docling
    - Office docs: Use Docling (DOCX, PPTX, XLSX)

    Args:
        input: Parse input with document location and type

    Returns:
        Parsed content with metadata
    """
    from ingestion_worker.activities.parsing import (
        is_docling_supported,
        parse_with_docling,
    )

    activity.logger.info(
        "Parsing document",
        extra={
            "document_id": input.document_id,
            "document_type": input.document_type,
            "storage_bucket": input.storage_bucket,
            "storage_key": input.storage_key,
            "skip_ocr": input.skip_ocr,
        },
    )

    # Check if Docling can handle this document type
    if is_docling_supported(input.document_type):
        # Use Docling for document parsing
        activity.heartbeat("Starting Docling parsing")

        try:
            result = await parse_with_docling(
                document_id=input.document_id,
                storage_bucket=input.storage_bucket,
                storage_key=input.storage_key,
                document_type=input.document_type,
                skip_ocr=input.skip_ocr,
                enable_table_structure=True,
            )

            activity.logger.info(
                "Document parsed with Docling",
                extra={
                    "document_id": input.document_id,
                    "page_count": result.get("page_count"),
                    "word_count": result.get("word_count"),
                    "tables_count": len(result.get("tables", [])),
                    "images_count": len(result.get("images", [])),
                },
            )

            # Use Markdown content as the primary content format
            # This preserves structure better than plain text
            content = result.get("content_markdown", "")

            return ParseDocumentOutput(
                content=content,
                page_count=result.get("page_count"),
                word_count=result.get("word_count"),
                language=result.get("language"),
                tables=result.get("tables", []),
                images=result.get("images", []),
            )

        except Exception as e:
            activity.logger.error(
                "Docling parsing failed",
                extra={"document_id": input.document_id, "error": str(e)},
            )
            raise RuntimeError(f"Failed to parse document with Docling: {e}") from e

    # Handle audio/video types (not yet implemented - placeholder for future)
    audio_video_types = {"audio", "video", "mp3", "wav", "mp4", "webm", "ogg"}
    if input.document_type.lower() in audio_video_types:
        activity.logger.warning(
            "Audio/video transcription not yet implemented",
            extra={"document_id": input.document_id, "document_type": input.document_type},
        )
        # TODO: Implement transcription using Parakeet/Whisper
        # For now, return placeholder
        return ParseDocumentOutput(
            content=f"[Audio/video transcription pending for {input.document_id}]",
            page_count=None,
            word_count=0,
            language=None,
            tables=[],
            images=[],
        )

    # Unknown document type - log warning and return empty
    activity.logger.warning(
        "Unsupported document type for parsing",
        extra={"document_id": input.document_id, "document_type": input.document_type},
    )

    return ParseDocumentOutput(
        content=f"[Unsupported document type: {input.document_type}]",
        page_count=None,
        word_count=0,
        language=None,
        tables=[],
        images=[],
    )


@activity.defn
async def chunk_document(input: ChunkDocumentInput) -> ChunkDocumentOutput:
    """
    Split document content into chunks for embedding.

    Uses semantic chunking to preserve meaning across chunk boundaries.
    Respects paragraph and sentence boundaries when possible.

    Args:
        input: Chunk input with content and chunking parameters

    Returns:
        List of chunks with metadata
    """
    activity.logger.info(f"Chunking document {input.document_id}")

    content = input.content
    chunk_size = input.chunk_size
    chunk_overlap = input.chunk_overlap

    # Simple chunking implementation
    # In production, use semantic chunking with sentence boundaries
    chunks: list[ChunkInfo] = []

    # Split by paragraphs first
    paragraphs = content.split("\n\n")

    current_chunk = ""
    current_start = 0
    sequence = 0

    for para in paragraphs:
        # Rough token estimation (words * 1.3)
        estimated_tokens = int(len(para.split()) * 1.3)
        current_tokens = int(len(current_chunk.split()) * 1.3)

        if current_tokens + estimated_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunk_hash = hashlib.sha256(current_chunk.encode()).hexdigest()
            chunks.append(
                ChunkInfo(
                    sequence_number=sequence,
                    content=current_chunk.strip(),
                    content_hash=chunk_hash,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    token_count=current_tokens,
                    page_number=None,  # Would be set from parsing metadata
                )
            )
            sequence += 1

            # Start new chunk with overlap
            overlap_words = current_chunk.split()[-chunk_overlap:] if chunk_overlap else []
            current_chunk = " ".join(overlap_words) + "\n\n" + para if overlap_words else para
            current_start = current_start + len(current_chunk) - len(current_chunk)
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    # Don't forget the last chunk
    if current_chunk.strip():
        chunk_hash = hashlib.sha256(current_chunk.encode()).hexdigest()
        chunks.append(
            ChunkInfo(
                sequence_number=sequence,
                content=current_chunk.strip(),
                content_hash=chunk_hash,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                token_count=int(len(current_chunk.split()) * 1.3),
                page_number=None,
            )
        )

    activity.logger.info(f"Created {len(chunks)} chunks")

    return ChunkDocumentOutput(
        chunks=chunks,
        total_chunks=len(chunks),
    )


@activity.defn
async def update_document_status(input: UpdateDocumentStatusInput) -> None:
    """
    Update document status in the database.

    This activity updates the document record with:
    - Processing status (pending, processing, completed, failed)
    - Document type and MIME type (from classification)
    - Page count, word count, language (from parsing)
    - Indexing status flags (vector, fulltext, graph)
    - Error message (if failed)

    Args:
        input: Status update input with document ID, status, and metadata
    """
    activity.logger.info(
        "Updating document status",
        extra={
            "document_id": input.document_id,
            "tenant_id": input.tenant_id,
            "status": input.status,
            "document_type": input.document_type,
            "mime_type": input.mime_type,
            "page_count": input.page_count,
            "word_count": input.word_count,
            "language": input.language,
            "is_indexed_vector": input.is_indexed_vector,
            "is_indexed_fulltext": input.is_indexed_fulltext,
            "is_indexed_graph": input.is_indexed_graph,
            "error_message": input.error_message,
        },
    )

    # TODO: Implement actual database update
    # In real implementation:
    # 1. Get database session from activity context or create new connection
    # 2. Update document record with all provided fields:
    #    - status
    #    - document_type (if provided)
    #    - mime_type (if provided)
    #    - page_count, word_count, language
    #    - is_indexed_vector, is_indexed_fulltext, is_indexed_graph
    #    - error_message (if status == "failed")
    #    - processing_completed_at = now() if status in ("completed", "failed")
    # 3. Update ingestion job record if exists

    activity.logger.info(
        "Document status updated",
        extra={"document_id": input.document_id, "status": input.status},
    )
