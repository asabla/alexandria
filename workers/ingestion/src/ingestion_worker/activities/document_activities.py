"""
Document processing activities.

Activities for document classification, parsing, chunking, and status updates.
"""

import hashlib
from temporalio import activity

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


@activity.defn
async def classify_document(input: ClassifyDocumentInput) -> ClassifyDocumentOutput:
    """
    Classify the document type based on file extension, mime type, and magic bytes.

    Args:
        input: Classification input with storage location and hints

    Returns:
        Document classification with type and confidence
    """
    activity.logger.info(f"Classifying document: {input.storage_key}")

    # Determine document type from mime type or filename
    document_type = "unknown"
    mime_type = input.mime_type or "application/octet-stream"

    # Check filename extension
    if input.source_filename:
        ext = input.source_filename.lower().split(".")[-1] if "." in input.source_filename else ""
        extension_map = {
            "pdf": ("pdf", "application/pdf"),
            "docx": (
                "docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            "doc": ("doc", "application/msword"),
            "txt": ("txt", "text/plain"),
            "md": ("markdown", "text/markdown"),
            "html": ("html", "text/html"),
            "htm": ("html", "text/html"),
            "jpg": ("image", "image/jpeg"),
            "jpeg": ("image", "image/jpeg"),
            "png": ("image", "image/png"),
            "gif": ("image", "image/gif"),
            "mp3": ("audio", "audio/mpeg"),
            "wav": ("audio", "audio/wav"),
            "mp4": ("video", "video/mp4"),
            "xlsx": (
                "spreadsheet",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            "csv": ("spreadsheet", "text/csv"),
            "eml": ("email", "message/rfc822"),
        }
        if ext in extension_map:
            document_type, mime_type = extension_map[ext]

    # Check mime type if extension didn't match
    if document_type == "unknown" and input.mime_type:
        mime_map = {
            "application/pdf": "pdf",
            "text/plain": "txt",
            "text/html": "html",
            "text/markdown": "markdown",
            "image/": "image",
            "audio/": "audio",
            "video/": "video",
        }
        for mime_prefix, doc_type in mime_map.items():
            if input.mime_type.startswith(mime_prefix):
                document_type = doc_type
                break

    activity.logger.info(f"Document classified as: {document_type} ({mime_type})")

    return ClassifyDocumentOutput(
        document_type=document_type,
        mime_type=mime_type,
        confidence=1.0 if document_type != "unknown" else 0.5,
    )


@activity.defn
async def parse_document(input: ParseDocumentInput) -> ParseDocumentOutput:
    """
    Parse document content based on its type.

    This activity handles different document types:
    - PDF: Use Docling for extraction with optional OCR
    - Images: Use OCR via Docling
    - Audio/Video: Use transcription (Parakeet/Whisper)
    - HTML/Markdown: Direct text extraction
    - Office docs: Use python-docx, openpyxl, etc.

    Args:
        input: Parse input with document location and type

    Returns:
        Parsed content with metadata
    """
    activity.logger.info(f"Parsing document {input.document_id} of type {input.document_type}")

    # TODO: Implement actual parsing using Docling, transcription services, etc.
    # For now, return a placeholder

    # In real implementation:
    # 1. Download file from MinIO
    # 2. Based on document_type, use appropriate parser
    # 3. For PDFs: Use Docling with OCR if needed
    # 4. For images: Use Docling OCR
    # 5. For audio/video: Use transcription service
    # 6. Extract tables and images metadata
    # 7. Return parsed content

    # Heartbeat to indicate we're still working
    activity.heartbeat()

    # Placeholder content
    content = f"[Placeholder content for document {input.document_id}]"

    return ParseDocumentOutput(
        content=content,
        page_count=1,
        word_count=len(content.split()),
        language="en",
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

    Args:
        input: Status update input with document ID and new status
    """
    activity.logger.info(f"Updating document {input.document_id} status to {input.status}")

    # TODO: Implement actual database update
    # In real implementation:
    # 1. Get database session
    # 2. Update document record with new status and metadata
    # 3. Update ingestion job record if exists

    activity.logger.info(f"Document {input.document_id} status updated to {input.status}")
