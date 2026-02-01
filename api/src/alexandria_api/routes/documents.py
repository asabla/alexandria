"""Document management routes."""

import hashlib
from datetime import datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from alexandria_api.config import Settings, get_settings
from alexandria_api.dependencies import get_db_session, MinIO, Temporal, Context
from alexandria_db import DocumentRepository, ChunkRepository, IngestionJobRepository
from alexandria_db.models import DocumentModel, IngestionJobModel

router = APIRouter(prefix="/documents", tags=["Documents"])


# ============================================================
# Response Models
# ============================================================


class DocumentResponse(BaseModel):
    """Document response model."""

    id: UUID
    tenant_id: UUID
    title: str
    description: str | None
    document_type: str
    source_url: str | None
    source_filename: str | None
    mime_type: str | None
    storage_key: str
    storage_bucket: str
    file_size_bytes: int
    file_hash: str | None
    status: str
    error_message: str | None
    processing_started_at: datetime | None
    processing_completed_at: datetime | None
    page_count: int | None
    word_count: int | None
    language: str | None
    summary: str | None
    is_indexed_vector: bool
    is_indexed_fulltext: bool
    is_indexed_graph: bool
    metadata: dict
    chunk_count: int  # Computed field
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    items: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class ChunkResponse(BaseModel):
    """Chunk response model."""

    id: UUID
    document_id: UUID
    sequence_number: int
    content: str
    content_hash: str
    page_number: int | None
    start_char: int | None
    end_char: int | None
    chunk_type: str
    token_count: int
    embedding_model: str | None
    embedding_id: str | None
    metadata: dict
    created_at: datetime


class IngestionJobResponse(BaseModel):
    """Ingestion job response model."""

    id: UUID
    document_id: UUID
    workflow_id: str
    workflow_run_id: str | None
    status: str
    current_step: str | None
    progress_percent: int
    chunks_created: int
    entities_extracted: int
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class UploadResponse(BaseModel):
    """Response after uploading a document."""

    document: DocumentResponse
    job: IngestionJobResponse | None


class DocumentStatsResponse(BaseModel):
    """Document statistics."""

    total: int
    by_status: dict[str, int]


# ============================================================
# Request Models
# ============================================================


class UpdateDocumentRequest(BaseModel):
    """Request to update document metadata."""

    title: str | None = None
    description: str | None = None
    language: str | None = None
    metadata: dict | None = None


# ============================================================
# Helper Functions
# ============================================================


def _document_to_response(doc: DocumentModel, chunk_count: int = 0) -> DocumentResponse:
    """Convert a DocumentModel to DocumentResponse."""
    return DocumentResponse(
        id=doc.id,
        tenant_id=doc.tenant_id,
        title=doc.title,
        description=doc.description,
        document_type=doc.document_type,
        source_url=doc.source_url,
        source_filename=doc.source_filename,
        mime_type=doc.mime_type,
        storage_key=doc.storage_key,
        storage_bucket=doc.storage_bucket,
        file_size_bytes=doc.file_size_bytes,
        file_hash=doc.file_hash,
        status=doc.status,
        error_message=doc.error_message,
        processing_started_at=doc.processing_started_at,
        processing_completed_at=doc.processing_completed_at,
        page_count=doc.page_count,
        word_count=doc.word_count,
        language=doc.language,
        summary=doc.summary,
        is_indexed_vector=doc.is_indexed_vector,
        is_indexed_fulltext=doc.is_indexed_fulltext,
        is_indexed_graph=doc.is_indexed_graph,
        metadata=doc.metadata_ or {},
        chunk_count=chunk_count,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


def _chunk_to_response(chunk) -> ChunkResponse:
    """Convert a ChunkModel to ChunkResponse."""
    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        sequence_number=chunk.sequence_number,
        content=chunk.content,
        content_hash=chunk.content_hash,
        page_number=chunk.page_number,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        chunk_type=chunk.chunk_type,
        token_count=chunk.token_count,
        embedding_model=chunk.embedding_model,
        embedding_id=chunk.embedding_id,
        metadata=chunk.metadata_ or {},
        created_at=chunk.created_at,
    )


def _job_to_response(job: IngestionJobModel) -> IngestionJobResponse:
    """Convert an IngestionJobModel to IngestionJobResponse."""
    return IngestionJobResponse(
        id=job.id,
        document_id=job.document_id,
        workflow_id=job.workflow_id,
        workflow_run_id=job.workflow_run_id,
        status=job.status,
        current_step=job.current_step,
        progress_percent=job.progress_percent,
        chunks_created=job.chunks_created,
        entities_extracted=job.entities_extracted,
        error_message=job.error_message,
        started_at=job.started_at,
        completed_at=job.completed_at,
        created_at=job.created_at,
    )


# ============================================================
# Endpoints
# ============================================================


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: str | None = Query(default=None, alias="status"),
    project_id: UUID | None = Query(default=None),
) -> DocumentListResponse:
    """
    List all documents for the current tenant.

    Supports pagination and filtering by status or project.
    """
    repo = DocumentRepository(db, ctx.tenant_id)
    chunk_repo = ChunkRepository(db, ctx.tenant_id)

    if project_id:
        documents = await repo.get_by_project(project_id, limit=limit, offset=offset)
    elif status_filter:
        documents = await repo.get_by_status(status_filter, limit=limit)
    else:
        documents = await repo.get_all(limit=limit, offset=offset)

    # Get chunk counts for each document
    items = []
    for doc in documents:
        chunk_count = await chunk_repo.count_by_document(doc.id)
        items.append(_document_to_response(doc, chunk_count))

    # Get total count
    if status_filter:
        counts = await repo.count_by_status()
        total = counts.get(status_filter, 0)
    else:
        total = await repo.count()

    return DocumentListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/stats", response_model=DocumentStatsResponse)
async def get_document_stats(
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentStatsResponse:
    """Get document statistics for the current tenant."""
    repo = DocumentRepository(db, ctx.tenant_id)

    total = await repo.count()
    by_status = await repo.count_by_status()

    return DocumentStatsResponse(
        total=total,
        by_status=by_status,
    )


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    minio: MinIO,
    temporal: Temporal,
    settings: Annotated[Settings, Depends(get_settings)],
    file: UploadFile = File(...),
    project_id: UUID | None = Form(default=None),
    title: str | None = Form(default=None),
    description: str | None = Form(default=None),
    skip_ingestion: bool = Form(default=False),
) -> UploadResponse:
    """
    Upload a new document.

    The document will be stored in MinIO and an ingestion job will be created
    to process it (extract text, create chunks, etc.).

    Args:
        file: The file to upload
        project_id: Optional project to add the document to
        title: Optional title (defaults to filename)
        description: Optional description
        skip_ingestion: If True, skip creating an ingestion job (useful for testing)
    """
    # Read file content
    content = await file.read()
    file_size = len(content)

    # Calculate file hash for deduplication
    file_hash = hashlib.sha256(content).hexdigest()

    # Check for duplicate
    repo = DocumentRepository(db, ctx.tenant_id)
    existing = await repo.get_by_hash(file_hash)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document with same content already exists: {existing.id}",
        )

    # Generate unique storage key
    unique_id = uuid4()
    original_filename = file.filename or "unknown"
    ext = original_filename.rsplit(".", 1)[-1] if "." in original_filename else ""
    storage_key = (
        f"documents/{ctx.tenant_id}/{unique_id}.{ext}"
        if ext
        else f"documents/{ctx.tenant_id}/{unique_id}"
    )

    # Determine document type from extension
    document_type = _get_document_type(ext, file.content_type)

    # Upload to MinIO
    bucket = "documents"
    try:
        await minio.ensure_bucket(bucket)
        await minio.upload_object(
            bucket=bucket,
            key=storage_key,
            data=content,
            content_type=file.content_type or "application/octet-stream",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file to storage: {e}",
        )

    # Create document record
    document = DocumentModel(
        tenant_id=ctx.tenant_id,
        title=title or original_filename,
        description=description,
        document_type=document_type,
        source_filename=original_filename,
        mime_type=file.content_type,
        storage_key=storage_key,
        storage_bucket=bucket,
        file_size_bytes=file_size,
        file_hash=file_hash,
        status="pending",
    )
    document = await repo.create(document)

    # Add to project if specified
    if project_id:
        await repo.add_to_project(document.id, project_id)

    # Create ingestion job
    job = None
    if not skip_ingestion:
        job_repo = IngestionJobRepository(db, ctx.tenant_id)
        workflow_id = f"ingest-{document.id}"
        job = IngestionJobModel(
            tenant_id=ctx.tenant_id,
            document_id=document.id,
            workflow_id=workflow_id,
            status="pending",
        )
        job = await job_repo.create(job)

        # Start Temporal workflow for ingestion
        try:
            handle = await temporal.start_workflow(
                "DocumentIngestionWorkflow",
                {
                    "document_id": str(document.id),
                    "tenant_id": str(ctx.tenant_id),
                    "storage_bucket": bucket,
                    "storage_key": storage_key,
                    "source_filename": original_filename,
                    "mime_type": file.content_type,
                    "project_id": str(project_id) if project_id else None,
                },
                id=workflow_id,
                task_queue="ingestion",
            )
            # Update job with run ID
            job.workflow_run_id = handle.result_run_id
            job.status = "running"
            job = await job_repo.update(job)
        except Exception as e:
            # Log error but don't fail the upload
            # The job can be retried later
            job.status = "failed"
            job.error_message = f"Failed to start workflow: {e}"
            job = await job_repo.update(job)

    return UploadResponse(
        document=_document_to_response(document),
        job=_job_to_response(job) if job else None,
    )


def _get_document_type(ext: str, content_type: str | None) -> str:
    """Determine document type from extension or content type."""
    ext_map = {
        "pdf": "pdf",
        "doc": "word",
        "docx": "word",
        "txt": "text",
        "md": "markdown",
        "html": "html",
        "htm": "html",
        "csv": "csv",
        "xlsx": "excel",
        "xls": "excel",
        "json": "json",
        "xml": "xml",
        "jpg": "image",
        "jpeg": "image",
        "png": "image",
        "gif": "image",
        "tiff": "image",
    }
    if ext.lower() in ext_map:
        return ext_map[ext.lower()]

    if content_type:
        if "pdf" in content_type:
            return "pdf"
        if "word" in content_type or "msword" in content_type:
            return "word"
        if "text" in content_type:
            return "text"
        if "html" in content_type:
            return "html"
        if "image" in content_type:
            return "image"

    return "unknown"


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentResponse:
    """Get a specific document by ID."""
    repo = DocumentRepository(db, ctx.tenant_id)
    chunk_repo = ChunkRepository(db, ctx.tenant_id)

    document = await repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    chunk_count = await chunk_repo.count_by_document(document_id)
    return _document_to_response(document, chunk_count)


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: UUID,
    request: UpdateDocumentRequest,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentResponse:
    """Update document metadata."""
    repo = DocumentRepository(db, ctx.tenant_id)
    chunk_repo = ChunkRepository(db, ctx.tenant_id)

    document = await repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    # Update fields if provided
    if request.title is not None:
        document.title = request.title
    if request.description is not None:
        document.description = request.description
    if request.language is not None:
        document.language = request.language
    if request.metadata is not None:
        document.metadata_ = request.metadata

    document = await repo.update(document)
    chunk_count = await chunk_repo.count_by_document(document_id)

    return _document_to_response(document, chunk_count)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    minio: MinIO,
    hard_delete: bool = Query(default=False),
) -> None:
    """
    Delete a document.

    By default, performs a soft delete (sets deleted_at).
    Use hard_delete=true to permanently remove the document and its storage.
    """
    repo = DocumentRepository(db, ctx.tenant_id)

    document = await repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    if hard_delete:
        # Delete from MinIO
        try:
            await minio.delete_object(document.storage_bucket, document.storage_key)
        except Exception:
            pass  # Ignore storage errors during deletion

        # Delete chunks
        chunk_repo = ChunkRepository(db, ctx.tenant_id)
        await chunk_repo.delete_by_document(document_id)

        # Hard delete document
        await repo.delete(document)
    else:
        # Soft delete
        await repo.soft_delete(document)


@router.get("/{document_id}/chunks", response_model=list[ChunkResponse])
async def get_document_chunks(
    document_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[ChunkResponse]:
    """Get all chunks for a document."""
    # Verify document exists
    doc_repo = DocumentRepository(db, ctx.tenant_id)
    document = await doc_repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    chunk_repo = ChunkRepository(db, ctx.tenant_id)
    chunks = await chunk_repo.get_by_document(document_id, limit=limit, offset=offset)

    return [_chunk_to_response(chunk) for chunk in chunks]


@router.get("/{document_id}/jobs", response_model=list[IngestionJobResponse])
async def get_document_jobs(
    document_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    limit: int = Query(default=10, ge=1, le=50),
) -> list[IngestionJobResponse]:
    """Get ingestion jobs for a document."""
    # Verify document exists
    doc_repo = DocumentRepository(db, ctx.tenant_id)
    document = await doc_repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    job_repo = IngestionJobRepository(db, ctx.tenant_id)
    jobs = await job_repo.get_by_document(document_id, limit=limit)

    return [_job_to_response(job) for job in jobs]


class WorkflowStatusResponse(BaseModel):
    """Workflow status response model."""

    workflow_id: str
    run_id: str | None
    status: str
    current_step: str | None = None
    progress_percent: int | None = None


@router.get("/{document_id}/jobs/{job_id}/status", response_model=WorkflowStatusResponse)
async def get_job_workflow_status(
    document_id: UUID,
    job_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    temporal: Temporal,
) -> WorkflowStatusResponse:
    """
    Get the real-time status of an ingestion job's workflow.

    Queries Temporal directly for the current workflow state.
    """
    # Verify document exists
    doc_repo = DocumentRepository(db, ctx.tenant_id)
    document = await doc_repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    # Get the job
    job_repo = IngestionJobRepository(db, ctx.tenant_id)
    job = await job_repo.get_by_id(job_id)
    if job is None or job.document_id != document_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    try:
        # Get workflow info from Temporal
        workflow_info = await temporal.describe_workflow(job.workflow_id)

        # Try to get detailed status via query if workflow is running
        current_step = None
        progress_percent = None
        if workflow_info.status == "running":
            try:
                status_result = await temporal.query_workflow(
                    job.workflow_id,
                    "get_status",
                )
                if status_result:
                    current_step = status_result.get("current_step")
                    progress_percent = status_result.get("progress_percent")
            except Exception:
                pass  # Query may fail if workflow doesn't support it

        return WorkflowStatusResponse(
            workflow_id=workflow_info.workflow_id,
            run_id=workflow_info.run_id,
            status=workflow_info.status,
            current_step=current_step,
            progress_percent=progress_percent,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow status: {e}",
        )


@router.post("/{document_id}/jobs/{job_id}/cancel", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(
    document_id: UUID,
    job_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    temporal: Temporal,
) -> None:
    """
    Cancel a running ingestion job.

    Sends a cancel signal to the Temporal workflow.
    """
    # Verify document exists
    doc_repo = DocumentRepository(db, ctx.tenant_id)
    document = await doc_repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    # Get the job
    job_repo = IngestionJobRepository(db, ctx.tenant_id)
    job = await job_repo.get_by_id(job_id)
    if job is None or job.document_id != document_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    try:
        # Send cancel signal to workflow
        await temporal.signal_workflow(job.workflow_id, "cancel")

        # Update job status
        job.status = "cancelling"
        await job_repo.update(job)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel workflow: {e}",
        )


@router.get("/{document_id}/download")
async def download_document(
    document_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    minio: MinIO,
) -> dict:
    """
    Get a presigned URL to download the document.

    Returns a URL that's valid for 1 hour.
    """
    repo = DocumentRepository(db, ctx.tenant_id)

    document = await repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    try:
        url = await minio.get_presigned_url(
            bucket=document.storage_bucket,
            key=document.storage_key,
            expires_in=3600,
        )
        return {
            "url": url,
            "filename": document.source_filename,
            "content_type": document.mime_type,
            "expires_in": 3600,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate download URL: {e}",
        )


@router.post("/{document_id}/reprocess", response_model=IngestionJobResponse)
async def reprocess_document(
    document_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    temporal: Temporal,
) -> IngestionJobResponse:
    """
    Trigger reprocessing of a document.

    Creates a new ingestion job and starts the workflow.
    """
    doc_repo = DocumentRepository(db, ctx.tenant_id)

    document = await doc_repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    # Reset document status
    document.status = "pending"
    document.is_indexed_vector = False
    document.is_indexed_fulltext = False
    document.is_indexed_graph = False
    await doc_repo.update(document)

    # Create new ingestion job
    job_repo = IngestionJobRepository(db, ctx.tenant_id)
    workflow_id = f"ingest-{document.id}-{uuid4()}"
    job = IngestionJobModel(
        tenant_id=ctx.tenant_id,
        document_id=document.id,
        workflow_id=workflow_id,
        status="pending",
    )
    job = await job_repo.create(job)

    # Start Temporal workflow for ingestion
    try:
        handle = await temporal.start_workflow(
            "DocumentIngestionWorkflow",
            {
                "document_id": str(document.id),
                "tenant_id": str(ctx.tenant_id),
                "storage_bucket": document.storage_bucket,
                "storage_key": document.storage_key,
                "source_filename": document.source_filename,
                "mime_type": document.mime_type,
                "force_reprocess": True,
            },
            id=workflow_id,
            task_queue="ingestion",
        )
        # Update job with run ID
        job.workflow_run_id = handle.result_run_id
        job.status = "running"
        job = await job_repo.update(job)
    except Exception as e:
        job.status = "failed"
        job.error_message = f"Failed to start workflow: {e}"
        job = await job_repo.update(job)

    return _job_to_response(job)
