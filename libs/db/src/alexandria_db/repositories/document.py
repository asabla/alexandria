"""Document, Chunk, and IngestionJob repositories."""

from datetime import datetime, UTC
from typing import Sequence
from uuid import UUID

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from alexandria_db.models import (
    DocumentModel,
    ChunkModel,
    IngestionJobModel,
    ProjectDocumentModel,
)
from alexandria_db.repositories.base import SoftDeleteRepository, BaseRepository


class DocumentRepository(SoftDeleteRepository[DocumentModel]):
    """Repository for document operations."""

    model_class = DocumentModel

    def __init__(self, session: AsyncSession, tenant_id: UUID):
        """Initialize with session and tenant scope."""
        super().__init__(session)
        self.tenant_id = tenant_id

    async def get_by_id(self, id: UUID) -> DocumentModel | None:
        """Get a document by ID (scoped to tenant)."""
        stmt = select(self.model_class).where(
            self.model_class.id == id,
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id_with_chunks(self, id: UUID) -> DocumentModel | None:
        """Get a document by ID with its chunks (scoped to tenant)."""
        stmt = (
            select(self.model_class)
            .options(selectinload(self.model_class.chunks))
            .where(
                self.model_class.id == id,
                self.model_class.tenant_id == self.tenant_id,
                self.model_class.deleted_at.is_(None),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        include_deleted: bool = False,
    ) -> Sequence[DocumentModel]:
        """Get all documents (scoped to tenant)."""
        stmt = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,
        )
        if not include_deleted:
            stmt = stmt.where(self.model_class.deleted_at.is_(None))
        stmt = stmt.order_by(self.model_class.created_at.desc())
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_project(
        self,
        project_id: UUID,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[DocumentModel]:
        """Get all documents in a project."""
        stmt = (
            select(self.model_class)
            .join(
                ProjectDocumentModel,
                ProjectDocumentModel.document_id == self.model_class.id,
            )
            .where(
                ProjectDocumentModel.project_id == project_id,
                self.model_class.tenant_id == self.tenant_id,
                self.model_class.deleted_at.is_(None),
            )
            .order_by(self.model_class.created_at.desc())
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_status(
        self,
        status: str,
        *,
        limit: int | None = None,
    ) -> Sequence[DocumentModel]:
        """Get documents by status (scoped to tenant)."""
        stmt = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.status == status,
            self.model_class.deleted_at.is_(None),
        )
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_hash(self, file_hash: str) -> DocumentModel | None:
        """Get a document by file hash (for deduplication)."""
        stmt = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.file_hash == file_hash,
            self.model_class.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def count_by_status(self) -> dict[str, int]:
        """Get count of documents by status."""
        stmt = (
            select(
                self.model_class.status,
                func.count(self.model_class.id),
            )
            .where(
                self.model_class.tenant_id == self.tenant_id,
                self.model_class.deleted_at.is_(None),
            )
            .group_by(self.model_class.status)
        )
        result = await self.session.execute(stmt)
        return dict(result.all())

    async def add_to_project(self, document_id: UUID, project_id: UUID) -> None:
        """Add a document to a project."""
        # Check if already in project
        stmt = select(ProjectDocumentModel).where(
            ProjectDocumentModel.document_id == document_id,
            ProjectDocumentModel.project_id == project_id,
        )
        result = await self.session.execute(stmt)
        if result.scalar_one_or_none() is not None:
            return  # Already in project

        link = ProjectDocumentModel(document_id=document_id, project_id=project_id)
        self.session.add(link)
        await self.session.flush()

    async def remove_from_project(self, document_id: UUID, project_id: UUID) -> bool:
        """Remove a document from a project. Returns True if removed."""
        stmt = select(ProjectDocumentModel).where(
            ProjectDocumentModel.document_id == document_id,
            ProjectDocumentModel.project_id == project_id,
        )
        result = await self.session.execute(stmt)
        link = result.scalar_one_or_none()
        if link is None:
            return False
        await self.session.delete(link)
        await self.session.flush()
        return True


class ChunkRepository(SoftDeleteRepository[ChunkModel]):
    """Repository for chunk operations."""

    model_class = ChunkModel

    def __init__(self, session: AsyncSession, tenant_id: UUID):
        """Initialize with session and tenant scope."""
        super().__init__(session)
        self.tenant_id = tenant_id

    async def get_by_document(
        self,
        document_id: UUID,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[ChunkModel]:
        """Get all chunks for a document."""
        stmt = (
            select(self.model_class)
            .where(
                self.model_class.document_id == document_id,
                self.model_class.tenant_id == self.tenant_id,
            )
            .order_by(self.model_class.sequence_number)
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_document_and_sequence(
        self,
        document_id: UUID,
        sequence_number: int,
    ) -> ChunkModel | None:
        """Get a specific chunk by document and sequence number."""
        stmt = select(self.model_class).where(
            self.model_class.document_id == document_id,
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.sequence_number == sequence_number,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def count_by_document(self, document_id: UUID) -> int:
        """Get the number of chunks for a document."""
        stmt = (
            select(func.count())
            .select_from(self.model_class)
            .where(
                self.model_class.document_id == document_id,
                self.model_class.tenant_id == self.tenant_id,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def delete_by_document(self, document_id: UUID) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        stmt = select(self.model_class).where(
            self.model_class.document_id == document_id,
            self.model_class.tenant_id == self.tenant_id,
        )
        result = await self.session.execute(stmt)
        chunks = result.scalars().all()
        count = len(chunks)
        for chunk in chunks:
            await self.session.delete(chunk)
        await self.session.flush()
        return count


class IngestionJobRepository(BaseRepository[IngestionJobModel]):
    """Repository for ingestion job operations."""

    model_class = IngestionJobModel

    def __init__(self, session: AsyncSession, tenant_id: UUID):
        """Initialize with session and tenant scope."""
        super().__init__(session)
        self.tenant_id = tenant_id

    async def get_by_document(
        self,
        document_id: UUID,
        *,
        limit: int | None = None,
    ) -> Sequence[IngestionJobModel]:
        """Get all jobs for a document."""
        stmt = (
            select(self.model_class)
            .where(
                self.model_class.document_id == document_id,
                self.model_class.tenant_id == self.tenant_id,
            )
            .order_by(self.model_class.created_at.desc())
        )
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_latest_by_document(self, document_id: UUID) -> IngestionJobModel | None:
        """Get the latest job for a document."""
        stmt = (
            select(self.model_class)
            .where(
                self.model_class.document_id == document_id,
                self.model_class.tenant_id == self.tenant_id,
            )
            .order_by(self.model_class.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_pending(
        self,
        *,
        limit: int | None = None,
    ) -> Sequence[IngestionJobModel]:
        """Get all pending jobs (scoped to tenant)."""
        stmt = (
            select(self.model_class)
            .where(
                self.model_class.tenant_id == self.tenant_id,
                self.model_class.status == "pending",
            )
            .order_by(self.model_class.created_at)
        )
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_running(
        self,
        *,
        limit: int | None = None,
    ) -> Sequence[IngestionJobModel]:
        """Get all running jobs (scoped to tenant)."""
        stmt = (
            select(self.model_class)
            .where(
                self.model_class.tenant_id == self.tenant_id,
                self.model_class.status == "running",
            )
            .order_by(self.model_class.started_at)
        )
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def start_job(self, job: IngestionJobModel) -> IngestionJobModel:
        """Mark a job as started."""
        job.status = "running"
        job.started_at = datetime.now(UTC)
        await self.session.flush()
        return job

    async def complete_job(
        self,
        job: IngestionJobModel,
        *,
        chunks_created: int = 0,
        error_message: str | None = None,
    ) -> IngestionJobModel:
        """Mark a job as completed."""
        job.status = "completed"
        job.completed_at = datetime.now(UTC)
        job.chunks_created = chunks_created
        job.error_message = error_message
        await self.session.flush()
        return job

    async def fail_job(
        self,
        job: IngestionJobModel,
        error_message: str,
    ) -> IngestionJobModel:
        """Mark a job as failed."""
        job.status = "failed"
        job.completed_at = datetime.now(UTC)
        job.error_message = error_message
        await self.session.flush()
        return job
