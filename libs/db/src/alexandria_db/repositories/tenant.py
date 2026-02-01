"""Tenant and Project repositories."""

from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from alexandria_db.models import TenantModel, ProjectModel
from alexandria_db.repositories.base import SoftDeleteRepository


class TenantRepository(SoftDeleteRepository[TenantModel]):
    """Repository for tenant operations."""

    model_class = TenantModel

    async def get_by_slug(self, slug: str) -> TenantModel | None:
        """Get a tenant by slug."""
        stmt = select(self.model_class).where(
            self.model_class.slug == slug,
            self.model_class.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[TenantModel]:
        """Get all active tenants."""
        stmt = select(self.model_class).where(
            self.model_class.is_active.is_(True),
            self.model_class.deleted_at.is_(None),
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def slug_exists(self, slug: str, exclude_id: UUID | None = None) -> bool:
        """Check if a slug already exists."""
        stmt = select(self.model_class.id).where(
            self.model_class.slug == slug,
            self.model_class.deleted_at.is_(None),
        )
        if exclude_id:
            stmt = stmt.where(self.model_class.id != exclude_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None


class ProjectRepository(SoftDeleteRepository[ProjectModel]):
    """Repository for project operations."""

    model_class = ProjectModel

    def __init__(self, session: AsyncSession, tenant_id: UUID):
        """Initialize with session and tenant scope."""
        super().__init__(session)
        self.tenant_id = tenant_id

    async def get_by_id(self, id: UUID) -> ProjectModel | None:
        """Get a project by ID (scoped to tenant)."""
        stmt = select(self.model_class).where(
            self.model_class.id == id,
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> ProjectModel | None:
        """Get a project by slug (scoped to tenant)."""
        stmt = select(self.model_class).where(
            self.model_class.slug == slug,
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        include_deleted: bool = False,
    ) -> Sequence[ProjectModel]:
        """Get all projects (scoped to tenant)."""
        stmt = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,
        )
        if not include_deleted:
            stmt = stmt.where(self.model_class.deleted_at.is_(None))
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_default(self) -> ProjectModel | None:
        """Get the default project for the tenant."""
        stmt = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.is_default.is_(True),
            self.model_class.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def slug_exists(self, slug: str, exclude_id: UUID | None = None) -> bool:
        """Check if a slug already exists for this tenant."""
        stmt = select(self.model_class.id).where(
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.slug == slug,
            self.model_class.deleted_at.is_(None),
        )
        if exclude_id:
            stmt = stmt.where(self.model_class.id != exclude_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def set_default(self, project: ProjectModel) -> ProjectModel:
        """Set a project as the default (unsets any existing default)."""
        # Unset current default
        current_default = await self.get_default()
        if current_default and current_default.id != project.id:
            current_default.is_default = False

        # Set new default
        project.is_default = True
        await self.session.flush()
        return project
