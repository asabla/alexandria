"""Repository base classes and utilities."""

from typing import Generic, TypeVar, Sequence
from uuid import UUID

from sqlalchemy import select, func, Select
from sqlalchemy.ext.asyncio import AsyncSession

from alexandria_db.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """
    Generic repository base class providing common CRUD operations.

    Usage:
        class UserRepository(BaseRepository[UserModel]):
            model_class = UserModel

        repo = UserRepository(db_session)
        user = await repo.get_by_id(user_id)
    """

    model_class: type[ModelT]

    def __init__(self, session: AsyncSession):
        """Initialize repository with a database session."""
        self.session = session

    async def get_by_id(self, id: UUID) -> ModelT | None:
        """Get a single record by ID."""
        return await self.session.get(self.model_class, id)

    async def get_all(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[ModelT]:
        """Get all records with optional pagination."""
        stmt: Select[tuple[ModelT]] = select(self.model_class)
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count(self) -> int:
        """Get total count of records."""
        stmt = select(func.count()).select_from(self.model_class)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def create(self, entity: ModelT) -> ModelT:
        """Create a new record."""
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def create_many(self, entities: Sequence[ModelT]) -> Sequence[ModelT]:
        """Create multiple records."""
        self.session.add_all(entities)
        await self.session.flush()
        return entities

    async def update(self, entity: ModelT) -> ModelT:
        """Update an existing record."""
        await self.session.flush()
        return entity

    async def delete(self, entity: ModelT) -> None:
        """Delete a record."""
        await self.session.delete(entity)
        await self.session.flush()

    async def delete_by_id(self, id: UUID) -> bool:
        """Delete a record by ID. Returns True if deleted, False if not found."""
        entity = await self.get_by_id(id)
        if entity is None:
            return False
        await self.delete(entity)
        return True


class TenantScopedRepository(BaseRepository[ModelT]):
    """
    Repository base class for tenant-scoped entities.

    All queries are automatically filtered by tenant_id.
    """

    def __init__(self, session: AsyncSession, tenant_id: UUID):
        """Initialize repository with a database session and tenant ID."""
        super().__init__(session)
        self.tenant_id = tenant_id

    async def get_by_id(self, id: UUID) -> ModelT | None:
        """Get a single record by ID (scoped to tenant)."""
        stmt = select(self.model_class).where(
            self.model_class.id == id,  # type: ignore
            self.model_class.tenant_id == self.tenant_id,  # type: ignore
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[ModelT]:
        """Get all records (scoped to tenant) with optional pagination."""
        stmt: Select[tuple[ModelT]] = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,  # type: ignore
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count(self) -> int:
        """Get total count of records (scoped to tenant)."""
        stmt = (
            select(func.count())
            .select_from(self.model_class)
            .where(
                self.model_class.tenant_id == self.tenant_id,  # type: ignore
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0


class SoftDeleteRepository(BaseRepository[ModelT]):
    """
    Repository mixin that handles soft deletion.

    Automatically filters out soft-deleted records.
    """

    async def get_by_id(self, id: UUID) -> ModelT | None:
        """Get a single record by ID (excluding soft-deleted)."""
        stmt = select(self.model_class).where(
            self.model_class.id == id,  # type: ignore
            self.model_class.deleted_at.is_(None),  # type: ignore
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        include_deleted: bool = False,
    ) -> Sequence[ModelT]:
        """Get all records with optional pagination (excluding soft-deleted by default)."""
        stmt: Select[tuple[ModelT]] = select(self.model_class)
        if not include_deleted:
            stmt = stmt.where(self.model_class.deleted_at.is_(None))  # type: ignore
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def soft_delete(self, entity: ModelT) -> ModelT:
        """Soft delete a record by setting deleted_at timestamp."""
        from datetime import datetime, UTC

        entity.deleted_at = datetime.now(UTC)  # type: ignore
        await self.session.flush()
        return entity

    async def soft_delete_by_id(self, id: UUID) -> bool:
        """Soft delete a record by ID. Returns True if deleted, False if not found."""
        entity = await self.get_by_id(id)
        if entity is None:
            return False
        await self.soft_delete(entity)
        return True

    async def restore(self, entity: ModelT) -> ModelT:
        """Restore a soft-deleted record."""
        entity.deleted_at = None  # type: ignore
        await self.session.flush()
        return entity
