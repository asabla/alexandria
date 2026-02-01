"""Entity, EntityMention, and Relationship repositories."""

from typing import Sequence
from uuid import UUID

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from alexandria_db.models import (
    EntityModel,
    EntityMentionModel,
    RelationshipModel,
    ProjectEntityModel,
)
from alexandria_db.repositories.base import SoftDeleteRepository


class EntityRepository(SoftDeleteRepository[EntityModel]):
    """Repository for entity operations."""

    model_class = EntityModel

    def __init__(self, session: AsyncSession, tenant_id: UUID):
        """Initialize with session and tenant scope."""
        super().__init__(session)
        self.tenant_id = tenant_id

    async def get_by_id(self, id: UUID) -> EntityModel | None:
        """Get an entity by ID (scoped to tenant)."""
        stmt = select(self.model_class).where(
            self.model_class.id == id,
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id_with_mentions(self, id: UUID) -> EntityModel | None:
        """Get an entity by ID with its mentions."""
        stmt = (
            select(self.model_class)
            .options(selectinload(self.model_class.mentions))
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
    ) -> Sequence[EntityModel]:
        """Get all entities (scoped to tenant)."""
        stmt = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,
        )
        if not include_deleted:
            stmt = stmt.where(self.model_class.deleted_at.is_(None))
        stmt = stmt.order_by(self.model_class.name)
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_type(
        self,
        entity_type: str,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[EntityModel]:
        """Get entities by type (scoped to tenant)."""
        stmt = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.entity_type == entity_type,
            self.model_class.deleted_at.is_(None),
        )
        stmt = stmt.order_by(self.model_class.name)
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def search_by_name(
        self,
        name: str,
        *,
        entity_type: str | None = None,
        limit: int | None = None,
    ) -> Sequence[EntityModel]:
        """Search entities by name (case-insensitive partial match)."""
        stmt = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.name.ilike(f"%{name}%"),
            self.model_class.deleted_at.is_(None),
        )
        if entity_type:
            stmt = stmt.where(self.model_class.entity_type == entity_type)
        stmt = stmt.order_by(self.model_class.name)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_canonical_name(
        self,
        canonical_name: str,
        entity_type: str,
    ) -> EntityModel | None:
        """Get entity by canonical name and type."""
        stmt = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.canonical_name == canonical_name,
            self.model_class.entity_type == entity_type,
            self.model_class.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_project(
        self,
        project_id: UUID,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[EntityModel]:
        """Get all entities in a project."""
        stmt = (
            select(self.model_class)
            .join(
                ProjectEntityModel,
                ProjectEntityModel.entity_id == self.model_class.id,
            )
            .where(
                ProjectEntityModel.project_id == project_id,
                self.model_class.tenant_id == self.tenant_id,
                self.model_class.deleted_at.is_(None),
            )
            .order_by(self.model_class.name)
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count_by_type(self) -> dict[str, int]:
        """Get count of entities by type."""
        stmt = (
            select(
                self.model_class.entity_type,
                func.count(self.model_class.id),
            )
            .where(
                self.model_class.tenant_id == self.tenant_id,
                self.model_class.deleted_at.is_(None),
            )
            .group_by(self.model_class.entity_type)
        )
        result = await self.session.execute(stmt)
        return dict(result.all())

    async def add_to_project(self, entity_id: UUID, project_id: UUID) -> None:
        """Add an entity to a project."""
        # Check if already in project
        stmt = select(ProjectEntityModel).where(
            ProjectEntityModel.entity_id == entity_id,
            ProjectEntityModel.project_id == project_id,
        )
        result = await self.session.execute(stmt)
        if result.scalar_one_or_none() is not None:
            return  # Already in project

        link = ProjectEntityModel(entity_id=entity_id, project_id=project_id)
        self.session.add(link)
        await self.session.flush()

    async def remove_from_project(self, entity_id: UUID, project_id: UUID) -> bool:
        """Remove an entity from a project. Returns True if removed."""
        stmt = select(ProjectEntityModel).where(
            ProjectEntityModel.entity_id == entity_id,
            ProjectEntityModel.project_id == project_id,
        )
        result = await self.session.execute(stmt)
        link = result.scalar_one_or_none()
        if link is None:
            return False
        await self.session.delete(link)
        await self.session.flush()
        return True


class EntityMentionRepository(SoftDeleteRepository[EntityMentionModel]):
    """Repository for entity mention operations."""

    model_class = EntityMentionModel

    def __init__(self, session: AsyncSession, tenant_id: UUID):
        """Initialize with session and tenant scope."""
        super().__init__(session)
        self.tenant_id = tenant_id

    async def get_by_entity(
        self,
        entity_id: UUID,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[EntityMentionModel]:
        """Get all mentions for an entity."""
        stmt = (
            select(self.model_class)
            .where(
                self.model_class.entity_id == entity_id,
                self.model_class.tenant_id == self.tenant_id,
            )
            .order_by(self.model_class.created_at.desc())
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_document(
        self,
        document_id: UUID,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[EntityMentionModel]:
        """Get all mentions in a document."""
        stmt = (
            select(self.model_class)
            .where(
                self.model_class.document_id == document_id,
                self.model_class.tenant_id == self.tenant_id,
            )
            .order_by(self.model_class.start_offset)
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_chunk(
        self,
        chunk_id: UUID,
    ) -> Sequence[EntityMentionModel]:
        """Get all mentions in a chunk."""
        stmt = (
            select(self.model_class)
            .where(
                self.model_class.chunk_id == chunk_id,
                self.model_class.tenant_id == self.tenant_id,
            )
            .order_by(self.model_class.start_offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count_by_entity(self, entity_id: UUID) -> int:
        """Get the number of mentions for an entity."""
        stmt = (
            select(func.count())
            .select_from(self.model_class)
            .where(
                self.model_class.entity_id == entity_id,
                self.model_class.tenant_id == self.tenant_id,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0


class RelationshipRepository(SoftDeleteRepository[RelationshipModel]):
    """Repository for relationship operations."""

    model_class = RelationshipModel

    def __init__(self, session: AsyncSession, tenant_id: UUID):
        """Initialize with session and tenant scope."""
        super().__init__(session)
        self.tenant_id = tenant_id

    async def get_by_entity(
        self,
        entity_id: UUID,
        *,
        limit: int | None = None,
    ) -> Sequence[RelationshipModel]:
        """Get all relationships involving an entity (as source or target)."""
        stmt = (
            select(self.model_class)
            .where(
                self.model_class.tenant_id == self.tenant_id,
                or_(
                    self.model_class.source_entity_id == entity_id,
                    self.model_class.target_entity_id == entity_id,
                ),
            )
            .order_by(self.model_class.created_at.desc())
        )
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_source(
        self,
        source_entity_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int | None = None,
    ) -> Sequence[RelationshipModel]:
        """Get all relationships from a source entity."""
        stmt = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.source_entity_id == source_entity_id,
        )
        if relationship_type:
            stmt = stmt.where(self.model_class.relationship_type == relationship_type)
        stmt = stmt.order_by(self.model_class.created_at.desc())
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_target(
        self,
        target_entity_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int | None = None,
    ) -> Sequence[RelationshipModel]:
        """Get all relationships to a target entity."""
        stmt = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.target_entity_id == target_entity_id,
        )
        if relationship_type:
            stmt = stmt.where(self.model_class.relationship_type == relationship_type)
        stmt = stmt.order_by(self.model_class.created_at.desc())
        if limit:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_between(
        self,
        entity_id_1: UUID,
        entity_id_2: UUID,
    ) -> Sequence[RelationshipModel]:
        """Get all relationships between two entities (in either direction)."""
        stmt = select(self.model_class).where(
            self.model_class.tenant_id == self.tenant_id,
            or_(
                and_(
                    self.model_class.source_entity_id == entity_id_1,
                    self.model_class.target_entity_id == entity_id_2,
                ),
                and_(
                    self.model_class.source_entity_id == entity_id_2,
                    self.model_class.target_entity_id == entity_id_1,
                ),
            ),
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def exists(
        self,
        source_entity_id: UUID,
        target_entity_id: UUID,
        relationship_type: str,
    ) -> bool:
        """Check if a specific relationship already exists."""
        stmt = select(self.model_class.id).where(
            self.model_class.tenant_id == self.tenant_id,
            self.model_class.source_entity_id == source_entity_id,
            self.model_class.target_entity_id == target_entity_id,
            self.model_class.relationship_type == relationship_type,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def count_by_type(self) -> dict[str, int]:
        """Get count of relationships by type."""
        stmt = (
            select(
                self.model_class.relationship_type,
                func.count(self.model_class.id),
            )
            .where(
                self.model_class.tenant_id == self.tenant_id,
            )
            .group_by(self.model_class.relationship_type)
        )
        result = await self.session.execute(stmt)
        return dict(result.all())
