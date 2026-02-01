"""Tenant and Project SQLAlchemy models."""

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, BigInteger, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from alexandria_db.models.base import AuditMixin, Base, SoftDeleteMixin, UUIDMixin

if TYPE_CHECKING:
    from alexandria_db.models.document import DocumentModel
    from alexandria_db.models.entity import EntityModel


class TenantModel(Base, UUIDMixin, AuditMixin, SoftDeleteMixin):
    """Tenant represents an organization or user account."""

    __tablename__ = "tenants"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(63), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Settings
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    max_storage_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    max_documents: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # Contact
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Metadata
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    # Relationships
    projects: Mapped[list["ProjectModel"]] = relationship(
        "ProjectModel", back_populates="tenant", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_tenants_is_active", "is_active"),)


class ProjectModel(Base, UUIDMixin, AuditMixin, SoftDeleteMixin):
    """Project represents a collection of documents and entities."""

    __tablename__ = "projects"

    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(63), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Settings
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Metadata
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    # Relationships
    tenant: Mapped["TenantModel"] = relationship("TenantModel", back_populates="projects")
    documents: Mapped[list["DocumentModel"]] = relationship(
        "DocumentModel",
        secondary="project_documents",
        back_populates="projects",
    )
    entities: Mapped[list["EntityModel"]] = relationship(
        "EntityModel",
        secondary="project_entities",
        back_populates="projects",
    )

    __table_args__ = (
        Index("ix_projects_tenant_slug", "tenant_id", "slug", unique=True),
        Index("ix_projects_is_active", "is_active"),
    )


class ProjectDocumentModel(Base, UUIDMixin):
    """Join table for Project-Document many-to-many relationship."""

    __tablename__ = "project_documents"

    project_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_project_documents_project", "project_id"),
        Index("ix_project_documents_document", "document_id"),
        Index("ix_project_documents_unique", "project_id", "document_id", unique=True),
    )


class ProjectEntityModel(Base, UUIDMixin):
    """Join table for Project-Entity many-to-many relationship."""

    __tablename__ = "project_entities"

    project_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    entity_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_project_entities_project", "project_id"),
        Index("ix_project_entities_entity", "entity_id"),
        Index("ix_project_entities_unique", "project_id", "entity_id", unique=True),
    )
