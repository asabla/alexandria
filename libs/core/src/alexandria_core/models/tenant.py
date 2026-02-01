"""Tenant and Project domain models."""

from uuid import UUID

from pydantic import Field

from alexandria_core.models.base import (
    AuditMixin,
    IdentifiableMixin,
    MetadataMixin,
    SoftDeleteMixin,
)


class Tenant(IdentifiableMixin, AuditMixin, SoftDeleteMixin, MetadataMixin):
    """
    Tenant represents an organization or user account.

    In multi-tenant mode, all resources are scoped to a tenant.
    In single-tenant mode, a default tenant is used.
    """

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=63, pattern=r"^[a-z0-9-]+$")
    description: str | None = None

    # Settings
    is_active: bool = True
    max_storage_bytes: int | None = None  # None = unlimited
    max_documents: int | None = None  # None = unlimited

    # Contact info (for multi-tenant)
    contact_email: str | None = None


class Project(IdentifiableMixin, AuditMixin, SoftDeleteMixin, MetadataMixin):
    """
    Project represents a collection of documents and entities for an investigation.

    Projects belong to a tenant and allow organizing work into separate contexts.
    Documents and entities can be shared across projects within the same tenant.
    """

    tenant_id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=63, pattern=r"^[a-z0-9-]+$")
    description: str | None = None

    # Settings
    is_active: bool = True
    is_default: bool = False  # Default project for the tenant


class ProjectDocument(IdentifiableMixin):
    """Join table for Project-Document many-to-many relationship."""

    project_id: UUID
    document_id: UUID


class ProjectEntity(IdentifiableMixin):
    """Join table for Project-Entity many-to-many relationship."""

    project_id: UUID
    entity_id: UUID
