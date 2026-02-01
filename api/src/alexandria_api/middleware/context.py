"""Tenant and request context for multi-tenancy support.

This module provides FastAPI dependencies for extracting tenant and project
information from incoming requests. It supports both multi-tenant and
single-tenant modes.

Multi-tenant mode (SINGLE_TENANT_MODE=false):
- Tenant is identified from the X-Tenant-ID header (required)
- Project is identified from the X-Project-ID header (optional, falls back to default)
- Validates that tenant exists and is active
- Validates that project belongs to tenant

Single-tenant mode (SINGLE_TENANT_MODE=true):
- Uses default tenant and project from config
- No headers required
- Creates default tenant/project if they don't exist
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from alexandria_api.config import Settings, get_settings
from alexandria_api.dependencies import get_db_session
from alexandria_db.models import TenantModel, ProjectModel


@dataclass(frozen=True)
class TenantContext:
    """
    Immutable context containing tenant information for the current request.

    This is the minimal context needed for basic tenant-scoped operations.
    """

    tenant_id: UUID
    tenant_slug: str
    is_active: bool


@dataclass(frozen=True)
class RequestContext:
    """
    Immutable context containing full request information.

    Includes tenant, project, and request metadata. This is the primary
    context object used throughout the application.
    """

    # Tenant info
    tenant_id: UUID
    tenant_slug: str
    tenant_name: str

    # Project info
    project_id: UUID
    project_slug: str
    project_name: str

    # Request metadata
    request_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Additional context
    is_single_tenant: bool = False


async def _get_or_create_default_tenant(
    db: AsyncSession,
    settings: Settings,
) -> TenantModel:
    """Get or create the default tenant for single-tenant mode."""
    stmt = select(TenantModel).where(TenantModel.slug == settings.default_tenant_id)
    result = await db.execute(stmt)
    tenant = result.scalar_one_or_none()

    if tenant is None:
        # Create default tenant
        tenant = TenantModel(
            name="Default Tenant",
            slug=settings.default_tenant_id,
            description="Default tenant for single-tenant mode",
            is_active=True,
        )
        db.add(tenant)
        await db.flush()  # Get the ID without committing

    return tenant


async def _get_or_create_default_project(
    db: AsyncSession,
    tenant_id: UUID,
    settings: Settings,
) -> ProjectModel:
    """Get or create the default project for single-tenant mode."""
    stmt = select(ProjectModel).where(
        ProjectModel.tenant_id == tenant_id,
        ProjectModel.slug == settings.default_project_id,
    )
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        # Create default project
        project = ProjectModel(
            tenant_id=tenant_id,
            name="Default Project",
            slug=settings.default_project_id,
            description="Default project for single-tenant mode",
            is_active=True,
            is_default=True,
        )
        db.add(project)
        await db.flush()  # Get the ID without committing

    return project


async def _get_tenant_by_id(
    db: AsyncSession,
    tenant_id: str,
) -> TenantModel | None:
    """Get tenant by ID or slug."""
    # Try to parse as UUID first
    try:
        uuid_value = UUID(tenant_id)
        stmt = select(TenantModel).where(
            TenantModel.id == uuid_value,
            TenantModel.deleted_at.is_(None),
        )
    except ValueError:
        # Not a UUID, try as slug
        stmt = select(TenantModel).where(
            TenantModel.slug == tenant_id,
            TenantModel.deleted_at.is_(None),
        )

    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def _get_project_by_id(
    db: AsyncSession,
    tenant_id: UUID,
    project_id: str | None,
) -> ProjectModel | None:
    """Get project by ID or slug, falling back to default project."""
    if project_id:
        # Try to parse as UUID first
        try:
            uuid_value = UUID(project_id)
            stmt = select(ProjectModel).where(
                ProjectModel.id == uuid_value,
                ProjectModel.tenant_id == tenant_id,
                ProjectModel.deleted_at.is_(None),
            )
        except ValueError:
            # Not a UUID, try as slug
            stmt = select(ProjectModel).where(
                ProjectModel.slug == project_id,
                ProjectModel.tenant_id == tenant_id,
                ProjectModel.deleted_at.is_(None),
            )
    else:
        # Get default project
        stmt = select(ProjectModel).where(
            ProjectModel.tenant_id == tenant_id,
            ProjectModel.is_default.is_(True),
            ProjectModel.deleted_at.is_(None),
        )

    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_tenant_context(
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    x_tenant_id: Annotated[str | None, Header()] = None,
) -> TenantContext:
    """
    FastAPI dependency that extracts tenant context from the request.

    In single-tenant mode, returns the default tenant.
    In multi-tenant mode, extracts tenant from X-Tenant-ID header.

    Usage:
        @router.get("/items")
        async def get_items(tenant: TenantContext = Depends(get_tenant_context)):
            # Use tenant.tenant_id for filtering
    """
    if settings.single_tenant_mode:
        # Single-tenant mode: use default tenant
        tenant = await _get_or_create_default_tenant(db, settings)
        return TenantContext(
            tenant_id=tenant.id,
            tenant_slug=tenant.slug,
            is_active=tenant.is_active,
        )

    # Multi-tenant mode: require X-Tenant-ID header
    if not x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Tenant-ID header is required",
        )

    tenant = await _get_tenant_by_id(db, x_tenant_id)

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {x_tenant_id}",
        )

    if not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant is inactive",
        )

    return TenantContext(
        tenant_id=tenant.id,
        tenant_slug=tenant.slug,
        is_active=tenant.is_active,
    )


async def get_request_context(
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    x_tenant_id: Annotated[str | None, Header()] = None,
    x_project_id: Annotated[str | None, Header()] = None,
    x_request_id: Annotated[str | None, Header()] = None,
) -> RequestContext:
    """
    FastAPI dependency that extracts full request context.

    Includes tenant, project, and request metadata. This is the primary
    dependency for most endpoints that need tenant/project scoping.

    In single-tenant mode:
    - Uses default tenant and project
    - Creates them if they don't exist

    In multi-tenant mode:
    - Requires X-Tenant-ID header
    - X-Project-ID is optional (falls back to default project)
    - Validates tenant is active
    - Validates project belongs to tenant

    Usage:
        @router.get("/documents")
        async def list_documents(ctx: RequestContext = Depends(get_request_context)):
            # Use ctx.tenant_id and ctx.project_id for filtering
    """
    if settings.single_tenant_mode:
        # Single-tenant mode: use defaults
        tenant = await _get_or_create_default_tenant(db, settings)
        project = await _get_or_create_default_project(db, tenant.id, settings)

        return RequestContext(
            tenant_id=tenant.id,
            tenant_slug=tenant.slug,
            tenant_name=tenant.name,
            project_id=project.id,
            project_slug=project.slug,
            project_name=project.name,
            request_id=x_request_id,
            is_single_tenant=True,
        )

    # Multi-tenant mode: require X-Tenant-ID header
    if not x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Tenant-ID header is required",
        )

    tenant = await _get_tenant_by_id(db, x_tenant_id)

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {x_tenant_id}",
        )

    if not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant is inactive",
        )

    # Get project (optional header, falls back to default)
    project = await _get_project_by_id(db, tenant.id, x_project_id)

    if project is None:
        if x_project_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project not found: {x_project_id}",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No default project found for tenant",
            )

    if not project.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Project is inactive",
        )

    return RequestContext(
        tenant_id=tenant.id,
        tenant_slug=tenant.slug,
        tenant_name=tenant.name,
        project_id=project.id,
        project_slug=project.slug,
        project_name=project.name,
        request_id=x_request_id,
        is_single_tenant=False,
    )


# Type aliases for cleaner dependency injection
Tenant = Annotated[TenantContext, Depends(get_tenant_context)]
Context = Annotated[RequestContext, Depends(get_request_context)]
