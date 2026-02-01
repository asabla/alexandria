"""Tenant and project management routes."""

from datetime import datetime, UTC
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from alexandria_api.config import Settings, get_settings
from alexandria_api.dependencies import get_db_session, Context, Tenant
from alexandria_api.middleware.context import RequestContext, TenantContext
from alexandria_db.models import TenantModel, ProjectModel

router = APIRouter(prefix="/tenants", tags=["Tenants"])


# ============================================================
# Response Models
# ============================================================


class TenantResponse(BaseModel):
    """Tenant response model."""

    id: UUID
    name: str
    slug: str
    description: str | None
    is_active: bool
    contact_email: str | None
    created_at: datetime
    updated_at: datetime


class ProjectResponse(BaseModel):
    """Project response model."""

    id: UUID
    tenant_id: UUID
    name: str
    slug: str
    description: str | None
    is_active: bool
    is_default: bool
    created_at: datetime
    updated_at: datetime


class ContextResponse(BaseModel):
    """Current request context response."""

    tenant_id: UUID
    tenant_slug: str
    tenant_name: str
    project_id: UUID
    project_slug: str
    project_name: str
    is_single_tenant: bool
    request_id: str | None


# ============================================================
# Request Models
# ============================================================


class CreateTenantRequest(BaseModel):
    """Request to create a new tenant."""

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=63, pattern=r"^[a-z0-9-]+$")
    description: str | None = None
    contact_email: str | None = None


class CreateProjectRequest(BaseModel):
    """Request to create a new project."""

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=63, pattern=r"^[a-z0-9-]+$")
    description: str | None = None
    is_default: bool = False


# ============================================================
# Endpoints
# ============================================================


@router.get("/current", response_model=ContextResponse)
async def get_current_context(ctx: Context) -> ContextResponse:
    """
    Get the current request context.

    Returns tenant and project information for the current request.
    Useful for debugging and verifying authentication.
    """
    return ContextResponse(
        tenant_id=ctx.tenant_id,
        tenant_slug=ctx.tenant_slug,
        tenant_name=ctx.tenant_name,
        project_id=ctx.project_id,
        project_slug=ctx.project_slug,
        project_name=ctx.project_name,
        is_single_tenant=ctx.is_single_tenant,
        request_id=ctx.request_id,
    )


@router.get("", response_model=list[TenantResponse])
async def list_tenants(
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[TenantResponse]:
    """
    List all tenants.

    Note: In production, this should be restricted to admin users.
    """
    stmt = select(TenantModel).where(TenantModel.deleted_at.is_(None))
    result = await db.execute(stmt)
    tenants = result.scalars().all()

    return [
        TenantResponse(
            id=t.id,
            name=t.name,
            slug=t.slug,
            description=t.description,
            is_active=t.is_active,
            contact_email=t.contact_email,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in tenants
    ]


@router.post("", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    request: CreateTenantRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TenantResponse:
    """
    Create a new tenant.

    Note: In production, this should be restricted to admin users.
    """
    # Check if slug already exists
    stmt = select(TenantModel).where(TenantModel.slug == request.slug)
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tenant with slug '{request.slug}' already exists",
        )

    tenant = TenantModel(
        name=request.name,
        slug=request.slug,
        description=request.description,
        contact_email=request.contact_email,
        is_active=True,
    )
    db.add(tenant)
    await db.flush()

    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        description=tenant.description,
        is_active=tenant.is_active,
        contact_email=tenant.contact_email,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(
    tenant_id: str,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> TenantResponse:
    """Get a specific tenant by ID or slug."""
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
    tenant = result.scalar_one_or_none()

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}",
        )

    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        description=tenant.description,
        is_active=tenant.is_active,
        contact_email=tenant.contact_email,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


@router.get("/{tenant_id}/projects", response_model=list[ProjectResponse])
async def list_projects(
    tenant_id: str,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> list[ProjectResponse]:
    """List all projects for a tenant."""
    # Get tenant first
    try:
        uuid_value = UUID(tenant_id)
        tenant_stmt = select(TenantModel).where(
            TenantModel.id == uuid_value,
            TenantModel.deleted_at.is_(None),
        )
    except ValueError:
        tenant_stmt = select(TenantModel).where(
            TenantModel.slug == tenant_id,
            TenantModel.deleted_at.is_(None),
        )

    result = await db.execute(tenant_stmt)
    tenant = result.scalar_one_or_none()

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}",
        )

    # Get projects
    stmt = select(ProjectModel).where(
        ProjectModel.tenant_id == tenant.id,
        ProjectModel.deleted_at.is_(None),
    )
    result = await db.execute(stmt)
    projects = result.scalars().all()

    return [
        ProjectResponse(
            id=p.id,
            tenant_id=p.tenant_id,
            name=p.name,
            slug=p.slug,
            description=p.description,
            is_active=p.is_active,
            is_default=p.is_default,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
        for p in projects
    ]


@router.post(
    "/{tenant_id}/projects",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_project(
    tenant_id: str,
    request: CreateProjectRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> ProjectResponse:
    """Create a new project for a tenant."""
    # Get tenant first
    try:
        uuid_value = UUID(tenant_id)
        tenant_stmt = select(TenantModel).where(
            TenantModel.id == uuid_value,
            TenantModel.deleted_at.is_(None),
        )
    except ValueError:
        tenant_stmt = select(TenantModel).where(
            TenantModel.slug == tenant_id,
            TenantModel.deleted_at.is_(None),
        )

    result = await db.execute(tenant_stmt)
    tenant = result.scalar_one_or_none()

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}",
        )

    # Check if slug already exists for this tenant
    slug_stmt = select(ProjectModel).where(
        ProjectModel.tenant_id == tenant.id,
        ProjectModel.slug == request.slug,
    )
    result = await db.execute(slug_stmt)
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project with slug '{request.slug}' already exists for this tenant",
        )

    # If this is the default project, unset other defaults
    if request.is_default:
        default_stmt = select(ProjectModel).where(
            ProjectModel.tenant_id == tenant.id,
            ProjectModel.is_default.is_(True),
        )
        result = await db.execute(default_stmt)
        for existing_default in result.scalars().all():
            existing_default.is_default = False

    project = ProjectModel(
        tenant_id=tenant.id,
        name=request.name,
        slug=request.slug,
        description=request.description,
        is_active=True,
        is_default=request.is_default,
    )
    db.add(project)
    await db.flush()

    return ProjectResponse(
        id=project.id,
        tenant_id=project.tenant_id,
        name=project.name,
        slug=project.slug,
        description=project.description,
        is_active=project.is_active,
        is_default=project.is_default,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )
