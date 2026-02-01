"""Entity and relationship management routes."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from alexandria_api.dependencies import get_db_session, Context
from alexandria_db import EntityRepository, EntityMentionRepository, RelationshipRepository
from alexandria_db.models import EntityModel, EntityMentionModel, RelationshipModel

router = APIRouter(prefix="/entities", tags=["Entities"])


# ============================================================
# Response Models
# ============================================================


class EntityResponse(BaseModel):
    """Entity response model."""

    id: UUID
    tenant_id: UUID
    name: str
    canonical_name: str | None
    aliases: list[str]
    entity_type: str
    description: str | None
    summary: str | None
    external_ids: dict
    confidence_score: float
    is_verified: bool
    neo4j_node_id: str | None
    mention_count: int
    metadata: dict
    created_at: datetime
    updated_at: datetime


class EntityListResponse(BaseModel):
    """Paginated list of entities."""

    items: list[EntityResponse]
    total: int
    limit: int
    offset: int


class EntityMentionResponse(BaseModel):
    """Entity mention response model."""

    id: UUID
    entity_id: UUID
    document_id: UUID
    chunk_id: UUID | None
    start_char: int | None
    end_char: int | None
    page_number: int | None
    context_text: str | None
    mention_text: str
    extraction_method: str
    confidence_score: float
    created_at: datetime


class RelationshipResponse(BaseModel):
    """Relationship response model."""

    id: UUID
    tenant_id: UUID
    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str
    description: str | None
    label: str | None
    document_id: UUID | None
    evidence_text: str | None
    confidence_score: float
    is_verified: bool
    start_date: str | None
    end_date: str | None
    is_current: bool | None
    neo4j_relationship_id: str | None
    metadata: dict
    created_at: datetime
    updated_at: datetime


class RelationshipWithEntitiesResponse(RelationshipResponse):
    """Relationship with source and target entity info."""

    source_entity: EntityResponse
    target_entity: EntityResponse


class EntityStatsResponse(BaseModel):
    """Entity statistics."""

    total: int
    by_type: dict[str, int]
    verified_count: int


class RelationshipStatsResponse(BaseModel):
    """Relationship statistics."""

    total: int
    by_type: dict[str, int]


# ============================================================
# Request Models
# ============================================================


class CreateEntityRequest(BaseModel):
    """Request to create an entity."""

    name: str = Field(..., min_length=1, max_length=500)
    entity_type: str = Field(default="unknown", max_length=50)
    canonical_name: str | None = Field(default=None, max_length=500)
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    external_ids: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class UpdateEntityRequest(BaseModel):
    """Request to update an entity."""

    name: str | None = Field(default=None, min_length=1, max_length=500)
    canonical_name: str | None = None
    aliases: list[str] | None = None
    entity_type: str | None = Field(default=None, max_length=50)
    description: str | None = None
    summary: str | None = None
    external_ids: dict | None = None
    is_verified: bool | None = None
    metadata: dict | None = None


class MergeEntitiesRequest(BaseModel):
    """Request to merge multiple entities into one."""

    source_entity_ids: list[UUID] = Field(..., min_length=1)
    target_entity_id: UUID
    merge_aliases: bool = True


class CreateRelationshipRequest(BaseModel):
    """Request to create a relationship."""

    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str = Field(default="associated_with", max_length=50)
    description: str | None = None
    label: str | None = Field(default=None, max_length=255)
    document_id: UUID | None = None
    evidence_text: str | None = None
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    start_date: str | None = Field(default=None, max_length=20)
    end_date: str | None = Field(default=None, max_length=20)
    is_current: bool | None = None
    metadata: dict = Field(default_factory=dict)


class UpdateRelationshipRequest(BaseModel):
    """Request to update a relationship."""

    relationship_type: str | None = Field(default=None, max_length=50)
    description: str | None = None
    label: str | None = None
    evidence_text: str | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    is_verified: bool | None = None
    start_date: str | None = None
    end_date: str | None = None
    is_current: bool | None = None
    metadata: dict | None = None


# ============================================================
# Helper Functions
# ============================================================


def _entity_to_response(entity: EntityModel) -> EntityResponse:
    """Convert an EntityModel to EntityResponse."""
    return EntityResponse(
        id=entity.id,
        tenant_id=entity.tenant_id,
        name=entity.name,
        canonical_name=entity.canonical_name,
        aliases=entity.aliases or [],
        entity_type=entity.entity_type,
        description=entity.description,
        summary=entity.summary,
        external_ids=entity.external_ids or {},
        confidence_score=entity.confidence_score,
        is_verified=entity.is_verified,
        neo4j_node_id=entity.neo4j_node_id,
        mention_count=entity.mention_count,
        metadata=entity.metadata_ or {},
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


def _mention_to_response(mention: EntityMentionModel) -> EntityMentionResponse:
    """Convert an EntityMentionModel to EntityMentionResponse."""
    return EntityMentionResponse(
        id=mention.id,
        entity_id=mention.entity_id,
        document_id=mention.document_id,
        chunk_id=mention.chunk_id,
        start_char=mention.start_char,
        end_char=mention.end_char,
        page_number=mention.page_number,
        context_text=mention.context_text,
        mention_text=mention.mention_text,
        extraction_method=mention.extraction_method,
        confidence_score=mention.confidence_score,
        created_at=mention.created_at,
    )


def _relationship_to_response(rel: RelationshipModel) -> RelationshipResponse:
    """Convert a RelationshipModel to RelationshipResponse."""
    return RelationshipResponse(
        id=rel.id,
        tenant_id=rel.tenant_id,
        source_entity_id=rel.source_entity_id,
        target_entity_id=rel.target_entity_id,
        relationship_type=rel.relationship_type,
        description=rel.description,
        label=rel.label,
        document_id=rel.document_id,
        evidence_text=rel.evidence_text,
        confidence_score=rel.confidence_score,
        is_verified=rel.is_verified,
        start_date=rel.start_date,
        end_date=rel.end_date,
        is_current=rel.is_current,
        neo4j_relationship_id=rel.neo4j_relationship_id,
        metadata=rel.metadata_ or {},
        created_at=rel.created_at,
        updated_at=rel.updated_at,
    )


# ============================================================
# Entity Endpoints
# ============================================================


@router.get("", response_model=EntityListResponse)
async def list_entities(
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    entity_type: str | None = Query(default=None, alias="type"),
    project_id: UUID | None = Query(default=None),
    search: str | None = Query(default=None, min_length=1),
) -> EntityListResponse:
    """
    List all entities for the current tenant.

    Supports pagination and filtering by type, project, or search term.
    """
    repo = EntityRepository(db, ctx.tenant_id)

    if search:
        entities = await repo.search_by_name(search, entity_type=entity_type, limit=limit)
        total = len(entities)
    elif project_id:
        entities = await repo.get_by_project(project_id, limit=limit, offset=offset)
        total = len(entities)  # TODO: Add proper count method
    elif entity_type:
        entities = await repo.get_by_type(entity_type, limit=limit, offset=offset)
        counts = await repo.count_by_type()
        total = counts.get(entity_type, 0)
    else:
        entities = await repo.get_all(limit=limit, offset=offset)
        total = await repo.count()

    return EntityListResponse(
        items=[_entity_to_response(e) for e in entities],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/stats", response_model=EntityStatsResponse)
async def get_entity_stats(
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> EntityStatsResponse:
    """Get entity statistics for the current tenant."""
    repo = EntityRepository(db, ctx.tenant_id)

    total = await repo.count()
    by_type = await repo.count_by_type()

    # TODO: Add verified count method to repository
    verified_count = 0

    return EntityStatsResponse(
        total=total,
        by_type=by_type,
        verified_count=verified_count,
    )


@router.post("", response_model=EntityResponse, status_code=status.HTTP_201_CREATED)
async def create_entity(
    request: CreateEntityRequest,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> EntityResponse:
    """
    Create a new entity.

    Entities represent named items extracted from documents (people, organizations,
    locations, etc.) or manually created for research purposes.
    """
    repo = EntityRepository(db, ctx.tenant_id)

    # Check for existing entity with same canonical name and type
    canonical = request.canonical_name or request.name.lower().strip()
    existing = await repo.get_by_canonical_name(canonical, request.entity_type)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Entity with canonical name '{canonical}' and type '{request.entity_type}' already exists: {existing.id}",
        )

    entity = EntityModel(
        tenant_id=ctx.tenant_id,
        name=request.name,
        canonical_name=canonical,
        aliases=request.aliases,
        entity_type=request.entity_type,
        description=request.description,
        external_ids=request.external_ids,
        metadata_=request.metadata,
    )
    entity = await repo.create(entity)

    return _entity_to_response(entity)


@router.get("/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> EntityResponse:
    """Get a specific entity by ID."""
    repo = EntityRepository(db, ctx.tenant_id)

    entity = await repo.get_by_id(entity_id)
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity not found: {entity_id}",
        )

    return _entity_to_response(entity)


@router.patch("/{entity_id}", response_model=EntityResponse)
async def update_entity(
    entity_id: UUID,
    request: UpdateEntityRequest,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> EntityResponse:
    """Update entity attributes."""
    repo = EntityRepository(db, ctx.tenant_id)

    entity = await repo.get_by_id(entity_id)
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity not found: {entity_id}",
        )

    # Update fields if provided
    if request.name is not None:
        entity.name = request.name
    if request.canonical_name is not None:
        entity.canonical_name = request.canonical_name
    if request.aliases is not None:
        entity.aliases = request.aliases
    if request.entity_type is not None:
        entity.entity_type = request.entity_type
    if request.description is not None:
        entity.description = request.description
    if request.summary is not None:
        entity.summary = request.summary
    if request.external_ids is not None:
        entity.external_ids = request.external_ids
    if request.is_verified is not None:
        entity.is_verified = request.is_verified
    if request.metadata is not None:
        entity.metadata_ = request.metadata

    entity = await repo.update(entity)

    return _entity_to_response(entity)


@router.delete("/{entity_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_entity(
    entity_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    hard_delete: bool = Query(default=False),
) -> None:
    """
    Delete an entity.

    By default, performs a soft delete (sets deleted_at).
    Use hard_delete=true to permanently remove the entity and its relationships.
    """
    repo = EntityRepository(db, ctx.tenant_id)

    entity = await repo.get_by_id(entity_id)
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity not found: {entity_id}",
        )

    if hard_delete:
        await repo.delete(entity)
    else:
        await repo.soft_delete(entity)


@router.get("/{entity_id}/mentions", response_model=list[EntityMentionResponse])
async def get_entity_mentions(
    entity_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[EntityMentionResponse]:
    """Get all mentions of an entity across documents."""
    # Verify entity exists
    entity_repo = EntityRepository(db, ctx.tenant_id)
    entity = await entity_repo.get_by_id(entity_id)
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity not found: {entity_id}",
        )

    mention_repo = EntityMentionRepository(db, ctx.tenant_id)
    mentions = await mention_repo.get_by_entity(entity_id, limit=limit, offset=offset)

    return [_mention_to_response(m) for m in mentions]


@router.get("/{entity_id}/relationships", response_model=list[RelationshipResponse])
async def get_entity_relationships(
    entity_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    limit: int = Query(default=50, ge=1, le=500),
) -> list[RelationshipResponse]:
    """Get all relationships involving an entity (as source or target)."""
    # Verify entity exists
    entity_repo = EntityRepository(db, ctx.tenant_id)
    entity = await entity_repo.get_by_id(entity_id)
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity not found: {entity_id}",
        )

    rel_repo = RelationshipRepository(db, ctx.tenant_id)
    relationships = await rel_repo.get_by_entity(entity_id, limit=limit)

    return [_relationship_to_response(r) for r in relationships]


@router.post("/{entity_id}/merge", response_model=EntityResponse)
async def merge_entities(
    entity_id: UUID,
    request: MergeEntitiesRequest,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> EntityResponse:
    """
    Merge multiple source entities into a target entity.

    This operation:
    - Transfers all mentions from source entities to the target
    - Updates relationships to point to the target entity
    - Optionally merges aliases
    - Soft-deletes the source entities
    """
    if entity_id != request.target_entity_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path entity_id must match target_entity_id in request body",
        )

    entity_repo = EntityRepository(db, ctx.tenant_id)
    mention_repo = EntityMentionRepository(db, ctx.tenant_id)

    # Get target entity
    target = await entity_repo.get_by_id(request.target_entity_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target entity not found: {request.target_entity_id}",
        )

    # Process each source entity
    merged_aliases: set[str] = set(target.aliases or [])

    for source_id in request.source_entity_ids:
        if source_id == request.target_entity_id:
            continue  # Skip if source is same as target

        source = await entity_repo.get_by_id(source_id)
        if source is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source entity not found: {source_id}",
            )

        # Merge aliases if requested
        if request.merge_aliases:
            merged_aliases.add(source.name)
            if source.canonical_name:
                merged_aliases.add(source.canonical_name)
            merged_aliases.update(source.aliases or [])

        # Transfer mentions
        mentions = await mention_repo.get_by_entity(source_id)
        for mention in mentions:
            mention.entity_id = request.target_entity_id
            await mention_repo.update(mention)

        # Update target mention count
        target.mention_count += source.mention_count

        # TODO: Update relationships to point to target entity

        # Soft delete source
        await entity_repo.soft_delete(source)

    # Update target with merged aliases
    if request.merge_aliases:
        # Remove target's own name from aliases
        merged_aliases.discard(target.name)
        if target.canonical_name:
            merged_aliases.discard(target.canonical_name)
        target.aliases = list(merged_aliases)

    target = await entity_repo.update(target)

    return _entity_to_response(target)


# ============================================================
# Relationship Endpoints
# ============================================================


relationships_router = APIRouter(prefix="/relationships", tags=["Relationships"])


@relationships_router.get("", response_model=list[RelationshipResponse])
async def list_relationships(
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    limit: int = Query(default=50, ge=1, le=500),
    source_id: UUID | None = Query(default=None),
    target_id: UUID | None = Query(default=None),
    relationship_type: str | None = Query(default=None, alias="type"),
) -> list[RelationshipResponse]:
    """
    List relationships with optional filtering.

    Can filter by source entity, target entity, or relationship type.
    """
    repo = RelationshipRepository(db, ctx.tenant_id)

    if source_id and target_id:
        relationships = await repo.get_between(source_id, target_id)
    elif source_id:
        relationships = await repo.get_by_source(
            source_id, relationship_type=relationship_type, limit=limit
        )
    elif target_id:
        relationships = await repo.get_by_target(
            target_id, relationship_type=relationship_type, limit=limit
        )
    else:
        # TODO: Add get_all method to RelationshipRepository
        relationships = []

    return [_relationship_to_response(r) for r in relationships]


@relationships_router.get("/stats", response_model=RelationshipStatsResponse)
async def get_relationship_stats(
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> RelationshipStatsResponse:
    """Get relationship statistics for the current tenant."""
    repo = RelationshipRepository(db, ctx.tenant_id)

    by_type = await repo.count_by_type()
    total = sum(by_type.values())

    return RelationshipStatsResponse(
        total=total,
        by_type=by_type,
    )


@relationships_router.post(
    "", response_model=RelationshipResponse, status_code=status.HTTP_201_CREATED
)
async def create_relationship(
    request: CreateRelationshipRequest,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> RelationshipResponse:
    """
    Create a relationship between two entities.

    Relationships represent connections like "works_for", "located_in",
    "associated_with", etc.
    """
    entity_repo = EntityRepository(db, ctx.tenant_id)
    rel_repo = RelationshipRepository(db, ctx.tenant_id)

    # Verify source entity exists
    source = await entity_repo.get_by_id(request.source_entity_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source entity not found: {request.source_entity_id}",
        )

    # Verify target entity exists
    target = await entity_repo.get_by_id(request.target_entity_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target entity not found: {request.target_entity_id}",
        )

    # Check for existing relationship
    exists = await rel_repo.exists(
        request.source_entity_id,
        request.target_entity_id,
        request.relationship_type,
    )
    if exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Relationship already exists between entities with type '{request.relationship_type}'",
        )

    relationship = RelationshipModel(
        tenant_id=ctx.tenant_id,
        source_entity_id=request.source_entity_id,
        target_entity_id=request.target_entity_id,
        relationship_type=request.relationship_type,
        description=request.description,
        label=request.label,
        document_id=request.document_id,
        evidence_text=request.evidence_text,
        confidence_score=request.confidence_score,
        start_date=request.start_date,
        end_date=request.end_date,
        is_current=request.is_current,
        metadata_=request.metadata,
    )
    relationship = await rel_repo.create(relationship)

    return _relationship_to_response(relationship)


@relationships_router.get("/{relationship_id}", response_model=RelationshipResponse)
async def get_relationship(
    relationship_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> RelationshipResponse:
    """Get a specific relationship by ID."""
    repo = RelationshipRepository(db, ctx.tenant_id)

    relationship = await repo.get_by_id(relationship_id)
    if relationship is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Relationship not found: {relationship_id}",
        )

    return _relationship_to_response(relationship)


@relationships_router.patch("/{relationship_id}", response_model=RelationshipResponse)
async def update_relationship(
    relationship_id: UUID,
    request: UpdateRelationshipRequest,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> RelationshipResponse:
    """Update relationship attributes."""
    repo = RelationshipRepository(db, ctx.tenant_id)

    relationship = await repo.get_by_id(relationship_id)
    if relationship is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Relationship not found: {relationship_id}",
        )

    # Update fields if provided
    if request.relationship_type is not None:
        relationship.relationship_type = request.relationship_type
    if request.description is not None:
        relationship.description = request.description
    if request.label is not None:
        relationship.label = request.label
    if request.evidence_text is not None:
        relationship.evidence_text = request.evidence_text
    if request.confidence_score is not None:
        relationship.confidence_score = request.confidence_score
    if request.is_verified is not None:
        relationship.is_verified = request.is_verified
    if request.start_date is not None:
        relationship.start_date = request.start_date
    if request.end_date is not None:
        relationship.end_date = request.end_date
    if request.is_current is not None:
        relationship.is_current = request.is_current
    if request.metadata is not None:
        relationship.metadata_ = request.metadata

    relationship = await repo.update(relationship)

    return _relationship_to_response(relationship)


@relationships_router.delete("/{relationship_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_relationship(
    relationship_id: UUID,
    ctx: Context,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> None:
    """Delete a relationship."""
    repo = RelationshipRepository(db, ctx.tenant_id)

    relationship = await repo.get_by_id(relationship_id)
    if relationship is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Relationship not found: {relationship_id}",
        )

    await repo.delete(relationship)


# Include relationships router in main router
router.include_router(relationships_router)
