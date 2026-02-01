"""FastAPI dependencies for database and service clients."""

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, async_sessionmaker

from alexandria_api.config import Settings, get_settings
from alexandria_db import (
    get_async_engine,
    get_async_sessionmaker,
    MinIOClient,
    QdrantClient,
    Neo4jClient,
    MeiliSearchClient,
    TemporalClient,
)


# ============================================================
# Database Session
# ============================================================

# Cache for engine and sessionmaker - keyed by database URL
_engine_cache: dict[str, AsyncEngine] = {}
_sessionmaker_cache: dict[str, async_sessionmaker[AsyncSession]] = {}


def _get_cached_sessionmaker(database_url: str, echo: bool) -> async_sessionmaker[AsyncSession]:
    """Get or create a cached sessionmaker for the given database URL."""
    if database_url not in _sessionmaker_cache:
        if database_url not in _engine_cache:
            _engine_cache[database_url] = get_async_engine(database_url, echo=echo)
        _sessionmaker_cache[database_url] = get_async_sessionmaker(_engine_cache[database_url])
    return _sessionmaker_cache[database_url]


async def get_db_session(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session.

    Automatically handles commit/rollback and cleanup.

    Usage:
        @router.get("/items")
        async def get_items(db: DbSession):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    sessionmaker = _get_cached_sessionmaker(settings.database_url, settings.debug)
    session = sessionmaker()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# Type alias for dependency injection
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


# ============================================================
# Service Clients
# ============================================================

# Cache for service clients
_minio_client: MinIOClient | None = None
_qdrant_client: QdrantClient | None = None
_neo4j_client: Neo4jClient | None = None
_meilisearch_client: MeiliSearchClient | None = None
_temporal_client: TemporalClient | None = None


def get_minio_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> MinIOClient:
    """Get a cached MinIO client."""
    global _minio_client
    if _minio_client is None:
        _minio_client = MinIOClient(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
    return _minio_client


def get_qdrant_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> QdrantClient:
    """Get a cached Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
        )
    return _qdrant_client


def get_neo4j_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Neo4jClient:
    """Get a cached Neo4j client."""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
    return _neo4j_client


def get_meilisearch_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> MeiliSearchClient:
    """Get a cached MeiliSearch client."""
    global _meilisearch_client
    if _meilisearch_client is None:
        _meilisearch_client = MeiliSearchClient(
            url=settings.meilisearch_url,
            api_key=settings.meilisearch_api_key,
        )
    return _meilisearch_client


async def get_temporal_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> TemporalClient:
    """Get a cached Temporal client."""
    global _temporal_client
    if _temporal_client is None:
        _temporal_client = await TemporalClient.connect(
            target_host=settings.temporal_address,
            namespace=settings.temporal_namespace,
            default_task_queue="ingestion",
        )
    return _temporal_client


# Type aliases for dependency injection
MinIO = Annotated[MinIOClient, Depends(get_minio_client)]
Qdrant = Annotated[QdrantClient, Depends(get_qdrant_client)]
Neo4j = Annotated[Neo4jClient, Depends(get_neo4j_client)]
MeiliSearch = Annotated[MeiliSearchClient, Depends(get_meilisearch_client)]
Temporal = Annotated[TemporalClient, Depends(get_temporal_client)]


# ============================================================
# Request Context (re-export from middleware)
# ============================================================

# Import context types for convenient access
# These are the primary dependencies for tenant/project scoping
from alexandria_api.middleware.context import (  # noqa: E402
    TenantContext,
    RequestContext,
    get_tenant_context,
    get_request_context,
    Tenant,
    Context,
)


# ============================================================
# Utility Functions for Health Checks
# ============================================================


async def check_database_health(settings: Settings) -> bool:
    """Check if PostgreSQL is accessible."""
    try:
        from sqlalchemy import text

        engine = get_async_engine(settings.database_url)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def check_minio_health(settings: Settings) -> bool:
    """Check if MinIO is accessible."""
    try:
        client = MinIOClient(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        return await client.health_check()
    except Exception:
        return False


async def check_qdrant_health(settings: Settings) -> bool:
    """Check if Qdrant is accessible."""
    try:
        client = QdrantClient(url=settings.qdrant_url)
        return await client.health_check()
    except Exception:
        return False


async def check_neo4j_health(settings: Settings) -> bool:
    """Check if Neo4j is accessible."""
    try:
        client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        result = await client.health_check()
        await client.close()
        return result
    except Exception:
        return False


async def check_meilisearch_health(settings: Settings) -> bool:
    """Check if MeiliSearch is accessible."""
    try:
        client = MeiliSearchClient(
            url=settings.meilisearch_url,
            api_key=settings.meilisearch_api_key,
        )
        return await client.health_check()
    except Exception:
        return False


async def check_temporal_health(settings: Settings) -> bool:
    """Check if Temporal is accessible."""
    try:
        from temporalio.client import Client

        client = await Client.connect(settings.temporal_address)
        await client.close()
        return True
    except Exception:
        return False
