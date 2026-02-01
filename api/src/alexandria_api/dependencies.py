"""FastAPI dependencies for database and service clients."""

from functools import lru_cache
from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from alexandria_api.config import Settings, get_settings
from alexandria_db import (
    get_async_engine,
    get_async_sessionmaker,
    MinIOClient,
    QdrantClient,
    Neo4jClient,
    MeiliSearchClient,
)


# ============================================================
# Database Session
# ============================================================


@lru_cache
def get_sessionmaker(
    settings: Settings = Depends(get_settings),
) -> async_sessionmaker[AsyncSession]:
    """Get a cached async sessionmaker."""
    engine = get_async_engine(settings.database_url, echo=settings.debug)
    return get_async_sessionmaker(engine)


async def get_db_session(
    sessionmaker: async_sessionmaker[AsyncSession] = Depends(get_sessionmaker),
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


@lru_cache
def get_minio_client(settings: Settings = Depends(get_settings)) -> MinIOClient:
    """Get a cached MinIO client."""
    return MinIOClient(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


@lru_cache
def get_qdrant_client(settings: Settings = Depends(get_settings)) -> QdrantClient:
    """Get a cached Qdrant client."""
    return QdrantClient(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
    )


@lru_cache
def get_neo4j_client(settings: Settings = Depends(get_settings)) -> Neo4jClient:
    """Get a cached Neo4j client."""
    return Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )


@lru_cache
def get_meilisearch_client(settings: Settings = Depends(get_settings)) -> MeiliSearchClient:
    """Get a cached MeiliSearch client."""
    return MeiliSearchClient(
        url=settings.meilisearch_url,
        api_key=settings.meilisearch_api_key,
    )


# Type aliases for dependency injection
MinIO = Annotated[MinIOClient, Depends(get_minio_client)]
Qdrant = Annotated[QdrantClient, Depends(get_qdrant_client)]
Neo4j = Annotated[Neo4jClient, Depends(get_neo4j_client)]
MeiliSearch = Annotated[MeiliSearchClient, Depends(get_meilisearch_client)]


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
