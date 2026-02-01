"""Database connection management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


@lru_cache
def get_async_engine(database_url: str, echo: bool = False) -> AsyncEngine:
    """
    Create and cache an async database engine.

    Args:
        database_url: PostgreSQL connection URL (must use asyncpg driver)
        echo: Whether to log SQL statements

    Returns:
        Cached AsyncEngine instance
    """
    return create_async_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


def get_async_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """
    Create an async session factory.

    Args:
        engine: AsyncEngine to bind sessions to

    Returns:
        async_sessionmaker configured for the engine
    """
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


@asynccontextmanager
async def get_async_session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.

    Automatically handles commit/rollback and cleanup.

    Args:
        sessionmaker: async_sessionmaker to create session from

    Yields:
        AsyncSession for database operations
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
DatabaseSession = Annotated[AsyncSession, "Database session"]
