"""Alexandria DB - Database clients and repositories."""

from alexandria_db.connection import (
    DatabaseSession,
    get_async_engine,
    get_async_session,
    get_async_sessionmaker,
)
from alexandria_db.models import Base
from alexandria_db.clients import (
    MinIOClient,
    QdrantClient,
    Neo4jClient,
    MeiliSearchClient,
)

__version__ = "0.1.0"

__all__ = [
    # Base and connection
    "Base",
    "DatabaseSession",
    "get_async_engine",
    "get_async_session",
    "get_async_sessionmaker",
    # Clients
    "MinIOClient",
    "QdrantClient",
    "Neo4jClient",
    "MeiliSearchClient",
]
