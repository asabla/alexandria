"""Database repositories for Alexandria."""

from alexandria_db.repositories.base import (
    BaseRepository,
    TenantScopedRepository,
    SoftDeleteRepository,
)
from alexandria_db.repositories.tenant import (
    TenantRepository,
    ProjectRepository,
)
from alexandria_db.repositories.document import (
    DocumentRepository,
    ChunkRepository,
    IngestionJobRepository,
)
from alexandria_db.repositories.entity import (
    EntityRepository,
    EntityMentionRepository,
    RelationshipRepository,
)

__all__ = [
    # Base
    "BaseRepository",
    "TenantScopedRepository",
    "SoftDeleteRepository",
    # Tenant
    "TenantRepository",
    "ProjectRepository",
    # Document
    "DocumentRepository",
    "ChunkRepository",
    "IngestionJobRepository",
    # Entity
    "EntityRepository",
    "EntityMentionRepository",
    "RelationshipRepository",
]
