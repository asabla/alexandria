"""Base model classes and mixins."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class BaseModel(PydanticBaseModel):
    """Base model with common configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
        validate_assignment=True,
    )


class TimestampMixin(BaseModel):
    """Mixin for created_at and updated_at timestamps."""

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class IdentifiableMixin(BaseModel):
    """Mixin for UUID-based identification."""

    id: UUID = Field(default_factory=uuid4)


class TenantScopedMixin(BaseModel):
    """Mixin for tenant-scoped resources."""

    tenant_id: UUID


class AuditMixin(TimestampMixin):
    """Mixin for audit fields including user tracking."""

    created_by: UUID | None = None
    updated_by: UUID | None = None


class SoftDeleteMixin(BaseModel):
    """Mixin for soft delete functionality."""

    deleted_at: datetime | None = None
    deleted_by: UUID | None = None

    @property
    def is_deleted(self) -> bool:
        """Check if the record is soft deleted."""
        return self.deleted_at is not None


class MetadataMixin(BaseModel):
    """Mixin for arbitrary metadata storage."""

    metadata: dict[str, Any] = Field(default_factory=dict)
