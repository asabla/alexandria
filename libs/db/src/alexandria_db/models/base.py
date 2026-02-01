"""Base SQLAlchemy model classes and mixins."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class UUIDMixin:
    """Mixin for UUID primary key."""

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
        nullable=True,
    )
    deleted_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        default=None,
        nullable=True,
    )


class AuditMixin(TimestampMixin):
    """Mixin for audit fields including user tracking."""

    created_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        default=None,
        nullable=True,
    )
    updated_by: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        default=None,
        nullable=True,
    )
