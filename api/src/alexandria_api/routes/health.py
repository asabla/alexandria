"""Health check endpoints."""

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthStatus(BaseModel):
    """Health check response model."""

    status: Literal["healthy", "unhealthy", "degraded"]
    timestamp: datetime
    version: str
    checks: dict[str, bool] = {}


class ReadinessStatus(BaseModel):
    """Readiness check response model."""

    ready: bool
    timestamp: datetime
    services: dict[str, bool] = {}


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Basic health check endpoint.

    Returns the application health status. Used by load balancers
    and container orchestrators for liveness probes.
    """
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(UTC),
        version="0.1.0",
        checks={
            "api": True,
        },
    )


@router.get("/ready", response_model=ReadinessStatus)
async def readiness_check() -> ReadinessStatus:
    """
    Readiness check endpoint.

    Checks if all required services are available.
    Used by container orchestrators for readiness probes.

    TODO: Implement actual service checks (PostgreSQL, MinIO, etc.)
    """
    # TODO: Check database connection
    # TODO: Check MinIO connection
    # TODO: Check Qdrant connection
    # TODO: Check Neo4j connection
    # TODO: Check MeiliSearch connection
    # TODO: Check Temporal connection

    services = {
        "database": True,  # TODO: Actual check
        "minio": True,  # TODO: Actual check
        "qdrant": True,  # TODO: Actual check
        "neo4j": True,  # TODO: Actual check
        "meilisearch": True,  # TODO: Actual check
        "temporal": True,  # TODO: Actual check
    }

    all_ready = all(services.values())

    return ReadinessStatus(
        ready=all_ready,
        timestamp=datetime.now(UTC),
        services=services,
    )


@router.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API info."""
    return {
        "name": "Alexandria API",
        "version": "0.1.0",
        "description": "Document research platform for investigative journalism",
        "docs": "/docs",
    }
