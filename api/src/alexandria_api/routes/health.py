"""Health check endpoints."""

import asyncio
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from alexandria_api.config import Settings, get_settings
from alexandria_api.dependencies import (
    check_database_health,
    check_minio_health,
    check_qdrant_health,
    check_neo4j_health,
    check_meilisearch_health,
    check_temporal_health,
)

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

    This endpoint always returns quickly and only checks if the API is running.
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
async def readiness_check(
    settings: Settings = Depends(get_settings),
) -> ReadinessStatus:
    """
    Readiness check endpoint.

    Checks if all required services are available.
    Used by container orchestrators for readiness probes.

    Services are checked in parallel for faster response times.
    """
    # Run all health checks in parallel
    results = await asyncio.gather(
        check_database_health(settings),
        check_minio_health(settings),
        check_qdrant_health(settings),
        check_neo4j_health(settings),
        check_meilisearch_health(settings),
        check_temporal_health(settings),
        return_exceptions=True,
    )

    services = {
        "database": results[0] if isinstance(results[0], bool) else False,
        "minio": results[1] if isinstance(results[1], bool) else False,
        "qdrant": results[2] if isinstance(results[2], bool) else False,
        "neo4j": results[3] if isinstance(results[3], bool) else False,
        "meilisearch": results[4] if isinstance(results[4], bool) else False,
        "temporal": results[5] if isinstance(results[5], bool) else False,
    }

    # Core services that must be healthy for the API to be ready
    core_services = ["database"]
    core_ready = all(services.get(s, False) for s in core_services)

    # Optional services - API can still work without them
    all_ready = all(services.values())

    return ReadinessStatus(
        ready=core_ready,  # Only require database for basic readiness
        timestamp=datetime.now(UTC),
        services=services,
    )


@router.get("/ready/strict", response_model=ReadinessStatus)
async def strict_readiness_check(
    settings: Settings = Depends(get_settings),
) -> ReadinessStatus:
    """
    Strict readiness check endpoint.

    Same as /ready but requires ALL services to be healthy.
    Use this for canary deployments or when full functionality is required.
    """
    # Run all health checks in parallel
    results = await asyncio.gather(
        check_database_health(settings),
        check_minio_health(settings),
        check_qdrant_health(settings),
        check_neo4j_health(settings),
        check_meilisearch_health(settings),
        check_temporal_health(settings),
        return_exceptions=True,
    )

    services = {
        "database": results[0] if isinstance(results[0], bool) else False,
        "minio": results[1] if isinstance(results[1], bool) else False,
        "qdrant": results[2] if isinstance(results[2], bool) else False,
        "neo4j": results[3] if isinstance(results[3], bool) else False,
        "meilisearch": results[4] if isinstance(results[4], bool) else False,
        "temporal": results[5] if isinstance(results[5], bool) else False,
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
