"""FastAPI application factory."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alexandria_api.config import Settings, get_settings
from alexandria_api.routes import health, tenants


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager for startup/shutdown events."""
    settings = get_settings()

    # Startup
    # TODO: Initialize database connections
    # TODO: Initialize service clients (MinIO, Qdrant, Neo4j, etc.)

    yield

    # Shutdown
    # TODO: Close database connections
    # TODO: Close service clients


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="Alexandria API",
        description="Document research platform for investigative journalism",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health.router, tags=["Health"])
    app.include_router(tenants.router)

    return app
