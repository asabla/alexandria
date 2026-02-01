"""Alexandria API - FastAPI backend for document research platform."""

from alexandria_api.app import create_app

__version__ = "0.1.0"

# Create default app instance
app = create_app()

__all__ = ["app", "create_app", "__version__"]
