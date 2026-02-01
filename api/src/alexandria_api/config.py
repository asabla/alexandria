"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "alexandria"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # CORS
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://alexandria:alexandria_dev@localhost:5432/alexandria"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_dev_password"

    # MeiliSearch
    meilisearch_url: str = "http://localhost:7700"
    meilisearch_api_key: str = "masterKey"

    # Temporal
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"

    # vLLM
    vllm_base_url: str = "http://localhost:8000/v1"

    # Multi-tenancy
    single_tenant_mode: bool = True
    default_tenant_id: str = "default"
    default_project_id: str = "default"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
