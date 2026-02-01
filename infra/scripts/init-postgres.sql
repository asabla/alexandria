-- =============================================================================
-- Alexandria PostgreSQL Initialization Script
-- =============================================================================
-- This script runs on first database creation
-- For schema changes, use Alembic migrations instead
-- =============================================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create Temporal schema (Temporal auto-setup will use this)
-- Note: Temporal creates its own tables, this just ensures the user has permissions

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE alexandria TO alexandria;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Alexandria database initialized successfully';
END $$;
