"""
Ingestion Worker configuration and entry point.

This module provides the worker setup for running Temporal activities
and workflows for document ingestion.
"""

import asyncio
import signal
import sys
from dataclasses import dataclass, field

from temporalio.client import Client
from temporalio.worker import Worker

from ingestion_worker.workflows import DocumentIngestionWorkflow
from ingestion_worker.activities import (
    classify_document,
    parse_document,
    chunk_document,
    update_document_status,
    generate_embeddings,
    index_vector,
    index_fulltext,
    extract_entities,
    extract_relationships,
    build_graph,
)


@dataclass
class WorkerConfig:
    """Configuration for the ingestion worker."""

    # Temporal connection
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"

    # Task queue
    task_queue: str = "ingestion-tasks"

    # Worker settings
    max_concurrent_activities: int = 10
    max_concurrent_workflow_tasks: int = 10

    # Debug settings
    debug: bool = False


# All activities to register
ACTIVITIES = [
    classify_document,
    parse_document,
    chunk_document,
    update_document_status,
    generate_embeddings,
    index_vector,
    index_fulltext,
    extract_entities,
    extract_relationships,
    build_graph,
]

# All workflows to register
WORKFLOWS = [
    DocumentIngestionWorkflow,
]


async def create_worker(
    client: Client,
    config: WorkerConfig | None = None,
) -> Worker:
    """
    Create an ingestion worker instance.

    Args:
        client: Temporal client connection
        config: Worker configuration (uses defaults if not provided)

    Returns:
        Configured Worker instance (not started)
    """
    if config is None:
        config = WorkerConfig()

    worker = Worker(
        client,
        task_queue=config.task_queue,
        workflows=WORKFLOWS,
        activities=ACTIVITIES,
        max_concurrent_activities=config.max_concurrent_activities,
        max_concurrent_workflow_task_polls=config.max_concurrent_workflow_tasks,
    )

    return worker


async def run_worker(config: WorkerConfig | None = None) -> None:
    """
    Run the ingestion worker.

    Connects to Temporal and runs until interrupted.

    Args:
        config: Worker configuration (uses defaults if not provided)
    """
    import structlog

    log = structlog.get_logger()

    if config is None:
        config = WorkerConfig()

    log.info(
        "Starting ingestion worker",
        temporal_address=config.temporal_address,
        task_queue=config.task_queue,
    )

    # Connect to Temporal
    client = await Client.connect(
        config.temporal_address,
        namespace=config.temporal_namespace,
    )

    # Create worker
    worker = await create_worker(client, config)

    # Set up graceful shutdown
    shutdown_event = asyncio.Event()

    def handle_shutdown(sig: signal.Signals) -> None:
        log.info(f"Received {sig.name}, shutting down...")
        shutdown_event.set()

    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_shutdown, sig)

    # Run worker
    log.info("Worker started, waiting for tasks...")

    async with worker:
        await shutdown_event.wait()

    log.info("Worker shut down gracefully")


def main() -> None:
    """Entry point for the ingestion worker."""
    import os

    # Configure from environment
    config = WorkerConfig(
        temporal_address=os.getenv("TEMPORAL_ADDRESS", "localhost:7233"),
        temporal_namespace=os.getenv("TEMPORAL_NAMESPACE", "default"),
        task_queue=os.getenv("TEMPORAL_TASK_QUEUE", "ingestion-tasks"),
        max_concurrent_activities=int(os.getenv("MAX_CONCURRENT_ACTIVITIES", "10")),
        max_concurrent_workflow_tasks=int(os.getenv("MAX_CONCURRENT_WORKFLOW_TASKS", "10")),
        debug=os.getenv("DEBUG", "false").lower() == "true",
    )

    # Configure logging
    import structlog

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()
            if config.debug
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Run worker
    try:
        asyncio.run(run_worker(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
