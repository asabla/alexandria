"""Temporal workflow client wrapper."""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Sequence
from uuid import UUID

import structlog
from temporalio.client import Client, WorkflowHandle, WorkflowExecutionStatus
from temporalio.common import RetryPolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

logger = structlog.get_logger(__name__)


@dataclass
class WorkflowInfo:
    """Information about a workflow execution."""

    workflow_id: str
    run_id: str
    workflow_type: str
    status: str
    start_time: str | None = None
    close_time: str | None = None
    execution_time: float | None = None


@dataclass
class WorkflowOptions:
    """Options for starting a workflow."""

    task_queue: str
    workflow_id: str | None = None
    retry_policy: RetryPolicy | None = None
    execution_timeout: timedelta | None = None
    run_timeout: timedelta | None = None
    task_timeout: timedelta | None = None
    id_reuse_policy: str = "ALLOW_DUPLICATE"
    memo: dict[str, Any] = field(default_factory=dict)
    search_attributes: dict[str, Any] = field(default_factory=dict)


class TemporalClient:
    """
    Wrapper for Temporal workflow operations.

    Provides a simplified interface for starting, querying, signaling,
    and managing Temporal workflows.
    """

    def __init__(self, client: Client, default_task_queue: str = "default"):
        """
        Initialize Temporal client wrapper.

        Args:
            client: Connected Temporal client instance
            default_task_queue: Default task queue for workflows
        """
        self._client = client
        self._default_task_queue = default_task_queue
        self._log = logger.bind(service="temporal")

    @classmethod
    async def connect(
        cls,
        target_host: str = "localhost:7233",
        namespace: str = "default",
        default_task_queue: str = "default",
    ) -> "TemporalClient":
        """
        Create a connected Temporal client.

        Args:
            target_host: Temporal server address
            namespace: Temporal namespace to use
            default_task_queue: Default task queue for workflows

        Returns:
            Connected TemporalClient instance
        """
        client = await Client.connect(target_host, namespace=namespace)
        return cls(client, default_task_queue)

    @property
    def client(self) -> Client:
        """Get the underlying Temporal client."""
        return self._client

    async def start_workflow(
        self,
        workflow: str | type,
        *args: Any,
        id: str | None = None,
        task_queue: str | None = None,
        execution_timeout: timedelta | None = None,
        run_timeout: timedelta | None = None,
        task_timeout: timedelta | None = None,
        retry_policy: RetryPolicy | None = None,
        memo: dict[str, Any] | None = None,
        search_attributes: dict[str, Any] | None = None,
    ) -> WorkflowHandle:
        """
        Start a new workflow execution.

        Args:
            workflow: Workflow class or name
            *args: Arguments to pass to the workflow
            id: Unique workflow ID (generated if not provided)
            task_queue: Task queue to use (defaults to default_task_queue)
            execution_timeout: Overall workflow execution timeout
            run_timeout: Single workflow run timeout
            task_timeout: Single task timeout
            retry_policy: Retry policy for the workflow
            memo: Memo fields to attach to the workflow
            search_attributes: Search attributes for the workflow

        Returns:
            WorkflowHandle for the started workflow
        """
        import uuid

        workflow_id = id or str(uuid.uuid4())
        queue = task_queue or self._default_task_queue

        try:
            handle = await self._client.start_workflow(
                workflow,
                *args,
                id=workflow_id,
                task_queue=queue,
                execution_timeout=execution_timeout,
                run_timeout=run_timeout,
                task_timeout=task_timeout,
                retry_policy=retry_policy,
                memo=memo or {},
                search_attributes=search_attributes or {},
            )

            self._log.info(
                "workflow_started",
                workflow_id=workflow_id,
                workflow_type=workflow if isinstance(workflow, str) else workflow.__name__,
                task_queue=queue,
                run_id=handle.result_run_id,
            )

            return handle
        except WorkflowAlreadyStartedError:
            self._log.warning("workflow_already_started", workflow_id=workflow_id)
            raise
        except Exception as e:
            self._log.error("workflow_start_failed", workflow_id=workflow_id, error=str(e))
            raise

    async def get_workflow_handle(
        self,
        workflow_id: str,
        run_id: str | None = None,
    ) -> WorkflowHandle:
        """
        Get a handle to an existing workflow.

        Args:
            workflow_id: The workflow ID
            run_id: Specific run ID (optional, uses latest if not provided)

        Returns:
            WorkflowHandle for the workflow
        """
        return self._client.get_workflow_handle(workflow_id, run_id=run_id)

    async def query_workflow(
        self,
        workflow_id: str,
        query: str,
        *args: Any,
        run_id: str | None = None,
    ) -> Any:
        """
        Query a workflow for its current state.

        Args:
            workflow_id: The workflow ID
            query: Name of the query method
            *args: Arguments to pass to the query
            run_id: Specific run ID (optional)

        Returns:
            Query result
        """
        try:
            handle = self._client.get_workflow_handle(workflow_id, run_id=run_id)
            result = await handle.query(query, *args)
            self._log.debug("workflow_queried", workflow_id=workflow_id, query=query)
            return result
        except Exception as e:
            self._log.error(
                "workflow_query_failed",
                workflow_id=workflow_id,
                query=query,
                error=str(e),
            )
            raise

    async def signal_workflow(
        self,
        workflow_id: str,
        signal: str,
        *args: Any,
        run_id: str | None = None,
    ) -> None:
        """
        Send a signal to a workflow.

        Args:
            workflow_id: The workflow ID
            signal: Name of the signal method
            *args: Arguments to pass to the signal
            run_id: Specific run ID (optional)
        """
        try:
            handle = self._client.get_workflow_handle(workflow_id, run_id=run_id)
            await handle.signal(signal, *args)
            self._log.info("workflow_signaled", workflow_id=workflow_id, signal=signal)
        except Exception as e:
            self._log.error(
                "workflow_signal_failed",
                workflow_id=workflow_id,
                signal=signal,
                error=str(e),
            )
            raise

    async def cancel_workflow(
        self,
        workflow_id: str,
        run_id: str | None = None,
    ) -> None:
        """
        Cancel a workflow execution.

        Args:
            workflow_id: The workflow ID
            run_id: Specific run ID (optional)
        """
        try:
            handle = self._client.get_workflow_handle(workflow_id, run_id=run_id)
            await handle.cancel()
            self._log.info("workflow_cancelled", workflow_id=workflow_id)
        except Exception as e:
            self._log.error(
                "workflow_cancel_failed",
                workflow_id=workflow_id,
                error=str(e),
            )
            raise

    async def terminate_workflow(
        self,
        workflow_id: str,
        reason: str = "Terminated via API",
        run_id: str | None = None,
    ) -> None:
        """
        Terminate a workflow execution immediately.

        Args:
            workflow_id: The workflow ID
            reason: Reason for termination
            run_id: Specific run ID (optional)
        """
        try:
            handle = self._client.get_workflow_handle(workflow_id, run_id=run_id)
            await handle.terminate(reason=reason)
            self._log.info("workflow_terminated", workflow_id=workflow_id, reason=reason)
        except Exception as e:
            self._log.error(
                "workflow_terminate_failed",
                workflow_id=workflow_id,
                error=str(e),
            )
            raise

    async def get_workflow_result(
        self,
        workflow_id: str,
        run_id: str | None = None,
        timeout: timedelta | None = None,
    ) -> Any:
        """
        Wait for and get the result of a workflow.

        Args:
            workflow_id: The workflow ID
            run_id: Specific run ID (optional)
            timeout: Maximum time to wait for result

        Returns:
            Workflow result
        """
        try:
            handle = self._client.get_workflow_handle(workflow_id, run_id=run_id)
            # Note: timeout is not directly supported in result(), would need async timeout wrapper
            result = await handle.result()
            self._log.info("workflow_result_retrieved", workflow_id=workflow_id)
            return result
        except Exception as e:
            self._log.error(
                "workflow_result_failed",
                workflow_id=workflow_id,
                error=str(e),
            )
            raise

    async def describe_workflow(
        self,
        workflow_id: str,
        run_id: str | None = None,
    ) -> WorkflowInfo:
        """
        Get detailed information about a workflow execution.

        Args:
            workflow_id: The workflow ID
            run_id: Specific run ID (optional)

        Returns:
            WorkflowInfo with execution details
        """
        try:
            handle = self._client.get_workflow_handle(workflow_id, run_id=run_id)
            desc = await handle.describe()

            # Map status to string
            status_map = {
                WorkflowExecutionStatus.RUNNING: "running",
                WorkflowExecutionStatus.COMPLETED: "completed",
                WorkflowExecutionStatus.FAILED: "failed",
                WorkflowExecutionStatus.CANCELED: "canceled",
                WorkflowExecutionStatus.TERMINATED: "terminated",
                WorkflowExecutionStatus.CONTINUED_AS_NEW: "continued_as_new",
                WorkflowExecutionStatus.TIMED_OUT: "timed_out",
            }

            return WorkflowInfo(
                workflow_id=workflow_id,
                run_id=desc.run_id,
                workflow_type=desc.workflow_type,
                status=status_map.get(desc.status, "unknown"),
                start_time=desc.start_time.isoformat() if desc.start_time else None,
                close_time=desc.close_time.isoformat() if desc.close_time else None,
                execution_time=(
                    (desc.close_time - desc.start_time).total_seconds()
                    if desc.close_time and desc.start_time
                    else None
                ),
            )
        except Exception as e:
            self._log.error(
                "workflow_describe_failed",
                workflow_id=workflow_id,
                error=str(e),
            )
            raise

    async def list_workflows(
        self,
        query: str | None = None,
        page_size: int = 100,
    ) -> Sequence[WorkflowInfo]:
        """
        List workflow executions.

        Args:
            query: Temporal visibility query (e.g., "WorkflowType='MyWorkflow'")
            page_size: Maximum number of results to return

        Returns:
            List of WorkflowInfo for matching workflows
        """
        try:
            workflows = []
            async for workflow in self._client.list_workflows(query=query):
                if len(workflows) >= page_size:
                    break

                status_map = {
                    WorkflowExecutionStatus.RUNNING: "running",
                    WorkflowExecutionStatus.COMPLETED: "completed",
                    WorkflowExecutionStatus.FAILED: "failed",
                    WorkflowExecutionStatus.CANCELED: "canceled",
                    WorkflowExecutionStatus.TERMINATED: "terminated",
                    WorkflowExecutionStatus.CONTINUED_AS_NEW: "continued_as_new",
                    WorkflowExecutionStatus.TIMED_OUT: "timed_out",
                }

                workflows.append(
                    WorkflowInfo(
                        workflow_id=workflow.id,
                        run_id=workflow.run_id,
                        workflow_type=workflow.workflow_type,
                        status=status_map.get(workflow.status, "unknown"),
                        start_time=workflow.start_time.isoformat() if workflow.start_time else None,
                        close_time=workflow.close_time.isoformat() if workflow.close_time else None,
                    )
                )

            self._log.debug("workflows_listed", count=len(workflows), query=query)
            return workflows
        except Exception as e:
            self._log.error("workflows_list_failed", query=query, error=str(e))
            raise

    async def health_check(self) -> bool:
        """
        Check if Temporal is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to list a single workflow as a health check
            async for _ in self._client.list_workflows(query="", page_size=1):
                break
            return True
        except Exception as e:
            self._log.warning("health_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the Temporal client connection."""
        # Note: The Temporal Python SDK client doesn't have an explicit close method
        # but we include this for API consistency
        self._log.info("client_closed")
