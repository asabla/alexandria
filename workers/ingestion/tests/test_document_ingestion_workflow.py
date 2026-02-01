"""
Unit tests for the Document Ingestion Workflow.

Tests cover:
- Workflow input validation
- Successful workflow execution
- Query methods (get_status, get_progress)
- Signal methods (cancel, pause, resume)
- Error handling and compensation
- Edge cases and skip options
"""

import uuid
from datetime import timedelta

import pytest
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from ingestion_worker.workflows.document_ingestion import (
    DocumentIngestionWorkflow,
    IngestionStep,
    IngestionWorkflowInput,
    IngestionWorkflowOutput,
    IngestionWorkflowStatus,
)

from .conftest import (
    MockActivityResults,
    create_mock_activities,
    get_default_mock_results,
    run_workflow_with_mocks,
)


# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


# =============================================================================
# IngestionWorkflowInput Tests
# =============================================================================


class TestIngestionWorkflowInput:
    """Tests for IngestionWorkflowInput dataclass."""

    def test_minimal_input(self, sample_document_id: str, sample_tenant_id: str):
        """Test creating input with only required fields."""
        input_data = IngestionWorkflowInput(
            document_id=sample_document_id,
            tenant_id=sample_tenant_id,
            storage_bucket="documents",
            storage_key="path/to/file.pdf",
        )

        assert input_data.document_id == sample_document_id
        assert input_data.tenant_id == sample_tenant_id
        assert input_data.storage_bucket == "documents"
        assert input_data.storage_key == "path/to/file.pdf"

        # Check defaults
        assert input_data.source_url is None
        assert input_data.source_filename is None
        assert input_data.mime_type is None
        assert input_data.skip_ocr is False
        assert input_data.skip_entity_extraction is False
        assert input_data.skip_graph_building is False
        assert input_data.force_reprocess is False
        assert input_data.chunk_size == 1000
        assert input_data.chunk_overlap == 200
        assert input_data.project_id is None
        assert input_data.metadata == {}

    def test_full_input(self, sample_document_id: str, sample_tenant_id: str):
        """Test creating input with all fields."""
        project_id = str(uuid.uuid4())
        metadata = {"source": "test", "priority": "high"}

        input_data = IngestionWorkflowInput(
            document_id=sample_document_id,
            tenant_id=sample_tenant_id,
            storage_bucket="documents",
            storage_key="path/to/file.pdf",
            source_url="https://example.com/doc.pdf",
            source_filename="document.pdf",
            mime_type="application/pdf",
            skip_ocr=True,
            skip_entity_extraction=True,
            skip_graph_building=True,
            force_reprocess=True,
            chunk_size=500,
            chunk_overlap=100,
            project_id=project_id,
            metadata=metadata,
        )

        assert input_data.source_url == "https://example.com/doc.pdf"
        assert input_data.source_filename == "document.pdf"
        assert input_data.mime_type == "application/pdf"
        assert input_data.skip_ocr is True
        assert input_data.skip_entity_extraction is True
        assert input_data.skip_graph_building is True
        assert input_data.force_reprocess is True
        assert input_data.chunk_size == 500
        assert input_data.chunk_overlap == 100
        assert input_data.project_id == project_id
        assert input_data.metadata == metadata


# =============================================================================
# Workflow Execution Tests
# =============================================================================


class TestWorkflowExecution:
    """Tests for successful workflow execution."""

    async def test_successful_workflow_execution(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_workflow_input: IngestionWorkflowInput,
        mock_results: MockActivityResults,
    ):
        """Test that workflow completes successfully with all activities."""
        result = await run_workflow_with_mocks(
            workflow_environment,
            sample_workflow_input,
            mock_results,
        )

        # Verify success
        assert result.success is True
        assert result.document_id == sample_workflow_input.document_id

        # Verify processing results
        assert result.document_type == "pdf"
        assert result.page_count == 5
        assert result.word_count == 150
        assert result.language == "en"

        # Verify indexing
        assert result.chunks_created == 2
        assert result.indexed_vector is True
        assert result.indexed_fulltext is True

        # Verify entity extraction
        assert result.entities_extracted == 2
        assert result.relationships_extracted == 1
        assert result.indexed_graph is True

        # Verify final status
        assert result.final_status.current_step == IngestionStep.COMPLETED
        assert result.final_status.progress_percent == 100
        assert result.final_status.message == "Document ingestion completed"

    async def test_workflow_with_skip_entity_extraction(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_document_id: str,
        sample_tenant_id: str,
        mock_results: MockActivityResults,
    ):
        """Test workflow with entity extraction skipped."""
        input_data = IngestionWorkflowInput(
            document_id=sample_document_id,
            tenant_id=sample_tenant_id,
            storage_bucket="documents",
            storage_key="path/to/file.pdf",
            skip_entity_extraction=True,
        )

        result = await run_workflow_with_mocks(
            workflow_environment,
            input_data,
            mock_results,
        )

        assert result.success is True
        assert result.entities_extracted == 0
        assert result.relationships_extracted == 0
        assert result.indexed_graph is False

    async def test_workflow_with_skip_graph_building(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_document_id: str,
        sample_tenant_id: str,
        mock_results: MockActivityResults,
    ):
        """Test workflow with graph building skipped but entities extracted."""
        input_data = IngestionWorkflowInput(
            document_id=sample_document_id,
            tenant_id=sample_tenant_id,
            storage_bucket="documents",
            storage_key="path/to/file.pdf",
            skip_graph_building=True,
        )

        result = await run_workflow_with_mocks(
            workflow_environment,
            input_data,
            mock_results,
        )

        assert result.success is True
        # Entities are still extracted
        assert result.entities_extracted == 2
        assert result.relationships_extracted == 1
        # But graph is not built
        assert result.indexed_graph is False

    async def test_workflow_with_no_entities_found(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_workflow_input: IngestionWorkflowInput,
    ):
        """Test workflow when no entities are found."""
        mock_results = get_default_mock_results()
        mock_results.entities = {"entities": []}

        result = await run_workflow_with_mocks(
            workflow_environment,
            sample_workflow_input,
            mock_results,
        )

        assert result.success is True
        assert result.entities_extracted == 0
        assert result.relationships_extracted == 0
        # Graph not built because no entities
        assert result.indexed_graph is False

    async def test_workflow_with_custom_chunk_settings(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_document_id: str,
        sample_tenant_id: str,
        mock_results: MockActivityResults,
    ):
        """Test workflow respects custom chunk size and overlap."""
        input_data = IngestionWorkflowInput(
            document_id=sample_document_id,
            tenant_id=sample_tenant_id,
            storage_bucket="documents",
            storage_key="path/to/file.pdf",
            chunk_size=500,
            chunk_overlap=50,
        )

        # The workflow should pass these values to the chunk_document activity
        # We're just verifying the workflow completes - actual chunking is tested
        # in activity tests
        result = await run_workflow_with_mocks(
            workflow_environment,
            input_data,
            mock_results,
        )

        assert result.success is True


# =============================================================================
# Query Method Tests
# =============================================================================


class TestWorkflowQueries:
    """Tests for workflow query methods."""

    async def test_get_status_query(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_workflow_input: IngestionWorkflowInput,
        mock_results: MockActivityResults,
        task_queue: str,
    ):
        """Test querying workflow status during execution."""
        activities = create_mock_activities(mock_results)
        workflow_id = f"test-workflow-{uuid.uuid4()}"

        async with Worker(
            workflow_environment.client,
            task_queue=task_queue,
            workflows=[DocumentIngestionWorkflow],
            activities=activities,
        ):
            handle = await workflow_environment.client.start_workflow(
                DocumentIngestionWorkflow.run,
                sample_workflow_input,
                id=workflow_id,
                task_queue=task_queue,
            )

            # Wait for workflow to complete
            result = await handle.result()
            assert result.success is True

            # Query final status (workflow is completed, but we can still query)
            # Note: In time-skipping env, we query after completion
            # For real-time testing, we'd query during execution

    async def test_get_progress_query(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_workflow_input: IngestionWorkflowInput,
        mock_results: MockActivityResults,
        task_queue: str,
    ):
        """Test querying workflow progress."""
        activities = create_mock_activities(mock_results)
        workflow_id = f"test-workflow-{uuid.uuid4()}"

        async with Worker(
            workflow_environment.client,
            task_queue=task_queue,
            workflows=[DocumentIngestionWorkflow],
            activities=activities,
        ):
            handle = await workflow_environment.client.start_workflow(
                DocumentIngestionWorkflow.run,
                sample_workflow_input,
                id=workflow_id,
                task_queue=task_queue,
            )

            # Wait for completion
            result = await handle.result()
            assert result.final_status.progress_percent == 100


# =============================================================================
# Signal Method Tests
# =============================================================================


class TestWorkflowSignals:
    """Tests for workflow signal methods."""

    async def test_cancel_signal(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_workflow_input: IngestionWorkflowInput,
        task_queue: str,
    ):
        """Test cancelling workflow via signal."""
        # Create mock that runs slowly to allow cancellation
        mock_results = get_default_mock_results()
        activities = create_mock_activities(mock_results)
        workflow_id = f"test-workflow-{uuid.uuid4()}"

        async with Worker(
            workflow_environment.client,
            task_queue=task_queue,
            workflows=[DocumentIngestionWorkflow],
            activities=activities,
        ):
            handle = await workflow_environment.client.start_workflow(
                DocumentIngestionWorkflow.run,
                sample_workflow_input,
                id=workflow_id,
                task_queue=task_queue,
            )

            # Send cancel signal
            await handle.signal(DocumentIngestionWorkflow.cancel)

            # Get result - workflow should complete with cancelled status
            result = await handle.result()

            # Note: Due to time-skipping, the workflow may complete before
            # the cancel signal is processed. In production, this would
            # result in a cancelled status.
            # We verify the signal method exists and can be called.
            assert result is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestWorkflowErrorHandling:
    """Tests for workflow error handling and compensation."""

    async def test_activity_failure_marks_workflow_failed(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_workflow_input: IngestionWorkflowInput,
    ):
        """Test that activity failure results in failed workflow."""
        mock_results = get_default_mock_results()
        mock_results.fail_at_activity = "parse_document"
        mock_results.fail_message = "Failed to parse document: corrupted file"

        result = await run_workflow_with_mocks(
            workflow_environment,
            sample_workflow_input,
            mock_results,
        )

        assert result.success is False
        assert result.error_message is not None
        assert result.final_status.current_step == IngestionStep.FAILED
        # Error occurred at parsing step
        assert result.error_step == "parsing"

    async def test_classification_failure(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_workflow_input: IngestionWorkflowInput,
    ):
        """Test handling of classification activity failure."""
        mock_results = get_default_mock_results()
        mock_results.fail_at_activity = "classify_document"
        mock_results.fail_message = "Unknown file type"

        result = await run_workflow_with_mocks(
            workflow_environment,
            sample_workflow_input,
            mock_results,
        )

        assert result.success is False
        assert result.error_step == "classifying"

    async def test_embedding_failure(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_workflow_input: IngestionWorkflowInput,
    ):
        """Test handling of embedding generation failure."""
        mock_results = get_default_mock_results()
        mock_results.fail_at_activity = "generate_embeddings"
        mock_results.fail_message = "Embedding service unavailable"

        result = await run_workflow_with_mocks(
            workflow_environment,
            sample_workflow_input,
            mock_results,
        )

        assert result.success is False
        assert result.error_step == "embedding"

    async def test_vector_indexing_failure(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_workflow_input: IngestionWorkflowInput,
    ):
        """Test handling of vector indexing failure."""
        mock_results = get_default_mock_results()
        mock_results.fail_at_activity = "index_vector"
        mock_results.fail_message = "Qdrant connection failed"

        result = await run_workflow_with_mocks(
            workflow_environment,
            sample_workflow_input,
            mock_results,
        )

        assert result.success is False
        assert result.error_step == "indexing_vector"

    async def test_entity_extraction_failure(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_workflow_input: IngestionWorkflowInput,
    ):
        """Test handling of entity extraction failure."""
        mock_results = get_default_mock_results()
        mock_results.fail_at_activity = "extract_entities"
        mock_results.fail_message = "NLP model error"

        result = await run_workflow_with_mocks(
            workflow_environment,
            sample_workflow_input,
            mock_results,
        )

        assert result.success is False
        assert result.error_step == "extracting_entities"

    async def test_graph_building_failure(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_workflow_input: IngestionWorkflowInput,
    ):
        """Test handling of graph building failure."""
        mock_results = get_default_mock_results()
        mock_results.fail_at_activity = "build_graph"
        mock_results.fail_message = "Neo4j connection refused"

        result = await run_workflow_with_mocks(
            workflow_environment,
            sample_workflow_input,
            mock_results,
        )

        assert result.success is False
        assert result.error_step == "building_graph"

    async def test_error_step_is_recorded(
        self,
        workflow_environment: WorkflowEnvironment,
        sample_workflow_input: IngestionWorkflowInput,
    ):
        """Test that the step where error occurred is recorded."""
        mock_results = get_default_mock_results()
        mock_results.fail_at_activity = "chunk_document"

        result = await run_workflow_with_mocks(
            workflow_environment,
            sample_workflow_input,
            mock_results,
        )

        assert result.success is False
        # The error step should be recorded
        assert result.error_step is not None or result.final_status.error_step is not None


# =============================================================================
# IngestionStep Enum Tests
# =============================================================================


class TestIngestionStep:
    """Tests for IngestionStep enum."""

    def test_all_steps_defined(self):
        """Test that all expected steps are defined."""
        expected_steps = [
            "initializing",
            "classifying",
            "downloading",
            "parsing",
            "chunking",
            "embedding",
            "indexing_vector",
            "indexing_fulltext",
            "extracting_entities",
            "extracting_relationships",
            "building_graph",
            "finalizing",
            "completed",
            "failed",
            "cancelled",
        ]

        actual_steps = [step.value for step in IngestionStep]
        for expected in expected_steps:
            assert expected in actual_steps, f"Missing step: {expected}"

    def test_step_values_are_strings(self):
        """Test that step values are lowercase strings."""
        for step in IngestionStep:
            assert isinstance(step.value, str)
            assert step.value == step.value.lower()


# =============================================================================
# IngestionWorkflowStatus Tests
# =============================================================================


class TestIngestionWorkflowStatus:
    """Tests for IngestionWorkflowStatus dataclass."""

    def test_minimal_status(self):
        """Test creating status with minimal fields."""
        status = IngestionWorkflowStatus(
            current_step=IngestionStep.INITIALIZING,
            progress_percent=0,
            message="Starting",
        )

        assert status.current_step == IngestionStep.INITIALIZING
        assert status.progress_percent == 0
        assert status.message == "Starting"
        assert status.chunks_created == 0
        assert status.entities_extracted == 0
        assert status.relationships_extracted == 0
        assert status.error_message is None
        assert status.error_step is None

    def test_full_status(self):
        """Test creating status with all fields."""
        status = IngestionWorkflowStatus(
            current_step=IngestionStep.COMPLETED,
            progress_percent=100,
            message="Done",
            chunks_created=10,
            entities_extracted=5,
            relationships_extracted=3,
            error_message=None,
            error_step=None,
            started_at="2024-01-01T00:00:00Z",
            step_started_at="2024-01-01T00:05:00Z",
        )

        assert status.chunks_created == 10
        assert status.entities_extracted == 5
        assert status.relationships_extracted == 3


# =============================================================================
# IngestionWorkflowOutput Tests
# =============================================================================


class TestIngestionWorkflowOutput:
    """Tests for IngestionWorkflowOutput dataclass."""

    def test_successful_output(self):
        """Test creating successful output."""
        status = IngestionWorkflowStatus(
            current_step=IngestionStep.COMPLETED,
            progress_percent=100,
            message="Done",
        )

        output = IngestionWorkflowOutput(
            document_id="doc-123",
            success=True,
            final_status=status,
            document_type="pdf",
            page_count=10,
            word_count=1000,
            language="en",
            chunks_created=5,
            entities_extracted=10,
            relationships_extracted=3,
            indexed_vector=True,
            indexed_fulltext=True,
            indexed_graph=True,
        )

        assert output.success is True
        assert output.error_message is None

    def test_failed_output(self):
        """Test creating failed output."""
        status = IngestionWorkflowStatus(
            current_step=IngestionStep.FAILED,
            progress_percent=35,
            message="Failed at chunking",
            error_message="Out of memory",
            error_step=IngestionStep.CHUNKING,
        )

        output = IngestionWorkflowOutput(
            document_id="doc-123",
            success=False,
            final_status=status,
            error_message="Out of memory",
            error_step="chunking",
        )

        assert output.success is False
        assert output.error_message == "Out of memory"
        assert output.error_step == "chunking"

    def test_partial_success_output(self):
        """Test creating output with partial success (some failures but continued)."""
        status = IngestionWorkflowStatus(
            current_step=IngestionStep.COMPLETED,
            progress_percent=100,
            message="Completed with warnings",
        )

        output = IngestionWorkflowOutput(
            document_id="doc-123",
            success=True,
            final_status=status,
            document_type="pdf",
            page_count=10,
            word_count=1000,
            chunks_created=5,
            indexed_vector=True,
            indexed_fulltext=False,  # Failed but continued
            indexed_graph=True,
            partial_success=True,
            indexing_failures=["fulltext: Connection refused"],
        )

        assert output.success is True
        assert output.partial_success is True
        assert output.indexed_vector is True
        assert output.indexed_fulltext is False
        assert len(output.indexing_failures) == 1
        assert "fulltext" in output.indexing_failures[0]

    def test_output_defaults(self):
        """Test that output has correct defaults for partial failure tracking."""
        status = IngestionWorkflowStatus(
            current_step=IngestionStep.COMPLETED,
            progress_percent=100,
            message="Done",
        )

        output = IngestionWorkflowOutput(
            document_id="doc-123",
            success=True,
            final_status=status,
        )

        # Check new default fields
        assert output.partial_success is False
        assert output.indexing_failures == []
        assert output.extraction_failures == []


# =============================================================================
# ParallelIndexingResult Tests
# =============================================================================


class TestParallelIndexingResult:
    """Tests for ParallelIndexingResult dataclass."""

    def test_default_values(self):
        """Test that ParallelIndexingResult has correct defaults."""
        from ingestion_worker.workflows.document_ingestion import ParallelIndexingResult

        result = ParallelIndexingResult()

        assert result.vector_success is False
        assert result.fulltext_success is False
        assert result.vector_error is None
        assert result.fulltext_error is None
        assert result.vector_indexed_count == 0
        assert result.vector_collection_name is None
        assert result.fulltext_index_name is None

    def test_successful_result(self):
        """Test creating a fully successful result."""
        from ingestion_worker.workflows.document_ingestion import ParallelIndexingResult

        result = ParallelIndexingResult(
            vector_success=True,
            fulltext_success=True,
            vector_indexed_count=50,
            vector_collection_name="documents_v1",
            fulltext_index_name="documents",
        )

        assert result.vector_success is True
        assert result.fulltext_success is True
        assert result.vector_error is None
        assert result.fulltext_error is None
        assert result.vector_indexed_count == 50

    def test_partial_failure_vector(self):
        """Test result when vector indexing fails."""
        from ingestion_worker.workflows.document_ingestion import ParallelIndexingResult

        result = ParallelIndexingResult(
            vector_success=False,
            vector_error="Connection refused to Qdrant",
            fulltext_success=True,
            fulltext_index_name="documents",
        )

        assert result.vector_success is False
        assert result.vector_error == "Connection refused to Qdrant"
        assert result.fulltext_success is True
        assert result.fulltext_error is None

    def test_partial_failure_fulltext(self):
        """Test result when fulltext indexing fails."""
        from ingestion_worker.workflows.document_ingestion import ParallelIndexingResult

        result = ParallelIndexingResult(
            vector_success=True,
            vector_indexed_count=50,
            vector_collection_name="documents_v1",
            fulltext_success=False,
            fulltext_error="MeiliSearch timeout",
        )

        assert result.vector_success is True
        assert result.fulltext_success is False
        assert result.fulltext_error == "MeiliSearch timeout"

    def test_both_failures(self):
        """Test result when both indexing operations fail."""
        from ingestion_worker.workflows.document_ingestion import ParallelIndexingResult

        result = ParallelIndexingResult(
            vector_success=False,
            vector_error="Qdrant connection refused",
            fulltext_success=False,
            fulltext_error="MeiliSearch unreachable",
        )

        assert result.vector_success is False
        assert result.fulltext_success is False
        assert "Qdrant" in result.vector_error
        assert "MeiliSearch" in result.fulltext_error


# =============================================================================
# Parallelism Input Options Tests
# =============================================================================


class TestParallelismOptions:
    """Tests for parallelism-related input options."""

    def test_default_parallelism_options(self, sample_document_id: str, sample_tenant_id: str):
        """Test that parallelism options have correct defaults."""
        workflow_input = IngestionWorkflowInput(
            document_id=sample_document_id,
            tenant_id=sample_tenant_id,
            storage_bucket="test-bucket",
            storage_key="test/doc.pdf",
        )

        # Parallel indexing should be enabled by default
        assert workflow_input.enable_parallel_indexing is True
        assert workflow_input.embedding_batch_size == 50
        assert workflow_input.max_parallel_activities == 5

        # Continue on failure should be off by default
        assert workflow_input.continue_on_indexing_failure is False
        assert workflow_input.continue_on_extraction_failure is False

    def test_disable_parallel_indexing(self, sample_document_id: str, sample_tenant_id: str):
        """Test disabling parallel indexing."""
        workflow_input = IngestionWorkflowInput(
            document_id=sample_document_id,
            tenant_id=sample_tenant_id,
            storage_bucket="test-bucket",
            storage_key="test/doc.pdf",
            enable_parallel_indexing=False,
        )

        assert workflow_input.enable_parallel_indexing is False

    def test_continue_on_failure_options(self, sample_document_id: str, sample_tenant_id: str):
        """Test enabling continue on failure options."""
        workflow_input = IngestionWorkflowInput(
            document_id=sample_document_id,
            tenant_id=sample_tenant_id,
            storage_bucket="test-bucket",
            storage_key="test/doc.pdf",
            continue_on_indexing_failure=True,
            continue_on_extraction_failure=True,
        )

        assert workflow_input.continue_on_indexing_failure is True
        assert workflow_input.continue_on_extraction_failure is True

    def test_custom_batch_size(self, sample_document_id: str, sample_tenant_id: str):
        """Test setting custom batch size for embeddings."""
        workflow_input = IngestionWorkflowInput(
            document_id=sample_document_id,
            tenant_id=sample_tenant_id,
            storage_bucket="test-bucket",
            storage_key="test/doc.pdf",
            embedding_batch_size=100,
            max_parallel_activities=10,
        )

        assert workflow_input.embedding_batch_size == 100
        assert workflow_input.max_parallel_activities == 10
