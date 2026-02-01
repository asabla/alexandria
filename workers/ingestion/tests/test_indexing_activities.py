"""Tests for vector indexing activities."""

import hashlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ingestion_worker.activities.indexing_activities import (
    index_vector,
    index_vector_mock,
    _generate_point_id,
    _build_point_payload,
)
from ingestion_worker.workflows.document_ingestion import (
    IndexVectorInput,
    IndexVectorOutput,
    ChunkInfo,
    EmbeddingInfo,
    SparseVector,
)


class TestGeneratePointId:
    """Tests for deterministic point ID generation."""

    def test_deterministic_id_same_input(self):
        """Same document_id and chunk_sequence should produce same ID."""
        doc_id = "doc-123"
        chunk_seq = 5

        id1 = _generate_point_id(doc_id, chunk_seq)
        id2 = _generate_point_id(doc_id, chunk_seq)

        assert id1 == id2

    def test_deterministic_id_different_chunks(self):
        """Different chunk sequences should produce different IDs."""
        doc_id = "doc-123"

        id1 = _generate_point_id(doc_id, 0)
        id2 = _generate_point_id(doc_id, 1)

        assert id1 != id2

    def test_deterministic_id_different_documents(self):
        """Different documents should produce different IDs."""
        chunk_seq = 5

        id1 = _generate_point_id("doc-123", chunk_seq)
        id2 = _generate_point_id("doc-456", chunk_seq)

        assert id1 != id2

    def test_id_format_uuid_like(self):
        """Generated ID should be UUID-like format."""
        point_id = _generate_point_id("doc-123", 0)

        # Check UUID-like format: 8-4-4-4-12 hex chars
        parts = point_id.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

    def test_id_is_hex(self):
        """Generated ID should contain only hex characters and dashes."""
        point_id = _generate_point_id("doc-123", 42)

        # Remove dashes and check hex
        hex_only = point_id.replace("-", "")
        int(hex_only, 16)  # Should not raise


class TestBuildPointPayload:
    """Tests for building Qdrant point payloads."""

    @pytest.fixture
    def sample_chunk(self) -> ChunkInfo:
        """Create a sample chunk for testing."""
        return ChunkInfo(
            sequence_number=0,
            content="This is test content",
            content_hash="abc123",
            start_char=0,
            end_char=20,
            token_count=5,
            page_number=1,
            heading_context=["Section 1", "Subsection A"],
            chunk_type="paragraph",
            metadata={"source": "test", "language": "en"},
        )

    @pytest.fixture
    def sample_embedding(self) -> EmbeddingInfo:
        """Create a sample embedding for testing."""
        return EmbeddingInfo(
            chunk_sequence=0,
            embedding_id="emb-123",
            model="bge-base-en-v1.5",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector=SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2]),
            dimensions=3,
        )

    def test_basic_payload_fields(self, sample_chunk, sample_embedding):
        """Test that basic fields are included in payload."""
        payload = _build_point_payload(
            chunk=sample_chunk,
            embedding=sample_embedding,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
        )

        assert payload["document_id"] == "doc-123"
        assert payload["tenant_id"] == "tenant-abc"
        assert payload["chunk_sequence"] == 0
        assert payload["content"] == "This is test content"
        assert payload["content_hash"] == "abc123"
        assert payload["embedding_id"] == "emb-123"
        assert payload["embedding_model"] == "bge-base-en-v1.5"
        assert payload["start_char"] == 0
        assert payload["end_char"] == 20
        assert payload["token_count"] == 5

    def test_payload_with_project_id(self, sample_chunk, sample_embedding):
        """Test that project_id is included as list."""
        payload = _build_point_payload(
            chunk=sample_chunk,
            embedding=sample_embedding,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id="project-xyz",
        )

        assert payload["project_ids"] == ["project-xyz"]

    def test_payload_without_project_id(self, sample_chunk, sample_embedding):
        """Test that project_ids is not included when None."""
        payload = _build_point_payload(
            chunk=sample_chunk,
            embedding=sample_embedding,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
        )

        assert "project_ids" not in payload

    def test_payload_with_page_number(self, sample_chunk, sample_embedding):
        """Test that page_number is included."""
        payload = _build_point_payload(
            chunk=sample_chunk,
            embedding=sample_embedding,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
        )

        assert payload["page_number"] == 1

    def test_payload_without_page_number(self, sample_embedding):
        """Test payload when page_number is None."""
        chunk = ChunkInfo(
            sequence_number=0,
            content="Test",
            content_hash="hash",
            start_char=0,
            end_char=4,
            token_count=1,
            page_number=None,
        )

        payload = _build_point_payload(
            chunk=chunk,
            embedding=sample_embedding,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
        )

        assert "page_number" not in payload

    def test_payload_with_heading_context(self, sample_chunk, sample_embedding):
        """Test that heading_context is included."""
        payload = _build_point_payload(
            chunk=sample_chunk,
            embedding=sample_embedding,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
        )

        assert payload["heading_context"] == ["Section 1", "Subsection A"]

    def test_payload_with_metadata_prefix(self, sample_chunk, sample_embedding):
        """Test that metadata is flattened with prefix."""
        payload = _build_point_payload(
            chunk=sample_chunk,
            embedding=sample_embedding,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
        )

        assert payload["meta_source"] == "test"
        assert payload["meta_language"] == "en"


class TestIndexVectorMock:
    """Tests for mock vector indexing activity."""

    @pytest.fixture
    def sample_input(self) -> IndexVectorInput:
        """Create sample input for testing."""
        chunks = [
            ChunkInfo(
                sequence_number=i,
                content=f"Chunk {i} content",
                content_hash=f"hash{i}",
                start_char=i * 100,
                end_char=(i + 1) * 100,
                token_count=10,
            )
            for i in range(5)
        ]

        embeddings = [
            EmbeddingInfo(
                chunk_sequence=i,
                embedding_id=f"emb-{i}",
                model="bge-base-en-v1.5",
                dense_vector=[0.1] * 768,
                dimensions=768,
            )
            for i in range(5)
        ]

        return IndexVectorInput(
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id="project-xyz",
            chunks=chunks,
            embeddings=embeddings,
        )

    @pytest.mark.asyncio
    async def test_mock_returns_correct_count(self, sample_input):
        """Test that mock activity returns correct indexed count."""
        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                result = await index_vector_mock(sample_input)

        assert isinstance(result, IndexVectorOutput)
        assert result.indexed_count == 5

    @pytest.mark.asyncio
    async def test_mock_returns_collection_name(self, sample_input):
        """Test that mock activity returns collection name."""
        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                result = await index_vector_mock(sample_input)

        assert result.collection_name == "documents"

    @pytest.mark.asyncio
    async def test_mock_with_custom_collection(self, sample_input):
        """Test mock activity with custom collection from env."""
        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                with patch.dict("os.environ", {"QDRANT_COLLECTION": "custom_collection"}):
                    result = await index_vector_mock(sample_input)

        assert result.collection_name == "custom_collection"


class TestIndexVector:
    """Tests for the main vector indexing activity."""

    @pytest.fixture
    def sample_input(self) -> IndexVectorInput:
        """Create sample input for testing."""
        chunks = [
            ChunkInfo(
                sequence_number=i,
                content=f"Chunk {i} content",
                content_hash=f"hash{i}",
                start_char=i * 100,
                end_char=(i + 1) * 100,
                token_count=10,
                page_number=1,
                heading_context=["Test Section"],
                chunk_type="paragraph",
            )
            for i in range(3)
        ]

        embeddings = [
            EmbeddingInfo(
                chunk_sequence=i,
                embedding_id=f"emb-{i}",
                model="bge-base-en-v1.5",
                dense_vector=[0.1] * 768,
                sparse_vector=SparseVector(indices=[1, 2], values=[1.0, 0.5]),
                dimensions=768,
            )
            for i in range(3)
        ]

        return IndexVectorInput(
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id="project-xyz",
            chunks=chunks,
            embeddings=embeddings,
        )

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock QdrantClient."""
        client = MagicMock()
        client.ensure_collection_hybrid = AsyncMock(return_value=False)
        client.upsert_hybrid = AsyncMock(return_value=3)
        return client

    @pytest.mark.asyncio
    async def test_index_vector_success(self, sample_input, mock_qdrant_client):
        """Test successful vector indexing with mocked client.

        Note: This test verifies the activity works correctly when the
        alexandria_db.clients module is available. In CI/CD, this would
        be tested with the actual client or a more complete mock setup.
        """
        # Create mock module
        mock_clients_module = MagicMock()
        mock_clients_module.QdrantClient = MagicMock(return_value=mock_qdrant_client)
        mock_clients_module.HybridPoint = MagicMock()
        mock_clients_module.SparseVector = MagicMock()

        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                with patch.dict("sys.modules", {"alexandria_db.clients": mock_clients_module}):
                    result = await index_vector(sample_input)

        assert isinstance(result, IndexVectorOutput)
        assert result.indexed_count == 3
        assert result.collection_name == "documents"
        mock_qdrant_client.ensure_collection_hybrid.assert_called_once()
        mock_qdrant_client.upsert_hybrid.assert_called()

    @pytest.mark.asyncio
    async def test_index_vector_creates_collection(self, sample_input, mock_qdrant_client):
        """Test that collection is created if it doesn't exist."""
        mock_qdrant_client.ensure_collection_hybrid = AsyncMock(return_value=True)

        mock_clients_module = MagicMock()
        mock_clients_module.QdrantClient = MagicMock(return_value=mock_qdrant_client)
        mock_clients_module.HybridPoint = MagicMock()
        mock_clients_module.SparseVector = MagicMock()

        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                with patch.dict("sys.modules", {"alexandria_db.clients": mock_clients_module}):
                    await index_vector(sample_input)

        mock_qdrant_client.ensure_collection_hybrid.assert_called_once()


class TestIndexVectorIdempotency:
    """Tests for idempotent indexing behavior."""

    def test_same_document_same_ids(self):
        """Re-indexing same document should produce same point IDs."""
        doc_id = "doc-123"
        chunks = range(5)

        # First indexing
        ids_first = [_generate_point_id(doc_id, i) for i in chunks]

        # Second indexing (simulating re-index)
        ids_second = [_generate_point_id(doc_id, i) for i in chunks]

        assert ids_first == ids_second

    def test_different_documents_different_ids(self):
        """Different documents should have different point IDs."""
        chunks = range(3)

        ids_doc1 = [_generate_point_id("doc-1", i) for i in chunks]
        ids_doc2 = [_generate_point_id("doc-2", i) for i in chunks]

        # No overlap between documents
        assert set(ids_doc1).isdisjoint(set(ids_doc2))


class TestIndexVectorEdgeCases:
    """Tests for edge cases in vector indexing."""

    def test_empty_chunks(self):
        """Test with empty chunk list."""
        input_data = IndexVectorInput(
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            chunks=[],
            embeddings=[],
        )

        assert len(input_data.chunks) == 0

    def test_missing_embedding_for_chunk(self):
        """Test when some chunks don't have embeddings."""
        chunks = [
            ChunkInfo(
                sequence_number=i,
                content=f"Chunk {i}",
                content_hash=f"hash{i}",
                start_char=0,
                end_char=10,
                token_count=2,
            )
            for i in range(5)
        ]

        # Only provide embeddings for 3 of 5 chunks
        embeddings = [
            EmbeddingInfo(
                chunk_sequence=i,
                embedding_id=f"emb-{i}",
                model="model",
                dense_vector=[0.1],
                dimensions=1,
            )
            for i in [0, 2, 4]  # Missing 1 and 3
        ]

        input_data = IndexVectorInput(
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            chunks=chunks,
            embeddings=embeddings,
        )

        # Build embedding map like the activity does
        embedding_map = {e.chunk_sequence: e for e in input_data.embeddings}

        # Should find missing embeddings
        missing = [c.sequence_number for c in chunks if c.sequence_number not in embedding_map]
        assert missing == [1, 3]

    def test_chunk_without_optional_fields(self):
        """Test chunk with minimal fields."""
        chunk = ChunkInfo(
            sequence_number=0,
            content="Test",
            content_hash="hash",
            start_char=0,
            end_char=4,
            token_count=1,
        )

        embedding = EmbeddingInfo(
            chunk_sequence=0,
            embedding_id="emb-0",
            model="model",
            dense_vector=[0.1],
            dimensions=1,
        )

        payload = _build_point_payload(
            chunk=chunk,
            embedding=embedding,
            document_id="doc",
            tenant_id="tenant",
            project_id=None,
        )

        # Required fields should be present
        assert payload["document_id"] == "doc"
        assert payload["tenant_id"] == "tenant"
        assert payload["content"] == "Test"

        # Optional fields should not be present
        assert "page_number" not in payload
        # heading_context is a list and defaults to empty, so it won't be in payload
        assert "project_ids" not in payload


class TestSparseVectorHandling:
    """Tests for sparse vector handling in indexing."""

    def test_embedding_with_sparse_vector(self):
        """Test embedding that has sparse vector."""
        embedding = EmbeddingInfo(
            chunk_sequence=0,
            embedding_id="emb-0",
            model="model",
            dense_vector=[0.1, 0.2],
            sparse_vector=SparseVector(indices=[10, 20, 30], values=[1.0, 0.5, 0.25]),
            dimensions=2,
        )

        assert embedding.sparse_vector is not None
        assert embedding.sparse_vector.indices == [10, 20, 30]
        assert embedding.sparse_vector.values == [1.0, 0.5, 0.25]

    def test_embedding_without_sparse_vector(self):
        """Test embedding without sparse vector."""
        embedding = EmbeddingInfo(
            chunk_sequence=0,
            embedding_id="emb-0",
            model="model",
            dense_vector=[0.1, 0.2],
            sparse_vector=None,
            dimensions=2,
        )

        assert embedding.sparse_vector is None

    def test_empty_sparse_vector(self):
        """Test embedding with empty sparse vector."""
        embedding = EmbeddingInfo(
            chunk_sequence=0,
            embedding_id="emb-0",
            model="model",
            dense_vector=[0.1, 0.2],
            sparse_vector=SparseVector(indices=[], values=[]),
            dimensions=2,
        )

        assert embedding.sparse_vector is not None
        assert len(embedding.sparse_vector.indices) == 0
