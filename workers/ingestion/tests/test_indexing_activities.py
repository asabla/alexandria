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


# ============================================================================
# MeiliSearch Indexing Activity Tests
# ============================================================================

from ingestion_worker.activities.indexing_activities import (
    index_fulltext,
    index_fulltext_mock,
    _generate_meili_doc_id,
    _build_meili_document,
)
from ingestion_worker.workflows.document_ingestion import (
    IndexFulltextInput,
    IndexFulltextOutput,
)


class TestGenerateMeiliDocId:
    """Tests for deterministic MeiliSearch document ID generation."""

    def test_deterministic_id_same_input(self):
        """Same document_id and chunk_sequence should produce same ID."""
        doc_id = "doc-123"
        chunk_seq = 5

        id1 = _generate_meili_doc_id(doc_id, chunk_seq)
        id2 = _generate_meili_doc_id(doc_id, chunk_seq)

        assert id1 == id2

    def test_deterministic_id_different_chunks(self):
        """Different chunk sequences should produce different IDs."""
        doc_id = "doc-123"

        id1 = _generate_meili_doc_id(doc_id, 0)
        id2 = _generate_meili_doc_id(doc_id, 1)

        assert id1 != id2

    def test_deterministic_id_different_documents(self):
        """Different documents should produce different IDs."""
        chunk_seq = 5

        id1 = _generate_meili_doc_id("doc-123", chunk_seq)
        id2 = _generate_meili_doc_id("doc-456", chunk_seq)

        assert id1 != id2

    def test_id_format(self):
        """Generated ID should follow expected format."""
        doc_id = "doc-123"
        chunk_seq = 0

        meili_id = _generate_meili_doc_id(doc_id, chunk_seq)

        assert meili_id == "doc-123_chunk_0"

    def test_id_format_with_uuid_document_id(self):
        """ID should work with UUID-style document IDs."""
        doc_id = "550e8400-e29b-41d4-a716-446655440000"
        chunk_seq = 42

        meili_id = _generate_meili_doc_id(doc_id, chunk_seq)

        assert meili_id == f"{doc_id}_chunk_42"


class TestBuildMeiliDocument:
    """Tests for building MeiliSearch documents from chunks."""

    @pytest.fixture
    def sample_chunk(self) -> ChunkInfo:
        """Create a sample chunk for testing."""
        return ChunkInfo(
            sequence_number=0,
            content="This is test content for MeiliSearch indexing.",
            content_hash="abc123",
            start_char=0,
            end_char=47,
            token_count=10,
            page_number=1,
            heading_context=["Chapter 1", "Section A"],
            chunk_type="paragraph",
            metadata={"source": "test"},
        )

    @pytest.fixture
    def sample_metadata(self) -> dict:
        """Create sample metadata for testing."""
        return {
            "language": "en",
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T11:00:00Z",
            "description": "Test document description",
            "summary": "Test summary",
            "entities": ["John Doe", "Acme Corp"],
            "entity_types": ["PERSON", "ORG"],
            "status": "processed",
        }

    def test_basic_document_fields(self, sample_chunk):
        """Test that basic fields are included in document."""
        doc = _build_meili_document(
            chunk=sample_chunk,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Test Document",
            document_type="pdf",
            metadata={},
        )

        assert doc["id"] == "doc-123_chunk_0"
        assert doc["title"] == "Test Document"
        assert doc["content"] == "This is test content for MeiliSearch indexing."
        assert doc["tenant_id"] == "tenant-abc"
        assert doc["document_type"] == "pdf"
        assert doc["document_id"] == "doc-123"
        assert doc["chunk_sequence"] == 0
        assert doc["chunk_type"] == "paragraph"

    def test_document_with_project_id(self, sample_chunk):
        """Test that project_id is included as list."""
        doc = _build_meili_document(
            chunk=sample_chunk,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id="project-xyz",
            title="Test Document",
            document_type="pdf",
            metadata={},
        )

        assert doc["project_ids"] == ["project-xyz"]

    def test_document_without_project_id(self, sample_chunk):
        """Test that project_ids is empty list when no project."""
        doc = _build_meili_document(
            chunk=sample_chunk,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Test Document",
            document_type="pdf",
            metadata={},
        )

        assert doc["project_ids"] == []

    def test_document_with_heading_context(self, sample_chunk):
        """Test that heading_context is joined with separator."""
        doc = _build_meili_document(
            chunk=sample_chunk,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Test Document",
            document_type="pdf",
            metadata={},
        )

        assert doc["heading_context"] == "Chapter 1 > Section A"

    def test_document_without_heading_context(self):
        """Test document when heading_context is empty."""
        chunk = ChunkInfo(
            sequence_number=0,
            content="Test",
            content_hash="hash",
            start_char=0,
            end_char=4,
            token_count=1,
            heading_context=[],
        )

        doc = _build_meili_document(
            chunk=chunk,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Test Document",
            document_type="pdf",
            metadata={},
        )

        assert "heading_context" not in doc

    def test_document_with_all_metadata_fields(self, sample_chunk, sample_metadata):
        """Test that all metadata fields are included."""
        doc = _build_meili_document(
            chunk=sample_chunk,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Test Document",
            document_type="pdf",
            metadata=sample_metadata,
        )

        assert doc["language"] == "en"
        assert doc["created_at"] == "2024-01-15T10:30:00Z"
        assert doc["updated_at"] == "2024-01-15T11:00:00Z"
        assert doc["description"] == "Test document description"
        assert doc["summary"] == "Test summary"
        assert doc["entities"] == ["John Doe", "Acme Corp"]
        assert doc["entity_types"] == ["PERSON", "ORG"]
        assert doc["status"] == "processed"

    def test_document_without_optional_metadata(self, sample_chunk):
        """Test document with empty metadata."""
        doc = _build_meili_document(
            chunk=sample_chunk,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Test Document",
            document_type="pdf",
            metadata={},
        )

        # Optional fields should not be present
        assert "language" not in doc
        assert "created_at" not in doc
        assert "description" not in doc
        assert "summary" not in doc
        assert "entities" not in doc
        assert "entity_types" not in doc
        assert "status" not in doc


class TestIndexFulltextMock:
    """Tests for mock fulltext indexing activity."""

    @pytest.fixture
    def sample_chunks(self) -> list[ChunkInfo]:
        """Create sample chunks for testing."""
        return [
            ChunkInfo(
                sequence_number=i,
                content=f"Chunk {i} content for testing",
                content_hash=f"hash{i}",
                start_char=i * 100,
                end_char=(i + 1) * 100,
                token_count=10,
            )
            for i in range(5)
        ]

    @pytest.fixture
    def sample_input(self, sample_chunks) -> IndexFulltextInput:
        """Create sample input for testing."""
        return IndexFulltextInput(
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id="project-xyz",
            title="Test Document",
            document_type="pdf",
            chunks=sample_chunks,
            metadata={"language": "en"},
        )

    @pytest.mark.asyncio
    async def test_mock_returns_correct_count(self, sample_input):
        """Test that mock activity returns correct indexed count."""
        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                result = await index_fulltext_mock(sample_input)

        assert isinstance(result, IndexFulltextOutput)
        assert result.indexed_count == 5

    @pytest.mark.asyncio
    async def test_mock_returns_indexed_true(self, sample_input):
        """Test that mock activity returns indexed=True."""
        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                result = await index_fulltext_mock(sample_input)

        assert result.indexed is True

    @pytest.mark.asyncio
    async def test_mock_returns_index_name(self, sample_input):
        """Test that mock activity returns index name."""
        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                result = await index_fulltext_mock(sample_input)

        assert result.index_name == "documents"

    @pytest.mark.asyncio
    async def test_mock_with_custom_index_name(self, sample_input):
        """Test mock activity with custom index from env."""
        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                with patch.dict("os.environ", {"MEILISEARCH_INDEX": "custom_index"}):
                    result = await index_fulltext_mock(sample_input)

        assert result.index_name == "custom_index"

    @pytest.mark.asyncio
    async def test_mock_with_empty_chunks(self):
        """Test mock activity with empty chunk list."""
        input_data = IndexFulltextInput(
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Empty Document",
            document_type="pdf",
            chunks=[],
            metadata={},
        )

        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                result = await index_fulltext_mock(input_data)

        assert result.indexed_count == 0
        assert result.indexed is True


class TestIndexFulltext:
    """Tests for the main fulltext indexing activity."""

    @pytest.fixture
    def sample_chunks(self) -> list[ChunkInfo]:
        """Create sample chunks for testing."""
        return [
            ChunkInfo(
                sequence_number=i,
                content=f"Chunk {i} content for testing MeiliSearch",
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

    @pytest.fixture
    def sample_input(self, sample_chunks) -> IndexFulltextInput:
        """Create sample input for testing."""
        return IndexFulltextInput(
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id="project-xyz",
            title="Test Document",
            document_type="pdf",
            chunks=sample_chunks,
            metadata={"language": "en", "status": "processed"},
        )

    @pytest.fixture
    def mock_meilisearch_client(self):
        """Create a mock MeiliSearchClient."""
        client = MagicMock()
        client.ensure_index = AsyncMock(return_value=False)
        client.index_documents = AsyncMock(return_value=3)
        return client

    @pytest.mark.asyncio
    async def test_index_fulltext_success(self, sample_input, mock_meilisearch_client):
        """Test successful fulltext indexing with mocked client."""
        mock_clients_module = MagicMock()
        mock_clients_module.MeiliSearchClient = MagicMock(return_value=mock_meilisearch_client)

        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                with patch.dict("sys.modules", {"alexandria_db.clients": mock_clients_module}):
                    result = await index_fulltext(sample_input)

        assert isinstance(result, IndexFulltextOutput)
        assert result.indexed is True
        assert result.indexed_count == 3
        assert result.index_name == "documents"
        mock_meilisearch_client.ensure_index.assert_called_once()
        mock_meilisearch_client.index_documents.assert_called()

    @pytest.mark.asyncio
    async def test_index_fulltext_creates_index(self, sample_input, mock_meilisearch_client):
        """Test that index is created if it doesn't exist."""
        mock_meilisearch_client.ensure_index = AsyncMock(return_value=True)

        mock_clients_module = MagicMock()
        mock_clients_module.MeiliSearchClient = MagicMock(return_value=mock_meilisearch_client)

        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                with patch.dict("sys.modules", {"alexandria_db.clients": mock_clients_module}):
                    await index_fulltext(sample_input)

        mock_meilisearch_client.ensure_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_fulltext_custom_config(self, sample_input, mock_meilisearch_client):
        """Test activity respects environment configuration."""
        mock_clients_module = MagicMock()
        mock_client_class = MagicMock(return_value=mock_meilisearch_client)
        mock_clients_module.MeiliSearchClient = mock_client_class

        env_vars = {
            "MEILISEARCH_URL": "http://custom:7700",
            "MEILISEARCH_API_KEY": "custom_key",
            "MEILISEARCH_INDEX": "custom_index",
        }

        with patch("temporalio.activity.heartbeat"):
            with patch("temporalio.activity.logger"):
                with patch.dict("sys.modules", {"alexandria_db.clients": mock_clients_module}):
                    with patch.dict("os.environ", env_vars):
                        result = await index_fulltext(sample_input)

        # Verify client was created with custom config
        mock_client_class.assert_called_once_with(
            url="http://custom:7700",
            api_key="custom_key",
            index="custom_index",
        )
        assert result.index_name == "custom_index"


class TestIndexFulltextIdempotency:
    """Tests for idempotent fulltext indexing behavior."""

    def test_same_document_same_ids(self):
        """Re-indexing same document should produce same document IDs."""
        doc_id = "doc-123"
        chunks = range(5)

        # First indexing
        ids_first = [_generate_meili_doc_id(doc_id, i) for i in chunks]

        # Second indexing (simulating re-index)
        ids_second = [_generate_meili_doc_id(doc_id, i) for i in chunks]

        assert ids_first == ids_second

    def test_different_documents_different_ids(self):
        """Different documents should have different document IDs."""
        chunks = range(3)

        ids_doc1 = [_generate_meili_doc_id("doc-1", i) for i in chunks]
        ids_doc2 = [_generate_meili_doc_id("doc-2", i) for i in chunks]

        # No overlap between documents
        assert set(ids_doc1).isdisjoint(set(ids_doc2))


class TestIndexFulltextEdgeCases:
    """Tests for edge cases in fulltext indexing."""

    def test_empty_chunks_list(self):
        """Test with empty chunk list."""
        input_data = IndexFulltextInput(
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Empty Document",
            document_type="pdf",
            chunks=[],
            metadata={},
        )

        assert len(input_data.chunks) == 0

    def test_single_chunk(self):
        """Test with single chunk."""
        chunk = ChunkInfo(
            sequence_number=0,
            content="Single chunk content",
            content_hash="hash",
            start_char=0,
            end_char=20,
            token_count=4,
        )

        input_data = IndexFulltextInput(
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Single Chunk Document",
            document_type="pdf",
            chunks=[chunk],
            metadata={},
        )

        assert len(input_data.chunks) == 1

    def test_chunk_with_special_characters(self):
        """Test chunk with special characters in content."""
        chunk = ChunkInfo(
            sequence_number=0,
            content="Content with <special> & \"characters\" 'quotes' © ® ™",
            content_hash="hash",
            start_char=0,
            end_char=50,
            token_count=8,
        )

        doc = _build_meili_document(
            chunk=chunk,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Special Characters",
            document_type="pdf",
            metadata={},
        )

        assert doc["content"] == "Content with <special> & \"characters\" 'quotes' © ® ™"

    def test_chunk_with_unicode_content(self):
        """Test chunk with Unicode content."""
        chunk = ChunkInfo(
            sequence_number=0,
            content="日本語テキスト 中文文本 العربية",
            content_hash="hash",
            start_char=0,
            end_char=30,
            token_count=5,
        )

        doc = _build_meili_document(
            chunk=chunk,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Unicode Document",
            document_type="pdf",
            metadata={},
        )

        assert doc["content"] == "日本語テキスト 中文文本 العربية"

    def test_chunk_with_minimal_fields(self):
        """Test chunk with only required fields."""
        chunk = ChunkInfo(
            sequence_number=0,
            content="Minimal content",
            content_hash="hash",
            start_char=0,
            end_char=15,
            token_count=2,
        )

        doc = _build_meili_document(
            chunk=chunk,
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Minimal",
            document_type="txt",
            metadata={},
        )

        # Required fields present
        assert doc["id"] == "doc-123_chunk_0"
        assert doc["content"] == "Minimal content"
        assert doc["tenant_id"] == "tenant-abc"
        assert doc["chunk_type"] == "text"  # Default value

    def test_large_batch_chunking(self):
        """Test that large number of chunks can be handled."""
        chunks = [
            ChunkInfo(
                sequence_number=i,
                content=f"Chunk {i} " * 50,  # ~300 chars each
                content_hash=f"hash{i}",
                start_char=i * 300,
                end_char=(i + 1) * 300,
                token_count=50,
            )
            for i in range(200)  # 200 chunks
        ]

        input_data = IndexFulltextInput(
            document_id="doc-123",
            tenant_id="tenant-abc",
            project_id=None,
            title="Large Document",
            document_type="pdf",
            chunks=chunks,
            metadata={},
        )

        assert len(input_data.chunks) == 200

        # Build all documents
        documents = [
            _build_meili_document(
                chunk=chunk,
                document_id="doc-123",
                tenant_id="tenant-abc",
                project_id=None,
                title="Large Document",
                document_type="pdf",
                metadata={},
            )
            for chunk in chunks
        ]

        assert len(documents) == 200
        # Each document should have unique ID
        ids = [d["id"] for d in documents]
        assert len(set(ids)) == 200
