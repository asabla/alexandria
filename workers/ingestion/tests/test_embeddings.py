"""
Unit tests for the embedding generation module.

Tests cover:
- EmbeddingConfig configuration
- EmbeddingClient batch processing
- BM25Tokenizer for sparse vectors
- Retry logic
- Mock embedding generation
"""

import asyncio
import hashlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ingestion_worker.activities.embeddings import (
    EmbeddingConfig,
    EmbeddingClient,
    EmbeddingModel,
    EmbeddingResult,
    BatchEmbeddingResult,
    DenseEmbedding,
    SparseEmbedding,
    BM25Tokenizer,
    EmbeddingError,
    MODEL_DIMENSIONS,
    get_embedding_client,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> EmbeddingConfig:
    """Create a default configuration."""
    return EmbeddingConfig()


@pytest.fixture
def custom_config() -> EmbeddingConfig:
    """Create a custom configuration."""
    return EmbeddingConfig(
        base_url="http://custom:8080/v1",
        api_key="test-key",
        model=EmbeddingModel.BGE_LARGE_EN,
        batch_size=16,
        max_retries=5,
        timeout=30.0,
    )


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for embedding."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables text understanding.",
    ]


@pytest.fixture
def mock_api_response() -> dict:
    """Mock API response for embeddings."""
    return {
        "data": [
            {"embedding": [0.1] * 768, "index": 0},
            {"embedding": [0.2] * 768, "index": 1},
            {"embedding": [0.3] * 768, "index": 2},
        ],
        "model": "bge-base-en-v1.5",
        "usage": {"total_tokens": 30},
    }


# =============================================================================
# Test EmbeddingConfig
# =============================================================================


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        assert config.base_url == "http://localhost:8000/v1"
        assert config.api_key is None
        assert config.model == EmbeddingModel.BGE_BASE_EN
        assert config.batch_size == 32
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.timeout == 60.0
        assert config.generate_sparse is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            base_url="http://custom:9000/v1",
            api_key="my-key",
            model=EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
            batch_size=64,
            max_retries=5,
            generate_sparse=False,
        )
        assert config.base_url == "http://custom:9000/v1"
        assert config.api_key == "my-key"
        assert config.model == EmbeddingModel.TEXT_EMBEDDING_3_SMALL
        assert config.batch_size == 64
        assert config.max_retries == 5
        assert config.generate_sparse is False


# =============================================================================
# Test EmbeddingModel
# =============================================================================


class TestEmbeddingModel:
    """Tests for EmbeddingModel enum."""

    def test_bge_models(self):
        """Test BGE model values."""
        assert EmbeddingModel.BGE_SMALL_EN == "bge-small-en-v1.5"
        assert EmbeddingModel.BGE_BASE_EN == "bge-base-en-v1.5"
        assert EmbeddingModel.BGE_LARGE_EN == "bge-large-en-v1.5"

    def test_openai_models(self):
        """Test OpenAI model values."""
        assert EmbeddingModel.TEXT_EMBEDDING_3_SMALL == "text-embedding-3-small"
        assert EmbeddingModel.TEXT_EMBEDDING_ADA == "text-embedding-ada-002"

    def test_model_dimensions(self):
        """Test model dimension mapping."""
        assert MODEL_DIMENSIONS[EmbeddingModel.BGE_SMALL_EN] == 384
        assert MODEL_DIMENSIONS[EmbeddingModel.BGE_BASE_EN] == 768
        assert MODEL_DIMENSIONS[EmbeddingModel.BGE_LARGE_EN] == 1024
        assert MODEL_DIMENSIONS[EmbeddingModel.TEXT_EMBEDDING_3_SMALL] == 1536


# =============================================================================
# Test BM25Tokenizer
# =============================================================================


class TestBM25Tokenizer:
    """Tests for BM25Tokenizer."""

    def test_basic_tokenization(self):
        """Test basic text tokenization."""
        tokenizer = BM25Tokenizer()
        result = tokenizer.tokenize("hello world")

        assert isinstance(result, SparseEmbedding)
        assert len(result.indices) == 2
        assert len(result.values) == 2
        assert "hello" in result.tokens
        assert "world" in result.tokens

    def test_stopword_filtering(self):
        """Test that stopwords are filtered."""
        tokenizer = BM25Tokenizer()
        result = tokenizer.tokenize("the quick brown fox")

        # "the" should be filtered
        assert "the" not in result.tokens
        assert "quick" in result.tokens
        assert "brown" in result.tokens
        assert "fox" in result.tokens

    def test_short_word_filtering(self):
        """Test that single-character words are filtered."""
        tokenizer = BM25Tokenizer()
        result = tokenizer.tokenize("I am a test")

        # Single-char words should be filtered
        assert "i" not in result.tokens
        assert "a" not in result.tokens

    def test_term_frequency_weights(self):
        """Test that repeated terms get higher weights."""
        tokenizer = BM25Tokenizer()
        result = tokenizer.tokenize("test test test unique")

        # Find indices for test and unique
        test_idx = result.tokens.index("test") if "test" in result.tokens else -1
        unique_idx = result.tokens.index("unique") if "unique" in result.tokens else -1

        if test_idx >= 0 and unique_idx >= 0:
            # "test" appears 3 times, should have higher TF weight
            assert result.values[test_idx] > result.values[unique_idx]

    def test_empty_text(self):
        """Test tokenization of empty text."""
        tokenizer = BM25Tokenizer()
        result = tokenizer.tokenize("")

        assert result.indices == []
        assert result.values == []
        assert result.tokens == []

    def test_only_stopwords(self):
        """Test text with only stopwords."""
        tokenizer = BM25Tokenizer()
        result = tokenizer.tokenize("the a an is are")

        assert result.indices == []
        assert result.values == []

    def test_consistent_hashing(self):
        """Test that same token always gets same index."""
        tokenizer = BM25Tokenizer()
        result1 = tokenizer.tokenize("hello world")
        result2 = tokenizer.tokenize("hello universe")

        # "hello" should have same index in both
        idx1 = result1.indices[result1.tokens.index("hello")]
        idx2 = result2.indices[result2.tokens.index("hello")]
        assert idx1 == idx2

    def test_vocab_size_limit(self):
        """Test that indices are within vocab size."""
        tokenizer = BM25Tokenizer(vocab_size=1000)
        result = tokenizer.tokenize("quick brown fox jumps over lazy dog")

        for idx in result.indices:
            assert 0 <= idx < 1000


# =============================================================================
# Test DenseEmbedding
# =============================================================================


class TestDenseEmbedding:
    """Tests for DenseEmbedding dataclass."""

    def test_basic_embedding(self):
        """Test creating a dense embedding."""
        embedding = DenseEmbedding(
            vector=[0.1, 0.2, 0.3],
            model="test-model",
            dimensions=3,
        )
        assert embedding.vector == [0.1, 0.2, 0.3]
        assert embedding.model == "test-model"
        assert embedding.dimensions == 3


# =============================================================================
# Test SparseEmbedding
# =============================================================================


class TestSparseEmbedding:
    """Tests for SparseEmbedding dataclass."""

    def test_basic_sparse(self):
        """Test creating a sparse embedding."""
        sparse = SparseEmbedding(
            indices=[1, 5, 10],
            values=[0.5, 0.3, 0.2],
            tokens=["word1", "word2", "word3"],
        )
        assert sparse.indices == [1, 5, 10]
        assert sparse.values == [0.5, 0.3, 0.2]
        assert len(sparse.tokens) == 3

    def test_default_tokens(self):
        """Test that tokens default to empty list."""
        sparse = SparseEmbedding(indices=[1], values=[0.5])
        assert sparse.tokens == []


# =============================================================================
# Test EmbeddingResult
# =============================================================================


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_basic_result(self):
        """Test creating an embedding result."""
        dense = DenseEmbedding(vector=[0.1] * 768, model="test", dimensions=768)
        result = EmbeddingResult(
            text_hash="abc123",
            dense=dense,
        )
        assert result.text_hash == "abc123"
        assert result.dense.dimensions == 768
        assert result.sparse is None
        assert result.token_count == 0

    def test_result_with_sparse(self):
        """Test result with sparse embedding."""
        dense = DenseEmbedding(vector=[0.1] * 768, model="test", dimensions=768)
        sparse = SparseEmbedding(indices=[1, 2], values=[0.5, 0.5])
        result = EmbeddingResult(
            text_hash="abc123",
            dense=dense,
            sparse=sparse,
            token_count=10,
        )
        assert result.sparse is not None
        assert result.token_count == 10


# =============================================================================
# Test EmbeddingClient
# =============================================================================


class TestEmbeddingClient:
    """Tests for EmbeddingClient."""

    def test_client_initialization(self):
        """Test client initialization."""
        config = EmbeddingConfig(base_url="http://test:8000/v1")
        client = EmbeddingClient(config)

        assert client.config.base_url == "http://test:8000/v1"
        assert client._client is None  # Not initialized yet

    def test_get_model_dimensions(self):
        """Test getting model dimensions."""
        config = EmbeddingConfig(model=EmbeddingModel.BGE_LARGE_EN)
        client = EmbeddingClient(config)

        assert client.get_model_dimensions() == 1024

    def test_get_model_dimensions_unknown(self):
        """Test getting dimensions for unknown model."""
        config = EmbeddingConfig(model="unknown-model")
        client = EmbeddingClient(config)

        # Should return default
        assert client.get_model_dimensions() == 768

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        """Test embedding empty text list."""
        config = EmbeddingConfig()
        client = EmbeddingClient(config)

        result = await client.embed_texts([])

        assert result.embeddings == []
        assert result.total_tokens == 0
        assert result.batch_count == 0

    @pytest.mark.asyncio
    async def test_embed_texts_with_mock(self, sample_texts, mock_api_response):
        """Test embedding texts with mocked API."""
        config = EmbeddingConfig(generate_sparse=True)

        with patch("httpx.AsyncClient") as mock_client_class:
            # Set up mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_api_response

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.is_closed = False
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            async with EmbeddingClient(config) as client:
                client._client = mock_client
                result = await client.embed_texts(sample_texts)

            assert len(result.embeddings) == 3
            assert result.model == config.model
            assert all(e.dense.dimensions == 768 for e in result.embeddings)
            # Should have sparse vectors
            assert all(e.sparse is not None for e in result.embeddings)

    @pytest.mark.asyncio
    async def test_embed_single_text_with_mock(self, mock_api_response):
        """Test embedding single text."""
        config = EmbeddingConfig()
        single_response = {
            "data": [{"embedding": [0.1] * 768, "index": 0}],
            "model": "bge-base-en-v1.5",
            "usage": {"total_tokens": 5},
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = single_response

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.is_closed = False
            mock_client_class.return_value = mock_client

            async with EmbeddingClient(config) as client:
                client._client = mock_client
                result = await client.embed_text("Hello world")

            assert isinstance(result, EmbeddingResult)
            assert result.dense.dimensions == 768

    @pytest.mark.asyncio
    async def test_batch_processing(self, mock_api_response):
        """Test that texts are processed in batches."""
        config = EmbeddingConfig(batch_size=2)

        # 5 texts should result in 3 batches (2+2+1)
        texts = ["text1", "text2", "text3", "text4", "text5"]
        batch_counts = []

        def on_batch(batch_num, total, processed):
            batch_counts.append(batch_num)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200

            def make_response(batch_size):
                return {
                    "data": [{"embedding": [0.1] * 768, "index": i} for i in range(batch_size)],
                    "model": "bge-base-en-v1.5",
                    "usage": {"total_tokens": batch_size * 5},
                }

            # Return appropriate response based on batch size
            responses = [make_response(2), make_response(2), make_response(1)]
            mock_response.json.side_effect = responses

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.is_closed = False
            mock_client_class.return_value = mock_client

            async with EmbeddingClient(config) as client:
                client._client = mock_client
                result = await client.embed_texts(texts, on_batch_complete=on_batch)

            assert result.batch_count == 3
            assert len(batch_counts) == 3

    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Test handling of API errors."""
        config = EmbeddingConfig(max_retries=1, retry_delay=0.01)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.is_closed = False
            mock_client_class.return_value = mock_client

            async with EmbeddingClient(config) as client:
                client._client = mock_client
                with pytest.raises(EmbeddingError):
                    await client.embed_text("test")

    @pytest.mark.asyncio
    async def test_client_error_no_retry(self):
        """Test that 4xx errors don't trigger retries."""
        config = EmbeddingConfig(max_retries=3, retry_delay=0.01)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad Request"

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.is_closed = False
            mock_client_class.return_value = mock_client

            async with EmbeddingClient(config) as client:
                client._client = mock_client
                with pytest.raises(EmbeddingError) as exc_info:
                    await client.embed_text("test")

                assert exc_info.value.status_code == 400


# =============================================================================
# Test get_embedding_client
# =============================================================================


class TestGetEmbeddingClient:
    """Tests for get_embedding_client factory function."""

    def test_default_client(self):
        """Test creating client with defaults."""
        client = get_embedding_client()

        assert client.config.base_url == "http://localhost:8000/v1"
        assert client.config.model == EmbeddingModel.BGE_BASE_EN

    def test_client_with_overrides(self):
        """Test creating client with overrides."""
        client = get_embedding_client(
            base_url="http://custom:9000/v1",
            model="custom-model",
        )

        assert client.config.base_url == "http://custom:9000/v1"
        assert client.config.model == "custom-model"

    def test_client_from_env(self):
        """Test creating client from environment variables."""
        import os

        with patch.dict(
            os.environ,
            {
                "VLLM_BASE_URL": "http://env:8080/v1",
                "EMBEDDING_MODEL": "env-model",
            },
        ):
            client = get_embedding_client()

            assert client.config.base_url == "http://env:8080/v1"
            assert client.config.model == "env-model"


# =============================================================================
# Test EmbeddingError
# =============================================================================


class TestEmbeddingError:
    """Tests for EmbeddingError exception."""

    def test_basic_error(self):
        """Test creating basic error."""
        error = EmbeddingError("Test error")
        assert str(error) == "Test error"
        assert error.status_code is None

    def test_error_with_status(self):
        """Test error with status code."""
        error = EmbeddingError("Server error", status_code=500)
        assert error.status_code == 500


# =============================================================================
# Test Integration (without real API)
# =============================================================================


class TestIntegration:
    """Integration tests without real API calls."""

    @pytest.mark.asyncio
    async def test_full_embedding_pipeline(self, sample_texts, mock_api_response):
        """Test full pipeline from config to results."""
        config = EmbeddingConfig(
            model=EmbeddingModel.BGE_BASE_EN,
            batch_size=10,
            generate_sparse=True,
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_api_response

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.is_closed = False
            mock_client_class.return_value = mock_client

            async with EmbeddingClient(config) as client:
                client._client = mock_client
                result = await client.embed_texts(sample_texts)

            # Verify complete result structure
            assert isinstance(result, BatchEmbeddingResult)
            assert len(result.embeddings) == len(sample_texts)
            assert result.model == config.model

            for emb in result.embeddings:
                # Check dense embedding
                assert isinstance(emb.dense, DenseEmbedding)
                assert len(emb.dense.vector) == 768

                # Check sparse embedding
                assert isinstance(emb.sparse, SparseEmbedding)
                assert len(emb.sparse.indices) > 0

                # Check hash
                assert len(emb.text_hash) == 16

    def test_text_hash_deterministic(self):
        """Test that text hashes are deterministic."""
        text = "Hello, world!"
        expected_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Same text should always produce same hash
        hash1 = hashlib.sha256(text.encode()).hexdigest()[:16]
        hash2 = hashlib.sha256(text.encode()).hexdigest()[:16]

        assert hash1 == hash2 == expected_hash
