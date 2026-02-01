"""
Embedding generation module.

This module provides embedding generation capabilities using OpenAI-compatible APIs
(such as vLLM, OpenAI, or other compatible services).

Features:
- Dense embeddings via BGE or other models
- Sparse vectors via BM25 tokenization for hybrid search
- Batch processing with configurable batch sizes
- Retry logic with exponential backoff
- Async HTTP client for efficient API calls

The implementation is designed to work within Temporal activities with proper
heartbeat support for long-running operations.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable

# Lazy import httpx to avoid Temporal sandbox issues
# httpx imports urllib.request which is blocked in workflow sandboxes
if TYPE_CHECKING:
    import httpx


class EmbeddingModel(StrEnum):
    """Supported embedding models."""

    # BGE models (BAAI)
    BGE_SMALL_EN = "bge-small-en-v1.5"
    BGE_BASE_EN = "bge-base-en-v1.5"
    BGE_LARGE_EN = "bge-large-en-v1.5"

    # OpenAI models
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA = "text-embedding-ada-002"

    # Sentence Transformers
    ALL_MINILM_L6 = "all-MiniLM-L6-v2"


# Model dimension mapping
MODEL_DIMENSIONS: dict[str, int] = {
    EmbeddingModel.BGE_SMALL_EN: 384,
    EmbeddingModel.BGE_BASE_EN: 768,
    EmbeddingModel.BGE_LARGE_EN: 1024,
    EmbeddingModel.TEXT_EMBEDDING_3_SMALL: 1536,
    EmbeddingModel.TEXT_EMBEDDING_3_LARGE: 3072,
    EmbeddingModel.TEXT_EMBEDDING_ADA: 1536,
    EmbeddingModel.ALL_MINILM_L6: 384,
}


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation.

    Attributes:
        base_url: Base URL for the embedding API (OpenAI-compatible)
        api_key: API key for authentication (optional for local vLLM)
        model: Model name to use for embeddings
        batch_size: Number of texts to embed in a single API call
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)
        timeout: Request timeout in seconds
        generate_sparse: Whether to generate sparse vectors (BM25)
    """

    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    model: str = EmbeddingModel.BGE_BASE_EN
    batch_size: int = 32
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0
    generate_sparse: bool = True


@dataclass
class DenseEmbedding:
    """A dense embedding vector.

    Attributes:
        vector: The embedding vector
        model: Model used to generate the embedding
        dimensions: Number of dimensions
    """

    vector: list[float]
    model: str
    dimensions: int


@dataclass
class SparseEmbedding:
    """A sparse embedding (BM25-style).

    Uses term frequency representation for hybrid search.

    Attributes:
        indices: Token indices (vocabulary positions)
        values: Token weights (TF-IDF style)
        tokens: Original tokens (for debugging)
    """

    indices: list[int]
    values: list[float]
    tokens: list[str] = field(default_factory=list)


@dataclass
class EmbeddingResult:
    """Result of embedding a single text.

    Attributes:
        text_hash: Hash of the input text for deduplication
        dense: Dense embedding vector
        sparse: Sparse embedding (optional, for hybrid search)
        token_count: Number of tokens in the input
    """

    text_hash: str
    dense: DenseEmbedding
    sparse: SparseEmbedding | None = None
    token_count: int = 0


@dataclass
class BatchEmbeddingResult:
    """Result of embedding a batch of texts.

    Attributes:
        embeddings: List of embedding results
        model: Model used
        total_tokens: Total tokens processed
        batch_count: Number of batches processed
    """

    embeddings: list[EmbeddingResult]
    model: str
    total_tokens: int
    batch_count: int


class EmbeddingError(Exception):
    """Error during embedding generation."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class EmbeddingClient:
    """Client for generating embeddings via OpenAI-compatible API.

    Supports vLLM, OpenAI, and other compatible embedding services.

    Example:
        >>> config = EmbeddingConfig(base_url="http://localhost:8000/v1")
        >>> client = EmbeddingClient(config)
        >>> result = await client.embed_text("Hello, world!")
        >>> print(f"Embedding dimensions: {result.dense.dimensions}")
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """Initialize the embedding client.

        Args:
            config: Embedding configuration. Uses defaults if not provided.
        """
        self.config = config or EmbeddingConfig()
        self._client: Any = None  # httpx.AsyncClient, typed as Any to avoid import
        self._sparse_tokenizer = BM25Tokenizer()

    async def __aenter__(self) -> "EmbeddingClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> Any:
        """Ensure HTTP client is initialized.

        Returns:
            httpx.AsyncClient instance
        """
        import httpx  # Lazy import to avoid Temporal sandbox issues

        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with dense (and optionally sparse) embedding
        """
        results = await self.embed_texts([text])
        return results.embeddings[0]

    async def embed_texts(
        self,
        texts: list[str],
        on_batch_complete: Callable[[int, int, int], None] | None = None,
    ) -> BatchEmbeddingResult:
        """Embed multiple texts with batching.

        Args:
            texts: List of texts to embed
            on_batch_complete: Optional callback after each batch (for heartbeats)

        Returns:
            BatchEmbeddingResult with all embeddings
        """
        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                model=self.config.model,
                total_tokens=0,
                batch_count=0,
            )

        embeddings: list[EmbeddingResult] = []
        total_tokens = 0
        batch_count = 0

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            batch_results = await self._embed_batch(batch)

            embeddings.extend(batch_results)
            total_tokens += sum(e.token_count for e in batch_results)
            batch_count += 1

            if on_batch_complete:
                on_batch_complete(batch_count, len(texts), len(embeddings))

        return BatchEmbeddingResult(
            embeddings=embeddings,
            model=self.config.model,
            total_tokens=total_tokens,
            batch_count=batch_count,
        )

    async def _embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a batch of texts with retry logic.

        Args:
            texts: Batch of texts to embed

        Returns:
            List of embedding results
        """
        import httpx  # Lazy import to avoid Temporal sandbox issues

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                return await self._call_embedding_api(texts)
            except EmbeddingError as e:
                last_error = e
                # Don't retry on 4xx errors (client errors)
                if e.status_code and 400 <= e.status_code < 500:
                    raise

                # Exponential backoff
                delay = self.config.retry_delay * (2**attempt)
                await asyncio.sleep(delay)
            except httpx.HTTPError as e:
                last_error = e
                delay = self.config.retry_delay * (2**attempt)
                await asyncio.sleep(delay)

        raise EmbeddingError(f"Failed after {self.config.max_retries} attempts: {last_error}")

    async def _call_embedding_api(self, texts: list[str]) -> list[EmbeddingResult]:
        """Call the embedding API.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding results
        """
        client = await self._ensure_client()

        # OpenAI-compatible embedding request
        request_body = {
            "model": self.config.model,
            "input": texts,
            "encoding_format": "float",
        }

        response = await client.post("/embeddings", json=request_body)

        if response.status_code != 200:
            raise EmbeddingError(
                f"Embedding API error: {response.text}",
                status_code=response.status_code,
            )

        data = response.json()
        results: list[EmbeddingResult] = []

        # Extract embeddings from response
        for i, item in enumerate(data.get("data", [])):
            text = texts[i]
            vector = item.get("embedding", [])

            # Create text hash for deduplication
            text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

            # Create dense embedding
            dense = DenseEmbedding(
                vector=vector,
                model=self.config.model,
                dimensions=len(vector),
            )

            # Generate sparse embedding if configured
            sparse = None
            if self.config.generate_sparse:
                sparse = self._sparse_tokenizer.tokenize(text)

            # Estimate token count (from API response if available)
            token_count = 0
            if "usage" in data:
                # Approximate per-text token count
                total_tokens = data["usage"].get("total_tokens", 0)
                token_count = total_tokens // len(texts)

            results.append(
                EmbeddingResult(
                    text_hash=text_hash,
                    dense=dense,
                    sparse=sparse,
                    token_count=token_count,
                )
            )

        return results

    def get_model_dimensions(self) -> int:
        """Get the expected dimensions for the configured model.

        Returns:
            Number of dimensions for the model's embeddings
        """
        return MODEL_DIMENSIONS.get(self.config.model, 768)


class BM25Tokenizer:
    """Simple BM25-style tokenizer for sparse embeddings.

    Creates sparse vectors based on term frequency for hybrid search.
    This is a simplified implementation suitable for most RAG use cases.
    """

    # Simple word tokenization pattern
    WORD_PATTERN = re.compile(r"\b[a-zA-Z0-9]+\b")

    # Common English stopwords to filter
    STOPWORDS = frozenset(
        [
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "what",
            "which",
            "who",
            "whom",
            "when",
            "where",
            "why",
            "how",
        ]
    )

    def __init__(self, vocab_size: int = 30000):
        """Initialize the tokenizer.

        Args:
            vocab_size: Maximum vocabulary size for sparse indices
        """
        self.vocab_size = vocab_size
        # Simple hash-based vocabulary mapping
        self._vocab_cache: dict[str, int] = {}

    def tokenize(self, text: str) -> SparseEmbedding:
        """Tokenize text into a sparse embedding.

        Args:
            text: Text to tokenize

        Returns:
            SparseEmbedding with indices and values
        """
        # Extract words
        words = self.WORD_PATTERN.findall(text.lower())

        # Filter stopwords and short words
        tokens = [w for w in words if w not in self.STOPWORDS and len(w) > 1]

        if not tokens:
            return SparseEmbedding(indices=[], values=[], tokens=[])

        # Count term frequencies
        term_counts = Counter(tokens)

        # Convert to sparse representation
        indices: list[int] = []
        values: list[float] = []
        result_tokens: list[str] = []

        for token, count in term_counts.items():
            # Hash token to vocabulary index
            idx = self._get_vocab_index(token)
            indices.append(idx)

            # TF weight (log-normalized)
            tf = 1 + math.log(count) if count > 0 else 0
            values.append(tf)
            result_tokens.append(token)

        return SparseEmbedding(
            indices=indices,
            values=values,
            tokens=result_tokens,
        )

    def _get_vocab_index(self, token: str) -> int:
        """Get vocabulary index for a token.

        Uses consistent hashing to map tokens to indices.

        Args:
            token: Token to get index for

        Returns:
            Vocabulary index
        """
        if token in self._vocab_cache:
            return self._vocab_cache[token]

        # Hash-based index (consistent across runs)
        hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
        idx = hash_val % self.vocab_size

        # Cache for performance
        if len(self._vocab_cache) < 100000:
            self._vocab_cache[token] = idx

        return idx


def get_embedding_client(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> EmbeddingClient:
    """Create an embedding client from environment configuration.

    Args:
        base_url: Override base URL (defaults to VLLM_BASE_URL env var)
        api_key: Override API key (defaults to OPENAI_API_KEY env var)
        model: Override model (defaults to EMBEDDING_MODEL env var)

    Returns:
        Configured EmbeddingClient
    """
    config = EmbeddingConfig(
        base_url=base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        model=model or os.getenv("EMBEDDING_MODEL", EmbeddingModel.BGE_BASE_EN),
    )
    return EmbeddingClient(config)
