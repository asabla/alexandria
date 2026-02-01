"""Qdrant vector database client wrapper."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import structlog
from qdrant_client import QdrantClient as BaseQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = structlog.get_logger(__name__)


@dataclass
class SparseVector:
    """Sparse vector representation for hybrid search.

    Attributes:
        indices: Non-zero indices in the sparse vector
        values: Values at the corresponding indices
    """

    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)


@dataclass
class HybridPoint:
    """Point with both dense and sparse vectors for hybrid search.

    Attributes:
        id: Unique point ID
        dense_vector: Dense embedding vector
        sparse_vector: Sparse vector for keyword matching (optional)
        payload: Metadata dictionary
    """

    id: str
    dense_vector: list[float]
    sparse_vector: SparseVector | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """Result from a vector search query."""

    id: str
    score: float
    payload: dict[str, Any]


@dataclass
class CollectionInfo:
    """Information about a Qdrant collection."""

    name: str
    vectors_count: int
    points_count: int
    status: str


class QdrantClient:
    """
    Wrapper for Qdrant vector database operations.

    Provides a simplified interface for vector indexing and search,
    with support for hybrid search (dense + sparse vectors).

    Hybrid search combines:
    - Dense vectors: Semantic similarity from embeddings (e.g., BGE)
    - Sparse vectors: Keyword matching via BM25-style tokenization

    Example:
        >>> client = QdrantClient("http://localhost:6333")
        >>> await client.ensure_collection_hybrid(vector_size=768)
        >>> points = [HybridPoint(
        ...     id="chunk-1",
        ...     dense_vector=[0.1, 0.2, ...],
        ...     sparse_vector=SparseVector(indices=[10, 50], values=[1.0, 0.5]),
        ...     payload={"document_id": "doc-1", "content": "..."}
        ... )]
        >>> await client.upsert_hybrid(points)
    """

    DEFAULT_COLLECTION = "documents"
    DEFAULT_VECTOR_SIZE = 768  # BGE-base embedding size
    DEFAULT_SPARSE_VECTOR_NAME = "sparse"
    DEFAULT_DENSE_VECTOR_NAME = "dense"

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        collection: str = DEFAULT_COLLECTION,
    ):
        """
        Initialize Qdrant client.

        Args:
            url: Qdrant server URL (e.g., "http://localhost:6333")
            api_key: Optional API key for authentication
            collection: Default collection name to use
        """
        self._client = BaseQdrantClient(url=url, api_key=api_key)
        self._url = url
        self._default_collection = collection
        self._log = logger.bind(service="qdrant", url=url)

    async def ensure_collection_hybrid(
        self,
        collection: str | None = None,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: str = "Cosine",
        on_disk: bool = False,
        sparse_on_disk: bool = False,
    ) -> bool:
        """
        Ensure a collection exists with hybrid search support (dense + sparse).

        Args:
            collection: Collection name (uses default if not specified)
            vector_size: Size of dense vectors
            distance: Distance metric for dense vectors (Cosine, Euclid, Dot)
            on_disk: Whether to store dense vectors on disk
            sparse_on_disk: Whether to store sparse vectors on disk

        Returns:
            True if collection was created, False if it already existed
        """
        collection = collection or self._default_collection

        try:
            collections = self._client.get_collections()
            existing = [c.name for c in collections.collections]

            if collection in existing:
                self._log.debug("collection_exists", collection=collection)
                return False

            # Create collection with named vectors for hybrid search
            self._client.create_collection(
                collection_name=collection,
                vectors_config={
                    self.DEFAULT_DENSE_VECTOR_NAME: models.VectorParams(
                        size=vector_size,
                        distance=getattr(models.Distance, distance.upper()),
                        on_disk=on_disk,
                    ),
                },
                sparse_vectors_config={
                    self.DEFAULT_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=sparse_on_disk,
                        ),
                    ),
                },
                # Enable quantization for memory efficiency on dense vectors
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            )

            # Create payload indexes for common filters
            self._create_payload_indexes(collection)

            self._log.info(
                "collection_created_hybrid",
                collection=collection,
                vector_size=vector_size,
                distance=distance,
            )
            return True

        except UnexpectedResponse as e:
            self._log.error("collection_ensure_failed", collection=collection, error=str(e))
            raise

    def _create_payload_indexes(self, collection: str) -> None:
        """Create standard payload indexes for filtering.

        Args:
            collection: Collection name
        """
        self._client.create_payload_index(
            collection_name=collection,
            field_name="tenant_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=collection,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=collection,
            field_name="project_ids",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=collection,
            field_name="chunk_sequence",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

    async def ensure_collection(
        self,
        collection: str | None = None,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: str = "Cosine",
        on_disk: bool = False,
    ) -> bool:
        """
        Ensure a collection exists with proper configuration (dense vectors only).

        For hybrid search with sparse vectors, use ensure_collection_hybrid instead.

        Args:
            collection: Collection name (uses default if not specified)
            vector_size: Size of dense vectors
            distance: Distance metric (Cosine, Euclid, Dot)
            on_disk: Whether to store vectors on disk

        Returns:
            True if collection was created, False if it already existed
        """
        collection = collection or self._default_collection

        try:
            collections = self._client.get_collections()
            existing = [c.name for c in collections.collections]

            if collection in existing:
                return False

            # Create collection with dense vectors
            self._client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=getattr(models.Distance, distance.upper()),
                    on_disk=on_disk,
                ),
                # Enable quantization for memory efficiency
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            )

            # Create payload indexes for common filters
            self._create_payload_indexes(collection)

            self._log.info("collection_created", collection=collection, vector_size=vector_size)
            return True

        except UnexpectedResponse as e:
            self._log.error("collection_ensure_failed", collection=collection, error=str(e))
            raise

    async def upsert_hybrid(
        self,
        points: list[HybridPoint],
        collection: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """
        Upsert points with both dense and sparse vectors for hybrid search.

        Args:
            points: List of HybridPoint objects
            collection: Target collection (uses default if not specified)
            batch_size: Number of points per batch

        Returns:
            Number of points upserted
        """
        collection = collection or self._default_collection

        try:
            total_upserted = 0

            # Process in batches
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                qdrant_points = []

                for p in batch:
                    # Build vectors dict with dense
                    vectors: dict[str, Any] = {
                        self.DEFAULT_DENSE_VECTOR_NAME: p.dense_vector,
                    }

                    # Add sparse vector if present
                    if p.sparse_vector and p.sparse_vector.indices:
                        vectors[self.DEFAULT_SPARSE_VECTOR_NAME] = models.SparseVector(
                            indices=p.sparse_vector.indices,
                            values=p.sparse_vector.values,
                        )

                    qdrant_points.append(
                        models.PointStruct(
                            id=p.id,
                            vector=vectors,
                            payload=p.payload,
                        )
                    )

                self._client.upsert(
                    collection_name=collection,
                    points=qdrant_points,
                    wait=True,
                )
                total_upserted += len(batch)

            self._log.debug(
                "hybrid_vectors_upserted",
                collection=collection,
                count=total_upserted,
            )
            return total_upserted

        except UnexpectedResponse as e:
            self._log.error(
                "hybrid_upsert_failed",
                collection=collection,
                count=len(points),
                error=str(e),
            )
            raise

    async def upsert_vectors(
        self,
        points: list[dict[str, Any]],
        collection: str | None = None,
    ) -> int:
        """
        Upsert vectors into the collection.

        Args:
            points: List of points, each containing:
                - id: Unique point ID (UUID or string)
                - vector: Dense vector (list of floats)
                - payload: Metadata dictionary
            collection: Target collection (uses default if not specified)

        Returns:
            Number of points upserted
        """
        collection = collection or self._default_collection

        try:
            qdrant_points = [
                models.PointStruct(
                    id=str(p["id"]) if isinstance(p["id"], UUID) else p["id"],
                    vector=p["vector"],
                    payload=p.get("payload", {}),
                )
                for p in points
            ]

            self._client.upsert(
                collection_name=collection,
                points=qdrant_points,
                wait=True,
            )

            self._log.debug(
                "vectors_upserted",
                collection=collection,
                count=len(points),
            )
            return len(points)

        except UnexpectedResponse as e:
            self._log.error(
                "vectors_upsert_failed",
                collection=collection,
                count=len(points),
                error=str(e),
            )
            raise

    async def search(
        self,
        vector: list[float],
        limit: int = 10,
        filter_conditions: dict[str, Any] | None = None,
        collection: str | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for similar vectors.

        Args:
            vector: Query vector
            limit: Maximum number of results
            filter_conditions: Payload filters (e.g., {"tenant_id": "xxx"})
            collection: Collection to search (uses default if not specified)
            score_threshold: Minimum similarity score

        Returns:
            List of VectorSearchResult sorted by similarity
        """
        collection = collection or self._default_collection

        try:
            # Build filter from conditions
            qdrant_filter = None
            if filter_conditions:
                must_conditions = []
                for key, value in filter_conditions.items():
                    if isinstance(value, list):
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value),
                            )
                        )
                    else:
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=str(value)),
                            )
                        )
                qdrant_filter = models.Filter(must=must_conditions)

            results = self._client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
            )

            return [
                VectorSearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {},
                )
                for r in results
            ]

        except UnexpectedResponse as e:
            self._log.error("search_failed", collection=collection, error=str(e))
            raise

    async def search_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector | None = None,
        limit: int = 10,
        filter_conditions: dict[str, Any] | None = None,
        collection: str | None = None,
        score_threshold: float | None = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[VectorSearchResult]:
        """
        Hybrid search combining dense and sparse vectors.

        Uses Reciprocal Rank Fusion (RRF) to combine results from both
        dense (semantic) and sparse (keyword) searches.

        Args:
            dense_vector: Dense query vector
            sparse_vector: Sparse query vector (optional)
            limit: Maximum number of results
            filter_conditions: Payload filters
            collection: Collection to search
            score_threshold: Minimum similarity score
            dense_weight: Weight for dense vector results (default 0.7)
            sparse_weight: Weight for sparse vector results (default 0.3)

        Returns:
            List of VectorSearchResult sorted by combined score
        """
        collection = collection or self._default_collection

        try:
            # Build filter
            qdrant_filter = self._build_filter(filter_conditions)

            # Prepare prefetch queries
            prefetch = [
                models.Prefetch(
                    query=dense_vector,
                    using=self.DEFAULT_DENSE_VECTOR_NAME,
                    limit=limit * 2,  # Fetch more for fusion
                ),
            ]

            # Add sparse prefetch if provided
            if sparse_vector and sparse_vector.indices:
                prefetch.append(
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vector.indices,
                            values=sparse_vector.values,
                        ),
                        using=self.DEFAULT_SPARSE_VECTOR_NAME,
                        limit=limit * 2,
                    ),
                )

            # Use query with RRF fusion
            results = self._client.query_points(
                collection_name=collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
            )

            return [
                VectorSearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {},
                )
                for r in results.points
            ]

        except UnexpectedResponse as e:
            self._log.error("hybrid_search_failed", collection=collection, error=str(e))
            raise

    def _build_filter(self, filter_conditions: dict[str, Any] | None) -> models.Filter | None:
        """Build Qdrant filter from conditions dict.

        Args:
            filter_conditions: Dict of field -> value conditions

        Returns:
            Qdrant Filter object or None
        """
        if not filter_conditions:
            return None

        must_conditions = []
        for key, value in filter_conditions.items():
            if isinstance(value, list):
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=str(value)),
                    )
                )
        return models.Filter(must=must_conditions)

    async def delete_by_filter(
        self,
        filter_conditions: dict[str, Any],
        collection: str | None = None,
    ) -> int:
        """
        Delete points matching filter conditions.

        Args:
            filter_conditions: Payload filters
            collection: Target collection

        Returns:
            Number of points deleted (approximate)
        """
        collection = collection or self._default_collection

        try:
            must_conditions = []
            for key, value in filter_conditions.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=str(value)),
                    )
                )

            self._client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(filter=models.Filter(must=must_conditions)),
                wait=True,
            )

            self._log.info(
                "points_deleted",
                collection=collection,
                filter=filter_conditions,
            )
            return -1  # Qdrant doesn't return count

        except UnexpectedResponse as e:
            self._log.error(
                "delete_failed",
                collection=collection,
                filter=filter_conditions,
                error=str(e),
            )
            raise

    async def get_collection_info(self, collection: str | None = None) -> CollectionInfo:
        """
        Get information about a collection.

        Args:
            collection: Collection name

        Returns:
            CollectionInfo with stats about the collection
        """
        collection = collection or self._default_collection

        try:
            info = self._client.get_collection(collection)
            return CollectionInfo(
                name=collection,
                vectors_count=info.vectors_count or 0,
                points_count=info.points_count or 0,
                status=info.status.value if info.status else "unknown",
            )
        except UnexpectedResponse as e:
            self._log.error("get_collection_failed", collection=collection, error=str(e))
            raise

    async def health_check(self) -> bool:
        """
        Check if Qdrant is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self._client.get_collections()
            return True
        except Exception as e:
            self._log.warning("health_check_failed", error=str(e))
            return False
