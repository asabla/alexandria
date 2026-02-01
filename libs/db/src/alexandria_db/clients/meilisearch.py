"""MeiliSearch full-text search client wrapper."""

from dataclasses import dataclass
from typing import Any

import structlog
from meilisearch import Client as BaseMeiliSearchClient
from meilisearch.errors import MeilisearchApiError

logger = structlog.get_logger(__name__)


@dataclass
class SearchHit:
    """A single search result."""

    id: str
    score: float | None
    document: dict[str, Any]
    formatted: dict[str, Any] | None = None


@dataclass
class SearchResponse:
    """Response from a search query."""

    hits: list[SearchHit]
    total_hits: int
    processing_time_ms: int
    query: str
    facets: dict[str, dict[str, int]] | None = None


@dataclass
class IndexInfo:
    """Information about a MeiliSearch index."""

    uid: str
    primary_key: str | None
    created_at: str
    updated_at: str
    number_of_documents: int


class MeiliSearchClient:
    """
    Wrapper for MeiliSearch full-text search operations.

    Provides a simplified interface for indexing and searching
    documents with faceted filtering.
    """

    DEFAULT_INDEX = "documents"

    def __init__(
        self,
        url: str,
        api_key: str,
        index: str = DEFAULT_INDEX,
    ):
        """
        Initialize MeiliSearch client.

        Args:
            url: MeiliSearch server URL (e.g., "http://localhost:7700")
            api_key: Master key or API key for authentication
            index: Default index name to use
        """
        self._client = BaseMeiliSearchClient(url, api_key)
        self._url = url
        self._default_index = index
        self._log = logger.bind(service="meilisearch", url=url)

    async def ensure_index(
        self,
        index: str | None = None,
        primary_key: str = "id",
    ) -> bool:
        """
        Ensure an index exists with proper configuration.

        Args:
            index: Index name (uses default if not specified)
            primary_key: Primary key field name

        Returns:
            True if index was created, False if it already existed
        """
        index = index or self._default_index

        try:
            self._client.get_index(index)
            return False
        except MeilisearchApiError as e:
            if e.code == "index_not_found":
                self._client.create_index(index, {"primaryKey": primary_key})
                self._log.info("index_created", index=index, primary_key=primary_key)

                # Configure default settings
                idx = self._client.index(index)

                # Set searchable attributes
                idx.update_searchable_attributes(
                    [
                        "title",
                        "content",
                        "description",
                        "summary",
                        "entities",
                    ]
                )

                # Set filterable attributes for faceted search
                idx.update_filterable_attributes(
                    [
                        "tenant_id",
                        "project_ids",
                        "document_type",
                        "status",
                        "language",
                        "created_at",
                        "entity_types",
                    ]
                )

                # Set sortable attributes
                idx.update_sortable_attributes(
                    [
                        "created_at",
                        "updated_at",
                        "title",
                    ]
                )

                # Configure facets
                idx.update_faceting_settings(
                    {
                        "maxValuesPerFacet": 100,
                    }
                )

                return True
            raise

    async def index_documents(
        self,
        documents: list[dict[str, Any]],
        index: str | None = None,
        wait: bool = True,
    ) -> int:
        """
        Index documents for search.

        Args:
            documents: List of documents to index (must have "id" field)
            index: Target index (uses default if not specified)
            wait: Whether to wait for indexing to complete

        Returns:
            Number of documents indexed
        """
        index_name = index or self._default_index

        try:
            idx = self._client.index(index_name)
            task = idx.add_documents(documents)

            if wait:
                self._client.wait_for_task(task.task_uid)

            self._log.debug(
                "documents_indexed",
                index=index_name,
                count=len(documents),
                task_uid=task.task_uid,
            )
            return len(documents)

        except MeilisearchApiError as e:
            self._log.error(
                "index_documents_failed",
                index=index_name,
                count=len(documents),
                error=str(e),
            )
            raise

    async def search(
        self,
        query: str,
        filter_expr: str | None = None,
        facets: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
        index: str | None = None,
        attributes_to_retrieve: list[str] | None = None,
        attributes_to_highlight: list[str] | None = None,
        sort: list[str] | None = None,
    ) -> SearchResponse:
        """
        Search documents.

        Args:
            query: Search query string
            filter_expr: MeiliSearch filter expression (e.g., "tenant_id = 'xxx'")
            facets: List of facet fields to return counts for
            limit: Maximum results to return
            offset: Results offset for pagination
            index: Index to search (uses default if not specified)
            attributes_to_retrieve: Fields to include in results
            attributes_to_highlight: Fields to highlight matches
            sort: Sort expressions (e.g., ["created_at:desc"])

        Returns:
            SearchResponse with hits and facets
        """
        index_name = index or self._default_index

        try:
            idx = self._client.index(index_name)

            search_params: dict[str, Any] = {
                "limit": limit,
                "offset": offset,
                "showRankingScore": True,
            }

            if filter_expr:
                search_params["filter"] = filter_expr

            if facets:
                search_params["facets"] = facets

            if attributes_to_retrieve:
                search_params["attributesToRetrieve"] = attributes_to_retrieve

            if attributes_to_highlight:
                search_params["attributesToHighlight"] = attributes_to_highlight
                search_params["highlightPreTag"] = "<mark>"
                search_params["highlightPostTag"] = "</mark>"

            if sort:
                search_params["sort"] = sort

            result = idx.search(query, search_params)

            hits = [
                SearchHit(
                    id=str(hit.get("id")),
                    score=hit.get("_rankingScore"),
                    document=hit,
                    formatted=hit.get("_formatted"),
                )
                for hit in result.get("hits", [])
            ]

            return SearchResponse(
                hits=hits,
                total_hits=result.get("estimatedTotalHits", len(hits)),
                processing_time_ms=result.get("processingTimeMs", 0),
                query=query,
                facets=result.get("facetDistribution"),
            )

        except MeilisearchApiError as e:
            self._log.error(
                "search_failed",
                index=index_name,
                query=query,
                error=str(e),
            )
            raise

    async def delete_document(
        self,
        document_id: str,
        index: str | None = None,
        wait: bool = True,
    ) -> None:
        """
        Delete a document from the index.

        Args:
            document_id: ID of document to delete
            index: Target index
            wait: Whether to wait for deletion to complete
        """
        index_name = index or self._default_index

        try:
            idx = self._client.index(index_name)
            task = idx.delete_document(document_id)

            if wait:
                self._client.wait_for_task(task.task_uid)

            self._log.info(
                "document_deleted",
                index=index_name,
                document_id=document_id,
            )

        except MeilisearchApiError as e:
            self._log.error(
                "delete_document_failed",
                index=index_name,
                document_id=document_id,
                error=str(e),
            )
            raise

    async def delete_documents_by_filter(
        self,
        filter_expr: str,
        index: str | None = None,
        wait: bool = True,
    ) -> None:
        """
        Delete documents matching a filter.

        Args:
            filter_expr: MeiliSearch filter expression
            index: Target index
            wait: Whether to wait for deletion to complete
        """
        index_name = index or self._default_index

        try:
            idx = self._client.index(index_name)
            task = idx.delete_documents_by_filter(filter_expr)

            if wait:
                self._client.wait_for_task(task.task_uid)

            self._log.info(
                "documents_deleted_by_filter",
                index=index_name,
                filter=filter_expr,
            )

        except MeilisearchApiError as e:
            self._log.error(
                "delete_by_filter_failed",
                index=index_name,
                filter=filter_expr,
                error=str(e),
            )
            raise

    async def get_document(
        self,
        document_id: str,
        index: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Get a document by ID.

        Args:
            document_id: Document ID
            index: Target index

        Returns:
            Document data or None if not found
        """
        index_name = index or self._default_index

        try:
            idx = self._client.index(index_name)
            return idx.get_document(document_id)
        except MeilisearchApiError as e:
            if e.code == "document_not_found":
                return None
            raise

    async def get_index_info(self, index: str | None = None) -> IndexInfo:
        """
        Get information about an index.

        Args:
            index: Index name

        Returns:
            IndexInfo with stats about the index
        """
        index_name = index or self._default_index

        try:
            idx = self._client.get_index(index_name)
            stats = idx.get_stats()

            return IndexInfo(
                uid=idx.uid,
                primary_key=idx.primary_key,
                created_at=idx.created_at,
                updated_at=idx.updated_at,
                number_of_documents=stats.number_of_documents,
            )

        except MeilisearchApiError as e:
            self._log.error("get_index_info_failed", index=index_name, error=str(e))
            raise

    async def health_check(self) -> bool:
        """
        Check if MeiliSearch is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self._client.health()
            return True
        except Exception as e:
            self._log.warning("health_check_failed", error=str(e))
            return False
