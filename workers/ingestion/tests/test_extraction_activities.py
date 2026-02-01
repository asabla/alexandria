"""
Tests for entity extraction activities.

Tests cover:
- Entity name normalization
- spaCy label mapping
- Entity validation
- Per-chunk extraction
- Mention grouping
- Full activity with mocked spaCy
- Mock activity for testing
- Edge cases
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ingestion_worker.activities.extraction_activities import (
    CONTEXT_WINDOW,
    EXTRACTED_ENTITY_TYPES,
    MAX_ENTITY_LENGTH,
    MIN_ENTITY_LENGTH,
    SPACY_LABEL_TO_ENTITY_TYPE,
    _extract_context,
    _extract_entities_from_chunk,
    _group_entity_mentions,
    _is_valid_entity,
    _map_spacy_label,
    _normalize_entity_name,
)
from ingestion_worker.workflows.document_ingestion import (
    ChunkInfo,
    EntityInfo,
    ExtractEntitiesInput,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_chunk() -> ChunkInfo:
    """Create a sample chunk for testing."""
    return ChunkInfo(
        sequence_number=0,
        content="John Doe works at Acme Corp in New York City.",
        content_hash="abc123",
        start_char=0,
        end_char=45,
        token_count=10,
        page_number=1,
    )


@pytest.fixture
def sample_chunks() -> list[ChunkInfo]:
    """Create multiple sample chunks for testing."""
    return [
        ChunkInfo(
            sequence_number=0,
            content="John Doe is the CEO of Acme Corporation.",
            content_hash="hash1",
            start_char=0,
            end_char=40,
            token_count=9,
            page_number=1,
        ),
        ChunkInfo(
            sequence_number=1,
            content="Mr. Doe founded Acme Corp in 2010.",
            content_hash="hash2",
            start_char=41,
            end_char=74,
            token_count=8,
            page_number=1,
        ),
        ChunkInfo(
            sequence_number=2,
            content="The company is headquartered in San Francisco.",
            content_hash="hash3",
            start_char=75,
            end_char=120,
            token_count=7,
            page_number=2,
        ),
    ]


@pytest.fixture
def mock_spacy_doc():
    """Create a mock spaCy Doc with entities."""

    @dataclass
    class MockEntity:
        text: str
        label_: str
        start_char: int
        end_char: int

    @dataclass
    class MockDoc:
        ents: list[MockEntity] = field(default_factory=list)

    return MockDoc, MockEntity


# =============================================================================
# Test _normalize_entity_name
# =============================================================================


class TestNormalizeEntityName:
    """Tests for entity name normalization."""

    def test_basic_normalization(self):
        """Test basic whitespace and case normalization."""
        assert _normalize_entity_name("John Doe") == "john doe"
        assert _normalize_entity_name("JOHN DOE") == "john doe"
        assert _normalize_entity_name("john doe") == "john doe"

    def test_whitespace_handling(self):
        """Test handling of various whitespace."""
        assert _normalize_entity_name("  John   Doe  ") == "john doe"
        assert _normalize_entity_name("John\tDoe") == "john doe"
        assert _normalize_entity_name("John\nDoe") == "john doe"

    def test_possessive_removal(self):
        """Test removal of possessive suffixes."""
        assert _normalize_entity_name("John's") == "john"
        assert _normalize_entity_name("Acme Corp's") == "acme corp"
        assert _normalize_entity_name("James'") == "james"
        # Smart quotes
        assert _normalize_entity_name("John's") == "john"

    def test_unicode_normalization(self):
        """Test Unicode NFKC normalization."""
        # Different representations of the same character
        assert _normalize_entity_name("café") == "café"
        # Full-width characters
        assert _normalize_entity_name("ABC") == "abc"

    def test_empty_and_whitespace(self):
        """Test empty and whitespace-only strings."""
        assert _normalize_entity_name("") == ""
        assert _normalize_entity_name("   ") == ""


# =============================================================================
# Test _map_spacy_label
# =============================================================================


class TestMapSpacyLabel:
    """Tests for spaCy label mapping."""

    def test_person_mapping(self):
        """Test PERSON label mapping."""
        assert _map_spacy_label("PERSON") == "person"

    def test_organization_mappings(self):
        """Test organization-related label mappings."""
        assert _map_spacy_label("ORG") == "organization"
        assert _map_spacy_label("NORP") == "organization"

    def test_location_mappings(self):
        """Test location-related label mappings."""
        assert _map_spacy_label("GPE") == "location"
        assert _map_spacy_label("LOC") == "location"
        assert _map_spacy_label("FAC") == "location"

    def test_date_mappings(self):
        """Test date/time label mappings."""
        assert _map_spacy_label("DATE") == "date"
        assert _map_spacy_label("TIME") == "date"

    def test_other_mappings(self):
        """Test other specific label mappings."""
        assert _map_spacy_label("EVENT") == "event"
        assert _map_spacy_label("MONEY") == "money"
        assert _map_spacy_label("LAW") == "law"
        assert _map_spacy_label("PRODUCT") == "product"

    def test_unknown_label(self):
        """Test unknown labels map to 'unknown'."""
        assert _map_spacy_label("CARDINAL") == "unknown"
        assert _map_spacy_label("QUANTITY") == "unknown"
        assert _map_spacy_label("NONEXISTENT") == "unknown"

    def test_all_mapped_labels_exist_in_constant(self):
        """Verify all mapped labels are in the constant."""
        for label in SPACY_LABEL_TO_ENTITY_TYPE:
            result = _map_spacy_label(label)
            assert result is not None


# =============================================================================
# Test _is_valid_entity
# =============================================================================


class TestIsValidEntity:
    """Tests for entity validation."""

    def test_valid_entities(self):
        """Test valid entities are accepted."""
        assert _is_valid_entity("John Doe", "person") is True
        assert _is_valid_entity("Acme Corp", "organization") is True
        assert _is_valid_entity("New York", "location") is True
        assert _is_valid_entity("$100", "money") is True

    def test_empty_name_rejected(self):
        """Test empty names are rejected."""
        assert _is_valid_entity("", "person") is False
        assert _is_valid_entity("   ", "person") is False

    def test_too_short_name_rejected(self):
        """Test names shorter than MIN_ENTITY_LENGTH are rejected."""
        assert _is_valid_entity("X", "person") is False
        assert MIN_ENTITY_LENGTH == 2  # Verify our assumption

    def test_too_long_name_rejected(self):
        """Test names longer than MAX_ENTITY_LENGTH are rejected."""
        long_name = "A" * (MAX_ENTITY_LENGTH + 1)
        assert _is_valid_entity(long_name, "person") is False

    def test_unknown_type_rejected(self):
        """Test entities with unknown type are rejected."""
        assert _is_valid_entity("John Doe", "unknown") is False
        assert "unknown" not in EXTRACTED_ENTITY_TYPES

    def test_numeric_entities_rejected(self):
        """Test purely numeric entities are rejected (except money)."""
        assert _is_valid_entity("12345", "person") is False
        assert _is_valid_entity("100 200", "organization") is False
        # Money with numbers is OK
        assert _is_valid_entity("$100", "money") is True

    def test_extracted_entity_types(self):
        """Test all extracted entity types are valid."""
        for entity_type in EXTRACTED_ENTITY_TYPES:
            assert _is_valid_entity("Valid Name", entity_type) is True


# =============================================================================
# Test _extract_context
# =============================================================================


class TestExtractContext:
    """Tests for context extraction."""

    def test_basic_context_extraction(self):
        """Test basic context extraction."""
        text = "The quick brown fox jumps over the lazy dog."
        # "fox" starts at 16, ends at 19
        context = _extract_context(text, 16, 19, window=10)
        # Should include 10 chars before and after
        assert "fox" in context
        assert "brown" in context
        assert "jumps" in context

    def test_context_at_start(self):
        """Test context extraction at document start."""
        text = "John Doe works at Acme."
        context = _extract_context(text, 0, 8, window=10)
        assert "John Doe" in context
        # Should have ellipsis at end if truncated
        assert context.startswith("John Doe")

    def test_context_at_end(self):
        """Test context extraction at document end."""
        text = "This is some text. Works at Acme Corp"
        # "Acme Corp" is at the end
        context = _extract_context(text, 28, 37, window=10)
        assert "Acme Corp" in context
        # Should have ellipsis at start since we're truncating
        assert "..." in context

    def test_context_with_ellipsis(self):
        """Test ellipsis are added when truncated."""
        text = "A" * 100 + "ENTITY" + "B" * 100
        context = _extract_context(text, 100, 106, window=10)
        assert "..." in context
        assert "ENTITY" in context

    def test_context_default_window(self):
        """Test default context window size."""
        text = "X" * 200
        context = _extract_context(text, 100, 105)
        # Should use CONTEXT_WINDOW (50 by default)
        assert len(context) <= CONTEXT_WINDOW * 2 + 5 + 6  # window*2 + entity + ellipsis


# =============================================================================
# Test _extract_entities_from_chunk
# =============================================================================


class TestExtractEntitiesFromChunk:
    """Tests for per-chunk entity extraction."""

    def test_extracts_entities_from_mock_doc(self, sample_chunk, mock_spacy_doc):
        """Test extraction from a mock spaCy doc."""
        MockDoc, MockEntity = mock_spacy_doc

        # Create mock NLP that returns a doc with entities
        mock_nlp = MagicMock()
        mock_doc = MockDoc(
            ents=[
                MockEntity(text="John Doe", label_="PERSON", start_char=0, end_char=8),
                MockEntity(text="Acme Corp", label_="ORG", start_char=18, end_char=27),
            ]
        )
        mock_nlp.return_value = mock_doc

        mentions = _extract_entities_from_chunk(mock_nlp, sample_chunk, 0)

        assert len(mentions) == 2
        assert mentions[0]["name"] == "John Doe"
        assert mentions[0]["entity_type"] == "person"
        assert mentions[0]["chunk_index"] == 0
        assert mentions[1]["name"] == "Acme Corp"
        assert mentions[1]["entity_type"] == "organization"

    def test_filters_invalid_entities(self, sample_chunk, mock_spacy_doc):
        """Test that invalid entities are filtered out."""
        MockDoc, MockEntity = mock_spacy_doc

        mock_nlp = MagicMock()
        mock_doc = MockDoc(
            ents=[
                MockEntity(text="John Doe", label_="PERSON", start_char=0, end_char=8),
                MockEntity(text="X", label_="PERSON", start_char=10, end_char=11),  # Too short
                MockEntity(
                    text="123", label_="CARDINAL", start_char=12, end_char=15
                ),  # Unknown type
            ]
        )
        mock_nlp.return_value = mock_doc

        mentions = _extract_entities_from_chunk(mock_nlp, sample_chunk, 0)

        assert len(mentions) == 1
        assert mentions[0]["name"] == "John Doe"

    def test_includes_context_in_mentions(self, sample_chunk, mock_spacy_doc):
        """Test that context is included in mentions."""
        MockDoc, MockEntity = mock_spacy_doc

        mock_nlp = MagicMock()
        mock_doc = MockDoc(
            ents=[
                MockEntity(text="Acme Corp", label_="ORG", start_char=18, end_char=27),
            ]
        )
        mock_nlp.return_value = mock_doc

        mentions = _extract_entities_from_chunk(mock_nlp, sample_chunk, 0)

        assert len(mentions) == 1
        assert "context" in mentions[0]
        assert "Acme Corp" in mentions[0]["context"]

    def test_calculates_global_positions(self, mock_spacy_doc):
        """Test that global positions are calculated correctly."""
        MockDoc, MockEntity = mock_spacy_doc

        # Chunk starting at position 100
        chunk = ChunkInfo(
            sequence_number=1,
            content="John Doe is here.",
            content_hash="hash",
            start_char=100,
            end_char=117,
            token_count=5,
        )

        mock_nlp = MagicMock()
        mock_doc = MockDoc(
            ents=[
                MockEntity(text="John Doe", label_="PERSON", start_char=0, end_char=8),
            ]
        )
        mock_nlp.return_value = mock_doc

        mentions = _extract_entities_from_chunk(mock_nlp, chunk, 1)

        assert mentions[0]["global_start"] == 100  # chunk.start_char + entity.start_char
        assert mentions[0]["global_end"] == 108  # chunk.start_char + entity.end_char


# =============================================================================
# Test _group_entity_mentions
# =============================================================================


class TestGroupEntityMentions:
    """Tests for mention grouping."""

    def test_groups_same_entity_mentions(self):
        """Test mentions of same entity are grouped."""
        mentions = [
            {
                "name": "John Doe",
                "entity_type": "person",
                "chunk_index": 0,
                "start": 0,
                "end": 8,
                "context": "John Doe...",
            },
            {
                "name": "John Doe",
                "entity_type": "person",
                "chunk_index": 1,
                "start": 5,
                "end": 13,
                "context": "...John Doe...",
            },
        ]

        entities = _group_entity_mentions(mentions)

        assert len(entities) == 1
        assert entities[0].name == "John Doe"
        assert len(entities[0].mentions) == 2

    def test_groups_normalized_variants(self):
        """Test name variants are normalized and grouped."""
        mentions = [
            {
                "name": "John Doe",
                "entity_type": "person",
                "chunk_index": 0,
                "start": 0,
                "end": 8,
                "context": "John Doe...",
            },
            {
                "name": "JOHN DOE",
                "entity_type": "person",
                "chunk_index": 1,
                "start": 5,
                "end": 13,
                "context": "...JOHN DOE...",
            },
            {
                "name": "john doe",
                "entity_type": "person",
                "chunk_index": 2,
                "start": 0,
                "end": 8,
                "context": "john doe...",
            },
        ]

        entities = _group_entity_mentions(mentions)

        assert len(entities) == 1
        assert len(entities[0].mentions) == 3

    def test_separates_different_entities(self):
        """Test different entities are kept separate."""
        mentions = [
            {
                "name": "John Doe",
                "entity_type": "person",
                "chunk_index": 0,
                "start": 0,
                "end": 8,
                "context": "John Doe...",
            },
            {
                "name": "Jane Smith",
                "entity_type": "person",
                "chunk_index": 0,
                "start": 20,
                "end": 30,
                "context": "...Jane Smith...",
            },
        ]

        entities = _group_entity_mentions(mentions)

        assert len(entities) == 2
        names = {e.name for e in entities}
        assert "John Doe" in names
        assert "Jane Smith" in names

    def test_separates_same_name_different_types(self):
        """Test same name with different types creates separate entities."""
        mentions = [
            {
                "name": "Apple",
                "entity_type": "organization",
                "chunk_index": 0,
                "start": 0,
                "end": 5,
                "context": "Apple Inc...",
            },
            {
                "name": "Apple",
                "entity_type": "product",
                "chunk_index": 1,
                "start": 0,
                "end": 5,
                "context": "Apple fruit...",
            },
        ]

        entities = _group_entity_mentions(mentions)

        assert len(entities) == 2

    def test_confidence_increases_with_mentions(self):
        """Test confidence increases with more mentions."""
        # Single mention
        single_mention = [
            {
                "name": "John Doe",
                "entity_type": "person",
                "chunk_index": 0,
                "start": 0,
                "end": 8,
                "context": "...",
            },
        ]

        # Multiple mentions
        multiple_mentions = [
            {
                "name": "Jane Smith",
                "entity_type": "person",
                "chunk_index": i,
                "start": 0,
                "end": 10,
                "context": "...",
            }
            for i in range(5)
        ]

        single_entity = _group_entity_mentions(single_mention)[0]
        multi_entity = _group_entity_mentions(multiple_mentions)[0]

        assert multi_entity.confidence > single_entity.confidence

    def test_uses_most_common_name_variant(self):
        """Test the most common name variant is used as canonical."""
        mentions = [
            {
                "name": "John Doe",
                "entity_type": "person",
                "chunk_index": 0,
                "start": 0,
                "end": 8,
                "context": "...",
            },
            {
                "name": "John Doe",
                "entity_type": "person",
                "chunk_index": 1,
                "start": 0,
                "end": 8,
                "context": "...",
            },
            {
                "name": "JOHN DOE",
                "entity_type": "person",
                "chunk_index": 2,
                "start": 0,
                "end": 8,
                "context": "...",
            },
        ]

        entities = _group_entity_mentions(mentions)

        # "John Doe" appears twice, "JOHN DOE" appears once
        assert entities[0].name == "John Doe"

    def test_sorted_by_confidence_then_mentions(self):
        """Test entities are sorted by confidence and mention count."""
        mentions = [
            # Low mention count
            {
                "name": "John Doe",
                "entity_type": "person",
                "chunk_index": 0,
                "start": 0,
                "end": 8,
                "context": "...",
            },
            # High mention count (5 mentions)
            *[
                {
                    "name": "Acme Corp",
                    "entity_type": "organization",
                    "chunk_index": i,
                    "start": 0,
                    "end": 9,
                    "context": "...",
                }
                for i in range(5)
            ],
        ]

        entities = _group_entity_mentions(mentions)

        # Acme Corp should come first (more mentions = higher confidence)
        assert entities[0].name == "Acme Corp"

    def test_empty_mentions_returns_empty_list(self):
        """Test empty input returns empty list."""
        entities = _group_entity_mentions([])
        assert entities == []


# =============================================================================
# Test extract_entities Activity
# =============================================================================


class TestExtractEntitiesActivity:
    """Tests for the extract_entities activity."""

    @pytest.mark.asyncio
    async def test_extracts_entities_with_mocked_spacy(self, sample_chunks, mock_spacy_doc):
        """Test full activity with mocked spaCy."""
        MockDoc, MockEntity = mock_spacy_doc

        # Import the activity function
        from ingestion_worker.activities.extraction_activities import extract_entities

        # Create mock NLP model
        mock_nlp = MagicMock()

        def create_doc(text):
            # Create mock entities based on text content
            ents = []
            if "John Doe" in text:
                idx = text.index("John Doe")
                ents.append(
                    MockEntity(text="John Doe", label_="PERSON", start_char=idx, end_char=idx + 8)
                )
            if "Acme" in text:
                idx = text.index("Acme")
                end_idx = idx + 4
                if "Corporation" in text[idx:]:
                    end_idx = idx + 16
                elif "Corp" in text[idx:]:
                    end_idx = idx + 9
                ents.append(
                    MockEntity(
                        text=text[idx:end_idx], label_="ORG", start_char=idx, end_char=end_idx
                    )
                )
            if "San Francisco" in text:
                idx = text.index("San Francisco")
                ents.append(
                    MockEntity(
                        text="San Francisco", label_="GPE", start_char=idx, end_char=idx + 13
                    )
                )
            return MockDoc(ents=ents)

        mock_nlp.side_effect = create_doc

        # Patch the model loader and activity context
        with (
            patch(
                "ingestion_worker.activities.extraction_activities._get_spacy_model",
                return_value=mock_nlp,
            ),
            patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity,
        ):
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractEntitiesInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="Full content here",
                chunks=sample_chunks,
            )

            result = await extract_entities(input_data)

        # Verify results
        assert len(result.entities) > 0

        # Should have extracted John Doe, Acme variants, and San Francisco
        entity_names = {e.name.lower() for e in result.entities}
        entity_types = {e.entity_type for e in result.entities}

        assert "person" in entity_types
        assert "organization" in entity_types
        assert "location" in entity_types

    @pytest.mark.asyncio
    async def test_handles_empty_chunks(self):
        """Test activity handles empty chunk list."""
        from ingestion_worker.activities.extraction_activities import extract_entities

        with (
            patch(
                "ingestion_worker.activities.extraction_activities._get_spacy_model",
                return_value=MagicMock(),
            ),
            patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity,
        ):
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractEntitiesInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="",
                chunks=[],
            )

            result = await extract_entities(input_data)

        assert result.entities == []

    @pytest.mark.asyncio
    async def test_heartbeats_periodically(self, mock_spacy_doc):
        """Test activity sends heartbeats for long documents."""
        MockDoc, MockEntity = mock_spacy_doc

        from ingestion_worker.activities.extraction_activities import extract_entities

        mock_nlp = MagicMock()
        mock_nlp.return_value = MockDoc(ents=[])

        # Create 25 chunks to trigger multiple heartbeats
        chunks = [
            ChunkInfo(
                sequence_number=i,
                content=f"Chunk {i} content.",
                content_hash=f"hash{i}",
                start_char=i * 20,
                end_char=(i + 1) * 20,
                token_count=5,
            )
            for i in range(25)
        ]

        with (
            patch(
                "ingestion_worker.activities.extraction_activities._get_spacy_model",
                return_value=mock_nlp,
            ),
            patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity,
        ):
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractEntitiesInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="Content",
                chunks=chunks,
            )

            await extract_entities(input_data)

        # Should have heartbeat calls (at least 2 for 25 chunks, plus final)
        assert mock_activity.heartbeat.call_count >= 2


# =============================================================================
# Test extract_entities_mock Activity
# =============================================================================


class TestExtractEntitiesMockActivity:
    """Tests for the mock entity extraction activity."""

    @pytest.mark.asyncio
    async def test_returns_person_for_john(self):
        """Test mock returns person entity for 'john' in content."""
        from ingestion_worker.activities.extraction_activities import extract_entities_mock

        with patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractEntitiesInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="John works here.",
                chunks=[],
            )

            result = await extract_entities_mock(input_data)

        assert any(e.name == "John Doe" and e.entity_type == "person" for e in result.entities)

    @pytest.mark.asyncio
    async def test_returns_organization_for_acme(self):
        """Test mock returns organization entity for 'acme' in content."""
        from ingestion_worker.activities.extraction_activities import extract_entities_mock

        with patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractEntitiesInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="Acme is a company.",
                chunks=[],
            )

            result = await extract_entities_mock(input_data)

        assert any(
            e.name == "Acme Corp" and e.entity_type == "organization" for e in result.entities
        )

    @pytest.mark.asyncio
    async def test_returns_location_for_new_york(self):
        """Test mock returns location entity for 'new york' in content."""
        from ingestion_worker.activities.extraction_activities import extract_entities_mock

        with patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractEntitiesInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="Located in New York.",
                chunks=[],
            )

            result = await extract_entities_mock(input_data)

        assert any(e.name == "New York" and e.entity_type == "location" for e in result.entities)

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_patterns(self):
        """Test mock returns empty list when no patterns match."""
        from ingestion_worker.activities.extraction_activities import extract_entities_mock

        with patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractEntitiesInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="This content has no recognizable patterns.",
                chunks=[],
            )

            result = await extract_entities_mock(input_data)

        assert result.entities == []


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestExtractEntitiesEdgeCases:
    """Tests for edge cases in entity extraction."""

    def test_chunk_dict_conversion(self, mock_spacy_doc):
        """Test that dict chunks are properly converted to ChunkInfo."""
        MockDoc, MockEntity = mock_spacy_doc

        mock_nlp = MagicMock()
        mock_nlp.return_value = MockDoc(ents=[])

        # Create a chunk as a dict (as it might come from workflow)
        chunk_dict = {
            "sequence_number": 0,
            "content": "Some content",
            "content_hash": "hash",
            "start_char": 0,
            "end_char": 12,
            "token_count": 3,
        }

        # Should not raise
        from ingestion_worker.activities.extraction_activities import ChunkInfo

        chunk = ChunkInfo(**chunk_dict)
        mentions = _extract_entities_from_chunk(mock_nlp, chunk, 0)
        assert isinstance(mentions, list)

    def test_entity_with_special_characters(self):
        """Test entities with special characters are normalized."""
        # Test various special characters
        assert _normalize_entity_name("O'Brien") == "o'brien"
        assert _normalize_entity_name("São Paulo") == "são paulo"
        assert _normalize_entity_name("Müller") == "müller"

    def test_very_long_context_truncation(self):
        """Test that very long context is properly truncated."""
        long_text = "A" * 1000 + "ENTITY" + "B" * 1000
        context = _extract_context(long_text, 1000, 1006, window=CONTEXT_WINDOW)

        # Context should be reasonably sized
        assert len(context) < 200

    def test_confidence_caps_at_maximum(self):
        """Test confidence doesn't exceed maximum."""
        # Create many mentions of the same entity
        mentions = [
            {
                "name": "John Doe",
                "entity_type": "person",
                "chunk_index": i,
                "start": 0,
                "end": 8,
                "context": "...",
            }
            for i in range(100)  # 100 mentions
        ]

        entities = _group_entity_mentions(mentions)

        # Confidence should be capped
        assert entities[0].confidence <= 0.98

    def test_handles_unicode_entities(self, mock_spacy_doc):
        """Test extraction handles Unicode entity names."""
        MockDoc, MockEntity = mock_spacy_doc

        mock_nlp = MagicMock()
        mock_doc = MockDoc(
            ents=[
                MockEntity(text="北京", label_="GPE", start_char=0, end_char=2),
                MockEntity(text="François Müller", label_="PERSON", start_char=5, end_char=20),
            ]
        )
        mock_nlp.return_value = mock_doc

        chunk = ChunkInfo(
            sequence_number=0,
            content="北京 is François Müller's city.",
            content_hash="hash",
            start_char=0,
            end_char=30,
            token_count=6,
        )

        mentions = _extract_entities_from_chunk(mock_nlp, chunk, 0)

        assert len(mentions) == 2
        assert any(m["name"] == "北京" for m in mentions)
        assert any(m["name"] == "François Müller" for m in mentions)
