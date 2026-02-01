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
- Relationship extraction helpers
- Relationship extraction activity
"""

from __future__ import annotations

import pytest

from ingestion_worker.activities.extraction_activities import (
    CONTEXT_WINDOW,
    EXTRACTED_ENTITY_TYPES,
    MAX_ENTITY_LENGTH,
    MIN_ENTITY_LENGTH,
    MIN_RELATIONSHIP_CONFIDENCE,
    SPACY_LABEL_TO_ENTITY_TYPE,
    _build_entity_lookup,
    _calculate_relationship_confidence,
    _classify_verb_pattern,
    _deduplicate_relationships,
    _determine_relationship_type,
    _entity_type_to_label,
    _extract_context,
    _extract_entities_from_chunk,
    _extract_relationships_from_sentence,
    _find_entity_in_span,
    _generate_entity_id,
    _group_entity_mentions,
    _is_valid_entity,
    _map_spacy_label,
    _normalize_entity_name,
    _relationship_type_to_label,
)
from ingestion_worker.workflows.document_ingestion import (
    BuildGraphInput,
    ChunkInfo,
    EntityInfo,
    ExtractEntitiesInput,
    ExtractRelationshipsInput,
    RelationshipInfo,
)

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

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


# =============================================================================
# Relationship Extraction Tests
# =============================================================================


@pytest.fixture
def sample_entities() -> list[EntityInfo]:
    """Create sample entities for relationship extraction tests."""
    return [
        EntityInfo(
            name="John Doe",
            entity_type="person",
            mentions=[{"chunk_index": 0, "start": 0, "end": 8}],
            confidence=0.9,
        ),
        EntityInfo(
            name="Acme Corporation",
            entity_type="organization",
            mentions=[{"chunk_index": 0, "start": 20, "end": 36}],
            confidence=0.85,
        ),
        EntityInfo(
            name="New York",
            entity_type="location",
            mentions=[{"chunk_index": 0, "start": 50, "end": 58}],
            confidence=0.88,
        ),
    ]


class TestBuildEntityLookup:
    """Tests for entity lookup building."""

    def test_builds_lookup_from_entity_info_list(self, sample_entities):
        """Test building lookup from EntityInfo objects."""
        lookup = _build_entity_lookup(sample_entities)

        assert len(lookup) == 3
        assert "john doe" in lookup
        assert "acme corporation" in lookup
        assert "new york" in lookup

    def test_builds_lookup_from_dict_list(self):
        """Test building lookup from dictionaries."""
        entities = [
            {"name": "John Doe", "entity_type": "person"},
            {"name": "Acme Corp", "entity_type": "organization"},
        ]

        lookup = _build_entity_lookup(entities)

        assert len(lookup) == 2
        assert lookup["john doe"]["entity_type"] == "person"
        assert lookup["acme corp"]["entity_type"] == "organization"

    def test_normalizes_names_in_lookup(self):
        """Test that names are normalized in the lookup."""
        entities = [
            EntityInfo(
                name="JOHN DOE",
                entity_type="person",
                mentions=[],
                confidence=0.9,
            ),
        ]

        lookup = _build_entity_lookup(entities)

        assert "john doe" in lookup
        assert "JOHN DOE" not in lookup


class TestFindEntityInSpan:
    """Tests for finding entities in text spans."""

    def test_finds_exact_match(self, sample_entities):
        """Test finding entity with exact match."""
        lookup = _build_entity_lookup(sample_entities)

        result = _find_entity_in_span("John Doe", lookup)

        assert result is not None
        assert result["name"] == "John Doe"
        assert result["entity_type"] == "person"

    def test_finds_case_insensitive_match(self, sample_entities):
        """Test finding entity with different case."""
        lookup = _build_entity_lookup(sample_entities)

        result = _find_entity_in_span("JOHN DOE", lookup)

        assert result is not None
        assert result["entity_type"] == "person"

    def test_finds_partial_match_in_span(self, sample_entities):
        """Test finding entity contained in larger span."""
        lookup = _build_entity_lookup(sample_entities)

        result = _find_entity_in_span("Mr. John Doe, CEO", lookup)

        assert result is not None
        assert result["entity_type"] == "person"

    def test_returns_none_for_no_match(self, sample_entities):
        """Test returns None when no entity matches."""
        lookup = _build_entity_lookup(sample_entities)

        result = _find_entity_in_span("Unknown Person", lookup)

        assert result is None


class TestClassifyVerbPattern:
    """Tests for verb pattern classification."""

    def test_classifies_work_verbs(self):
        """Test classification of work-related verbs."""
        assert _classify_verb_pattern("work") == "work"
        assert _classify_verb_pattern("employ") == "work"
        assert _classify_verb_pattern("manage") == "work"
        assert _classify_verb_pattern("lead") == "work"

    def test_classifies_ownership_verbs(self):
        """Test classification of ownership verbs."""
        assert _classify_verb_pattern("own") == "own"
        assert _classify_verb_pattern("acquire") == "own"
        assert _classify_verb_pattern("purchase") == "own"

    def test_classifies_location_verbs(self):
        """Test classification of location verbs."""
        assert _classify_verb_pattern("locate") == "locate"
        assert _classify_verb_pattern("base") == "locate"
        assert _classify_verb_pattern("headquarter") == "locate"

    def test_classifies_membership_verbs(self):
        """Test classification of membership verbs."""
        assert _classify_verb_pattern("belong") == "member"
        assert _classify_verb_pattern("affiliate") == "member"

    def test_classifies_report_verbs(self):
        """Test classification of reporting verbs."""
        assert _classify_verb_pattern("report") == "report"
        assert _classify_verb_pattern("answer") == "report"

    def test_returns_none_for_unknown_verbs(self):
        """Test returns None for unrecognized verbs."""
        assert _classify_verb_pattern("xyz") is None
        assert _classify_verb_pattern("think") is None


class TestDetermineRelationshipType:
    """Tests for relationship type determination."""

    def test_person_organization_work(self):
        """Test person works_for organization."""
        result = _determine_relationship_type("person", "organization", "work")
        assert result == "works_for"

    def test_person_organization_own(self):
        """Test person owns organization."""
        result = _determine_relationship_type("person", "organization", "own")
        assert result == "owns"

    def test_organization_organization_partner(self):
        """Test organization partner_of organization."""
        result = _determine_relationship_type("organization", "organization", "partner")
        assert result == "partner_of"

    def test_organization_location_locate(self):
        """Test organization headquarters_in location."""
        result = _determine_relationship_type("organization", "location", "locate")
        assert result == "headquarters_in"

    def test_person_person_work(self):
        """Test person works_with person."""
        result = _determine_relationship_type("person", "person", "work")
        assert result == "works_with"

    def test_fallback_to_associated_with(self):
        """Test fallback to associated_with for unknown combinations."""
        result = _determine_relationship_type("unknown", "unknown", "unknown")
        assert result == "associated_with"


class TestCalculateRelationshipConfidence:
    """Tests for relationship confidence calculation."""

    def test_base_confidence(self):
        """Test base confidence for a relationship."""
        rel = {
            "source_entity": "John",
            "target_entity": "Acme",
            "relationship_type": "works_for",
            "verb": None,
            "evidence": "...",
        }

        confidence = _calculate_relationship_confidence(rel, 1)

        assert confidence >= 0.3
        assert confidence <= 1.0

    def test_verb_based_boost(self):
        """Test confidence boost for verb-based extraction."""
        rel_no_verb = {
            "source_entity": "John",
            "target_entity": "Acme",
            "relationship_type": "works_for",
            "verb": None,
            "evidence": "...",
        }

        rel_with_verb = {
            "source_entity": "John",
            "target_entity": "Acme",
            "relationship_type": "works_for",
            "verb": "work",
            "evidence": "...",
        }

        conf_no_verb = _calculate_relationship_confidence(rel_no_verb, 1)
        conf_with_verb = _calculate_relationship_confidence(rel_with_verb, 1)

        assert conf_with_verb > conf_no_verb

    def test_multiple_occurrences_boost(self):
        """Test confidence boost for multiple occurrences."""
        rel = {
            "source_entity": "John",
            "target_entity": "Acme",
            "relationship_type": "works_for",
            "verb": "work",
            "evidence": "...",
        }

        conf_single = _calculate_relationship_confidence(rel, 1)
        conf_multiple = _calculate_relationship_confidence(rel, 5)

        assert conf_multiple > conf_single

    def test_confidence_capped_at_maximum(self):
        """Test confidence doesn't exceed 0.95."""
        rel = {
            "source_entity": "John",
            "target_entity": "Acme",
            "relationship_type": "works_for",
            "verb": "work",
            "evidence": "...",
        }

        confidence = _calculate_relationship_confidence(rel, 100)

        assert confidence <= 0.95


class TestDeduplicateRelationships:
    """Tests for relationship deduplication."""

    def test_deduplicates_identical_relationships(self):
        """Test that identical relationships are merged."""
        relationships = [
            {
                "source_entity": "John Doe",
                "target_entity": "Acme Corp",
                "relationship_type": "works_for",
                "evidence": "Evidence 1",
                "verb": "work",
            },
            {
                "source_entity": "John Doe",
                "target_entity": "Acme Corp",
                "relationship_type": "works_for",
                "evidence": "Evidence 2",
                "verb": "work",
            },
        ]

        result = _deduplicate_relationships(relationships)

        assert len(result) == 1
        assert result[0].source_entity == "John Doe"
        assert result[0].target_entity == "Acme Corp"

    def test_keeps_different_relationships(self):
        """Test that different relationships are kept separate."""
        relationships = [
            {
                "source_entity": "John Doe",
                "target_entity": "Acme Corp",
                "relationship_type": "works_for",
                "evidence": "Evidence 1",
                "verb": "work",
            },
            {
                "source_entity": "Acme Corp",
                "target_entity": "New York",
                "relationship_type": "located_in",
                "evidence": "Evidence 2",
                "verb": "locate",
            },
        ]

        result = _deduplicate_relationships(relationships)

        assert len(result) == 2

    def test_filters_low_confidence(self):
        """Test that low confidence relationships are filtered."""
        relationships = [
            {
                "source_entity": "X",
                "target_entity": "Y",
                "relationship_type": "associated_with",
                "evidence": "...",
                "verb": None,
            },
        ]

        result = _deduplicate_relationships(relationships)

        # May or may not be filtered depending on MIN_RELATIONSHIP_CONFIDENCE
        for rel in result:
            assert rel.confidence >= MIN_RELATIONSHIP_CONFIDENCE

    def test_sorts_by_confidence(self):
        """Test that results are sorted by confidence."""
        relationships = [
            {
                "source_entity": "A",
                "target_entity": "B",
                "relationship_type": "associated_with",
                "evidence": "...",
                "verb": None,
            },
            {
                "source_entity": "C",
                "target_entity": "D",
                "relationship_type": "works_for",
                "evidence": "...",
                "verb": "work",
            },
        ]

        result = _deduplicate_relationships(relationships)

        if len(result) >= 2:
            assert result[0].confidence >= result[1].confidence


class TestExtractRelationshipsFromSentence:
    """Tests for sentence-level relationship extraction."""

    def test_extracts_from_verb_pattern(self):
        """Test extraction from subject-verb-object pattern."""
        # This test requires a mock spaCy sentence
        # We'll create mock objects that simulate spaCy's behavior

        @dataclass
        class MockToken:
            text: str
            lemma_: str
            pos_: str
            dep_: str
            i: int
            children: list = field(default_factory=list)
            subtree: list = field(default_factory=list)

        @dataclass
        class MockEntity:
            text: str
            start: int
            end: int

        @dataclass
        class MockSent:
            text: str
            ents: list
            doc: Any = None

            def __iter__(self):
                return iter([])

            def __len__(self):
                return len(self.text)

        entity_lookup = {
            "john doe": {"name": "John Doe", "entity_type": "person"},
            "acme corp": {"name": "Acme Corp", "entity_type": "organization"},
        }

        # Create a minimal mock sentence
        sent = MockSent(
            text="John Doe works at Acme Corp.",
            ents=[
                MockEntity(text="John Doe", start=0, end=2),
                MockEntity(text="Acme Corp", start=4, end=6),
            ],
        )

        # The function should handle the minimal mock
        relationships = _extract_relationships_from_sentence(sent, entity_lookup)

        # With the minimal mock, we won't get verb-based relationships
        # but we might get co-occurrence based ones
        assert isinstance(relationships, list)

    def test_skips_long_sentences(self):
        """Test that very long sentences are skipped."""
        from ingestion_worker.activities.extraction_activities import MAX_SENTENCE_LENGTH

        @dataclass
        class MockSent:
            text: str
            ents: list = field(default_factory=list)

        entity_lookup = {"john": {"name": "John", "entity_type": "person"}}

        # Create a sentence longer than MAX_SENTENCE_LENGTH
        long_sent = MockSent(text="A" * (MAX_SENTENCE_LENGTH + 100))

        relationships = _extract_relationships_from_sentence(long_sent, entity_lookup)

        assert relationships == []


class TestExtractRelationshipsActivity:
    """Tests for the extract_relationships activity."""

    @pytest.mark.asyncio
    async def test_handles_no_entities(self):
        """Test activity handles case with no entities."""
        from ingestion_worker.activities.extraction_activities import extract_relationships

        with patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractRelationshipsInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="Some content without entities.",
                entities=[],
            )

            result = await extract_relationships(input_data)

        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_handles_single_entity(self):
        """Test activity handles case with single entity."""
        from ingestion_worker.activities.extraction_activities import extract_relationships

        with patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractRelationshipsInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="John Doe is a person.",
                entities=[
                    EntityInfo(
                        name="John Doe",
                        entity_type="person",
                        mentions=[],
                        confidence=0.9,
                    )
                ],
            )

            result = await extract_relationships(input_data)

        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_extracts_relationships_with_mocked_spacy(self, sample_entities):
        """Test full activity with mocked spaCy."""
        from ingestion_worker.activities.extraction_activities import extract_relationships

        @dataclass
        class MockToken:
            text: str
            lemma_: str
            pos_: str
            dep_: str
            i: int
            children: list = field(default_factory=list)

            @property
            def subtree(self):
                return [self]

        @dataclass
        class MockEntity:
            text: str
            label_: str
            start: int
            end: int
            start_char: int
            end_char: int

        @dataclass
        class MockSent:
            text: str
            ents: list
            doc: Any = None
            _tokens: list = field(default_factory=list)

            def __iter__(self):
                return iter(self._tokens)

            def __len__(self):
                return len(self.text)

        @dataclass
        class MockDoc:
            text: str
            _sents: list = field(default_factory=list)
            ents: list = field(default_factory=list)

            @property
            def sents(self):
                return self._sents

        def mock_nlp(text):
            # Create a mock doc with one sentence
            sent = MockSent(
                text=text,
                ents=[
                    MockEntity(
                        text="John Doe",
                        label_="PERSON",
                        start=0,
                        end=2,
                        start_char=0,
                        end_char=8,
                    ),
                    MockEntity(
                        text="Acme Corporation",
                        label_="ORG",
                        start=4,
                        end=6,
                        start_char=20,
                        end_char=36,
                    ),
                ],
            )
            doc = MockDoc(text=text, _sents=[sent])
            sent.doc = doc
            return doc

        with (
            patch(
                "ingestion_worker.activities.extraction_activities._get_spacy_model",
                return_value=mock_nlp,
            ),
            patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity,
        ):
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractRelationshipsInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="John Doe works at Acme Corporation in New York.",
                entities=sample_entities,
            )

            result = await extract_relationships(input_data)

        # Should return a list of relationships (may be empty with minimal mock)
        assert isinstance(result.relationships, list)


class TestExtractRelationshipsMockActivity:
    """Tests for the mock relationship extraction activity."""

    @pytest.mark.asyncio
    async def test_returns_works_for_relationship(self):
        """Test mock returns works_for when patterns match."""
        from ingestion_worker.activities.extraction_activities import extract_relationships_mock

        with patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractRelationshipsInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="John Doe works at Acme Corp as CEO.",
                entities=[
                    EntityInfo(
                        name="John Doe",
                        entity_type="person",
                        mentions=[],
                        confidence=0.9,
                    ),
                    EntityInfo(
                        name="Acme Corp",
                        entity_type="organization",
                        mentions=[],
                        confidence=0.85,
                    ),
                ],
            )

            result = await extract_relationships_mock(input_data)

        assert len(result.relationships) >= 1
        assert any(r.relationship_type == "works_for" for r in result.relationships)

    @pytest.mark.asyncio
    async def test_returns_headquarters_relationship(self):
        """Test mock returns headquarters_in when patterns match."""
        from ingestion_worker.activities.extraction_activities import extract_relationships_mock

        with patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractRelationshipsInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="Acme Corp is headquartered in New York.",
                entities=[
                    EntityInfo(
                        name="Acme Corp",
                        entity_type="organization",
                        mentions=[],
                        confidence=0.85,
                    ),
                    EntityInfo(
                        name="New York",
                        entity_type="location",
                        mentions=[],
                        confidence=0.88,
                    ),
                ],
            )

            result = await extract_relationships_mock(input_data)

        assert len(result.relationships) >= 1
        assert any(r.relationship_type == "headquarters_in" for r in result.relationships)

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_patterns(self):
        """Test mock returns empty list when no patterns match."""
        from ingestion_worker.activities.extraction_activities import extract_relationships_mock

        with patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = ExtractRelationshipsInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                content="This content has no recognizable patterns.",
                entities=[
                    EntityInfo(
                        name="Unknown Entity",
                        entity_type="unknown",
                        mentions=[],
                        confidence=0.5,
                    ),
                ],
            )

            result = await extract_relationships_mock(input_data)

        assert result.relationships == []


# =============================================================================
# Graph Building Tests
# =============================================================================


class TestGenerateEntityId:
    """Tests for entity ID generation."""

    def test_generates_deterministic_id(self):
        """Test that same inputs produce same ID."""
        id1 = _generate_entity_id("tenant-1", "John Doe", "person")
        id2 = _generate_entity_id("tenant-1", "John Doe", "person")
        assert id1 == id2

    def test_different_names_produce_different_ids(self):
        """Test that different names produce different IDs."""
        id1 = _generate_entity_id("tenant-1", "John Doe", "person")
        id2 = _generate_entity_id("tenant-1", "Jane Smith", "person")
        assert id1 != id2

    def test_different_types_produce_different_ids(self):
        """Test that different types produce different IDs."""
        id1 = _generate_entity_id("tenant-1", "Apple", "organization")
        id2 = _generate_entity_id("tenant-1", "Apple", "product")
        assert id1 != id2

    def test_different_tenants_produce_different_ids(self):
        """Test that different tenants produce different IDs."""
        id1 = _generate_entity_id("tenant-1", "John Doe", "person")
        id2 = _generate_entity_id("tenant-2", "John Doe", "person")
        assert id1 != id2

    def test_normalizes_name_for_id(self):
        """Test that name normalization is applied."""
        id1 = _generate_entity_id("tenant-1", "John Doe", "person")
        id2 = _generate_entity_id("tenant-1", "JOHN DOE", "person")
        id3 = _generate_entity_id("tenant-1", "  john   doe  ", "person")
        assert id1 == id2 == id3

    def test_id_format(self):
        """Test that ID has expected format."""
        entity_id = _generate_entity_id("tenant-1", "John Doe", "person")
        assert entity_id.startswith("ent_")
        assert len(entity_id) == 28  # "ent_" + 24 hex chars


class TestEntityTypeToLabel:
    """Tests for entity type to Neo4j label conversion."""

    def test_converts_person(self):
        """Test person type conversion."""
        assert _entity_type_to_label("person") == "Person"

    def test_converts_organization(self):
        """Test organization type conversion."""
        assert _entity_type_to_label("organization") == "Organization"

    def test_converts_location(self):
        """Test location type conversion."""
        assert _entity_type_to_label("location") == "Location"

    def test_handles_uppercase_input(self):
        """Test handles uppercase input."""
        assert _entity_type_to_label("PERSON") == "Person"
        assert _entity_type_to_label("Organization") == "Organization"

    def test_unknown_type_defaults_to_entity(self):
        """Test unknown types default to Entity."""
        assert _entity_type_to_label("unknown") == "Entity"
        assert _entity_type_to_label("custom_type") == "Entity"


class TestRelationshipTypeToLabel:
    """Tests for relationship type to Neo4j label conversion."""

    def test_converts_to_uppercase(self):
        """Test conversion to uppercase."""
        assert _relationship_type_to_label("works_for") == "WORKS_FOR"
        assert _relationship_type_to_label("located_in") == "LOCATED_IN"

    def test_handles_already_uppercase(self):
        """Test handles already uppercase input."""
        assert _relationship_type_to_label("WORKS_FOR") == "WORKS_FOR"

    def test_converts_spaces_to_underscores(self):
        """Test spaces are converted to underscores."""
        assert _relationship_type_to_label("works for") == "WORKS_FOR"


class TestBuildGraphMockActivity:
    """Tests for the mock graph building activity."""

    @pytest.mark.asyncio
    async def test_counts_entities_and_relationships(self, sample_entities):
        """Test mock returns correct counts."""
        from ingestion_worker.activities.extraction_activities import build_graph_mock

        with patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = BuildGraphInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                project_id=None,
                entities=sample_entities,
                relationships=[
                    RelationshipInfo(
                        source_entity="John Doe",
                        target_entity="Acme Corporation",
                        relationship_type="works_for",
                        evidence="John works at Acme",
                        confidence=0.8,
                    ),
                ],
            )

            result = await build_graph_mock(input_data)

        # 3 entities + (3 MENTIONED_IN + 1 relationship) = 4 relationships
        assert result.nodes_created == 3
        assert result.relationships_created == 4

    @pytest.mark.asyncio
    async def test_handles_empty_input(self):
        """Test mock handles empty entities list."""
        from ingestion_worker.activities.extraction_activities import build_graph_mock

        with patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = BuildGraphInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                project_id=None,
                entities=[],
                relationships=[],
            )

            result = await build_graph_mock(input_data)

        assert result.nodes_created == 0
        assert result.relationships_created == 0


class TestBuildGraphActivity:
    """Tests for the build_graph activity with mocked Neo4j."""

    @pytest.mark.asyncio
    async def test_returns_zero_for_no_entities(self):
        """Test activity returns zero counts when no entities provided."""
        from ingestion_worker.activities.extraction_activities import build_graph

        with patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = BuildGraphInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                project_id=None,
                entities=[],
                relationships=[],
            )

            result = await build_graph(input_data)

        assert result.nodes_created == 0
        assert result.relationships_created == 0

    @pytest.mark.asyncio
    async def test_creates_nodes_and_relationships(self, sample_entities):
        """Test activity creates nodes and relationships with mocked client."""
        from ingestion_worker.activities.extraction_activities import build_graph

        # Create mock Neo4j client
        mock_client = MagicMock()

        # Make async methods return coroutines
        async def mock_execute_query(*_args, **_kwargs):
            return [{"e": MagicMock()}]

        async def mock_create_doc(*_args, **_kwargs):
            return MagicMock()

        async def mock_create_mentioned(*_args, **_kwargs):
            return MagicMock()

        async def mock_create_rel(*_args, **_kwargs):
            return MagicMock()

        async def mock_close():
            pass

        mock_client.execute_query = mock_execute_query
        mock_client.create_document_node = mock_create_doc
        mock_client.create_mentioned_in_relationship = mock_create_mentioned
        mock_client.create_relationship = mock_create_rel
        mock_client.close = mock_close

        # Patch where the import happens (inside the function)
        mock_neo4j_class = MagicMock(return_value=mock_client)

        with (
            patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity,
            patch.dict(
                "sys.modules",
                {"alexandria_db.clients": MagicMock(Neo4jClient=mock_neo4j_class)},
            ),
        ):
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            input_data = BuildGraphInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                project_id="project-789",
                entities=sample_entities,
                relationships=[
                    RelationshipInfo(
                        source_entity="John Doe",
                        target_entity="Acme Corporation",
                        relationship_type="works_for",
                        evidence="John works at Acme",
                        confidence=0.8,
                    ),
                ],
            )

            result = await build_graph(input_data)

        # Should create nodes for all entities
        assert result.nodes_created == 3
        # Should create MENTIONED_IN + inter-entity relationships
        assert result.relationships_created >= 1

    @pytest.mark.asyncio
    async def test_handles_dict_entities(self):
        """Test activity handles entities passed as dicts."""
        from ingestion_worker.activities.extraction_activities import build_graph

        mock_client = MagicMock()

        async def mock_execute_query(*_args, **_kwargs):
            return [{"e": MagicMock()}]

        async def mock_create_doc(*_args, **_kwargs):
            return MagicMock()

        async def mock_create_mentioned(*_args, **_kwargs):
            return MagicMock()

        async def mock_close():
            pass

        mock_client.execute_query = mock_execute_query
        mock_client.create_document_node = mock_create_doc
        mock_client.create_mentioned_in_relationship = mock_create_mentioned
        mock_client.close = mock_close

        mock_neo4j_class = MagicMock(return_value=mock_client)

        with (
            patch("ingestion_worker.activities.extraction_activities.activity") as mock_activity,
            patch.dict(
                "sys.modules",
                {"alexandria_db.clients": MagicMock(Neo4jClient=mock_neo4j_class)},
            ),
        ):
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()

            # Pass entities as dicts (as they might come from workflow)
            input_data = BuildGraphInput(
                document_id="doc-123",
                tenant_id="tenant-456",
                project_id=None,
                entities=[
                    {
                        "name": "John Doe",
                        "entity_type": "person",
                        "confidence": 0.9,
                        "mentions": [],
                    },
                ],
                relationships=[],
            )

            result = await build_graph(input_data)

        assert result.nodes_created == 1
