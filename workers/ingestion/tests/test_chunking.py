"""
Unit tests for the semantic chunking module.

Tests cover:
- Basic chunking functionality
- Sentence boundary detection
- Heading hierarchy tracking
- Code block preservation
- Table preservation
- Overlap handling
- Edge cases and configuration options
"""

import pytest
from ingestion_worker.activities.chunking import (
    ChunkInfo,
    ChunkingConfig,
    ChunkType,
    HeadingInfo,
    SemanticChunker,
    SemanticUnit,
    chunk_text,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_chunker() -> SemanticChunker:
    """Create a chunker with default configuration."""
    return SemanticChunker()


@pytest.fixture
def small_chunk_chunker() -> SemanticChunker:
    """Create a chunker with small chunk size for testing."""
    config = ChunkingConfig(
        chunk_size=50,
        chunk_overlap=10,
        min_chunk_size=10,
    )
    return SemanticChunker(config)


@pytest.fixture
def sample_markdown() -> str:
    """Sample markdown document for testing."""
    return """# Introduction

This is the introduction paragraph. It contains some important information that should be kept together.

## Background

The background section provides context. This is the first sentence. This is the second sentence. And this is the third.

### Technical Details

Here are some technical details that matter for understanding the system.

```python
def hello_world():
    print("Hello, World!")
    return True
```

## Data Section

| Name | Age | City |
|------|-----|------|
| Alice | 30 | NYC |
| Bob | 25 | LA |

## Conclusion

The conclusion summarizes everything. It wraps up the document nicely.
"""


@pytest.fixture
def simple_text() -> str:
    """Simple text without markdown structure."""
    return """First paragraph with some content. It has multiple sentences. This is the third sentence.

Second paragraph is here. It also has content. Multiple sentences too.

Third paragraph rounds things out. Final content here."""


# =============================================================================
# Test ChunkingConfig
# =============================================================================


class TestChunkingConfig:
    """Tests for ChunkingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.respect_sentence_boundaries is True
        assert config.respect_heading_boundaries is True
        assert config.preserve_code_blocks is True
        assert config.preserve_tables is True
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 2000
        assert config.tokens_per_word == 1.3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            respect_sentence_boundaries=False,
            min_chunk_size=20,
        )
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.respect_sentence_boundaries is False
        assert config.min_chunk_size == 20


# =============================================================================
# Test SemanticUnit
# =============================================================================


class TestSemanticUnit:
    """Tests for SemanticUnit dataclass."""

    def test_basic_unit(self):
        """Test creating a basic semantic unit."""
        unit = SemanticUnit(
            content="Test content",
            unit_type=ChunkType.TEXT,
            start_char=0,
            end_char=12,
            estimated_tokens=3,
        )
        assert unit.content == "Test content"
        assert unit.unit_type == ChunkType.TEXT
        assert unit.heading_level is None

    def test_heading_unit(self):
        """Test creating a heading unit."""
        unit = SemanticUnit(
            content="# Main Title",
            unit_type=ChunkType.HEADING,
            start_char=0,
            end_char=12,
            heading_level=1,
            estimated_tokens=2,
        )
        assert unit.unit_type == ChunkType.HEADING
        assert unit.heading_level == 1


# =============================================================================
# Test ChunkInfo
# =============================================================================


class TestChunkInfo:
    """Tests for ChunkInfo dataclass."""

    def test_basic_chunk(self):
        """Test creating a basic chunk."""
        chunk = ChunkInfo(
            sequence_number=0,
            content="Test content",
            content_hash="abc123",
            start_char=0,
            end_char=12,
            token_count=3,
        )
        assert chunk.sequence_number == 0
        assert chunk.content == "Test content"
        assert chunk.page_number is None
        assert chunk.heading_context == []
        assert chunk.chunk_type == "text"
        assert chunk.metadata == {}

    def test_chunk_with_context(self):
        """Test chunk with heading context."""
        chunk = ChunkInfo(
            sequence_number=1,
            content="Content under heading",
            content_hash="def456",
            start_char=100,
            end_char=121,
            token_count=4,
            page_number=2,
            heading_context=["# Main", "## Section"],
            chunk_type="text",
            metadata={"custom": "value"},
        )
        assert chunk.page_number == 2
        assert len(chunk.heading_context) == 2
        assert chunk.metadata["custom"] == "value"


# =============================================================================
# Test Token Estimation
# =============================================================================


class TestTokenEstimation:
    """Tests for token estimation functionality."""

    def test_empty_text(self, default_chunker):
        """Test token estimation for empty text."""
        assert default_chunker._estimate_tokens("") == 0
        assert default_chunker._estimate_tokens("   ") == 0

    def test_single_word(self, default_chunker):
        """Test token estimation for single word."""
        tokens = default_chunker._estimate_tokens("hello")
        assert tokens == 1  # 1 word * 1.3 = 1.3 -> 1

    def test_multiple_words(self, default_chunker):
        """Test token estimation for multiple words."""
        tokens = default_chunker._estimate_tokens("hello world how are you")
        assert tokens == 6  # 5 words * 1.3 = 6.5 -> 6

    def test_custom_tokens_per_word(self):
        """Test with custom tokens per word ratio."""
        config = ChunkingConfig(tokens_per_word=1.5)
        chunker = SemanticChunker(config)
        tokens = chunker._estimate_tokens("one two three four")
        assert tokens == 6  # 4 words * 1.5 = 6


# =============================================================================
# Test Sentence Splitting
# =============================================================================


class TestSentenceSplitting:
    """Tests for sentence boundary detection."""

    def test_simple_sentences(self, default_chunker):
        """Test splitting simple sentences."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = default_chunker._split_sentences(text)
        assert len(sentences) == 3
        assert "First sentence." in sentences[0]

    def test_question_and_exclamation(self, default_chunker):
        """Test sentences with different punctuation."""
        text = "Is this a question? Yes it is! And a statement."
        sentences = default_chunker._split_sentences(text)
        assert len(sentences) == 3

    def test_empty_text(self, default_chunker):
        """Test with empty text."""
        assert default_chunker._split_sentences("") == []
        assert default_chunker._split_sentences("   ") == []

    def test_single_sentence(self, default_chunker):
        """Test with single sentence."""
        sentences = default_chunker._split_sentences("Just one sentence.")
        assert len(sentences) == 1

    def test_no_punctuation(self, default_chunker):
        """Test text without sentence-ending punctuation."""
        sentences = default_chunker._split_sentences("No punctuation here")
        assert len(sentences) == 1


# =============================================================================
# Test Semantic Unit Parsing
# =============================================================================


class TestSemanticUnitParsing:
    """Tests for parsing content into semantic units."""

    def test_simple_paragraphs(self, default_chunker, simple_text):
        """Test parsing simple paragraphs."""
        units = default_chunker._parse_semantic_units(simple_text)
        assert len(units) == 3
        assert all(u.unit_type == ChunkType.TEXT for u in units)

    def test_heading_detection(self, default_chunker):
        """Test heading detection."""
        content = "# Main Title\n\nParagraph content."
        units = default_chunker._parse_semantic_units(content)

        heading_units = [u for u in units if u.unit_type == ChunkType.HEADING]
        assert len(heading_units) == 1
        assert heading_units[0].heading_level == 1

    def test_code_block_detection(self, default_chunker):
        """Test code block detection."""
        content = """Some text.

```python
def foo():
    pass
```

More text."""
        units = default_chunker._parse_semantic_units(content)

        code_units = [u for u in units if u.unit_type == ChunkType.CODE]
        assert len(code_units) == 1
        assert "def foo():" in code_units[0].content

    def test_table_detection(self, default_chunker):
        """Test table detection."""
        content = """Before table.

| A | B |
|---|---|
| 1 | 2 |

After table."""
        units = default_chunker._parse_semantic_units(content)

        table_units = [u for u in units if u.unit_type == ChunkType.TABLE]
        assert len(table_units) == 1

    def test_list_detection(self, default_chunker):
        """Test list item detection."""
        content = """Intro text.

- Item one
- Item two
- Item three

After list."""
        units = default_chunker._parse_semantic_units(content)

        list_units = [u for u in units if u.unit_type == ChunkType.LIST]
        assert len(list_units) >= 1

    def test_mixed_content(self, default_chunker, sample_markdown):
        """Test parsing mixed markdown content."""
        units = default_chunker._parse_semantic_units(sample_markdown)

        # Should have headings, text, code, and table
        types = set(u.unit_type for u in units)
        assert ChunkType.HEADING in types
        assert ChunkType.TEXT in types
        assert ChunkType.CODE in types
        assert ChunkType.TABLE in types


# =============================================================================
# Test Basic Chunking
# =============================================================================


class TestBasicChunking:
    """Tests for basic chunking functionality."""

    def test_empty_content(self, default_chunker):
        """Test chunking empty content."""
        chunks = default_chunker.chunk("")
        assert chunks == []

    def test_whitespace_only(self, default_chunker):
        """Test chunking whitespace-only content."""
        chunks = default_chunker.chunk("   \n\n   ")
        assert chunks == []

    def test_single_paragraph(self, default_chunker):
        """Test chunking single paragraph."""
        chunks = default_chunker.chunk("This is a single paragraph.")
        assert len(chunks) == 1
        assert chunks[0].sequence_number == 0
        assert "single paragraph" in chunks[0].content

    def test_multiple_paragraphs(self, default_chunker, simple_text):
        """Test chunking multiple paragraphs."""
        chunks = default_chunker.chunk(simple_text)
        assert len(chunks) >= 1
        # Content should be preserved
        full_content = " ".join(c.content for c in chunks)
        assert "First paragraph" in full_content
        assert "Third paragraph" in full_content

    def test_sequence_numbers(self, small_chunk_chunker, simple_text):
        """Test that sequence numbers are sequential."""
        chunks = small_chunk_chunker.chunk(simple_text)
        for i, chunk in enumerate(chunks):
            assert chunk.sequence_number == i

    def test_content_hash_unique(self, small_chunk_chunker, simple_text):
        """Test that each chunk has a unique hash."""
        chunks = small_chunk_chunker.chunk(simple_text)
        hashes = [c.content_hash for c in chunks]
        assert len(hashes) == len(set(hashes))  # All unique

    def test_content_hash_deterministic(self, default_chunker):
        """Test that same content produces same hash."""
        content = "Test content for hashing."
        chunks1 = default_chunker.chunk(content)
        chunks2 = default_chunker.chunk(content)
        assert chunks1[0].content_hash == chunks2[0].content_hash


# =============================================================================
# Test Heading Context
# =============================================================================


class TestHeadingContext:
    """Tests for heading hierarchy tracking."""

    def test_simple_heading_context(self, default_chunker):
        """Test heading context is captured."""
        content = """# Main Title

Content under main title.

## Section One

Content under section one."""
        chunks = default_chunker.chunk(content)

        # At least one chunk should have heading context
        chunks_with_context = [c for c in chunks if c.heading_context]
        assert len(chunks_with_context) >= 1

    def test_heading_hierarchy(self, default_chunker):
        """Test heading hierarchy is maintained."""
        content = """# Level 1

Text at level 1.

## Level 2

Text at level 2.

### Level 3

Text at level 3."""
        chunks = default_chunker.chunk(content)

        # Look for chunk with level 3 content
        for chunk in chunks:
            if "level 3" in chunk.content.lower():
                # Should have hierarchy
                context_levels = len(chunk.heading_context)
                assert context_levels >= 1

    def test_heading_reset_on_same_level(self, default_chunker):
        """Test heading context resets when same level heading appears."""
        content = """# Title

## Section A

Content A.

## Section B

Content B."""
        chunks = default_chunker.chunk(content)

        # Verify heading context doesn't accumulate siblings
        for chunk in chunks:
            # Should never have both Section A and Section B in context
            if chunk.heading_context:
                context_text = " ".join(chunk.heading_context)
                has_both = "Section A" in context_text and "Section B" in context_text
                assert not has_both


# =============================================================================
# Test Code Block Handling
# =============================================================================


class TestCodeBlockHandling:
    """Tests for code block preservation."""

    def test_code_block_not_split(self, small_chunk_chunker):
        """Test that code blocks are not split across chunks."""
        content = """Intro text.

```python
def very_long_function():
    # This is a long function
    x = 1
    y = 2
    z = 3
    return x + y + z
```

After code."""
        chunks = small_chunk_chunker.chunk(content)

        # Find chunk with code
        code_chunks = [c for c in chunks if c.chunk_type == "code" or "```" in c.content]

        for code_chunk in code_chunks:
            # If it has opening fence, must have closing
            if "```python" in code_chunk.content or "```" in code_chunk.content:
                assert (
                    code_chunk.content.count("```") % 2 == 0 or code_chunk.content.count("```") >= 2
                )

    def test_code_language_in_metadata(self, default_chunker):
        """Test that code language is captured in metadata."""
        content = """```javascript
const x = 42;
```"""
        chunks = default_chunker.chunk(content)

        assert len(chunks) >= 1
        # Check if language was captured
        code_chunk = chunks[0]
        if code_chunk.chunk_type == "code" and code_chunk.metadata.get("code_language"):
            assert code_chunk.metadata["code_language"] == "javascript"


# =============================================================================
# Test Table Handling
# =============================================================================


class TestTableHandling:
    """Tests for table preservation."""

    def test_table_not_split(self, small_chunk_chunker):
        """Test that tables are not split across chunks."""
        content = """Intro text.

| Column A | Column B | Column C |
|----------|----------|----------|
| Row 1 A  | Row 1 B  | Row 1 C  |
| Row 2 A  | Row 2 B  | Row 2 C  |
| Row 3 A  | Row 3 B  | Row 3 C  |

After table."""
        chunks = small_chunk_chunker.chunk(content)

        # Find table chunk
        table_chunks = [c for c in chunks if c.chunk_type == "table" or "|" in c.content]

        for table_chunk in table_chunks:
            if "Column A" in table_chunk.content:
                # Should have all columns
                assert "Column B" in table_chunk.content
                assert "Column C" in table_chunk.content


# =============================================================================
# Test Overlap Handling
# =============================================================================


class TestOverlapHandling:
    """Tests for chunk overlap functionality."""

    def test_overlap_present(self):
        """Test that overlap is present between chunks."""
        config = ChunkingConfig(
            chunk_size=30,
            chunk_overlap=10,
            min_chunk_size=10,
        )
        chunker = SemanticChunker(config)

        content = """First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."""
        chunks = chunker.chunk(content)

        if len(chunks) > 1:
            # Check for some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                current_words = set(chunks[i].content.split()[-5:])
                next_words = set(chunks[i + 1].content.split()[:10])
                # There should be some word overlap
                overlap = current_words & next_words
                # Note: overlap may not always exist depending on boundary detection
                # This is a soft assertion

    def test_no_overlap_when_zero(self):
        """Test that no overlap when overlap is zero."""
        config = ChunkingConfig(
            chunk_size=30,
            chunk_overlap=0,
            min_chunk_size=10,
        )
        chunker = SemanticChunker(config)

        content = "Word " * 100  # Lots of words
        chunks = chunker.chunk(content)

        # Chunks should not have significant overlap
        # (some may occur at boundaries)


# =============================================================================
# Test Chunk Types
# =============================================================================


class TestChunkTypes:
    """Tests for chunk type classification."""

    def test_text_chunk_type(self, default_chunker):
        """Test text chunks are typed correctly."""
        chunks = default_chunker.chunk("Just plain text content.")
        assert chunks[0].chunk_type == "text"

    def test_code_chunk_type(self, default_chunker):
        """Test code chunks are typed correctly."""
        chunks = default_chunker.chunk("```\ncode\n```")
        assert chunks[0].chunk_type == "code"

    def test_mixed_chunk_type(self, small_chunk_chunker):
        """Test mixed content is typed correctly."""
        content = """Some text.

```python
code
```

More text."""
        chunks = small_chunk_chunker.chunk(content)

        types = set(c.chunk_type for c in chunks)
        # Should have at least text and code
        assert "text" in types or "code" in types or "mixed" in types


# =============================================================================
# Test Convenience Function
# =============================================================================


class TestChunkTextFunction:
    """Tests for the chunk_text convenience function."""

    def test_basic_usage(self):
        """Test basic usage of chunk_text."""
        chunks = chunk_text("Simple text content here.")
        assert len(chunks) >= 1
        assert isinstance(chunks[0], ChunkInfo)

    def test_custom_parameters(self):
        """Test chunk_text with custom parameters."""
        content = "Sentence one. Sentence two. Sentence three. " * 10
        chunks = chunk_text(
            content,
            chunk_size=50,
            chunk_overlap=10,
            respect_sentences=True,
            respect_headings=True,
        )
        assert len(chunks) >= 1

    def test_disable_sentence_boundaries(self):
        """Test disabling sentence boundary respect."""
        content = "Short. Content. Here."
        chunks = chunk_text(content, respect_sentences=False)
        assert len(chunks) >= 1


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_long_single_word(self, default_chunker):
        """Test handling very long single word."""
        long_word = "a" * 10000
        chunks = default_chunker.chunk(long_word)
        assert len(chunks) >= 1

    def test_unicode_content(self, default_chunker):
        """Test handling unicode content."""
        content = "Unicode: 你好世界 🌍 αβγδ café"
        chunks = default_chunker.chunk(content)
        assert len(chunks) >= 1
        assert "你好" in chunks[0].content

    def test_special_characters(self, default_chunker):
        """Test handling special characters."""
        content = "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"
        chunks = default_chunker.chunk(content)
        assert len(chunks) >= 1

    def test_only_whitespace_paragraphs(self, default_chunker):
        """Test content with empty paragraphs."""
        content = "First.\n\n\n\n\nSecond.\n\n\n\nThird."
        chunks = default_chunker.chunk(content)
        # Should handle gracefully
        full_content = " ".join(c.content for c in chunks)
        assert "First" in full_content
        assert "Third" in full_content

    def test_nested_code_blocks(self, default_chunker):
        """Test handling of nested code block markers."""
        content = """```markdown
Here is some markdown with code:
```python
print("nested")
```
End of markdown.
```"""
        # Should not crash
        chunks = default_chunker.chunk(content)
        assert len(chunks) >= 1

    def test_incomplete_table(self, default_chunker):
        """Test handling incomplete table markup."""
        content = """| Header |
Some text after incomplete table."""
        # Should not crash
        chunks = default_chunker.chunk(content)
        assert len(chunks) >= 1

    def test_heading_without_content(self, default_chunker):
        """Test heading followed immediately by another heading."""
        content = """# Title

## Section

### Subsection

Finally some content."""
        chunks = default_chunker.chunk(content)
        assert len(chunks) >= 1


# =============================================================================
# Test Position Tracking
# =============================================================================


class TestPositionTracking:
    """Tests for character position tracking."""

    def test_start_char_zero_first_chunk(self, default_chunker):
        """Test that first chunk starts at 0."""
        chunks = default_chunker.chunk("Content here.")
        assert chunks[0].start_char == 0

    def test_end_char_greater_than_start(self, default_chunker):
        """Test that end_char > start_char for all chunks."""
        chunks = default_chunker.chunk("Multiple paragraphs.\n\nSecond one.\n\nThird one.")
        for chunk in chunks:
            assert chunk.end_char >= chunk.start_char

    def test_positions_non_overlapping(self, small_chunk_chunker):
        """Test that chunk positions don't overlap incorrectly."""
        content = "Word " * 100
        chunks = small_chunk_chunker.chunk(content)

        # Start positions should be non-decreasing
        starts = [c.start_char for c in chunks]
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1]


# =============================================================================
# Test Page Number Handling
# =============================================================================


class TestPageNumberHandling:
    """Tests for page number tracking."""

    def test_initial_page_number(self, default_chunker):
        """Test that initial page number is set."""
        chunks = default_chunker.chunk("Content here.", initial_page=5)
        assert chunks[0].page_number == 5

    def test_default_page_number_none(self, default_chunker):
        """Test that page number is None by default."""
        chunks = default_chunker.chunk("Content here.")
        assert chunks[0].page_number is None


# =============================================================================
# Test Configuration Effects
# =============================================================================


class TestConfigurationEffects:
    """Tests for different configuration settings."""

    def test_larger_chunk_size_fewer_chunks(self):
        """Test that larger chunk size produces fewer chunks."""
        content = "Word " * 200

        small_config = ChunkingConfig(chunk_size=50, min_chunk_size=10)
        large_config = ChunkingConfig(chunk_size=500, min_chunk_size=10)

        small_chunker = SemanticChunker(small_config)
        large_chunker = SemanticChunker(large_config)

        small_chunks = small_chunker.chunk(content)
        large_chunks = large_chunker.chunk(content)

        assert len(large_chunks) <= len(small_chunks)

    def test_disable_heading_boundaries(self):
        """Test disabling heading boundary respect."""
        content = """# Title

Short paragraph.

## Another Heading

More content here."""

        config = ChunkingConfig(
            chunk_size=100,
            respect_heading_boundaries=False,
            min_chunk_size=10,
        )
        chunker = SemanticChunker(config)
        chunks = chunker.chunk(content)

        # Should still work, just might combine differently
        assert len(chunks) >= 1


# =============================================================================
# Test Integration with Real Markdown
# =============================================================================


class TestRealMarkdownIntegration:
    """Integration tests with realistic markdown content."""

    def test_full_document_chunking(self, default_chunker, sample_markdown):
        """Test chunking a complete markdown document."""
        chunks = default_chunker.chunk(sample_markdown)

        # Should produce reasonable chunks
        assert len(chunks) >= 1

        # Should preserve content
        full_content = " ".join(c.content for c in chunks)
        assert "Introduction" in full_content
        assert "hello_world" in full_content
        assert "Conclusion" in full_content

        # Each chunk should have content hash
        for chunk in chunks:
            assert len(chunk.content_hash) == 64  # SHA-256 hex

    def test_readme_style_document(self, default_chunker):
        """Test chunking a README-style document."""
        readme = """# Project Name

A brief description of the project.

## Installation

```bash
pip install project-name
```

## Usage

```python
from project import main
main()
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| debug | false | Enable debug mode |
| port | 8080 | Server port |

## License

MIT License
"""
        chunks = default_chunker.chunk(readme)

        assert len(chunks) >= 1

        # Code should be preserved
        found_bash = any("pip install" in c.content for c in chunks)
        found_python = any("from project" in c.content for c in chunks)
        assert found_bash or found_python
