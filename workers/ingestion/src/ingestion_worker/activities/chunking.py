"""
Semantic chunking module for document processing.

This module provides intelligent document chunking that respects semantic
boundaries such as sentences, paragraphs, headings, code blocks, and tables.

Features:
- Sentence boundary detection using regex-based tokenization
- Heading hierarchy tracking for context preservation
- Code block integrity (never splits code blocks)
- Table integrity (never splits tables)
- Configurable chunk size and overlap
- Token estimation for LLM compatibility

The implementation is optimized for RAG (Retrieval Augmented Generation)
use cases where maintaining semantic coherence is critical for retrieval quality.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ChunkType(StrEnum):
    """Type of content in a chunk."""

    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"
    MIXED = "mixed"  # Contains multiple types


@dataclass
class ChunkingConfig:
    """Configuration for semantic chunking.

    Attributes:
        chunk_size: Target chunk size in tokens (approximate)
        chunk_overlap: Number of tokens to overlap between chunks
        respect_sentence_boundaries: Try to break at sentence boundaries
        respect_heading_boundaries: Keep headings with their content
        preserve_code_blocks: Never split code blocks
        preserve_tables: Never split tables
        min_chunk_size: Minimum chunk size to create (avoid tiny chunks)
        max_chunk_size: Maximum chunk size (hard limit)
        tokens_per_word: Estimation factor for token counting
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    respect_sentence_boundaries: bool = True
    respect_heading_boundaries: bool = True
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    tokens_per_word: float = 1.3  # Common approximation for English text


@dataclass
class HeadingInfo:
    """Information about a heading in the document.

    Attributes:
        level: Heading level (1-6)
        text: Heading text
        start_char: Start character position in original document
    """

    level: int
    text: str
    start_char: int


@dataclass
class SemanticUnit:
    """A semantic unit of content from the document.

    Represents a coherent piece of content that should not be split,
    such as a paragraph, code block, table, or heading with its content.

    Attributes:
        content: The text content
        unit_type: Type of content (text, code, table, etc.)
        start_char: Start position in original document
        end_char: End position in original document
        heading_level: If this is a heading, its level (1-6)
        estimated_tokens: Estimated token count
    """

    content: str
    unit_type: ChunkType
    start_char: int
    end_char: int
    heading_level: int | None = None
    estimated_tokens: int = 0


@dataclass
class ChunkInfo:
    """Information about a single chunk.

    Enhanced from basic chunking to include semantic context.

    Attributes:
        sequence_number: Position of this chunk (0-indexed)
        content: The chunk text content
        content_hash: SHA-256 hash of content for deduplication
        start_char: Start character position in original document
        end_char: End character position in original document
        token_count: Estimated token count
        page_number: Page number if known
        heading_context: List of headings above this chunk (hierarchy path)
        chunk_type: Type of content (text, code, table, etc.)
        metadata: Additional metadata dictionary
    """

    sequence_number: int
    content: str
    content_hash: str
    start_char: int
    end_char: int
    token_count: int
    page_number: int | None = None
    # New semantic fields
    heading_context: list[str] = field(default_factory=list)
    chunk_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticChunker:
    """Semantic document chunker that respects content boundaries.

    This chunker analyzes document structure and creates chunks that:
    - Preserve sentence boundaries when possible
    - Keep headings with their following content
    - Never split code blocks or tables
    - Track heading hierarchy for context
    - Support configurable chunk sizes and overlap

    Example:
        >>> config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        >>> chunker = SemanticChunker(config)
        >>> chunks = chunker.chunk(markdown_content)
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.sequence_number}: {chunk.token_count} tokens")
    """

    # Regex patterns for parsing
    SENTENCE_END_PATTERN = re.compile(
        r"(?<=[.!?])"  # After sentence-ending punctuation
        r"(?:\s|$)"  # Followed by whitespace or end
        r'(?!["\'])'  # Not followed by quote (e.g., "Hello!" she said)
    )

    # Markdown patterns
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    TABLE_PATTERN = re.compile(
        r"^\|.*\|$\n"  # Header row
        r"^\|[-:\s|]+\|$\n"  # Separator row
        r"(?:^\|.*\|$\n?)+",  # Data rows
        re.MULTILINE,
    )
    LIST_ITEM_PATTERN = re.compile(r"^[\s]*[-*+]\s+.+$|^[\s]*\d+\.\s+.+$", re.MULTILINE)

    def __init__(self, config: ChunkingConfig | None = None):
        """Initialize the semantic chunker.

        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self.config = config or ChunkingConfig()
        self._heading_stack: list[HeadingInfo] = []

    def chunk(self, content: str, initial_page: int | None = None) -> list[ChunkInfo]:
        """Chunk document content into semantic units.

        Args:
            content: The document content (Markdown preferred)
            initial_page: Starting page number for page tracking

        Returns:
            List of ChunkInfo objects with semantic metadata
        """
        if not content or not content.strip():
            return []

        # Reset state
        self._heading_stack = []

        # Parse into semantic units
        units = self._parse_semantic_units(content)

        # Group units into chunks
        chunks = self._create_chunks(units, initial_page)

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in text.

        Uses a simple word-based heuristic. For more accuracy,
        consider using tiktoken or similar library.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        words = len(text.split())
        return int(words * self.config.tokens_per_word)

    def _parse_semantic_units(self, content: str) -> list[SemanticUnit]:
        """Parse content into semantic units.

        Identifies and extracts:
        - Code blocks (preserved intact)
        - Tables (preserved intact)
        - Headings (marked for context tracking)
        - Paragraphs and text

        Args:
            content: Document content

        Returns:
            List of semantic units in document order
        """
        units: list[SemanticUnit] = []
        current_pos = 0

        # Find all special blocks (code, tables) first
        special_blocks: list[tuple[int, int, str, ChunkType]] = []

        # Find code blocks
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            special_blocks.append((match.start(), match.end(), match.group(), ChunkType.CODE))

        # Find tables
        for match in self.TABLE_PATTERN.finditer(content):
            special_blocks.append((match.start(), match.end(), match.group(), ChunkType.TABLE))

        # Sort by position
        special_blocks.sort(key=lambda x: x[0])

        # Process content, respecting special blocks
        for block_start, block_end, block_content, block_type in special_blocks:
            # Process text before this block
            if current_pos < block_start:
                text_before = content[current_pos:block_start]
                units.extend(self._parse_text_units(text_before, current_pos))

            # Add the special block
            units.append(
                SemanticUnit(
                    content=block_content,
                    unit_type=block_type,
                    start_char=block_start,
                    end_char=block_end,
                    estimated_tokens=self._estimate_tokens(block_content),
                )
            )
            current_pos = block_end

        # Process remaining text
        if current_pos < len(content):
            remaining = content[current_pos:]
            units.extend(self._parse_text_units(remaining, current_pos))

        return units

    def _parse_text_units(self, text: str, offset: int) -> list[SemanticUnit]:
        """Parse regular text into semantic units.

        Handles:
        - Headings
        - Paragraphs
        - List items

        Args:
            text: Text content to parse
            offset: Character offset in original document

        Returns:
            List of semantic units
        """
        units: list[SemanticUnit] = []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text)
        current_pos = offset

        for para in paragraphs:
            if not para.strip():
                # Track empty space for position calculation
                current_pos += len(para) + 2  # +2 for the split delimiter
                continue

            para_start = current_pos
            para_end = current_pos + len(para)

            # Check if this is a heading
            heading_match = self.HEADING_PATTERN.match(para.strip())
            if heading_match:
                level = len(heading_match.group(1))
                units.append(
                    SemanticUnit(
                        content=para.strip(),
                        unit_type=ChunkType.HEADING,
                        start_char=para_start,
                        end_char=para_end,
                        heading_level=level,
                        estimated_tokens=self._estimate_tokens(para),
                    )
                )
            # Check if this is a list
            elif self.LIST_ITEM_PATTERN.match(para.strip()):
                units.append(
                    SemanticUnit(
                        content=para.strip(),
                        unit_type=ChunkType.LIST,
                        start_char=para_start,
                        end_char=para_end,
                        estimated_tokens=self._estimate_tokens(para),
                    )
                )
            else:
                # Regular text paragraph
                units.append(
                    SemanticUnit(
                        content=para.strip(),
                        unit_type=ChunkType.TEXT,
                        start_char=para_start,
                        end_char=para_end,
                        estimated_tokens=self._estimate_tokens(para),
                    )
                )

            # Move position, accounting for paragraph delimiter
            current_pos = para_end + 2  # +2 for newlines

        return units

    def _create_chunks(
        self, units: list[SemanticUnit], initial_page: int | None = None
    ) -> list[ChunkInfo]:
        """Create chunks from semantic units.

        Groups units into chunks respecting:
        - Token limits
        - Boundary preservation settings
        - Overlap requirements

        Args:
            units: List of semantic units
            initial_page: Starting page number

        Returns:
            List of chunks with metadata
        """
        if not units:
            return []

        chunks: list[ChunkInfo] = []
        current_units: list[SemanticUnit] = []
        current_tokens = 0
        sequence = 0

        # Track heading context
        heading_context: list[str] = []

        for unit in units:
            # Update heading context
            if unit.unit_type == ChunkType.HEADING and unit.heading_level:
                # Pop headings of same or lower level
                while heading_context and self._heading_stack:
                    if self._heading_stack[-1].level >= unit.heading_level:
                        self._heading_stack.pop()
                        heading_context.pop()
                    else:
                        break

                # Add new heading
                self._heading_stack.append(
                    HeadingInfo(
                        level=unit.heading_level,
                        text=unit.content,
                        start_char=unit.start_char,
                    )
                )
                heading_context.append(unit.content)

            # Check if adding this unit would exceed chunk size
            would_exceed = current_tokens + unit.estimated_tokens > self.config.chunk_size

            # Special handling for large units (code blocks, tables)
            is_large_unit = unit.estimated_tokens > self.config.chunk_size

            # Decide whether to start a new chunk
            should_start_new_chunk = False

            if would_exceed and current_units:
                should_start_new_chunk = True
            elif is_large_unit and current_units:
                # Large unit - flush current and handle separately
                should_start_new_chunk = True

            # Respect heading boundaries - don't split heading from content
            if (
                self.config.respect_heading_boundaries
                and unit.unit_type == ChunkType.HEADING
                and current_units
                and current_tokens > self.config.min_chunk_size
            ):
                should_start_new_chunk = True

            if should_start_new_chunk:
                # Create chunk from current units
                chunk = self._create_chunk_from_units(
                    current_units,
                    sequence,
                    heading_context.copy(),
                    initial_page,
                )
                chunks.append(chunk)
                sequence += 1

                # Handle overlap
                overlap_units = self._get_overlap_units(current_units)
                current_units = overlap_units
                current_tokens = sum(u.estimated_tokens for u in overlap_units)

            # Add current unit
            current_units.append(unit)
            current_tokens += unit.estimated_tokens

            # If this unit alone exceeds max size, create chunk immediately
            if is_large_unit and len(current_units) == 1:
                chunk = self._create_chunk_from_units(
                    current_units,
                    sequence,
                    heading_context.copy(),
                    initial_page,
                )
                chunks.append(chunk)
                sequence += 1
                current_units = []
                current_tokens = 0

        # Create final chunk
        if current_units:
            chunk = self._create_chunk_from_units(
                current_units,
                sequence,
                heading_context.copy(),
                initial_page,
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap_units(self, units: list[SemanticUnit]) -> list[SemanticUnit]:
        """Get units for overlap from the end of current chunk.

        Tries to get approximately `chunk_overlap` tokens worth of content,
        respecting sentence boundaries.

        Args:
            units: Units from current chunk

        Returns:
            Units to carry over for overlap
        """
        if not units or self.config.chunk_overlap <= 0:
            return []

        overlap_units: list[SemanticUnit] = []
        overlap_tokens = 0

        # Work backwards through units
        for unit in reversed(units):
            if overlap_tokens >= self.config.chunk_overlap:
                break

            # Don't carry over headings as overlap (they're context)
            if unit.unit_type == ChunkType.HEADING:
                continue

            # For text units, try to get just the last sentences
            if unit.unit_type == ChunkType.TEXT and self.config.respect_sentence_boundaries:
                sentences = self._split_sentences(unit.content)
                if len(sentences) > 1:
                    # Take last few sentences for overlap
                    overlap_sentences = []
                    sentence_tokens = 0
                    for sentence in reversed(sentences):
                        tokens = self._estimate_tokens(sentence)
                        if sentence_tokens + tokens > self.config.chunk_overlap:
                            break
                        overlap_sentences.insert(0, sentence)
                        sentence_tokens += tokens

                    if overlap_sentences:
                        overlap_content = " ".join(overlap_sentences)
                        overlap_units.insert(
                            0,
                            SemanticUnit(
                                content=overlap_content,
                                unit_type=ChunkType.TEXT,
                                start_char=unit.end_char - len(overlap_content),
                                end_char=unit.end_char,
                                estimated_tokens=sentence_tokens,
                            ),
                        )
                        overlap_tokens += sentence_tokens
                        continue

            # Add whole unit if small enough
            if overlap_tokens + unit.estimated_tokens <= self.config.chunk_overlap * 1.5:
                overlap_units.insert(0, unit)
                overlap_tokens += unit.estimated_tokens

        return overlap_units

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Uses regex-based sentence boundary detection.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if not text:
            return []

        # Split on sentence boundaries
        sentences = self.SENTENCE_END_PATTERN.split(text)

        # Clean up and filter empty
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _create_chunk_from_units(
        self,
        units: list[SemanticUnit],
        sequence: int,
        heading_context: list[str],
        page_number: int | None,
    ) -> ChunkInfo:
        """Create a ChunkInfo from a list of semantic units.

        Args:
            units: Units to combine into a chunk
            sequence: Chunk sequence number
            heading_context: Current heading hierarchy
            page_number: Page number if known

        Returns:
            ChunkInfo with combined content and metadata
        """
        # Combine content
        content_parts = [u.content for u in units]
        content = "\n\n".join(content_parts)

        # Calculate positions
        start_char = units[0].start_char if units else 0
        end_char = units[-1].end_char if units else 0

        # Determine chunk type
        types_present = set(u.unit_type for u in units)
        if len(types_present) == 1:
            chunk_type = types_present.pop()
        else:
            chunk_type = ChunkType.MIXED

        # Calculate token count
        token_count = sum(u.estimated_tokens for u in units)

        # Create content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Build metadata
        metadata: dict[str, Any] = {
            "unit_count": len(units),
            "types_present": [t.value for t in types_present] if len(types_present) > 1 else [],
        }

        # Add code language if this is a code chunk
        if chunk_type == ChunkType.CODE:
            # Try to extract language from code fence
            code_match = re.match(r"```(\w+)?", content)
            if code_match and code_match.group(1):
                metadata["code_language"] = code_match.group(1)

        return ChunkInfo(
            sequence_number=sequence,
            content=content,
            content_hash=content_hash,
            start_char=start_char,
            end_char=end_char,
            token_count=token_count,
            page_number=page_number,
            heading_context=heading_context,
            chunk_type=chunk_type.value if isinstance(chunk_type, ChunkType) else str(chunk_type),
            metadata=metadata,
        )


def chunk_text(
    content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    respect_sentences: bool = True,
    respect_headings: bool = True,
) -> list[ChunkInfo]:
    """Convenience function to chunk text content.

    Args:
        content: Text content to chunk
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        respect_sentences: Whether to respect sentence boundaries
        respect_headings: Whether to respect heading boundaries

    Returns:
        List of ChunkInfo objects
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        respect_sentence_boundaries=respect_sentences,
        respect_heading_boundaries=respect_headings,
    )
    chunker = SemanticChunker(config)
    return chunker.chunk(content)
