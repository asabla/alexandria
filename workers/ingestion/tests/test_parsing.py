"""
Unit tests for Docling document parsing activity.

These tests verify the parsing module functionality with mocked Docling components
to allow testing without the full Docling installation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pytest

from ingestion_worker.activities.parsing import (
    DoclingConfig,
    DoclingFormat,
    DoclingParseResult,
    OCREngine,
    ParsedImage,
    ParsedTable,
    SUPPORTED_DOCUMENT_TYPES,
    _count_words,
    _detect_language,
    _extract_images,
    _extract_tables,
    is_docling_supported,
    parse_document_with_docling,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_docling_document():
    """Create a mock Docling document object."""
    doc = MagicMock()
    doc.export_to_markdown.return_value = "# Test Document\n\nThis is test content."
    doc.export_to_text.return_value = "Test Document This is test content."
    doc.export_to_dict.return_value = {
        "schema_name": "docling_document",
        "version": "1.0.0",
        "body": {"text_content": "Test content"},
    }
    doc.pages = [MagicMock(), MagicMock()]  # 2 pages
    doc.title = "Test Document"
    doc.pictures = []

    # Mock iterate_items for table extraction
    doc.iterate_items.return_value = []

    return doc


@pytest.fixture
def mock_docling_converter(mock_docling_document):
    """Create a mock DocumentConverter."""
    converter = MagicMock()
    result = MagicMock()
    result.document = mock_docling_document
    converter.convert.return_value = result
    return converter


@pytest.fixture
def sample_pdf_file(tmp_path):
    """Create a sample PDF file for testing."""
    # Create a minimal PDF-like file (not a real PDF, just for path testing)
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 sample content")
    return pdf_path


@pytest.fixture
def sample_docx_file(tmp_path):
    """Create a sample DOCX file for testing."""
    docx_path = tmp_path / "sample.docx"
    docx_path.write_bytes(b"PK\x03\x04 docx content")  # DOCX magic bytes
    return docx_path


@pytest.fixture
def docling_config():
    """Create a default DoclingConfig."""
    return DoclingConfig()


# ============================================================================
# DoclingConfig Tests
# ============================================================================


class TestDoclingConfig:
    """Tests for DoclingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DoclingConfig()

        assert config.enable_ocr is True
        assert config.ocr_engine == OCREngine.EASYOCR
        assert config.enable_table_structure is True
        assert config.enable_cell_matching is True
        assert config.enable_code_enrichment is False
        assert config.generate_page_images is False
        assert config.generate_picture_images is True
        assert config.image_resolution_scale == 2.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DoclingConfig(
            enable_ocr=False,
            ocr_engine=OCREngine.TESSERACT,
            enable_table_structure=False,
            enable_cell_matching=False,
            image_resolution_scale=1.5,
        )

        assert config.enable_ocr is False
        assert config.ocr_engine == OCREngine.TESSERACT
        assert config.enable_table_structure is False
        assert config.enable_cell_matching is False
        assert config.image_resolution_scale == 1.5

    def test_ocr_engine_enum(self):
        """Test OCREngine enum values."""
        assert OCREngine.EASYOCR == "easyocr"
        assert OCREngine.TESSERACT == "tesseract"
        assert OCREngine.RAPID == "rapid"
        assert OCREngine.MAC == "mac"
        assert OCREngine.NONE == "none"


class TestDoclingFormat:
    """Tests for DoclingFormat enum."""

    def test_format_values(self):
        """Test DoclingFormat enum values."""
        assert DoclingFormat.MARKDOWN == "markdown"
        assert DoclingFormat.JSON == "json"
        assert DoclingFormat.TEXT == "text"


# ============================================================================
# ParsedTable Tests
# ============================================================================


class TestParsedTable:
    """Tests for ParsedTable dataclass."""

    def test_create_table(self):
        """Test creating a ParsedTable."""
        table = ParsedTable(
            index=0,
            page_number=1,
            markdown="| Col1 | Col2 |\n|------|------|\n| A    | B    |",
            html="<table><tr><td>A</td><td>B</td></tr></table>",
            rows=2,
            cols=2,
            caption="Test Table",
        )

        assert table.index == 0
        assert table.page_number == 1
        assert "Col1" in table.markdown
        assert "<table>" in table.html
        assert table.rows == 2
        assert table.cols == 2
        assert table.caption == "Test Table"

    def test_table_without_caption(self):
        """Test table with no caption."""
        table = ParsedTable(
            index=1,
            page_number=None,
            markdown="|A|B|",
            html="<table></table>",
            rows=1,
            cols=2,
        )

        assert table.caption is None
        assert table.page_number is None


# ============================================================================
# ParsedImage Tests
# ============================================================================


class TestParsedImage:
    """Tests for ParsedImage dataclass."""

    def test_create_image(self):
        """Test creating a ParsedImage."""
        image = ParsedImage(
            index=0,
            page_number=1,
            image_path="/tmp/image_0.png",
            mime_type="image/png",
            width=800,
            height=600,
            caption="Figure 1",
            alt_text="A test image",
        )

        assert image.index == 0
        assert image.page_number == 1
        assert image.image_path == "/tmp/image_0.png"
        assert image.mime_type == "image/png"
        assert image.width == 800
        assert image.height == 600
        assert image.caption == "Figure 1"
        assert image.alt_text == "A test image"

    def test_image_minimal(self):
        """Test image with minimal fields."""
        image = ParsedImage(index=0, page_number=None)

        assert image.index == 0
        assert image.page_number is None
        assert image.image_path is None
        assert image.mime_type is None
        assert image.width is None
        assert image.height is None
        assert image.caption is None
        assert image.alt_text is None


# ============================================================================
# DoclingParseResult Tests
# ============================================================================


class TestDoclingParseResult:
    """Tests for DoclingParseResult dataclass."""

    def test_create_result(self):
        """Test creating a parse result."""
        result = DoclingParseResult(
            content_markdown="# Title\n\nContent",
            content_json='{"text": "content"}',
            content_text="Title Content",
            page_count=5,
            word_count=100,
            language="en",
            title="Test Document",
            tables=[
                ParsedTable(index=0, page_number=1, markdown="|A|", html="<table>", rows=1, cols=1)
            ],
            images=[ParsedImage(index=0, page_number=1)],
            metadata={"source": "test"},
            processing_time_seconds=1.5,
        )

        assert "# Title" in result.content_markdown
        assert result.page_count == 5
        assert result.word_count == 100
        assert result.language == "en"
        assert result.title == "Test Document"
        assert len(result.tables) == 1
        assert len(result.images) == 1
        assert result.processing_time_seconds == 1.5

    def test_result_defaults(self):
        """Test parse result with default values."""
        result = DoclingParseResult(
            content_markdown="",
            content_json="{}",
            content_text="",
        )

        assert result.page_count is None
        assert result.word_count is None
        assert result.language is None
        assert result.title is None
        assert result.tables == []
        assert result.images == []
        assert result.metadata == {}
        assert result.processing_time_seconds == 0.0


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_count_words(self):
        """Test word counting."""
        assert _count_words("hello world") == 2
        assert _count_words("") == 0
        assert _count_words("one") == 1
        assert _count_words("  multiple   spaces  ") == 2
        assert _count_words("line1\nline2\nline3") == 3

    def test_detect_language(self):
        """Test language detection (simple fallback)."""
        # Currently returns 'en' as fallback
        assert _detect_language("Hello world") == "en"
        assert _detect_language("Bonjour le monde") == "en"  # Still returns en (fallback)
        assert _detect_language("") == "en"

    def test_is_docling_supported(self):
        """Test document type support checking."""
        # Supported types
        assert is_docling_supported("pdf") is True
        assert is_docling_supported("PDF") is True
        assert is_docling_supported("docx") is True
        assert is_docling_supported("pptx") is True
        assert is_docling_supported("xlsx") is True
        assert is_docling_supported("html") is True
        assert is_docling_supported("md") is True
        assert is_docling_supported("txt") is True
        assert is_docling_supported("png") is True
        assert is_docling_supported("jpg") is True
        assert is_docling_supported("jpeg") is True
        assert is_docling_supported("tiff") is True

        # Unsupported types
        assert is_docling_supported("mp3") is False
        assert is_docling_supported("mp4") is False
        assert is_docling_supported("unknown") is False


class TestSupportedDocumentTypes:
    """Tests for SUPPORTED_DOCUMENT_TYPES constant."""

    def test_pdf_types(self):
        """Test PDF MIME types."""
        assert "application/pdf" in SUPPORTED_DOCUMENT_TYPES["pdf"]

    def test_office_types(self):
        """Test Office document MIME types."""
        assert (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            in SUPPORTED_DOCUMENT_TYPES["docx"]
        )
        assert (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            in SUPPORTED_DOCUMENT_TYPES["pptx"]
        )
        assert (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            in SUPPORTED_DOCUMENT_TYPES["xlsx"]
        )

    def test_image_types(self):
        """Test image MIME types."""
        assert "image/png" in SUPPORTED_DOCUMENT_TYPES["png"]
        assert "image/jpeg" in SUPPORTED_DOCUMENT_TYPES["jpg"]
        assert "image/jpeg" in SUPPORTED_DOCUMENT_TYPES["jpeg"]
        assert "image/tiff" in SUPPORTED_DOCUMENT_TYPES["tiff"]

    def test_text_types(self):
        """Test text MIME types."""
        assert "text/plain" in SUPPORTED_DOCUMENT_TYPES["txt"]
        assert "text/html" in SUPPORTED_DOCUMENT_TYPES["html"]
        assert "text/markdown" in SUPPORTED_DOCUMENT_TYPES["md"]


# ============================================================================
# Table Extraction Tests
# ============================================================================


class TestTableExtraction:
    """Tests for table extraction function."""

    def test_extract_tables_empty(self):
        """Test extracting tables from document with no tables."""
        doc = MagicMock()
        doc.iterate_items.return_value = []

        tables = _extract_tables(doc)

        assert tables == []

    def test_extract_tables_with_error(self):
        """Test table extraction handles errors gracefully."""
        doc = MagicMock()
        doc.iterate_items.side_effect = Exception("Iteration failed")

        tables = _extract_tables(doc)

        assert tables == []


# ============================================================================
# Image Extraction Tests
# ============================================================================


class TestImageExtraction:
    """Tests for image extraction function."""

    def test_extract_images_empty(self):
        """Test extracting images from document with no images."""
        doc = MagicMock()
        doc.pictures = []

        images = _extract_images(doc)

        assert images == []

    def test_extract_images_with_pictures(self):
        """Test extracting images from document with pictures."""
        doc = MagicMock()

        # Create mock picture
        picture = MagicMock()
        picture.image = MagicMock()
        picture.image.width = 100
        picture.image.height = 200
        picture.prov = [MagicMock()]
        picture.prov[0].page_no = 1
        picture.caption = "Test caption"
        picture.alt_text = None

        doc.pictures = [picture]

        images = _extract_images(doc)

        assert len(images) == 1
        assert images[0].index == 0
        assert images[0].page_number == 1
        assert images[0].width == 100
        assert images[0].height == 200
        assert images[0].caption == "Test caption"

    def test_extract_images_handles_errors(self):
        """Test image extraction handles errors gracefully."""
        doc = MagicMock()
        # Accessing pictures raises an exception
        type(doc).pictures = property(lambda self: (_ for _ in ()).throw(Exception("No pictures")))

        images = _extract_images(doc)

        assert images == []

    def test_extract_images_saves_to_disk(self, tmp_path):
        """Test extracting images and saving to disk."""
        doc = MagicMock()

        # Create mock picture with saveable image
        picture = MagicMock()
        picture.image = MagicMock()
        picture.image.width = 100
        picture.image.height = 100
        picture.prov = []
        picture.caption = None
        picture.alt_text = None

        doc.pictures = [picture]

        images = _extract_images(doc, output_dir=tmp_path)

        # Should attempt to save
        assert len(images) == 1
        picture.image.save.assert_called_once()


# ============================================================================
# parse_document_with_docling Tests (Mocked)
# ============================================================================


class TestParseDocumentWithDocling:
    """Tests for the main parsing function."""

    def test_file_not_found(self, tmp_path):
        """Test handling of non-existent file."""
        non_existent = tmp_path / "does_not_exist.pdf"

        with pytest.raises(FileNotFoundError):
            parse_document_with_docling(non_existent)

    @patch("ingestion_worker.activities.parsing._create_document_converter")
    def test_successful_parse(self, mock_create_converter, sample_pdf_file, mock_docling_converter):
        """Test successful document parsing."""
        mock_create_converter.return_value = mock_docling_converter

        result = parse_document_with_docling(sample_pdf_file)

        assert isinstance(result, DoclingParseResult)
        assert result.content_markdown == "# Test Document\n\nThis is test content."
        assert result.page_count == 2
        assert result.title == "Test Document"
        mock_docling_converter.convert.assert_called_once()

    @patch("ingestion_worker.activities.parsing._create_document_converter")
    def test_parse_with_custom_config(
        self, mock_create_converter, sample_pdf_file, mock_docling_converter
    ):
        """Test parsing with custom configuration."""
        mock_create_converter.return_value = mock_docling_converter

        config = DoclingConfig(
            enable_ocr=False,
            enable_table_structure=False,
        )

        result = parse_document_with_docling(sample_pdf_file, config=config)

        assert isinstance(result, DoclingParseResult)
        mock_create_converter.assert_called_once()

    @patch("ingestion_worker.activities.parsing._create_document_converter")
    def test_parse_with_heartbeat_callback(
        self, mock_create_converter, sample_pdf_file, mock_docling_converter
    ):
        """Test parsing with heartbeat callback."""
        mock_create_converter.return_value = mock_docling_converter
        heartbeat_messages = []

        def heartbeat_callback(message: str) -> None:
            heartbeat_messages.append(message)

        result = parse_document_with_docling(
            sample_pdf_file,
            heartbeat_callback=heartbeat_callback,
        )

        assert isinstance(result, DoclingParseResult)
        assert len(heartbeat_messages) > 0
        assert "Starting document conversion" in heartbeat_messages
        assert "Parsing complete" in heartbeat_messages

    @patch("ingestion_worker.activities.parsing._create_document_converter")
    def test_parse_returns_json_export(
        self, mock_create_converter, sample_pdf_file, mock_docling_converter
    ):
        """Test that JSON export is included."""
        mock_create_converter.return_value = mock_docling_converter

        result = parse_document_with_docling(sample_pdf_file)

        # Verify JSON content
        json_data = json.loads(result.content_json)
        assert "schema_name" in json_data
        assert json_data["schema_name"] == "docling_document"

    @patch("ingestion_worker.activities.parsing._create_document_converter")
    def test_parse_calculates_word_count(
        self, mock_create_converter, sample_pdf_file, mock_docling_converter
    ):
        """Test that word count is calculated."""
        mock_create_converter.return_value = mock_docling_converter

        result = parse_document_with_docling(sample_pdf_file)

        assert result.word_count is not None
        assert result.word_count > 0

    @patch("ingestion_worker.activities.parsing._create_document_converter")
    def test_parse_records_processing_time(
        self, mock_create_converter, sample_pdf_file, mock_docling_converter
    ):
        """Test that processing time is recorded."""
        mock_create_converter.return_value = mock_docling_converter

        result = parse_document_with_docling(sample_pdf_file)

        assert result.processing_time_seconds >= 0

    @patch("ingestion_worker.activities.parsing._create_document_converter")
    def test_parse_includes_metadata(
        self, mock_create_converter, sample_pdf_file, mock_docling_converter
    ):
        """Test that metadata is included."""
        mock_create_converter.return_value = mock_docling_converter

        result = parse_document_with_docling(sample_pdf_file)

        assert "source_file" in result.metadata
        assert "file_size" in result.metadata
        assert result.metadata["source_file"] == "sample.pdf"


# ============================================================================
# parse_with_docling Activity Tests (Mocked)
# ============================================================================


class TestParseWithDoclingActivity:
    """Tests for the Temporal activity function."""

    @pytest.fixture
    def mock_activity_context(self):
        """Create a mock activity context."""
        with patch("ingestion_worker.activities.parsing.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()
            yield mock_activity

    @pytest.fixture
    def mock_minio_download(self):
        """Mock MinIO download function."""
        with patch("ingestion_worker.activities.parsing.download_file_from_minio") as mock_download:
            # Create a proper async mock
            async def download_side_effect(bucket, key, local_path):
                # Create the file
                Path(local_path).write_bytes(b"%PDF-1.4 test content")

            mock_download.side_effect = download_side_effect
            yield mock_download

    @pytest.mark.asyncio
    @patch("ingestion_worker.activities.parsing._create_document_converter")
    @patch("ingestion_worker.activities.parsing.activity")
    async def test_activity_calls_docling(
        self, mock_activity, mock_create_converter, mock_docling_converter
    ):
        """Test that the activity calls Docling correctly."""
        from ingestion_worker.activities.parsing import parse_with_docling

        mock_activity.logger = MagicMock()
        mock_activity.heartbeat = MagicMock()
        mock_create_converter.return_value = mock_docling_converter

        # Mock the MinIO download
        with patch("ingestion_worker.activities.parsing.download_file_from_minio") as mock_download:

            async def download_side_effect(bucket, key, local_path):
                Path(local_path).write_bytes(b"%PDF-1.4 test")

            mock_download.side_effect = download_side_effect

            result = await parse_with_docling(
                document_id="doc-123",
                storage_bucket="documents",
                storage_key="test.pdf",
                document_type="pdf",
                skip_ocr=False,
                enable_table_structure=True,
            )

            assert result["content_markdown"] == "# Test Document\n\nThis is test content."
            assert result["page_count"] == 2
            assert "tables" in result
            assert "images" in result

    @pytest.mark.asyncio
    @patch("ingestion_worker.activities.parsing.activity")
    async def test_activity_handles_download_failure(self, mock_activity):
        """Test that the activity handles MinIO download failures."""
        from ingestion_worker.activities.parsing import parse_with_docling

        mock_activity.logger = MagicMock()
        mock_activity.heartbeat = MagicMock()

        with patch("ingestion_worker.activities.parsing.download_file_from_minio") as mock_download:
            mock_download.side_effect = Exception("Download failed")

            with pytest.raises(RuntimeError, match="Failed to download file"):
                await parse_with_docling(
                    document_id="doc-123",
                    storage_bucket="documents",
                    storage_key="test.pdf",
                    document_type="pdf",
                )

    @pytest.mark.asyncio
    @patch("ingestion_worker.activities.parsing._create_document_converter")
    @patch("ingestion_worker.activities.parsing.activity")
    async def test_activity_sends_heartbeats(self, mock_activity, mock_create_converter):
        """Test that the activity sends heartbeats during processing."""
        from ingestion_worker.activities.parsing import parse_with_docling

        mock_activity.logger = MagicMock()
        mock_activity.heartbeat = MagicMock()

        # Create mock converter that takes time
        mock_converter = MagicMock()
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "content"
        mock_doc.export_to_text.return_value = "content"
        mock_doc.export_to_dict.return_value = {}
        mock_doc.pages = []
        mock_doc.title = None
        mock_doc.pictures = []
        mock_doc.iterate_items.return_value = []

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter.convert.return_value = mock_result
        mock_create_converter.return_value = mock_converter

        with patch("ingestion_worker.activities.parsing.download_file_from_minio") as mock_download:

            async def download_side_effect(bucket, key, local_path):
                Path(local_path).write_bytes(b"test")

            mock_download.side_effect = download_side_effect

            await parse_with_docling(
                document_id="doc-123",
                storage_bucket="documents",
                storage_key="test.pdf",
                document_type="pdf",
            )

            # Verify heartbeats were called
            assert mock_activity.heartbeat.call_count >= 2


# ============================================================================
# Integration with document_activities Tests
# ============================================================================


class TestParseDocumentIntegration:
    """Tests for integration between parse_document and Docling."""

    @pytest.fixture
    def mock_parse_document_context(self):
        """Set up mocks for parse_document activity."""
        with patch("ingestion_worker.activities.document_activities.activity") as mock_activity:
            mock_activity.logger = MagicMock()
            mock_activity.heartbeat = MagicMock()
            yield mock_activity

    @pytest.mark.asyncio
    @patch("ingestion_worker.activities.parsing.is_docling_supported")
    @patch("ingestion_worker.activities.parsing.parse_with_docling")
    @patch("ingestion_worker.activities.document_activities.activity")
    async def test_parse_document_uses_docling_for_pdf(
        self, mock_activity, mock_parse_docling, mock_is_supported
    ):
        """Test that parse_document uses Docling for supported types."""
        from ingestion_worker.activities.document_activities import parse_document
        from ingestion_worker.workflows.document_ingestion import (
            ParseDocumentInput,
        )

        mock_activity.logger = MagicMock()
        mock_activity.heartbeat = MagicMock()
        mock_is_supported.return_value = True
        mock_parse_docling.return_value = {
            "content_markdown": "# Parsed Content",
            "content_json": "{}",
            "content_text": "Parsed Content",
            "page_count": 3,
            "word_count": 50,
            "language": "en",
            "title": "Test",
            "tables": [],
            "images": [],
            "metadata": {},
            "processing_time_seconds": 1.0,
        }

        input_data = ParseDocumentInput(
            document_id="doc-123",
            storage_bucket="documents",
            storage_key="test.pdf",
            document_type="pdf",
            skip_ocr=False,
        )

        result = await parse_document(input_data)

        mock_parse_docling.assert_called_once()
        assert result.content == "# Parsed Content"
        assert result.page_count == 3
        assert result.word_count == 50

    @pytest.mark.asyncio
    @patch("ingestion_worker.activities.parsing.is_docling_supported")
    @patch("ingestion_worker.activities.document_activities.activity")
    async def test_parse_document_handles_unsupported_type(self, mock_activity, mock_is_supported):
        """Test that parse_document handles unsupported document types."""
        from ingestion_worker.activities.document_activities import parse_document
        from ingestion_worker.workflows.document_ingestion import (
            ParseDocumentInput,
        )

        mock_activity.logger = MagicMock()
        mock_activity.heartbeat = MagicMock()
        mock_is_supported.return_value = False

        input_data = ParseDocumentInput(
            document_id="doc-123",
            storage_bucket="documents",
            storage_key="test.xyz",
            document_type="xyz",  # Unsupported type
        )

        result = await parse_document(input_data)

        assert "Unsupported document type" in result.content
        assert result.word_count == 0

    @pytest.mark.asyncio
    @patch("ingestion_worker.activities.parsing.is_docling_supported")
    @patch("ingestion_worker.activities.document_activities.activity")
    async def test_parse_document_handles_audio_video(self, mock_activity, mock_is_supported):
        """Test that parse_document handles audio/video types."""
        from ingestion_worker.activities.document_activities import parse_document
        from ingestion_worker.workflows.document_ingestion import (
            ParseDocumentInput,
        )

        mock_activity.logger = MagicMock()
        mock_activity.heartbeat = MagicMock()
        mock_is_supported.return_value = False

        input_data = ParseDocumentInput(
            document_id="doc-123",
            storage_bucket="documents",
            storage_key="test.mp3",
            document_type="audio",
        )

        result = await parse_document(input_data)

        assert "transcription pending" in result.content.lower()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_document_type(self):
        """Test handling of empty document type."""
        assert is_docling_supported("") is False

    def test_whitespace_document_type(self):
        """Test handling of whitespace document type."""
        assert is_docling_supported("   ") is False

    def test_none_handling_in_parse_result(self):
        """Test that parse result handles None values correctly."""
        result = DoclingParseResult(
            content_markdown="",
            content_json="null",
            content_text="",
            page_count=None,
            word_count=None,
            language=None,
            title=None,
        )

        assert result.page_count is None
        assert result.word_count is None
        assert result.language is None
        assert result.title is None

    def test_large_word_count(self):
        """Test word counting with large text."""
        large_text = " ".join(["word"] * 100000)
        count = _count_words(large_text)
        assert count == 100000

    def test_unicode_word_count(self):
        """Test word counting with unicode text."""
        unicode_text = "Hello world"
        count = _count_words(unicode_text)
        assert count == 2  # Should treat unicode as regular words
