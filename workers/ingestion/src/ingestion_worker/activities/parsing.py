"""
Document parsing activities using Docling.

This module provides document parsing capabilities using IBM's Docling library.
Docling supports various document formats including PDF, DOCX, PPTX, XLSX, HTML,
images (PNG, JPEG, TIFF), and more.

Features:
- Unified document conversion across formats
- OCR support for scanned PDFs and images
- Table structure extraction
- Image extraction
- Markdown and JSON export

The implementation is designed to work within Temporal activities with proper
heartbeat support for long-running operations.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from temporalio import activity

# Type aliases for Docling imports (will be imported at runtime)
# This allows the module to be imported without Docling installed for testing


class DoclingFormat(StrEnum):
    """Supported output formats for Docling parsing."""

    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


class OCREngine(StrEnum):
    """Available OCR engines for Docling."""

    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    RAPID = "rapid"
    MAC = "mac"  # macOS Vision framework
    NONE = "none"  # Disable OCR


@dataclass
class DoclingConfig:
    """Configuration for Docling document processing.

    Attributes:
        enable_ocr: Whether to enable OCR for scanned documents
        ocr_engine: Which OCR engine to use
        enable_table_structure: Whether to detect table structure
        enable_cell_matching: Whether to match cells in tables (more accurate but slower)
        enable_code_enrichment: Whether to enrich code blocks
        generate_page_images: Whether to generate page images
        generate_picture_images: Whether to extract embedded images
        image_resolution_scale: Scale factor for image resolution (1.0 = 72 DPI)
    """

    enable_ocr: bool = True
    ocr_engine: OCREngine = OCREngine.EASYOCR
    enable_table_structure: bool = True
    enable_cell_matching: bool = True
    enable_code_enrichment: bool = False
    generate_page_images: bool = False
    generate_picture_images: bool = True
    image_resolution_scale: float = 2.0  # 144 DPI


@dataclass
class ParsedTable:
    """Extracted table from a document.

    Attributes:
        index: Table index in the document
        page_number: Page number where table appears (1-based)
        markdown: Markdown representation of the table
        html: HTML representation of the table
        rows: Number of rows
        cols: Number of columns
        caption: Table caption if available
    """

    index: int
    page_number: int | None
    markdown: str
    html: str
    rows: int
    cols: int
    caption: str | None = None


@dataclass
class ParsedImage:
    """Extracted image from a document.

    Attributes:
        index: Image index in the document
        page_number: Page number where image appears (1-based)
        image_path: Path to extracted image file (if saved)
        mime_type: Image MIME type
        width: Image width in pixels
        height: Image height in pixels
        caption: Image caption if available
        alt_text: Alternative text if available
    """

    index: int
    page_number: int | None
    image_path: str | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    caption: str | None = None
    alt_text: str | None = None


@dataclass
class DoclingParseResult:
    """Result of parsing a document with Docling.

    Attributes:
        content_markdown: Document content in Markdown format
        content_json: Document content in JSON format (Docling native)
        content_text: Plain text content
        page_count: Number of pages in the document
        word_count: Approximate word count
        language: Detected primary language
        title: Document title if detected
        tables: List of extracted tables
        images: List of extracted images
        metadata: Additional metadata from the document
        processing_time_seconds: Time taken to process the document
    """

    content_markdown: str
    content_json: str
    content_text: str
    page_count: int | None = None
    word_count: int | None = None
    language: str | None = None
    title: str | None = None
    tables: list[ParsedTable] = field(default_factory=list)
    images: list[ParsedImage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_time_seconds: float = 0.0


def _get_minio_client():
    """Get MinIO client from environment configuration."""
    from alexandria_db.clients.minio import MinIOClient

    return MinIOClient(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )


def _get_docling_config(config: DoclingConfig) -> Any:
    """Build Docling pipeline options from config.

    Args:
        config: Docling configuration

    Returns:
        Configured PdfPipelineOptions (or general pipeline options)
    """
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableStructureOptions,
    )

    # Configure table structure options
    table_options = TableStructureOptions(
        do_cell_matching=config.enable_cell_matching,
    )

    # Configure PDF pipeline options
    pdf_options = PdfPipelineOptions(
        do_ocr=config.enable_ocr,
        do_table_structure=config.enable_table_structure,
        table_structure_options=table_options,
        generate_page_images=config.generate_page_images,
        generate_picture_images=config.generate_picture_images,
        images_scale=config.image_resolution_scale,
    )

    # Configure OCR engine if OCR is enabled
    if config.enable_ocr and config.ocr_engine != OCREngine.NONE:
        # The OCR engine is configured through environment or Docling defaults
        # Each engine requires its optional dependency to be installed
        pass  # Docling auto-detects available OCR engines

    return pdf_options


def _create_document_converter(config: DoclingConfig) -> Any:
    """Create a configured Docling DocumentConverter.

    Args:
        config: Docling configuration

    Returns:
        Configured DocumentConverter instance
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat

    pdf_options = _get_docling_config(config)

    # Create converter with format-specific options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        }
    )

    return converter


def _extract_tables(doc: Any) -> list[ParsedTable]:
    """Extract tables from a Docling document.

    Args:
        doc: Docling document object

    Returns:
        List of ParsedTable objects
    """
    tables: list[ParsedTable] = []

    try:
        # Iterate through document items to find tables
        for i, (item, level) in enumerate(doc.iterate_items()):
            if hasattr(item, "export_to_markdown") and "table" in type(item).__name__.lower():
                try:
                    markdown = item.export_to_markdown()
                    html = getattr(item, "export_to_html", lambda: markdown)()

                    # Get table dimensions
                    rows = getattr(item, "num_rows", 0)
                    cols = getattr(item, "num_cols", 0)

                    # Get page number if available
                    page_num = None
                    if hasattr(item, "prov") and item.prov:
                        page_num = getattr(item.prov[0], "page_no", None)

                    tables.append(
                        ParsedTable(
                            index=len(tables),
                            page_number=page_num,
                            markdown=markdown,
                            html=html,
                            rows=rows,
                            cols=cols,
                            caption=getattr(item, "caption", None),
                        )
                    )
                except Exception:
                    # Skip tables that fail to export
                    pass
    except Exception:
        # If iteration fails, return empty list
        pass

    return tables


def _extract_images(doc: Any, output_dir: Path | None = None) -> list[ParsedImage]:
    """Extract images from a Docling document.

    Args:
        doc: Docling document object
        output_dir: Optional directory to save extracted images

    Returns:
        List of ParsedImage objects
    """
    images: list[ParsedImage] = []

    try:
        # Check for pictures in the document
        if hasattr(doc, "pictures"):
            for i, picture in enumerate(doc.pictures):
                image_path = None

                # Save image if output directory provided and image data available
                if output_dir and hasattr(picture, "image") and picture.image:
                    try:
                        image_filename = f"image_{i}.png"
                        image_path = str(output_dir / image_filename)
                        picture.image.save(image_path)
                    except Exception:
                        image_path = None

                # Get page number if available
                page_num = None
                if hasattr(picture, "prov") and picture.prov:
                    page_num = getattr(picture.prov[0], "page_no", None)

                # Get dimensions if available
                width = None
                height = None
                if hasattr(picture, "image") and picture.image:
                    width = picture.image.width
                    height = picture.image.height

                images.append(
                    ParsedImage(
                        index=i,
                        page_number=page_num,
                        image_path=image_path,
                        mime_type="image/png",
                        width=width,
                        height=height,
                        caption=getattr(picture, "caption", None),
                        alt_text=getattr(picture, "alt_text", None),
                    )
                )
    except Exception:
        # If extraction fails, return empty list
        pass

    return images


def _count_words(text: str) -> int:
    """Count words in text.

    Args:
        text: Text to count words in

    Returns:
        Word count
    """
    return len(text.split())


def _detect_language(text: str) -> str | None:
    """Detect the primary language of text.

    This is a simple heuristic. For production, consider using
    langdetect or similar library.

    Args:
        text: Text to detect language for

    Returns:
        ISO 639-1 language code or None
    """
    # Simple fallback - assume English
    # In production, use langdetect or similar
    return "en"


def parse_document_with_docling(
    file_path: str | Path,
    config: DoclingConfig | None = None,
    heartbeat_callback: callable | None = None,
) -> DoclingParseResult:
    """Parse a document using Docling.

    This is the core parsing function that can be called from activities
    or used directly in tests.

    Args:
        file_path: Path to the document file
        config: Docling configuration (uses defaults if not provided)
        heartbeat_callback: Optional callback to send heartbeats during processing

    Returns:
        DoclingParseResult with parsed content and metadata

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is not supported
    """
    import time

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    config = config or DoclingConfig()
    start_time = time.time()

    # Send initial heartbeat
    if heartbeat_callback:
        heartbeat_callback("Starting document conversion")

    # Create converter
    converter = _create_document_converter(config)

    # Send heartbeat before conversion (this can be slow)
    if heartbeat_callback:
        heartbeat_callback("Converting document with Docling")

    # Convert the document
    result = converter.convert(str(file_path))

    # Send heartbeat after conversion
    if heartbeat_callback:
        heartbeat_callback("Document converted, extracting content")

    # Get the document object
    doc = result.document

    # Export to different formats
    content_markdown = doc.export_to_markdown()

    if heartbeat_callback:
        heartbeat_callback("Exported to Markdown")

    # Export to JSON (Docling's native format)
    content_json = json.dumps(doc.export_to_dict(), indent=2, default=str)

    if heartbeat_callback:
        heartbeat_callback("Exported to JSON")

    # Export to plain text
    content_text = doc.export_to_text() if hasattr(doc, "export_to_text") else content_markdown

    # Get page count
    page_count = None
    if hasattr(doc, "pages"):
        page_count = len(doc.pages) if doc.pages else None

    # Get title if available
    title = None
    if hasattr(doc, "title") and doc.title:
        title = doc.title

    # Extract tables
    if heartbeat_callback:
        heartbeat_callback("Extracting tables")
    tables = _extract_tables(doc)

    # Extract images
    if heartbeat_callback:
        heartbeat_callback("Extracting images")
    images = _extract_images(doc)

    # Count words
    word_count = _count_words(content_text)

    # Detect language
    language = _detect_language(content_text[:5000])  # Use first 5000 chars for detection

    # Calculate processing time
    processing_time = time.time() - start_time

    # Build metadata
    metadata = {
        "source_file": file_path.name,
        "file_size": file_path.stat().st_size,
        "docling_version": _get_docling_version(),
    }

    if heartbeat_callback:
        heartbeat_callback("Parsing complete")

    return DoclingParseResult(
        content_markdown=content_markdown,
        content_json=content_json,
        content_text=content_text,
        page_count=page_count,
        word_count=word_count,
        language=language,
        title=title,
        tables=tables,
        images=images,
        metadata=metadata,
        processing_time_seconds=processing_time,
    )


def _get_docling_version() -> str:
    """Get the installed Docling version."""
    try:
        from docling import __version__

        return __version__
    except ImportError:
        return "unknown"


async def download_file_from_minio(
    bucket: str,
    key: str,
    local_path: Path,
) -> None:
    """Download a file from MinIO to local filesystem.

    Args:
        bucket: MinIO bucket name
        key: Object key in the bucket
        local_path: Local path to save the file
    """
    minio_client = _get_minio_client()
    await minio_client.download_file(bucket, key, str(local_path))


@activity.defn(name="parse_with_docling")
async def parse_with_docling(
    document_id: str,
    storage_bucket: str,
    storage_key: str,
    document_type: str,
    skip_ocr: bool = False,
    enable_table_structure: bool = True,
) -> dict[str, Any]:
    """Parse a document using Docling.

    This Temporal activity downloads a document from MinIO, processes it
    with Docling, and returns the parsed content.

    Args:
        document_id: Unique document identifier
        storage_bucket: MinIO bucket containing the document
        storage_key: Object key in the bucket
        document_type: Document type (pdf, docx, etc.)
        skip_ocr: Whether to skip OCR processing
        enable_table_structure: Whether to enable table structure detection

    Returns:
        Dictionary with parsed content and metadata
    """
    activity.logger.info(
        "Parsing document with Docling",
        extra={
            "document_id": document_id,
            "storage_bucket": storage_bucket,
            "storage_key": storage_key,
            "document_type": document_type,
            "skip_ocr": skip_ocr,
        },
    )

    # Create heartbeat callback
    def heartbeat(status: str) -> None:
        activity.heartbeat(status)
        activity.logger.debug(f"Heartbeat: {status}")

    heartbeat("Starting document parsing")

    # Create temp directory for processing
    with tempfile.TemporaryDirectory(prefix="docling_") as temp_dir:
        temp_path = Path(temp_dir)

        # Determine file extension from storage key or document type
        extension = Path(storage_key).suffix or f".{document_type}"
        local_file = temp_path / f"document{extension}"

        # Download file from MinIO
        heartbeat("Downloading file from storage")
        try:
            await download_file_from_minio(storage_bucket, storage_key, local_file)
        except Exception as e:
            activity.logger.error(f"Failed to download file: {e}")
            raise RuntimeError(f"Failed to download file from storage: {e}") from e

        heartbeat("File downloaded, starting Docling conversion")

        # Configure Docling
        config = DoclingConfig(
            enable_ocr=not skip_ocr,
            enable_table_structure=enable_table_structure,
            enable_cell_matching=True,
            generate_picture_images=True,
        )

        # Parse the document
        try:
            result = parse_document_with_docling(
                file_path=local_file,
                config=config,
                heartbeat_callback=heartbeat,
            )
        except FileNotFoundError as e:
            activity.logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            activity.logger.error(f"Docling parsing failed: {e}")
            raise RuntimeError(f"Document parsing failed: {e}") from e

    activity.logger.info(
        "Document parsed successfully",
        extra={
            "document_id": document_id,
            "page_count": result.page_count,
            "word_count": result.word_count,
            "tables_extracted": len(result.tables),
            "images_extracted": len(result.images),
            "processing_time_seconds": result.processing_time_seconds,
        },
    )

    # Return as dictionary for Temporal serialization
    return {
        "content_markdown": result.content_markdown,
        "content_json": result.content_json,
        "content_text": result.content_text,
        "page_count": result.page_count,
        "word_count": result.word_count,
        "language": result.language,
        "title": result.title,
        "tables": [
            {
                "index": t.index,
                "page_number": t.page_number,
                "markdown": t.markdown,
                "html": t.html,
                "rows": t.rows,
                "cols": t.cols,
                "caption": t.caption,
            }
            for t in result.tables
        ],
        "images": [
            {
                "index": i.index,
                "page_number": i.page_number,
                "mime_type": i.mime_type,
                "width": i.width,
                "height": i.height,
                "caption": i.caption,
                "alt_text": i.alt_text,
            }
            for i in result.images
        ],
        "metadata": result.metadata,
        "processing_time_seconds": result.processing_time_seconds,
    }


# Document type handlers for different formats
SUPPORTED_DOCUMENT_TYPES = {
    "pdf": ["application/pdf"],
    "docx": ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
    "doc": ["application/msword"],
    "pptx": ["application/vnd.openxmlformats-officedocument.presentationml.presentation"],
    "xlsx": ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
    "html": ["text/html"],
    "md": ["text/markdown"],
    "txt": ["text/plain"],
    "png": ["image/png"],
    "jpg": ["image/jpeg"],
    "jpeg": ["image/jpeg"],
    "tiff": ["image/tiff"],
    "tif": ["image/tiff"],
    "bmp": ["image/bmp"],
}


def is_docling_supported(document_type: str) -> bool:
    """Check if a document type is supported by Docling.

    Args:
        document_type: Document type string

    Returns:
        True if Docling can process this document type
    """
    return document_type.lower() in SUPPORTED_DOCUMENT_TYPES
