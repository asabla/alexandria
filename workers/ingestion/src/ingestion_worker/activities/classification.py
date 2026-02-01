"""
Document classification utilities.

Provides magic bytes detection, MIME type detection, and file extension mapping
for accurate document type classification.
"""

from dataclasses import dataclass
from enum import StrEnum


class DocumentType(StrEnum):
    """Supported document types for ingestion."""

    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"
    IMAGE = "image"  # jpg, png, gif, webp, tiff, bmp
    AUDIO = "audio"  # mp3, wav, flac, ogg, m4a
    VIDEO = "video"  # mp4, webm, avi, mov, mkv
    SPREADSHEET = "spreadsheet"  # xlsx, xls, csv, ods
    EMAIL = "email"  # eml, msg
    WEBPAGE = "webpage"  # scraped web pages
    PRESENTATION = "presentation"  # pptx, ppt, odp
    ARCHIVE = "archive"  # zip, tar, gz, 7z, rar
    RTF = "rtf"
    JSON = "json"
    XML = "xml"
    UNKNOWN = "unknown"


@dataclass
class MagicSignature:
    """Magic bytes signature for file type detection."""

    bytes_pattern: bytes
    offset: int = 0
    document_type: DocumentType = DocumentType.UNKNOWN
    mime_type: str = "application/octet-stream"
    description: str = ""


# Magic bytes signatures for common file types
# Order matters - more specific signatures should come first
MAGIC_SIGNATURES: list[MagicSignature] = [
    # PDF
    MagicSignature(
        bytes_pattern=b"%PDF",
        offset=0,
        document_type=DocumentType.PDF,
        mime_type="application/pdf",
        description="PDF document",
    ),
    # Microsoft Office Open XML formats (OOXML) - ZIP-based
    # These need special handling since they're ZIP files
    MagicSignature(
        bytes_pattern=b"PK\x03\x04",
        offset=0,
        document_type=DocumentType.UNKNOWN,  # Will be refined by extension
        mime_type="application/zip",
        description="ZIP-based format (OOXML or archive)",
    ),
    # Legacy Microsoft Office formats (OLE2 Compound Document)
    MagicSignature(
        bytes_pattern=b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
        offset=0,
        document_type=DocumentType.DOC,  # Could also be xls, ppt - refined by extension
        mime_type="application/msword",
        description="OLE2 Compound Document (legacy Office)",
    ),
    # Images
    MagicSignature(
        bytes_pattern=b"\xff\xd8\xff",
        offset=0,
        document_type=DocumentType.IMAGE,
        mime_type="image/jpeg",
        description="JPEG image",
    ),
    MagicSignature(
        bytes_pattern=b"\x89PNG\r\n\x1a\n",
        offset=0,
        document_type=DocumentType.IMAGE,
        mime_type="image/png",
        description="PNG image",
    ),
    MagicSignature(
        bytes_pattern=b"GIF87a",
        offset=0,
        document_type=DocumentType.IMAGE,
        mime_type="image/gif",
        description="GIF image (87a)",
    ),
    MagicSignature(
        bytes_pattern=b"GIF89a",
        offset=0,
        document_type=DocumentType.IMAGE,
        mime_type="image/gif",
        description="GIF image (89a)",
    ),
    MagicSignature(
        bytes_pattern=b"RIFF",
        offset=0,
        document_type=DocumentType.IMAGE,  # Could also be audio (WAV) - refined later
        mime_type="image/webp",
        description="RIFF container (WebP or WAV)",
    ),
    MagicSignature(
        bytes_pattern=b"II*\x00",
        offset=0,
        document_type=DocumentType.IMAGE,
        mime_type="image/tiff",
        description="TIFF image (little-endian)",
    ),
    MagicSignature(
        bytes_pattern=b"MM\x00*",
        offset=0,
        document_type=DocumentType.IMAGE,
        mime_type="image/tiff",
        description="TIFF image (big-endian)",
    ),
    MagicSignature(
        bytes_pattern=b"BM",
        offset=0,
        document_type=DocumentType.IMAGE,
        mime_type="image/bmp",
        description="BMP image",
    ),
    # Audio
    MagicSignature(
        bytes_pattern=b"ID3",
        offset=0,
        document_type=DocumentType.AUDIO,
        mime_type="audio/mpeg",
        description="MP3 audio (ID3 tag)",
    ),
    MagicSignature(
        bytes_pattern=b"\xff\xfb",
        offset=0,
        document_type=DocumentType.AUDIO,
        mime_type="audio/mpeg",
        description="MP3 audio (frame sync)",
    ),
    MagicSignature(
        bytes_pattern=b"\xff\xfa",
        offset=0,
        document_type=DocumentType.AUDIO,
        mime_type="audio/mpeg",
        description="MP3 audio (frame sync)",
    ),
    MagicSignature(
        bytes_pattern=b"fLaC",
        offset=0,
        document_type=DocumentType.AUDIO,
        mime_type="audio/flac",
        description="FLAC audio",
    ),
    MagicSignature(
        bytes_pattern=b"OggS",
        offset=0,
        document_type=DocumentType.AUDIO,  # Could also be video - refined by content
        mime_type="audio/ogg",
        description="Ogg container",
    ),
    # Video
    MagicSignature(
        bytes_pattern=b"\x00\x00\x00\x18ftypmp4",
        offset=0,
        document_type=DocumentType.VIDEO,
        mime_type="video/mp4",
        description="MP4 video (ftyp mp4)",
    ),
    MagicSignature(
        bytes_pattern=b"\x00\x00\x00\x1cftypisom",
        offset=0,
        document_type=DocumentType.VIDEO,
        mime_type="video/mp4",
        description="MP4 video (ftyp isom)",
    ),
    MagicSignature(
        bytes_pattern=b"\x00\x00\x00\x14ftyp",
        offset=0,
        document_type=DocumentType.VIDEO,
        mime_type="video/mp4",
        description="MP4 video (ftyp)",
    ),
    MagicSignature(
        bytes_pattern=b"\x1aE\xdf\xa3",
        offset=0,
        document_type=DocumentType.VIDEO,
        mime_type="video/webm",
        description="WebM/Matroska video",
    ),
    # RTF
    MagicSignature(
        bytes_pattern=b"{\\rtf",
        offset=0,
        document_type=DocumentType.RTF,
        mime_type="application/rtf",
        description="Rich Text Format",
    ),
    # Archives
    MagicSignature(
        bytes_pattern=b"\x1f\x8b",
        offset=0,
        document_type=DocumentType.ARCHIVE,
        mime_type="application/gzip",
        description="Gzip archive",
    ),
    MagicSignature(
        bytes_pattern=b"Rar!\x1a\x07",
        offset=0,
        document_type=DocumentType.ARCHIVE,
        mime_type="application/vnd.rar",
        description="RAR archive",
    ),
    MagicSignature(
        bytes_pattern=b"7z\xbc\xaf'\x1c",
        offset=0,
        document_type=DocumentType.ARCHIVE,
        mime_type="application/x-7z-compressed",
        description="7-Zip archive",
    ),
    # XML
    MagicSignature(
        bytes_pattern=b"<?xml",
        offset=0,
        document_type=DocumentType.XML,
        mime_type="application/xml",
        description="XML document",
    ),
    # HTML (common patterns)
    MagicSignature(
        bytes_pattern=b"<!DOCTYPE html",
        offset=0,
        document_type=DocumentType.HTML,
        mime_type="text/html",
        description="HTML document (DOCTYPE)",
    ),
    MagicSignature(
        bytes_pattern=b"<!doctype html",
        offset=0,
        document_type=DocumentType.HTML,
        mime_type="text/html",
        description="HTML document (doctype)",
    ),
    MagicSignature(
        bytes_pattern=b"<html",
        offset=0,
        document_type=DocumentType.HTML,
        mime_type="text/html",
        description="HTML document",
    ),
]

# File extension to document type mapping
EXTENSION_MAP: dict[str, tuple[DocumentType, str]] = {
    # Documents
    "pdf": (DocumentType.PDF, "application/pdf"),
    "doc": (DocumentType.DOC, "application/msword"),
    "docx": (
        DocumentType.DOCX,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    "odt": (DocumentType.DOCX, "application/vnd.oasis.opendocument.text"),
    "rtf": (DocumentType.RTF, "application/rtf"),
    "txt": (DocumentType.TXT, "text/plain"),
    "text": (DocumentType.TXT, "text/plain"),
    "md": (DocumentType.MARKDOWN, "text/markdown"),
    "markdown": (DocumentType.MARKDOWN, "text/markdown"),
    # Web
    "html": (DocumentType.HTML, "text/html"),
    "htm": (DocumentType.HTML, "text/html"),
    "xhtml": (DocumentType.HTML, "application/xhtml+xml"),
    "xml": (DocumentType.XML, "application/xml"),
    "json": (DocumentType.JSON, "application/json"),
    # Spreadsheets
    "xlsx": (
        DocumentType.SPREADSHEET,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    "xls": (DocumentType.SPREADSHEET, "application/vnd.ms-excel"),
    "csv": (DocumentType.SPREADSHEET, "text/csv"),
    "tsv": (DocumentType.SPREADSHEET, "text/tab-separated-values"),
    "ods": (DocumentType.SPREADSHEET, "application/vnd.oasis.opendocument.spreadsheet"),
    # Presentations
    "pptx": (
        DocumentType.PRESENTATION,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    "ppt": (DocumentType.PRESENTATION, "application/vnd.ms-powerpoint"),
    "odp": (DocumentType.PRESENTATION, "application/vnd.oasis.opendocument.presentation"),
    # Images
    "jpg": (DocumentType.IMAGE, "image/jpeg"),
    "jpeg": (DocumentType.IMAGE, "image/jpeg"),
    "png": (DocumentType.IMAGE, "image/png"),
    "gif": (DocumentType.IMAGE, "image/gif"),
    "webp": (DocumentType.IMAGE, "image/webp"),
    "tiff": (DocumentType.IMAGE, "image/tiff"),
    "tif": (DocumentType.IMAGE, "image/tiff"),
    "bmp": (DocumentType.IMAGE, "image/bmp"),
    "svg": (DocumentType.IMAGE, "image/svg+xml"),
    "ico": (DocumentType.IMAGE, "image/x-icon"),
    "heic": (DocumentType.IMAGE, "image/heic"),
    "heif": (DocumentType.IMAGE, "image/heif"),
    # Audio
    "mp3": (DocumentType.AUDIO, "audio/mpeg"),
    "wav": (DocumentType.AUDIO, "audio/wav"),
    "flac": (DocumentType.AUDIO, "audio/flac"),
    "ogg": (DocumentType.AUDIO, "audio/ogg"),
    "m4a": (DocumentType.AUDIO, "audio/mp4"),
    "aac": (DocumentType.AUDIO, "audio/aac"),
    "wma": (DocumentType.AUDIO, "audio/x-ms-wma"),
    # Video
    "mp4": (DocumentType.VIDEO, "video/mp4"),
    "webm": (DocumentType.VIDEO, "video/webm"),
    "mkv": (DocumentType.VIDEO, "video/x-matroska"),
    "avi": (DocumentType.VIDEO, "video/x-msvideo"),
    "mov": (DocumentType.VIDEO, "video/quicktime"),
    "wmv": (DocumentType.VIDEO, "video/x-ms-wmv"),
    "flv": (DocumentType.VIDEO, "video/x-flv"),
    # Email
    "eml": (DocumentType.EMAIL, "message/rfc822"),
    "msg": (DocumentType.EMAIL, "application/vnd.ms-outlook"),
    "mbox": (DocumentType.EMAIL, "application/mbox"),
    # Archives
    "zip": (DocumentType.ARCHIVE, "application/zip"),
    "tar": (DocumentType.ARCHIVE, "application/x-tar"),
    "gz": (DocumentType.ARCHIVE, "application/gzip"),
    "tgz": (DocumentType.ARCHIVE, "application/gzip"),
    "rar": (DocumentType.ARCHIVE, "application/vnd.rar"),
    "7z": (DocumentType.ARCHIVE, "application/x-7z-compressed"),
    "bz2": (DocumentType.ARCHIVE, "application/x-bzip2"),
    "xz": (DocumentType.ARCHIVE, "application/x-xz"),
}

# MIME type prefix to document type mapping (fallback)
MIME_PREFIX_MAP: dict[str, DocumentType] = {
    "application/pdf": DocumentType.PDF,
    "application/msword": DocumentType.DOC,
    "application/vnd.openxmlformats-officedocument.wordprocessingml": DocumentType.DOCX,
    "application/vnd.oasis.opendocument.text": DocumentType.DOCX,
    "application/rtf": DocumentType.RTF,
    "text/plain": DocumentType.TXT,
    "text/markdown": DocumentType.MARKDOWN,
    "text/html": DocumentType.HTML,
    "application/xhtml": DocumentType.HTML,
    "application/xml": DocumentType.XML,
    "text/xml": DocumentType.XML,
    "application/json": DocumentType.JSON,
    "application/vnd.openxmlformats-officedocument.spreadsheetml": DocumentType.SPREADSHEET,
    "application/vnd.ms-excel": DocumentType.SPREADSHEET,
    "application/vnd.oasis.opendocument.spreadsheet": DocumentType.SPREADSHEET,
    "text/csv": DocumentType.SPREADSHEET,
    "application/vnd.openxmlformats-officedocument.presentationml": DocumentType.PRESENTATION,
    "application/vnd.ms-powerpoint": DocumentType.PRESENTATION,
    "image/": DocumentType.IMAGE,
    "audio/": DocumentType.AUDIO,
    "video/": DocumentType.VIDEO,
    "message/rfc822": DocumentType.EMAIL,
    "application/vnd.ms-outlook": DocumentType.EMAIL,
    "application/zip": DocumentType.ARCHIVE,
    "application/x-tar": DocumentType.ARCHIVE,
    "application/gzip": DocumentType.ARCHIVE,
    "application/x-7z-compressed": DocumentType.ARCHIVE,
    "application/vnd.rar": DocumentType.ARCHIVE,
}


@dataclass
class ClassificationResult:
    """Result of document classification."""

    document_type: DocumentType
    mime_type: str
    confidence: float
    detection_method: str  # "magic_bytes", "extension", "mime_type", "content_analysis"
    description: str = ""


def detect_by_magic_bytes(data: bytes) -> ClassificationResult | None:
    """
    Detect document type by magic bytes (file signature).

    Args:
        data: First bytes of the file (at least 32 bytes recommended)

    Returns:
        ClassificationResult if detected, None otherwise
    """
    if not data or len(data) < 4:
        return None

    # Check against known signatures
    for sig in MAGIC_SIGNATURES:
        end_offset = sig.offset + len(sig.bytes_pattern)
        if len(data) >= end_offset:
            if data[sig.offset : end_offset] == sig.bytes_pattern:
                # Special handling for RIFF container (could be WebP or WAV)
                if sig.bytes_pattern == b"RIFF" and len(data) >= 12:
                    if data[8:12] == b"WEBP":
                        return ClassificationResult(
                            document_type=DocumentType.IMAGE,
                            mime_type="image/webp",
                            confidence=0.95,
                            detection_method="magic_bytes",
                            description="WebP image (RIFF/WEBP)",
                        )
                    elif data[8:12] == b"WAVE":
                        return ClassificationResult(
                            document_type=DocumentType.AUDIO,
                            mime_type="audio/wav",
                            confidence=0.95,
                            detection_method="magic_bytes",
                            description="WAV audio (RIFF/WAVE)",
                        )
                    elif data[8:12] == b"AVI ":
                        return ClassificationResult(
                            document_type=DocumentType.VIDEO,
                            mime_type="video/x-msvideo",
                            confidence=0.95,
                            detection_method="magic_bytes",
                            description="AVI video (RIFF/AVI)",
                        )

                # Skip generic ZIP signature - will be refined by extension
                if sig.bytes_pattern == b"PK\x03\x04":
                    continue

                return ClassificationResult(
                    document_type=sig.document_type,
                    mime_type=sig.mime_type,
                    confidence=0.95,
                    detection_method="magic_bytes",
                    description=sig.description,
                )

    return None


def detect_by_extension(filename: str | None) -> ClassificationResult | None:
    """
    Detect document type by file extension.

    Args:
        filename: Source filename

    Returns:
        ClassificationResult if detected, None otherwise
    """
    if not filename:
        return None

    # Extract extension
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if not ext:
        return None

    if ext in EXTENSION_MAP:
        doc_type, mime_type = EXTENSION_MAP[ext]
        return ClassificationResult(
            document_type=doc_type,
            mime_type=mime_type,
            confidence=0.8,
            detection_method="extension",
            description=f"Detected by .{ext} extension",
        )

    return None


def detect_by_mime_type(mime_type: str | None) -> ClassificationResult | None:
    """
    Detect document type by MIME type.

    Args:
        mime_type: MIME type string

    Returns:
        ClassificationResult if detected, None otherwise
    """
    if not mime_type:
        return None

    # Check exact match first
    for mime_prefix, doc_type in MIME_PREFIX_MAP.items():
        if mime_type == mime_prefix or mime_type.startswith(mime_prefix):
            return ClassificationResult(
                document_type=doc_type,
                mime_type=mime_type,
                confidence=0.7,
                detection_method="mime_type",
                description=f"Detected by MIME type: {mime_type}",
            )

    return None


def detect_text_content_type(data: bytes) -> ClassificationResult | None:
    """
    Analyze text content to determine if it's JSON, XML, HTML, or Markdown.

    Args:
        data: File content (should be text-decodable)

    Returns:
        ClassificationResult if detected, None otherwise
    """
    try:
        # Try to decode as UTF-8
        text = data[:4096].decode("utf-8", errors="ignore").strip()
        if not text:
            return None

        # Check for JSON
        if text.startswith(("{", "[")):
            return ClassificationResult(
                document_type=DocumentType.JSON,
                mime_type="application/json",
                confidence=0.7,
                detection_method="content_analysis",
                description="Detected JSON structure",
            )

        # Check for XML (not already detected by magic bytes)
        if text.startswith("<?xml") or text.startswith("<"):
            # Check if it's HTML
            lower_text = text.lower()
            if "<html" in lower_text or "<!doctype html" in lower_text:
                return ClassificationResult(
                    document_type=DocumentType.HTML,
                    mime_type="text/html",
                    confidence=0.7,
                    detection_method="content_analysis",
                    description="Detected HTML structure",
                )
            return ClassificationResult(
                document_type=DocumentType.XML,
                mime_type="application/xml",
                confidence=0.7,
                detection_method="content_analysis",
                description="Detected XML structure",
            )

        # Check for Markdown patterns
        md_patterns = ["# ", "## ", "### ", "```", "---\n", "* ", "- [ ]", "[", "!["]
        md_count = sum(1 for p in md_patterns if p in text)
        if md_count >= 2:
            return ClassificationResult(
                document_type=DocumentType.MARKDOWN,
                mime_type="text/markdown",
                confidence=0.6,
                detection_method="content_analysis",
                description="Detected Markdown patterns",
            )

    except Exception:
        pass

    return None


def classify_document(
    data: bytes | None = None,
    filename: str | None = None,
    mime_type: str | None = None,
) -> ClassificationResult:
    """
    Classify a document using multiple detection methods.

    Priority order:
    1. Magic bytes (most reliable)
    2. File extension (common and quick)
    3. MIME type (if provided)
    4. Content analysis (for text files)

    Args:
        data: First bytes of the file (at least 32 bytes recommended)
        filename: Source filename
        mime_type: Provided MIME type

    Returns:
        ClassificationResult with detected type and confidence
    """
    results: list[ClassificationResult] = []

    # Try magic bytes first (most reliable)
    if data:
        magic_result = detect_by_magic_bytes(data)
        if magic_result:
            # If it's a ZIP-based format, refine by extension
            if magic_result.document_type == DocumentType.UNKNOWN:
                ext_result = detect_by_extension(filename)
                if ext_result and ext_result.document_type in (
                    DocumentType.DOCX,
                    DocumentType.SPREADSHEET,
                    DocumentType.PRESENTATION,
                ):
                    ext_result.confidence = 0.9  # High confidence for ZIP + known extension
                    return ext_result
            else:
                results.append(magic_result)

    # Try extension
    ext_result = detect_by_extension(filename)
    if ext_result:
        results.append(ext_result)

    # Try MIME type
    mime_result = detect_by_mime_type(mime_type)
    if mime_result:
        results.append(mime_result)

    # Try content analysis for text files
    if data:
        content_result = detect_text_content_type(data)
        if content_result:
            results.append(content_result)

    # Return highest confidence result
    if results:
        best_result = max(results, key=lambda r: r.confidence)

        # Boost confidence if multiple methods agree
        agreeing_methods = [r for r in results if r.document_type == best_result.document_type]
        if len(agreeing_methods) > 1:
            best_result.confidence = min(0.99, best_result.confidence + 0.1)
            best_result.description += f" (confirmed by {len(agreeing_methods)} methods)"

        return best_result

    # Default to unknown
    return ClassificationResult(
        document_type=DocumentType.UNKNOWN,
        mime_type=mime_type or "application/octet-stream",
        confidence=0.0,
        detection_method="none",
        description="Unable to determine document type",
    )


def is_text_based(doc_type: DocumentType) -> bool:
    """Check if document type is text-based (can be directly read)."""
    return doc_type in (
        DocumentType.TXT,
        DocumentType.MARKDOWN,
        DocumentType.HTML,
        DocumentType.JSON,
        DocumentType.XML,
    )


def needs_ocr(doc_type: DocumentType) -> bool:
    """Check if document type likely needs OCR for text extraction."""
    return doc_type == DocumentType.IMAGE


def needs_transcription(doc_type: DocumentType) -> bool:
    """Check if document type needs audio/video transcription."""
    return doc_type in (DocumentType.AUDIO, DocumentType.VIDEO)


def is_compound_document(doc_type: DocumentType) -> bool:
    """Check if document type is a compound format (contains multiple parts)."""
    return doc_type in (
        DocumentType.PDF,
        DocumentType.DOCX,
        DocumentType.DOC,
        DocumentType.SPREADSHEET,
        DocumentType.PRESENTATION,
        DocumentType.EMAIL,
    )
