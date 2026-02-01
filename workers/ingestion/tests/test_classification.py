"""
Unit tests for document classification utilities.

Tests magic bytes detection, extension mapping, MIME type detection,
and the combined classification function.
"""

import pytest

from ingestion_worker.activities.classification import (
    ClassificationResult,
    DocumentType,
    EXTENSION_MAP,
    MAGIC_SIGNATURES,
    classify_document,
    detect_by_extension,
    detect_by_magic_bytes,
    detect_by_mime_type,
    detect_text_content_type,
    is_compound_document,
    is_text_based,
    needs_ocr,
    needs_transcription,
)


# =============================================================================
# DocumentType Enum Tests
# =============================================================================


class TestDocumentType:
    """Tests for DocumentType enum."""

    def test_all_types_are_strings(self):
        """All document types should be string values."""
        for doc_type in DocumentType:
            assert isinstance(doc_type.value, str)

    def test_key_types_exist(self):
        """Key document types should be defined."""
        expected = [
            "pdf",
            "docx",
            "doc",
            "txt",
            "html",
            "markdown",
            "image",
            "audio",
            "video",
            "spreadsheet",
            "email",
            "unknown",
        ]
        for expected_value in expected:
            assert any(dt.value == expected_value for dt in DocumentType)

    def test_unknown_is_default(self):
        """UNKNOWN should be a valid type for unrecognized files."""
        assert DocumentType.UNKNOWN.value == "unknown"


# =============================================================================
# Magic Bytes Detection Tests
# =============================================================================


class TestMagicBytesDetection:
    """Tests for magic bytes (file signature) detection."""

    def test_detect_pdf(self):
        """PDF files should be detected by %PDF signature."""
        pdf_data = b"%PDF-1.7\n"
        result = detect_by_magic_bytes(pdf_data)

        assert result is not None
        assert result.document_type == DocumentType.PDF
        assert result.mime_type == "application/pdf"
        assert result.confidence >= 0.9
        assert result.detection_method == "magic_bytes"

    def test_detect_jpeg(self):
        """JPEG files should be detected by FF D8 FF signature."""
        jpeg_data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        result = detect_by_magic_bytes(jpeg_data)

        assert result is not None
        assert result.document_type == DocumentType.IMAGE
        assert result.mime_type == "image/jpeg"

    def test_detect_png(self):
        """PNG files should be detected by their signature."""
        png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        result = detect_by_magic_bytes(png_data)

        assert result is not None
        assert result.document_type == DocumentType.IMAGE
        assert result.mime_type == "image/png"

    def test_detect_gif87a(self):
        """GIF87a files should be detected."""
        gif_data = b"GIF87a\x00\x00"
        result = detect_by_magic_bytes(gif_data)

        assert result is not None
        assert result.document_type == DocumentType.IMAGE
        assert result.mime_type == "image/gif"

    def test_detect_gif89a(self):
        """GIF89a files should be detected."""
        gif_data = b"GIF89a\x00\x00"
        result = detect_by_magic_bytes(gif_data)

        assert result is not None
        assert result.document_type == DocumentType.IMAGE
        assert result.mime_type == "image/gif"

    def test_detect_webp(self):
        """WebP files should be detected (RIFF container with WEBP)."""
        webp_data = b"RIFF\x00\x00\x00\x00WEBP"
        result = detect_by_magic_bytes(webp_data)

        assert result is not None
        assert result.document_type == DocumentType.IMAGE
        assert result.mime_type == "image/webp"

    def test_detect_wav(self):
        """WAV files should be detected (RIFF container with WAVE)."""
        wav_data = b"RIFF\x00\x00\x00\x00WAVE"
        result = detect_by_magic_bytes(wav_data)

        assert result is not None
        assert result.document_type == DocumentType.AUDIO
        assert result.mime_type == "audio/wav"

    def test_detect_avi(self):
        """AVI files should be detected (RIFF container with AVI)."""
        avi_data = b"RIFF\x00\x00\x00\x00AVI "
        result = detect_by_magic_bytes(avi_data)

        assert result is not None
        assert result.document_type == DocumentType.VIDEO
        assert result.mime_type == "video/x-msvideo"

    def test_detect_mp3_id3(self):
        """MP3 files with ID3 tag should be detected."""
        mp3_data = b"ID3\x04\x00\x00\x00\x00\x00\x00"
        result = detect_by_magic_bytes(mp3_data)

        assert result is not None
        assert result.document_type == DocumentType.AUDIO
        assert result.mime_type == "audio/mpeg"

    def test_detect_flac(self):
        """FLAC files should be detected."""
        flac_data = b"fLaC\x00\x00\x00"
        result = detect_by_magic_bytes(flac_data)

        assert result is not None
        assert result.document_type == DocumentType.AUDIO
        assert result.mime_type == "audio/flac"

    def test_detect_rtf(self):
        """RTF files should be detected."""
        rtf_data = b"{\\rtf1\\ansi"
        result = detect_by_magic_bytes(rtf_data)

        assert result is not None
        assert result.document_type == DocumentType.RTF
        assert result.mime_type == "application/rtf"

    def test_detect_gzip(self):
        """Gzip files should be detected."""
        gzip_data = b"\x1f\x8b\x08\x00\x00"
        result = detect_by_magic_bytes(gzip_data)

        assert result is not None
        assert result.document_type == DocumentType.ARCHIVE
        assert result.mime_type == "application/gzip"

    def test_detect_7zip(self):
        """7-Zip files should be detected."""
        sevenzip_data = b"7z\xbc\xaf'\x1c\x00\x00"
        result = detect_by_magic_bytes(sevenzip_data)

        assert result is not None
        assert result.document_type == DocumentType.ARCHIVE
        assert result.mime_type == "application/x-7z-compressed"

    def test_detect_ole2_doc(self):
        """Legacy Office (OLE2) files should be detected."""
        ole2_data = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1\x00\x00"
        result = detect_by_magic_bytes(ole2_data)

        assert result is not None
        assert result.document_type == DocumentType.DOC
        assert result.mime_type == "application/msword"

    def test_detect_html_doctype(self):
        """HTML files with DOCTYPE should be detected."""
        html_data = b"<!DOCTYPE html>\n<html>"
        result = detect_by_magic_bytes(html_data)

        assert result is not None
        assert result.document_type == DocumentType.HTML
        assert result.mime_type == "text/html"

    def test_detect_xml(self):
        """XML files should be detected."""
        xml_data = b'<?xml version="1.0"?>\n<root>'
        result = detect_by_magic_bytes(xml_data)

        assert result is not None
        assert result.document_type == DocumentType.XML
        assert result.mime_type == "application/xml"

    def test_empty_data_returns_none(self):
        """Empty data should return None."""
        result = detect_by_magic_bytes(b"")
        assert result is None

    def test_short_data_returns_none(self):
        """Data shorter than 4 bytes should return None."""
        result = detect_by_magic_bytes(b"abc")
        assert result is None

    def test_unknown_signature_returns_none(self):
        """Unknown signatures should return None."""
        result = detect_by_magic_bytes(b"UNKNOWN_DATA_PATTERN")
        assert result is None


# =============================================================================
# Extension Detection Tests
# =============================================================================


class TestExtensionDetection:
    """Tests for file extension detection."""

    @pytest.mark.parametrize(
        "filename,expected_type,expected_mime",
        [
            ("document.pdf", DocumentType.PDF, "application/pdf"),
            (
                "report.docx",
                DocumentType.DOCX,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            ("letter.doc", DocumentType.DOC, "application/msword"),
            ("notes.txt", DocumentType.TXT, "text/plain"),
            ("readme.md", DocumentType.MARKDOWN, "text/markdown"),
            ("page.html", DocumentType.HTML, "text/html"),
            ("page.htm", DocumentType.HTML, "text/html"),
            ("photo.jpg", DocumentType.IMAGE, "image/jpeg"),
            ("photo.jpeg", DocumentType.IMAGE, "image/jpeg"),
            ("image.png", DocumentType.IMAGE, "image/png"),
            ("animation.gif", DocumentType.IMAGE, "image/gif"),
            ("song.mp3", DocumentType.AUDIO, "audio/mpeg"),
            ("audio.wav", DocumentType.AUDIO, "audio/wav"),
            ("movie.mp4", DocumentType.VIDEO, "video/mp4"),
            (
                "data.xlsx",
                DocumentType.SPREADSHEET,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            ("data.csv", DocumentType.SPREADSHEET, "text/csv"),
            ("mail.eml", DocumentType.EMAIL, "message/rfc822"),
            ("archive.zip", DocumentType.ARCHIVE, "application/zip"),
            ("config.json", DocumentType.JSON, "application/json"),
            ("data.xml", DocumentType.XML, "application/xml"),
        ],
    )
    def test_common_extensions(self, filename, expected_type, expected_mime):
        """Common file extensions should be correctly detected."""
        result = detect_by_extension(filename)

        assert result is not None
        assert result.document_type == expected_type
        assert result.mime_type == expected_mime
        assert result.detection_method == "extension"

    def test_case_insensitive(self):
        """Extension detection should be case insensitive."""
        result_lower = detect_by_extension("file.pdf")
        result_upper = detect_by_extension("FILE.PDF")
        result_mixed = detect_by_extension("FiLe.PdF")

        assert result_lower is not None
        assert result_upper is not None
        assert result_mixed is not None
        assert (
            result_lower.document_type == result_upper.document_type == result_mixed.document_type
        )

    def test_no_extension_returns_none(self):
        """Files without extension should return None."""
        result = detect_by_extension("filename_no_extension")
        assert result is None

    def test_empty_filename_returns_none(self):
        """Empty filename should return None."""
        result = detect_by_extension("")
        assert result is None

    def test_none_filename_returns_none(self):
        """None filename should return None."""
        result = detect_by_extension(None)
        assert result is None

    def test_unknown_extension_returns_none(self):
        """Unknown extensions should return None."""
        result = detect_by_extension("file.xyz123")
        assert result is None

    def test_confidence_level(self):
        """Extension detection should have moderate confidence."""
        result = detect_by_extension("file.pdf")

        assert result is not None
        assert 0.7 <= result.confidence <= 0.9


# =============================================================================
# MIME Type Detection Tests
# =============================================================================


class TestMimeTypeDetection:
    """Tests for MIME type detection."""

    @pytest.mark.parametrize(
        "mime_type,expected_doc_type",
        [
            ("application/pdf", DocumentType.PDF),
            ("text/plain", DocumentType.TXT),
            ("text/html", DocumentType.HTML),
            ("text/markdown", DocumentType.MARKDOWN),
            ("image/jpeg", DocumentType.IMAGE),
            ("image/png", DocumentType.IMAGE),
            ("audio/mpeg", DocumentType.AUDIO),
            ("video/mp4", DocumentType.VIDEO),
            ("application/json", DocumentType.JSON),
        ],
    )
    def test_common_mime_types(self, mime_type, expected_doc_type):
        """Common MIME types should be correctly detected."""
        result = detect_by_mime_type(mime_type)

        assert result is not None
        assert result.document_type == expected_doc_type
        assert result.detection_method == "mime_type"

    def test_mime_prefix_matching(self):
        """MIME types should match by prefix for categories."""
        result = detect_by_mime_type("image/webp")

        assert result is not None
        assert result.document_type == DocumentType.IMAGE

    def test_none_mime_returns_none(self):
        """None MIME type should return None."""
        result = detect_by_mime_type(None)
        assert result is None

    def test_empty_mime_returns_none(self):
        """Empty MIME type should return None."""
        result = detect_by_mime_type("")
        assert result is None

    def test_unknown_mime_returns_none(self):
        """Unknown MIME types should return None."""
        result = detect_by_mime_type("application/x-custom-unknown")
        assert result is None


# =============================================================================
# Content Analysis Tests
# =============================================================================


class TestContentAnalysis:
    """Tests for text content analysis."""

    def test_detect_json_object(self):
        """JSON objects should be detected."""
        json_data = b'{"key": "value", "number": 123}'
        result = detect_text_content_type(json_data)

        assert result is not None
        assert result.document_type == DocumentType.JSON
        assert result.detection_method == "content_analysis"

    def test_detect_json_array(self):
        """JSON arrays should be detected."""
        json_data = b'[1, 2, 3, "four"]'
        result = detect_text_content_type(json_data)

        assert result is not None
        assert result.document_type == DocumentType.JSON

    def test_detect_html_in_content(self):
        """HTML content should be detected."""
        html_data = b"<html>\n<body>Content</body>\n</html>"
        result = detect_text_content_type(html_data)

        assert result is not None
        assert result.document_type == DocumentType.HTML

    def test_detect_html_with_doctype(self):
        """HTML with doctype in content should be detected."""
        html_data = b"<!doctype html>\n<html>\n<body>Content</body>\n</html>"
        result = detect_text_content_type(html_data)

        assert result is not None
        assert result.document_type == DocumentType.HTML

    def test_detect_xml_content(self):
        """XML content should be detected."""
        xml_data = b"<root><item>value</item></root>"
        result = detect_text_content_type(xml_data)

        assert result is not None
        assert result.document_type == DocumentType.XML

    def test_detect_markdown_content(self):
        """Markdown content should be detected by patterns."""
        md_data = b"# Heading\n\n## Subheading\n\n- List item\n- Another item\n\n```code```"
        result = detect_text_content_type(md_data)

        assert result is not None
        assert result.document_type == DocumentType.MARKDOWN

    def test_empty_returns_none(self):
        """Empty content should return None."""
        result = detect_text_content_type(b"")
        assert result is None

    def test_plain_text_returns_none(self):
        """Plain text without patterns should return None."""
        plain_data = b"This is just plain text without any special patterns."
        result = detect_text_content_type(plain_data)
        assert result is None


# =============================================================================
# Combined Classification Tests
# =============================================================================


class TestClassifyDocument:
    """Tests for the combined classify_document function."""

    def test_magic_bytes_takes_priority(self):
        """Magic bytes should take priority over extension."""
        # PDF data with .txt extension
        pdf_data = b"%PDF-1.7\nContent here"
        result = classify_document(
            data=pdf_data,
            filename="misleading.txt",
            mime_type="text/plain",
        )

        assert result.document_type == DocumentType.PDF
        assert result.mime_type == "application/pdf"

    def test_extension_used_when_no_magic(self):
        """Extension should be used when magic bytes don't match."""
        # Random data with .docx extension
        unknown_data = b"random data"
        result = classify_document(
            data=unknown_data,
            filename="document.docx",
        )

        assert result.document_type == DocumentType.DOCX

    def test_zip_refined_by_extension_docx(self):
        """ZIP signature should be refined to DOCX by extension."""
        zip_data = b"PK\x03\x04\x00\x00"
        result = classify_document(
            data=zip_data,
            filename="document.docx",
        )

        assert result.document_type == DocumentType.DOCX

    def test_zip_refined_by_extension_xlsx(self):
        """ZIP signature should be refined to XLSX by extension."""
        zip_data = b"PK\x03\x04\x00\x00"
        result = classify_document(
            data=zip_data,
            filename="spreadsheet.xlsx",
        )

        assert result.document_type == DocumentType.SPREADSHEET

    def test_mime_type_fallback(self):
        """MIME type should be used as fallback."""
        result = classify_document(
            data=None,
            filename=None,
            mime_type="application/pdf",
        )

        assert result.document_type == DocumentType.PDF

    def test_confidence_boost_for_agreement(self):
        """Confidence should be boosted when multiple methods agree."""
        # PDF data with .pdf extension
        pdf_data = b"%PDF-1.7\n"
        result = classify_document(
            data=pdf_data,
            filename="document.pdf",
            mime_type="application/pdf",
        )

        assert result.confidence > 0.95

    def test_unknown_when_nothing_matches(self):
        """UNKNOWN should be returned when nothing matches."""
        result = classify_document(
            data=b"random bytes",
            filename="no_extension",
            mime_type=None,
        )

        assert result.document_type == DocumentType.UNKNOWN
        assert result.confidence == 0.0

    def test_all_none_returns_unknown(self):
        """All None inputs should return UNKNOWN."""
        result = classify_document(
            data=None,
            filename=None,
            mime_type=None,
        )

        assert result.document_type == DocumentType.UNKNOWN


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper classification functions."""

    def test_is_text_based(self):
        """Text-based types should be correctly identified."""
        text_types = [
            DocumentType.TXT,
            DocumentType.MARKDOWN,
            DocumentType.HTML,
            DocumentType.JSON,
            DocumentType.XML,
        ]
        for doc_type in text_types:
            assert is_text_based(doc_type) is True

        non_text_types = [
            DocumentType.PDF,
            DocumentType.IMAGE,
            DocumentType.AUDIO,
            DocumentType.VIDEO,
        ]
        for doc_type in non_text_types:
            assert is_text_based(doc_type) is False

    def test_needs_ocr(self):
        """Image types should need OCR."""
        assert needs_ocr(DocumentType.IMAGE) is True
        assert needs_ocr(DocumentType.PDF) is False
        assert needs_ocr(DocumentType.TXT) is False

    def test_needs_transcription(self):
        """Audio and video types should need transcription."""
        assert needs_transcription(DocumentType.AUDIO) is True
        assert needs_transcription(DocumentType.VIDEO) is True
        assert needs_transcription(DocumentType.PDF) is False
        assert needs_transcription(DocumentType.IMAGE) is False

    def test_is_compound_document(self):
        """Compound document types should be correctly identified."""
        compound_types = [
            DocumentType.PDF,
            DocumentType.DOCX,
            DocumentType.DOC,
            DocumentType.SPREADSHEET,
            DocumentType.PRESENTATION,
            DocumentType.EMAIL,
        ]
        for doc_type in compound_types:
            assert is_compound_document(doc_type) is True

        simple_types = [
            DocumentType.TXT,
            DocumentType.MARKDOWN,
            DocumentType.IMAGE,
        ]
        for doc_type in simple_types:
            assert is_compound_document(doc_type) is False


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_binary_garbage_does_not_crash(self):
        """Random binary data should not crash classification."""
        import random

        garbage = bytes(random.randint(0, 255) for _ in range(1024))

        # Should not raise
        result = classify_document(data=garbage, filename=None, mime_type=None)
        assert result is not None

    def test_very_long_filename(self):
        """Very long filenames should be handled."""
        long_name = "a" * 1000 + ".pdf"
        result = detect_by_extension(long_name)

        assert result is not None
        assert result.document_type == DocumentType.PDF

    def test_multiple_extensions(self):
        """Files with multiple extensions should use the last one."""
        result = detect_by_extension("archive.tar.gz")

        assert result is not None
        assert result.document_type == DocumentType.ARCHIVE

    def test_hidden_file_with_extension(self):
        """Hidden files (starting with .) should be handled."""
        result = detect_by_extension(".hidden.pdf")

        assert result is not None
        assert result.document_type == DocumentType.PDF

    def test_unicode_filename(self):
        """Unicode filenames should be handled."""
        result = detect_by_extension("文档.pdf")

        assert result is not None
        assert result.document_type == DocumentType.PDF

    def test_extension_map_coverage(self):
        """All extension map entries should have valid document types."""
        for ext, (doc_type, mime) in EXTENSION_MAP.items():
            assert isinstance(doc_type, DocumentType)
            assert isinstance(mime, str)
            assert len(mime) > 0

    def test_magic_signatures_valid(self):
        """All magic signatures should have valid document types."""
        for sig in MAGIC_SIGNATURES:
            assert isinstance(sig.bytes_pattern, bytes)
            assert len(sig.bytes_pattern) > 0
            assert isinstance(sig.document_type, DocumentType)
            assert isinstance(sig.mime_type, str)
