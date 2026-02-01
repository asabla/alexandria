"""
File download activity for the ingestion workflow.

Handles downloading files from various sources:
- HTTP/HTTPS URLs with custom headers
- YouTube videos (via yt-dlp)
- Other supported protocols

Downloads are stored in MinIO for subsequent processing.
"""

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

import httpx
from temporalio import activity

# Constants
DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_CHUNK_SIZE = 8192  # 8KB chunks for streaming
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB default limit
HEARTBEAT_INTERVAL_BYTES = 1024 * 1024  # Heartbeat every 1MB downloaded

# User agent for HTTP requests
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; Alexandria/1.0; +https://github.com/alexandria-platform)"
)

# YouTube URL patterns
YOUTUBE_PATTERNS = [
    r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+",
    r"(?:https?://)?(?:www\.)?youtu\.be/[\w-]+",
    r"(?:https?://)?(?:www\.)?youtube\.com/shorts/[\w-]+",
]


class DownloadSource(StrEnum):
    """Source type for downloads."""

    HTTP = "http"
    YOUTUBE = "youtube"
    LOCAL = "local"  # For testing/development


@dataclass
class DownloadFileInput:
    """Input for file download activity."""

    document_id: str
    tenant_id: str
    source_url: str
    storage_bucket: str
    storage_key: str  # Target key in MinIO

    # Optional HTTP headers (e.g., Authorization, cookies)
    headers: dict[str, str] = field(default_factory=dict)

    # Download options
    timeout_seconds: int = DEFAULT_TIMEOUT
    max_file_size: int = MAX_FILE_SIZE
    follow_redirects: bool = True

    # YouTube-specific options
    youtube_format: str = "bestaudio/best"  # yt-dlp format string
    extract_audio_only: bool = False


@dataclass
class DownloadFileOutput:
    """Output from file download activity."""

    storage_bucket: str
    storage_key: str
    file_size: int
    content_type: str | None
    file_hash: str  # SHA-256 hash of downloaded content
    source_type: str  # http, youtube, etc.
    original_filename: str | None = None
    download_duration_seconds: float = 0.0


def _detect_source_type(url: str) -> DownloadSource:
    """Detect the source type from a URL."""
    # Check for YouTube URLs
    for pattern in YOUTUBE_PATTERNS:
        if re.match(pattern, url, re.IGNORECASE):
            return DownloadSource.YOUTUBE

    # Check for local file URLs (for testing)
    if url.startswith("file://"):
        return DownloadSource.LOCAL

    # Default to HTTP
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        return DownloadSource.HTTP

    raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")


def _extract_filename_from_url(url: str) -> str | None:
    """Extract filename from URL path."""
    parsed = urlparse(url)
    path = parsed.path
    if path and "/" in path:
        filename = path.rsplit("/", 1)[-1]
        if filename and "." in filename:
            return filename
    return None


def _extract_filename_from_headers(headers: httpx.Headers) -> str | None:
    """Extract filename from Content-Disposition header."""
    content_disposition = headers.get("content-disposition", "")
    if "filename=" in content_disposition:
        # Try to extract quoted filename
        match = re.search(r'filename="([^"]+)"', content_disposition)
        if match:
            return match.group(1)
        # Try unquoted filename
        match = re.search(r"filename=([^\s;]+)", content_disposition)
        if match:
            return match.group(1)
    return None


def _get_minio_client():
    """Get MinIO client from environment configuration."""
    from alexandria_db.clients.minio import MinIOClient

    return MinIOClient(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )


async def _download_http(
    input: DownloadFileInput,
) -> tuple[bytes, str | None, str | None]:
    """
    Download file via HTTP/HTTPS.

    Returns:
        Tuple of (file_bytes, content_type, original_filename)
    """
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        **input.headers,
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(input.timeout_seconds),
        follow_redirects=input.follow_redirects,
    ) as client:
        # First, do a HEAD request to check file size
        try:
            head_response = await client.head(input.source_url, headers=headers)
            content_length = head_response.headers.get("content-length")
            if content_length and int(content_length) > input.max_file_size:
                raise ValueError(
                    f"File size ({int(content_length)} bytes) exceeds maximum "
                    f"allowed size ({input.max_file_size} bytes)"
                )
        except httpx.HTTPError:
            # HEAD request failed, proceed with GET
            pass

        # Stream the download with progress tracking
        activity.logger.info(f"Starting HTTP download from {input.source_url}")

        async with client.stream("GET", input.source_url, headers=headers) as response:
            response.raise_for_status()

            content_type = response.headers.get("content-type")
            original_filename = _extract_filename_from_headers(
                response.headers
            ) or _extract_filename_from_url(input.source_url)

            # Download in chunks with heartbeat
            chunks = []
            total_bytes = 0
            bytes_since_heartbeat = 0

            async for chunk in response.aiter_bytes(chunk_size=DEFAULT_CHUNK_SIZE):
                chunks.append(chunk)
                total_bytes += len(chunk)
                bytes_since_heartbeat += len(chunk)

                # Check size limit
                if total_bytes > input.max_file_size:
                    raise ValueError(
                        f"Download exceeded maximum file size ({input.max_file_size} bytes)"
                    )

                # Send heartbeat periodically
                if bytes_since_heartbeat >= HEARTBEAT_INTERVAL_BYTES:
                    activity.heartbeat(f"Downloaded {total_bytes} bytes")
                    bytes_since_heartbeat = 0

            file_bytes = b"".join(chunks)
            activity.logger.info(f"HTTP download complete: {total_bytes} bytes")

            return file_bytes, content_type, original_filename


async def _download_youtube(
    input: DownloadFileInput,
) -> tuple[bytes, str | None, str | None]:
    """
    Download video/audio from YouTube using yt-dlp.

    Returns:
        Tuple of (file_bytes, content_type, original_filename)
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed. Install with: pip install yt-dlp")

    activity.logger.info(f"Starting YouTube download from {input.source_url}")

    # Create temp directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        output_template = str(Path(temp_dir) / "%(title)s.%(ext)s")

        ydl_opts = {
            "format": input.youtube_format,
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "extract_audio": input.extract_audio_only,
            "progress_hooks": [lambda d: _youtube_progress_hook(d)],
        }

        if input.extract_audio_only:
            ydl_opts["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ]

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(input.source_url, download=True)
            if info is None:
                raise ValueError(f"Could not extract info from {input.source_url}")

            # Find the downloaded file
            if input.extract_audio_only:
                # Audio extraction changes extension
                title = info.get("title", "video")
                downloaded_file = Path(temp_dir) / f"{title}.mp3"
                content_type = "audio/mpeg"
            else:
                # Get the actual downloaded filename
                ext = info.get("ext", "mp4")
                title = info.get("title", "video")
                downloaded_file = Path(temp_dir) / f"{title}.{ext}"
                content_type = f"video/{ext}" if ext in ("mp4", "webm", "mkv") else None

            # Find any file in temp dir if exact match fails
            if not downloaded_file.exists():
                files = list(Path(temp_dir).iterdir())
                if files:
                    downloaded_file = files[0]
                else:
                    raise FileNotFoundError("No file was downloaded")

            # Read the file
            file_bytes = downloaded_file.read_bytes()
            original_filename = downloaded_file.name

            # Check size limit
            if len(file_bytes) > input.max_file_size:
                raise ValueError(
                    f"Downloaded file size ({len(file_bytes)} bytes) exceeds "
                    f"maximum allowed size ({input.max_file_size} bytes)"
                )

            activity.logger.info(
                f"YouTube download complete: {len(file_bytes)} bytes, {original_filename}"
            )

            return file_bytes, content_type, original_filename


def _youtube_progress_hook(d: dict) -> None:
    """Progress hook for yt-dlp downloads."""
    if d["status"] == "downloading":
        downloaded = d.get("downloaded_bytes", 0)
        total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
        if total:
            percent = (downloaded / total) * 100
            activity.heartbeat(f"YouTube download: {percent:.1f}%")
        else:
            activity.heartbeat(f"YouTube download: {downloaded} bytes")
    elif d["status"] == "finished":
        activity.heartbeat("YouTube download finished, processing...")


async def _download_local(
    input: DownloadFileInput,
) -> tuple[bytes, str | None, str | None]:
    """
    Read file from local filesystem (for testing).

    Returns:
        Tuple of (file_bytes, content_type, original_filename)
    """
    # Remove file:// prefix
    file_path = input.source_url.replace("file://", "")
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Local file not found: {file_path}")

    if path.stat().st_size > input.max_file_size:
        raise ValueError(
            f"File size ({path.stat().st_size} bytes) exceeds maximum "
            f"allowed size ({input.max_file_size} bytes)"
        )

    file_bytes = path.read_bytes()
    original_filename = path.name

    # Guess content type from extension
    ext = path.suffix.lower().lstrip(".")
    content_type_map = {
        "pdf": "application/pdf",
        "txt": "text/plain",
        "html": "text/html",
        "json": "application/json",
        "xml": "application/xml",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "mp3": "audio/mpeg",
        "mp4": "video/mp4",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")

    return file_bytes, content_type, original_filename


@activity.defn
async def download_file(input: DownloadFileInput) -> DownloadFileOutput:
    """
    Download a file from a URL and store it in MinIO.

    This activity handles different source types:
    - HTTP/HTTPS: Standard web downloads with optional headers
    - YouTube: Video/audio downloads via yt-dlp
    - Local: File system access (for testing)

    The activity sends heartbeats during download to prevent timeout.

    Args:
        input: Download configuration including URL, storage location, and options

    Returns:
        DownloadFileOutput with storage location and file metadata
    """
    import time

    start_time = time.time()

    activity.logger.info(
        "Starting file download",
        extra={
            "document_id": input.document_id,
            "source_url": input.source_url,
            "storage_key": input.storage_key,
        },
    )

    # Detect source type
    source_type = _detect_source_type(input.source_url)
    activity.logger.info(f"Detected source type: {source_type}")

    # Download based on source type
    if source_type == DownloadSource.HTTP:
        file_bytes, content_type, original_filename = await _download_http(input)
    elif source_type == DownloadSource.YOUTUBE:
        file_bytes, content_type, original_filename = await _download_youtube(input)
    elif source_type == DownloadSource.LOCAL:
        file_bytes, content_type, original_filename = await _download_local(input)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    # Calculate file hash
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    activity.heartbeat("Calculated file hash")

    # Upload to MinIO
    activity.logger.info(f"Uploading to MinIO: {input.storage_bucket}/{input.storage_key}")

    minio_client = _get_minio_client()
    await minio_client.ensure_bucket(input.storage_bucket)
    await minio_client.upload_object(
        bucket=input.storage_bucket,
        key=input.storage_key,
        data=file_bytes,
        content_type=content_type or "application/octet-stream",
        metadata={
            "document_id": input.document_id,
            "source_url": input.source_url,
            "original_filename": original_filename or "",
            "file_hash": file_hash,
        },
    )

    download_duration = time.time() - start_time

    activity.logger.info(
        "File download complete",
        extra={
            "document_id": input.document_id,
            "file_size": len(file_bytes),
            "file_hash": file_hash,
            "duration_seconds": download_duration,
        },
    )

    return DownloadFileOutput(
        storage_bucket=input.storage_bucket,
        storage_key=input.storage_key,
        file_size=len(file_bytes),
        content_type=content_type,
        file_hash=file_hash,
        source_type=source_type.value,
        original_filename=original_filename,
        download_duration_seconds=download_duration,
    )
