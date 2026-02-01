"""MinIO/S3 object storage client wrapper."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from io import BytesIO
from typing import BinaryIO

import structlog
from minio import Minio
from minio.error import S3Error

logger = structlog.get_logger(__name__)


@dataclass
class ObjectInfo:
    """Information about a stored object."""

    bucket: str
    key: str
    size: int
    etag: str
    content_type: str | None = None


class MinIOClient:
    """
    Wrapper for MinIO/S3 object storage operations.

    Provides a simplified interface for common operations like
    uploading, downloading, and managing objects in MinIO buckets.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
    ):
        """
        Initialize MinIO client.

        Args:
            endpoint: MinIO server endpoint (e.g., "localhost:9000")
            access_key: Access key for authentication
            secret_key: Secret key for authentication
            secure: Whether to use HTTPS
        """
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._endpoint = endpoint
        self._secure = secure
        self._log = logger.bind(service="minio", endpoint=endpoint)

    async def ensure_bucket(self, bucket: str) -> bool:
        """
        Ensure a bucket exists, creating it if necessary.

        Args:
            bucket: Name of the bucket

        Returns:
            True if bucket was created, False if it already existed
        """
        try:
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                self._log.info("bucket_created", bucket=bucket)
                return True
            return False
        except S3Error as e:
            self._log.error("bucket_ensure_failed", bucket=bucket, error=str(e))
            raise

    async def upload_object(
        self,
        bucket: str,
        key: str,
        data: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> ObjectInfo:
        """
        Upload an object to MinIO.

        Args:
            bucket: Target bucket name
            key: Object key (path in bucket)
            data: Object data as bytes or file-like object
            content_type: MIME type of the object
            metadata: Optional metadata to attach to the object

        Returns:
            ObjectInfo with details about the uploaded object
        """
        try:
            if isinstance(data, bytes):
                data_stream = BytesIO(data)
                length = len(data)
            else:
                data.seek(0, 2)  # Seek to end
                length = data.tell()
                data.seek(0)  # Seek back to start
                data_stream = data

            result = self._client.put_object(
                bucket,
                key,
                data_stream,
                length=length,
                content_type=content_type,
                metadata=metadata or {},
            )

            self._log.info(
                "object_uploaded",
                bucket=bucket,
                key=key,
                size=length,
                etag=result.etag,
            )

            return ObjectInfo(
                bucket=bucket,
                key=key,
                size=length,
                etag=result.etag,
                content_type=content_type,
            )
        except S3Error as e:
            self._log.error("object_upload_failed", bucket=bucket, key=key, error=str(e))
            raise

    async def download_object(self, bucket: str, key: str) -> bytes:
        """
        Download an object from MinIO.

        Args:
            bucket: Source bucket name
            key: Object key

        Returns:
            Object data as bytes
        """
        try:
            response = self._client.get_object(bucket, key)
            data = response.read()
            response.close()
            response.release_conn()

            self._log.debug("object_downloaded", bucket=bucket, key=key, size=len(data))
            return data
        except S3Error as e:
            self._log.error("object_download_failed", bucket=bucket, key=key, error=str(e))
            raise

    @asynccontextmanager
    async def stream_object(self, bucket: str, key: str) -> AsyncGenerator[BinaryIO, None]:
        """
        Stream an object from MinIO.

        Args:
            bucket: Source bucket name
            key: Object key

        Yields:
            File-like object for streaming the data
        """
        response = None
        try:
            response = self._client.get_object(bucket, key)
            yield response
        finally:
            if response:
                response.close()
                response.release_conn()

    async def delete_object(self, bucket: str, key: str) -> None:
        """
        Delete an object from MinIO.

        Args:
            bucket: Bucket name
            key: Object key to delete
        """
        try:
            self._client.remove_object(bucket, key)
            self._log.info("object_deleted", bucket=bucket, key=key)
        except S3Error as e:
            self._log.error("object_delete_failed", bucket=bucket, key=key, error=str(e))
            raise

    async def get_presigned_url(
        self,
        bucket: str,
        key: str,
        expires: timedelta = timedelta(hours=1),
        method: str = "GET",
    ) -> str:
        """
        Generate a pre-signed URL for an object.

        Args:
            bucket: Bucket name
            key: Object key
            expires: URL expiration time
            method: HTTP method (GET or PUT)

        Returns:
            Pre-signed URL string
        """
        try:
            if method.upper() == "GET":
                url = self._client.presigned_get_object(bucket, key, expires=expires)
            elif method.upper() == "PUT":
                url = self._client.presigned_put_object(bucket, key, expires=expires)
            else:
                raise ValueError(f"Unsupported method: {method}")

            self._log.debug(
                "presigned_url_generated",
                bucket=bucket,
                key=key,
                method=method,
                expires_seconds=expires.total_seconds(),
            )
            return url
        except S3Error as e:
            self._log.error("presigned_url_failed", bucket=bucket, key=key, error=str(e))
            raise

    async def object_exists(self, bucket: str, key: str) -> bool:
        """
        Check if an object exists.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            True if object exists, False otherwise
        """
        try:
            self._client.stat_object(bucket, key)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise

    async def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        recursive: bool = True,
    ) -> list[ObjectInfo]:
        """
        List objects in a bucket.

        Args:
            bucket: Bucket name
            prefix: Filter by prefix
            recursive: Whether to list recursively

        Returns:
            List of ObjectInfo for matching objects
        """
        try:
            objects = self._client.list_objects(bucket, prefix=prefix, recursive=recursive)
            return [
                ObjectInfo(
                    bucket=bucket,
                    key=obj.object_name,
                    size=obj.size,
                    etag=obj.etag,
                )
                for obj in objects
                if not obj.is_dir
            ]
        except S3Error as e:
            self._log.error("list_objects_failed", bucket=bucket, prefix=prefix, error=str(e))
            raise

    async def health_check(self) -> bool:
        """
        Check if MinIO is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to list buckets as a health check
            list(self._client.list_buckets())
            return True
        except Exception as e:
            self._log.warning("health_check_failed", error=str(e))
            return False
