"""
Module-level convenience functions for S3 storage operations.
"""

from typing import Any, Dict, List, Optional
from utils.s3.client import S3Client

_default_client: Optional[S3Client] = None


def _get_client() -> S3Client:
    global _default_client
    if _default_client is None:
        _default_client = S3Client()
    return _default_client


def upload_file(
    local_file_path: str,
    s3_key: str,
    bucket_name: Optional[str] = None,
    extra_args: Optional[Dict[str, Any]] = None,
) -> str:
    """Uploads a local file to S3."""
    return _get_client().upload_file(
        local_file_path=local_file_path,
        s3_key=s3_key,
        bucket_name=bucket_name,
        extra_args=extra_args,
    )


def upload_bytes(
    data: bytes,
    s3_key: str,
    bucket_name: Optional[str] = None,
    content_type: Optional[str] = None,
    extra_args: Optional[Dict[str, Any]] = None,
) -> str:
    """Uploads raw bytes to S3."""
    return _get_client().upload_bytes(
        data=data,
        s3_key=s3_key,
        bucket_name=bucket_name,
        content_type=content_type,
        extra_args=extra_args,
    )


def download_file(
    s3_key: str,
    local_file_path: str,
    bucket_name: Optional[str] = None,
) -> str:
    """Downloads an S3 object to a local file path."""
    return _get_client().download_file(
        s3_key=s3_key,
        local_file_path=local_file_path,
        bucket_name=bucket_name,
    )


def download_bytes(s3_key: str, bucket_name: Optional[str] = None) -> bytes:
    """Downloads an S3 object content directly as bytes."""
    return _get_client().download_bytes(s3_key=s3_key, bucket_name=bucket_name)


def generate_presigned_url(
    s3_key: str,
    client_method: str = "get_object",
    expiration: int = 3600,
    bucket_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Generates a presigned URL for an S3 object action."""
    return _get_client().generate_presigned_url(
        s3_key=s3_key,
        client_method=client_method,
        expiration=expiration,
        bucket_name=bucket_name,
        params=params,
    )


def file_exists(s3_key: str, bucket_name: Optional[str] = None) -> bool:
    """Checks if an object exists in S3."""
    return _get_client().file_exists(s3_key=s3_key, bucket_name=bucket_name)


def delete_file(s3_key: str, bucket_name: Optional[str] = None) -> bool:
    """Deletes an object from S3."""
    return _get_client().delete_file(s3_key=s3_key, bucket_name=bucket_name)


def list_objects(
    prefix: str = "",
    bucket_name: Optional[str] = None,
    max_keys: int = 1000,
) -> List[str]:
    """Lists object keys matching a prefix in S3."""
    return _get_client().list_objects(
        prefix=prefix, bucket_name=bucket_name, max_keys=max_keys
    )
