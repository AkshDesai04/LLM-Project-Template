"""
S3 client wrapper for standard AWS operations and authentication.
"""

import os
from typing import Any, Dict, List, Optional

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

    class ClientError(Exception):  # type: ignore
        pass


from utils.env import get_secret
from utils.logging import LOGGING_MODE_LAMBDA, get_logger, get_logging_mode

logger = get_logger("S3Client")


class S3Client:
    """S3 client supporting IAM roles in LAMBDA mode and local credentials in NORMAL mode."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is not installed. Please install boto3 to use S3 functionality."
            )

        self.default_bucket = bucket_name or get_secret(
            "AWS_S3_BUCKET", raise_error=False
        )
        self.client = self._init_client(
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )

    def _init_client(
        self,
        region_name: Optional[str],
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        aws_session_token: Optional[str],
    ) -> Any:
        mode = get_logging_mode()
        region = (
            region_name
            or get_secret("AWS_REGION", raise_error=False)
            or os.getenv("AWS_DEFAULT_REGION")
        )

        if mode == LOGGING_MODE_LAMBDA:
            logger.info("Initializing S3 client in LAMBDA mode via IAM role credentials.")
            kwargs: Dict[str, Any] = {}
            if region:
                kwargs["region_name"] = region
            return boto3.client("s3", **kwargs)

        logger.info("Initializing S3 client in NORMAL mode using configured secrets/env.")
        key_id = aws_access_key_id or get_secret(
            "AWS_ACCESS_KEY_ID", raise_error=False
        )
        secret_key = aws_secret_access_key or get_secret(
            "AWS_SECRET_ACCESS_KEY", raise_error=False
        )
        session_token = aws_session_token or get_secret(
            "AWS_SESSION_TOKEN", raise_error=False
        )

        kwargs = {}
        if key_id and secret_key:
            kwargs["aws_access_key_id"] = key_id
            kwargs["aws_secret_access_key"] = secret_key
            if session_token:
                kwargs["aws_session_token"] = session_token
        if region:
            kwargs["region_name"] = region

        return boto3.client("s3", **kwargs)

    def _resolve_bucket(self, bucket_name: Optional[str]) -> str:
        target = bucket_name or self.default_bucket
        if not target:
            raise ValueError(
                "Bucket name must be provided or configured via default_bucket / AWS_S3_BUCKET."
            )
        return target

    def upload_file(
        self,
        local_file_path: str,
        s3_key: str,
        bucket_name: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Uploads a local file to S3."""
        bucket = self._resolve_bucket(bucket_name)
        logger.info(f"Uploading file '{local_file_path}' to s3://{bucket}/{s3_key}")
        self.client.upload_file(
            Filename=local_file_path,
            Bucket=bucket,
            Key=s3_key,
            ExtraArgs=extra_args,
        )
        return f"s3://{bucket}/{s3_key}"

    def upload_bytes(
        self,
        data: bytes,
        s3_key: str,
        bucket_name: Optional[str] = None,
        content_type: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Uploads raw bytes to S3."""
        bucket = self._resolve_bucket(bucket_name)
        logger.info(f"Uploading {len(data)} bytes to s3://{bucket}/{s3_key}")
        upload_args = dict(extra_args) if extra_args else {}
        if content_type:
            upload_args["ContentType"] = content_type
        self.client.put_object(
            Bucket=bucket, Key=s3_key, Body=data, **upload_args
        )
        return f"s3://{bucket}/{s3_key}"

    def download_file(
        self,
        s3_key: str,
        local_file_path: str,
        bucket_name: Optional[str] = None,
    ) -> str:
        """Downloads an S3 object to a local file path."""
        bucket = self._resolve_bucket(bucket_name)
        logger.info(f"Downloading s3://{bucket}/{s3_key} to '{local_file_path}'")
        os.makedirs(os.path.dirname(os.path.abspath(local_file_path)), exist_ok=True)
        self.client.download_file(Bucket=bucket, Key=s3_key, Filename=local_file_path)
        return local_file_path

    def download_bytes(
        self, s3_key: str, bucket_name: Optional[str] = None
    ) -> bytes:
        """Downloads an S3 object content directly as bytes."""
        bucket = self._resolve_bucket(bucket_name)
        logger.info(f"Downloading bytes from s3://{bucket}/{s3_key}")
        response = self.client.get_object(Bucket=bucket, Key=s3_key)
        return response["Body"].read()

    def generate_presigned_url(
        self,
        s3_key: str,
        client_method: str = "get_object",
        expiration: int = 3600,
        bucket_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generates a presigned URL for an S3 object action (default 'get_object')."""
        bucket = self._resolve_bucket(bucket_name)
        method_params = {"Bucket": bucket, "Key": s3_key}
        if params:
            method_params.update(params)
        logger.info(
            f"Generating presigned URL for '{client_method}' on s3://{bucket}/{s3_key} (expires in {expiration}s)"
        )
        return self.client.generate_presigned_url(
            ClientMethod=client_method,
            Params=method_params,
            ExpiresIn=expiration,
        )

    def file_exists(
        self, s3_key: str, bucket_name: Optional[str] = None
    ) -> bool:
        """Checks if an object exists in S3."""
        bucket = self._resolve_bucket(bucket_name)
        try:
            self.client.head_object(Bucket=bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return False
            raise

    def delete_file(
        self, s3_key: str, bucket_name: Optional[str] = None
    ) -> bool:
        """Deletes an object from S3."""
        bucket = self._resolve_bucket(bucket_name)
        logger.info(f"Deleting object s3://{bucket}/{s3_key}")
        self.client.delete_object(Bucket=bucket, Key=s3_key)
        return True

    def list_objects(
        self,
        prefix: str = "",
        bucket_name: Optional[str] = None,
        max_keys: int = 1000,
    ) -> List[str]:
        """Lists object keys in S3 matching a prefix."""
        bucket = self._resolve_bucket(bucket_name)
        response = self.client.list_objects_v2(
            Bucket=bucket, Prefix=prefix, MaxKeys=max_keys
        )
        contents = response.get("Contents", [])
        return [obj["Key"] for obj in contents]
