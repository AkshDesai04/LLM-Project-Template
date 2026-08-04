"""
Unit tests for S3 storage utility (S3Client and helper functions).
"""

from unittest.mock import MagicMock, patch
import pytest

from utils.s3 import (
    S3Client,
    delete_file,
    download_bytes,
    download_file,
    file_exists,
    generate_presigned_url,
    list_objects,
    upload_bytes,
    upload_file,
)


@pytest.fixture
def mock_boto3_client():
    with patch("boto3.client") as mock_client_fn:
        mock_s3 = MagicMock()
        mock_client_fn.return_value = mock_s3
        yield mock_s3


def test_s3_client_lambda_mode(mock_boto3_client):
    with patch("utils.s3.client.get_logging_mode", return_value="LAMBDA"):
        client = S3Client(bucket_name="test-bucket", region_name="us-east-1")
        assert client.default_bucket == "test-bucket"
        assert client.client == mock_boto3_client


def test_s3_client_normal_mode(mock_boto3_client):
    with patch("utils.s3.client.get_logging_mode", return_value="NORMAL"), patch(
        "utils.s3.client.get_secret"
    ) as mock_secret:
        mock_secret.side_effect = lambda key, raise_error=False: {
            "AWS_ACCESS_KEY_ID": "mock_id",
            "AWS_SECRET_ACCESS_KEY": "mock_secret",
            "AWS_REGION": "us-west-2",
            "AWS_S3_BUCKET": "env-bucket",
        }.get(key)

        client = S3Client()
        assert client.default_bucket == "env-bucket"


def test_upload_file(mock_boto3_client):
    with patch("utils.s3.client.get_logging_mode", return_value="LAMBDA"):
        client = S3Client(bucket_name="my-bucket")
        res = client.upload_file("local.txt", "remote/file.txt")
        assert res == "s3://my-bucket/remote/file.txt"
        mock_boto3_client.upload_file.assert_called_once_with(
            Filename="local.txt",
            Bucket="my-bucket",
            Key="remote/file.txt",
            ExtraArgs=None,
        )


def test_upload_bytes(mock_boto3_client):
    with patch("utils.s3.client.get_logging_mode", return_value="LAMBDA"):
        client = S3Client(bucket_name="my-bucket")
        res = client.upload_bytes(b"hello world", "data.txt", content_type="text/plain")
        assert res == "s3://my-bucket/data.txt"
        mock_boto3_client.put_object.assert_called_once_with(
            Bucket="my-bucket", Key="data.txt", Body=b"hello world", ContentType="text/plain"
        )


def test_download_file(mock_boto3_client, tmp_path):
    dest_path = str(tmp_path / "downloaded.txt")
    with patch("utils.s3.client.get_logging_mode", return_value="LAMBDA"):
        client = S3Client(bucket_name="my-bucket")
        res = client.download_file("remote/file.txt", dest_path)
        assert res == dest_path
        mock_boto3_client.download_file.assert_called_once_with(
            Bucket="my-bucket", Key="remote/file.txt", Filename=dest_path
        )


def test_download_bytes(mock_boto3_client):
    mock_body = MagicMock()
    mock_body.read.return_value = b"sample content"
    mock_boto3_client.get_object.return_value = {"Body": mock_body}

    with patch("utils.s3.client.get_logging_mode", return_value="LAMBDA"):
        client = S3Client(bucket_name="my-bucket")
        content = client.download_bytes("remote/file.txt")
        assert content == b"sample content"
        mock_boto3_client.get_object.assert_called_once_with(
            Bucket="my-bucket", Key="remote/file.txt"
        )


def test_generate_presigned_url(mock_boto3_client):
    mock_boto3_client.generate_presigned_url.return_value = "https://s3.amazonaws.com/signed-url"
    with patch("utils.s3.client.get_logging_mode", return_value="LAMBDA"):
        client = S3Client(bucket_name="my-bucket")
        url = client.generate_presigned_url("my-key.pdf", client_method="get_object", expiration=1800)
        assert url == "https://s3.amazonaws.com/signed-url"
        mock_boto3_client.generate_presigned_url.assert_called_once_with(
            ClientMethod="get_object",
            Params={"Bucket": "my-bucket", "Key": "my-key.pdf"},
            ExpiresIn=1800,
        )


def test_file_exists_true_and_false(mock_boto3_client):
    with patch("utils.s3.client.get_logging_mode", return_value="LAMBDA"):
        client = S3Client(bucket_name="my-bucket")

        # True case
        mock_boto3_client.head_object.return_value = {}
        assert client.file_exists("existing.txt") is True

        # False case (NoSuchKey / 404)
        from botocore.exceptions import ClientError
        error_resp = {"Error": {"Code": "404", "Message": "Not Found"}}
        mock_boto3_client.head_object.side_effect = ClientError(error_resp, "HeadObject")
        assert client.file_exists("missing.txt") is False


def test_delete_file(mock_boto3_client):
    with patch("utils.s3.client.get_logging_mode", return_value="LAMBDA"):
        client = S3Client(bucket_name="my-bucket")
        assert client.delete_file("obsolete.txt") is True
        mock_boto3_client.delete_object.assert_called_once_with(
            Bucket="my-bucket", Key="obsolete.txt"
        )


def test_list_objects(mock_boto3_client):
    mock_boto3_client.list_objects_v2.return_value = {
        "Contents": [{"Key": "folder/a.txt"}, {"Key": "folder/b.txt"}]
    }
    with patch("utils.s3.client.get_logging_mode", return_value="LAMBDA"):
        client = S3Client(bucket_name="my-bucket")
        keys = client.list_objects(prefix="folder/")
        assert keys == ["folder/a.txt", "folder/b.txt"]
        mock_boto3_client.list_objects_v2.assert_called_once_with(
            Bucket="my-bucket", Prefix="folder/", MaxKeys=1000
        )


def test_top_level_convenience_functions(mock_boto3_client):
    with patch("utils.s3.client.get_logging_mode", return_value="LAMBDA"), patch(
        "utils.s3.s3_ops._default_client", None
    ):
        mock_boto3_client.generate_presigned_url.return_value = "https://presigned"
        url = generate_presigned_url("test.png", bucket_name="top-bucket")
        assert url == "https://presigned"
