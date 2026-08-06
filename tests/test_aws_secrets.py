"""
Unit tests for AWS Secrets Manager utilities (utils.env.aws_secrets).
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from utils.env import aws_secrets
from utils.env.aws_secrets import (
    _get_secrets_manager_client,
    get_aws_secret,
    get_keys_dict,
    get_secret_dict,
)


@pytest.fixture(autouse=True)
def reset_aws_secrets_globals():
    aws_secrets._secrets_manager_client = None
    aws_secrets._aws_secrets_cache = {}
    aws_secrets._api_keys_dict = None
    yield
    aws_secrets._secrets_manager_client = None
    aws_secrets._aws_secrets_cache = {}
    aws_secrets._api_keys_dict = None


@pytest.mark.env
@pytest.mark.unit
def test_get_secrets_manager_client_missing_boto3():
    with patch("utils.env.aws_secrets.BOTO3_AVAILABLE", False):
        with pytest.raises(ImportError, match="boto3 is not installed"):
            _get_secrets_manager_client()


@pytest.mark.env
@pytest.mark.unit
def test_get_secrets_manager_client_missing_region():
    with patch("utils.env.aws_secrets.BOTO3_AVAILABLE", True), patch("os.getenv", return_value=None), patch(
        "utils.env.aws_secrets.load_dotenv"
    ):
        with pytest.raises(ValueError, match="AWS_REGION or AWS_DEFAULT_REGION"):
            _get_secrets_manager_client()


@pytest.mark.env
@pytest.mark.unit
def test_get_secrets_manager_client_success():
    mock_boto = MagicMock()
    with patch("utils.env.aws_secrets.BOTO3_AVAILABLE", True), patch(
        "os.getenv", side_effect=lambda k: "us-west-2" if k == "AWS_REGION" else None
    ), patch("boto3.client", return_value=mock_boto) as mock_client_fn:
        client = _get_secrets_manager_client()
        assert client == mock_boto
        mock_client_fn.assert_called_once_with("secretsmanager", region_name="us-west-2")


@pytest.mark.env
@pytest.mark.unit
def test_get_secret_dict_cache_and_fetch():
    mock_sm = MagicMock()
    mock_sm.get_secret_value.return_value = {
        "SecretString": json.dumps({"GEMINI_API_KEY": "g123", "OPENAI_API_KEY": "o123"})
    }
    with patch("utils.env.aws_secrets._get_secrets_manager_client", return_value=mock_sm):
        # First call fetches from AWS
        data1 = get_secret_dict("my-secret")
        assert data1 == {"GEMINI_API_KEY": "g123", "OPENAI_API_KEY": "o123"}
        assert mock_sm.get_secret_value.call_count == 1

        # Second call returns cached dict without calling AWS again
        data2 = get_secret_dict("my-secret")
        assert data2 == data1
        assert mock_sm.get_secret_value.call_count == 1


@pytest.mark.env
@pytest.mark.unit
def test_get_secret_dict_client_error():
    mock_sm = MagicMock()
    from botocore.exceptions import ClientError
    error_resp = {"Error": {"Code": "ResourceNotFoundException", "Message": "Not Found"}}
    mock_sm.get_secret_value.side_effect = ClientError(error_resp, "GetSecretValue")

    with patch("utils.env.aws_secrets._get_secrets_manager_client", return_value=mock_sm):
        with pytest.raises(ClientError):
            get_secret_dict("missing-secret")


@pytest.mark.env
@pytest.mark.unit
def test_get_aws_secret():
    with patch(
        "utils.env.aws_secrets.get_secret_dict", return_value={"KEY_A": "VAL_A"}
    ):
        assert get_aws_secret("KEY_A", "my-secret") == "VAL_A"
        assert get_aws_secret("KEY_B", "my-secret") is None


@pytest.mark.env
@pytest.mark.unit
def test_get_aws_secret_error_returns_none():
    from botocore.exceptions import ClientError
    error_resp = {"Error": {"Code": "AccessDenied", "Message": "Denied"}}
    with patch("utils.env.aws_secrets.get_secret_dict", side_effect=ClientError(error_resp, "GetSecretValue")):
        assert get_aws_secret("KEY_A", "my-secret") is None


@pytest.mark.env
@pytest.mark.unit
def test_get_keys_dict():
    mock_local = MagicMock(return_value="my-secret")
    with patch(
        "utils.env.aws_secrets.get_secret_dict", return_value={aws_secrets.GEMINI_API_KEY_NAME: "g-key", aws_secrets.OPENAI_API_KEY_NAME: "o-key"}
    ):
        keys = get_keys_dict(mock_local)
        assert keys[aws_secrets.GEMINI_API_KEY_NAME] == "g-key"
        assert keys[aws_secrets.OPENAI_API_KEY_NAME] == "o-key"

        # Test global caching of keys dict
        keys2 = get_keys_dict(mock_local)
        assert keys2 is keys
