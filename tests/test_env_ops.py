"""
Unit tests for environment operations (utils.env.env_ops).
"""

from unittest.mock import MagicMock, patch
import pytest

from utils.env.env_ops import (
    get_database_url,
    get_gemini_key_type,
    get_key_location,
    get_keys_dict,
    get_local_secret,
    get_secret,
    load_gemini_service_account_credentials,
    resolve_gemini_location,
    resolve_gemini_project,
)


@pytest.mark.env
@pytest.mark.unit
def test_get_local_secret_success():
    with patch("os.getenv", return_value="my_secret_val"):
        val = get_local_secret("TEST_KEY", raise_error=True)
        assert val == "my_secret_val"


@pytest.mark.env
@pytest.mark.unit
def test_get_local_secret_missing_raise():
    with patch("os.getenv", return_value=None):
        with pytest.raises(ValueError, match="TEST_KEY"):
            get_local_secret("TEST_KEY", raise_error=True)


@pytest.mark.env
@pytest.mark.unit
def test_get_local_secret_missing_no_raise():
    with patch("os.getenv", return_value=None):
        val = get_local_secret("TEST_KEY", raise_error=False)
        assert val is None


@pytest.mark.env
@pytest.mark.unit
def test_get_key_location_default():
    with patch("utils.env.env_ops.get_local_secret", return_value=None):
        assert get_key_location() == "LOCAL"


@pytest.mark.env
@pytest.mark.unit
def test_get_key_location_custom():
    with patch("utils.env.env_ops.get_local_secret", return_value="AWS_SM"):
        assert get_key_location() == "AWS_SM"


@pytest.mark.env
@pytest.mark.unit
def test_get_key_location_invalid():
    with patch("utils.env.env_ops.get_local_secret", return_value="INVALID_LOC"):
        with pytest.raises(ValueError, match="Invalid KEY_LOCATION"):
            get_key_location()


@pytest.mark.env
@pytest.mark.unit
def test_get_secret_local():
    with patch("utils.env.env_ops.get_key_location", return_value="LOCAL"), patch(
        "utils.env.env_ops.get_local_secret", return_value="local_val"
    ) as mock_local:
        res = get_secret("MY_KEY")
        assert res == "local_val"
        mock_local.assert_called_with("MY_KEY", raise_error=True)


@pytest.mark.env
@pytest.mark.unit
def test_get_secret_aws_sm_missing_secret_name():
    with patch("utils.env.env_ops.get_key_location", return_value="AWS_SM"), patch(
        "utils.env.env_ops.get_local_secret", return_value=None
    ):
        with pytest.raises(ValueError, match="requires SECRET_NAME"):
            get_secret("MY_KEY", raise_error=True)


@pytest.mark.env
@pytest.mark.unit
def test_get_secret_aws_sm_success():
    with patch("utils.env.env_ops.get_key_location", return_value="AWS_SM"), patch(
        "utils.env.env_ops.get_local_secret", return_value="my-aws-secret-bundle"
    ), patch("utils.env.env_ops.get_aws_secret", return_value="aws_val") as mock_aws:
        res = get_secret("MY_KEY")
        assert res == "aws_val"
        mock_aws.assert_called_with("MY_KEY", "my-aws-secret-bundle")


@pytest.mark.env
@pytest.mark.unit
def test_get_secret_aws_sm_not_found():
    with patch("utils.env.env_ops.get_key_location", return_value="AWS_SM"), patch(
        "utils.env.env_ops.get_local_secret", return_value="my-aws-secret-bundle"
    ), patch("utils.env.env_ops.get_aws_secret", return_value=None):
        with pytest.raises(ValueError, match="Secret 'MY_KEY' not found"):
            get_secret("MY_KEY", raise_error=True)


@pytest.mark.env
@pytest.mark.unit
def test_get_database_url():
    with patch("utils.env.env_ops.get_secret", return_value="sqlite:///:memory:") as mock_sec:
        res = get_database_url()
        assert res == "sqlite:///:memory:"
        mock_sec.assert_called_with("DATABASE_URL", raise_error=True)


@pytest.mark.env
@pytest.mark.unit
def test_gemini_wrappers():
    with patch("utils.env.gemini_credentials.get_gemini_key_type", return_value="GEMINI_KEY"):
        assert get_gemini_key_type() == "GEMINI_KEY"

    with patch("utils.env.gemini_credentials.resolve_gemini_location", return_value="us-central1"):
        assert resolve_gemini_location() == "us-central1"

    with patch("utils.env.gemini_credentials.resolve_gemini_project", return_value="proj-123"):
        assert resolve_gemini_project() == "proj-123"

    mock_creds = MagicMock()
    with patch("utils.env.gemini_credentials.load_gemini_service_account_credentials", return_value=mock_creds):
        assert load_gemini_service_account_credentials() == mock_creds


@pytest.mark.env
@pytest.mark.unit
def test_get_keys_dict_wrapper():
    with patch("utils.env.env_ops._get_keys_dict", return_value={"KEY": "VAL"}) as mock_keys:
        res = get_keys_dict()
        assert res == {"KEY": "VAL"}
