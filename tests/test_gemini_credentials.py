"""
Unit tests for Gemini & Vertex AI credentials resolution (utils.env.gemini_credentials).
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from utils.env.gemini_credentials import (
    get_gemini_key_type,
    load_gemini_service_account_credentials,
    resolve_gemini_location,
    resolve_gemini_project,
)


@pytest.mark.env
@pytest.mark.unit
def test_get_gemini_key_type_valid():
    mock_local = MagicMock(return_value="service_acc_json")
    assert get_gemini_key_type(mock_local) == "SERVICE_ACC_JSON"

    mock_local_default = MagicMock(return_value=None)
    assert get_gemini_key_type(mock_local_default) == "GEMINI_KEY"


@pytest.mark.env
@pytest.mark.unit
def test_get_gemini_key_type_invalid():
    mock_local = MagicMock(return_value="INVALID_TYPE")
    with pytest.raises(ValueError, match="Invalid GEMINI_KEY_TYPE"):
        get_gemini_key_type(mock_local)


@pytest.mark.env
@pytest.mark.unit
def test_load_gemini_service_account_from_file(tmp_path):
    json_file = tmp_path / "sa.json"
    json_file.write_text(json.dumps({"type": "service_account", "project_id": "p123"}))

    mock_local = MagicMock(side_effect=lambda k, **kw: str(json_file) if k == "GEMINI_SERVICE_ACCOUNT_FILE" else None)
    mock_sec = MagicMock(return_value=None)

    mock_oauth_creds = MagicMock()
    with patch("google.oauth2.service_account.Credentials.from_service_account_file", return_value=mock_oauth_creds) as mock_from_file:
        creds = load_gemini_service_account_credentials(mock_local, mock_sec)
        assert creds == mock_oauth_creds
        mock_from_file.assert_called_once()


@pytest.mark.env
@pytest.mark.unit
def test_load_gemini_service_account_missing_file():
    mock_local = MagicMock(side_effect=lambda k, **kw: "/non/existent/file.json" if k == "GEMINI_SERVICE_ACCOUNT_FILE" else None)
    mock_sec = MagicMock(return_value=None)

    with pytest.raises(FileNotFoundError, match="missing file"):
        load_gemini_service_account_credentials(mock_local, mock_sec)


@pytest.mark.env
@pytest.mark.unit
def test_load_gemini_service_account_from_inline_json():
    mock_local = MagicMock(return_value=None)
    mock_sec = MagicMock(side_effect=lambda k, **kw: json.dumps({"type": "service_account"}) if k == "GEMINI_SERVICE_ACCOUNT_JSON" else None)

    mock_oauth_creds = MagicMock()
    with patch("google.oauth2.service_account.Credentials.from_service_account_info", return_value=mock_oauth_creds) as mock_from_info:
        creds = load_gemini_service_account_credentials(mock_local, mock_sec)
        assert creds == mock_oauth_creds
        mock_from_info.assert_called_once()


@pytest.mark.env
@pytest.mark.unit
def test_load_gemini_service_account_invalid_inline_json():
    mock_local = MagicMock(return_value=None)
    mock_sec = MagicMock(side_effect=lambda k, **kw: "{invalid_json" if k == "GEMINI_SERVICE_ACCOUNT_JSON" else None)

    with pytest.raises(ValueError, match="is not valid JSON"):
        load_gemini_service_account_credentials(mock_local, mock_sec)


@pytest.mark.env
@pytest.mark.unit
def test_load_gemini_service_account_missing_both():
    mock_local = MagicMock(return_value=None)
    mock_sec = MagicMock(return_value=None)

    with pytest.raises(ValueError, match="requires either GEMINI_SERVICE_ACCOUNT_FILE"):
        load_gemini_service_account_credentials(mock_local, mock_sec)


@pytest.mark.env
@pytest.mark.unit
def test_resolve_gemini_project():
    # 1. From GEMINI_PROJECT_NAME
    mock_local = MagicMock(side_effect=lambda k, **kw: "my-project" if k == "GEMINI_PROJECT" else None)
    assert resolve_gemini_project(mock_local) == "my-project"

    # 2. From GOOGLE_CLOUD_PROJECT
    mock_local_gcp = MagicMock(side_effect=lambda k, **kw: "gcp-project" if k == "GOOGLE_CLOUD_PROJECT" else None)
    assert resolve_gemini_project(mock_local_gcp) == "gcp-project"

    # 3. From credentials.project_id
    mock_local_none = MagicMock(return_value=None)
    mock_creds = MagicMock()
    mock_creds.project_id = "cred-project"
    assert resolve_gemini_project(mock_local_none, credentials=mock_creds) == "cred-project"

    # 4. Missing project exception
    with pytest.raises(ValueError, match="requires a GCP project"):
        resolve_gemini_project(mock_local_none, credentials=None)


@pytest.mark.env
@pytest.mark.unit
def test_resolve_gemini_location():
    mock_local = MagicMock(side_effect=lambda k, **kw: "us-west1" if k == "GEMINI_LOCATION" else None)
    assert resolve_gemini_location(mock_local) == "us-west1"

    mock_local_default = MagicMock(return_value=None)
    assert resolve_gemini_location(mock_local_default) == "us-central1"
