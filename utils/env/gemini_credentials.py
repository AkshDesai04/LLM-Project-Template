"""
Google / Vertex AI credential resolution utilities.
"""

import os
import json
from utils.logging import get_logger
from utils.env.constants import (
    GEMINI_KEY_TYPE_NAME,
    GEMINI_KEY_TYPE_API_KEY,
    GEMINI_KEY_TYPE_SERVICE_ACC_JSON,
    VALID_GEMINI_KEY_TYPES,
    GEMINI_SERVICE_ACCOUNT_FILE_NAME,
    GEMINI_SERVICE_ACCOUNT_JSON_NAME,
    GEMINI_SERVICE_ACCOUNT_SCOPES,
    GEMINI_PROJECT_NAME,
    GEMINI_LOCATION_NAME,
    DEFAULT_GEMINI_LOCATION,
)

logger = get_logger("EnvOps.Gemini")


def get_gemini_key_type(get_local_secret_fn) -> str:
    """Reads GEMINI_KEY_TYPE from .env."""
    raw = get_local_secret_fn(GEMINI_KEY_TYPE_NAME, raise_error=False) or GEMINI_KEY_TYPE_API_KEY
    key_type = raw.strip().strip("'\"").upper()
    if key_type not in VALID_GEMINI_KEY_TYPES:
        raise ValueError(
            f"Invalid {GEMINI_KEY_TYPE_NAME}='{raw}'. "
            f"Expected one of: {', '.join(VALID_GEMINI_KEY_TYPES)}."
        )
    return key_type


def load_gemini_service_account_credentials(get_local_secret_fn, get_secret_fn):
    """Builds google.oauth2 service-account credentials for Vertex AI."""
    try:
        from google.oauth2 import service_account
    except ImportError as e:
        raise ImportError(
            f"google-auth is required for {GEMINI_KEY_TYPE_NAME}="
            f"{GEMINI_KEY_TYPE_SERVICE_ACC_JSON}. Run `pip install google-auth`."
        ) from e

    file_path = get_local_secret_fn(GEMINI_SERVICE_ACCOUNT_FILE_NAME, raise_error=False)
    if file_path:
        resolved = os.path.expanduser(file_path.strip().strip("'\""))
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"{GEMINI_SERVICE_ACCOUNT_FILE_NAME} points to a missing file: {resolved}"
            )
        logger.info(f"Loading Gemini service account credentials from file: {resolved}")
        return service_account.Credentials.from_service_account_file(
            resolved,
            scopes=list(GEMINI_SERVICE_ACCOUNT_SCOPES),
        )

    inline_json = get_secret_fn(GEMINI_SERVICE_ACCOUNT_JSON_NAME, raise_error=False)
    if inline_json:
        try:
            info = json.loads(inline_json)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{GEMINI_SERVICE_ACCOUNT_JSON_NAME} is not valid JSON: {e}"
            ) from e
        logger.info("Loading Gemini service account credentials from inline JSON.")
        return service_account.Credentials.from_service_account_info(
            info,
            scopes=list(GEMINI_SERVICE_ACCOUNT_SCOPES),
        )

    raise ValueError(
        f"{GEMINI_KEY_TYPE_NAME}={GEMINI_KEY_TYPE_SERVICE_ACC_JSON} requires either "
        f"{GEMINI_SERVICE_ACCOUNT_FILE_NAME} (path) or "
        f"{GEMINI_SERVICE_ACCOUNT_JSON_NAME} (inline JSON)."
    )


def resolve_gemini_project(get_local_secret_fn, credentials=None) -> str:
    """Resolves the GCP project for Vertex AI."""
    project = (
        get_local_secret_fn(GEMINI_PROJECT_NAME, raise_error=False)
        or get_local_secret_fn("GOOGLE_CLOUD_PROJECT", raise_error=False)
        or getattr(credentials, "project_id", None)
    )
    if not project:
        raise ValueError(
            f"{GEMINI_KEY_TYPE_NAME}={GEMINI_KEY_TYPE_SERVICE_ACC_JSON} requires a GCP "
            f"project. Set {GEMINI_PROJECT_NAME} (or GOOGLE_CLOUD_PROJECT) in .env, "
            f"or use a service-account JSON that includes project_id."
        )
    return project.strip().strip("'\"")


def resolve_gemini_location(get_local_secret_fn) -> str:
    """Resolves the Vertex AI location."""
    location = (
        get_local_secret_fn(GEMINI_LOCATION_NAME, raise_error=False)
        or get_local_secret_fn("GOOGLE_CLOUD_LOCATION", raise_error=False)
        or DEFAULT_GEMINI_LOCATION
    )
    return location.strip().strip("'\"")
