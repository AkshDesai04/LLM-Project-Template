import os
from typing import Optional

from dotenv import load_dotenv
from utils.logging import get_logger
from utils.env.constants import (
    KEY_LOCATION_NAME,
    KEY_LOCATION_LOCAL,
    KEY_LOCATION_AWS_SM,
    VALID_KEY_LOCATIONS,
    SECRET_NAME_KEY,
    GEMINI_API_KEY_NAME,
    OPENAI_API_KEY_NAME,
    GEMINI_KEY_TYPE_NAME,
    GEMINI_KEY_TYPE_API_KEY,
    GEMINI_KEY_TYPE_SERVICE_ACC_JSON,
    VALID_GEMINI_KEY_TYPES,
    GEMINI_SERVICE_ACCOUNT_FILE_NAME,
    GEMINI_SERVICE_ACCOUNT_JSON_NAME,
    GEMINI_PROJECT_NAME,
    GEMINI_LOCATION_NAME,
    GEMINI_SERVICE_ACCOUNT_SCOPES,
    DEFAULT_GEMINI_LOCATION,
    ROUTED_SECRET_NAMES,
)
from utils.env.aws_secrets import get_secret_dict, get_aws_secret, get_keys_dict as _get_keys_dict
import utils.env.gemini_credentials as gemini_cred

logger = get_logger("EnvOps")


def get_local_secret(key_name: str, raise_error: bool = True) -> Optional[str]:
    """Reads a local environment variable using load_dotenv."""
    load_dotenv()
    value = os.getenv(key_name)
    if not value and raise_error:
        logger.error(f"Local environment variable '{key_name}' not set.")
        raise ValueError(f"Local environment variable '{key_name}' not set.")
    return value


def get_key_location() -> str:
    """Reads KEY_LOCATION from .env."""
    raw = get_local_secret(KEY_LOCATION_NAME, raise_error=False) or KEY_LOCATION_LOCAL
    location = raw.strip().strip("'\"").upper()
    if location not in VALID_KEY_LOCATIONS:
        raise ValueError(
            f"Invalid {KEY_LOCATION_NAME}='{raw}'. "
            f"Expected one of: {', '.join(VALID_KEY_LOCATIONS)}."
        )
    return location


def get_secret(key_name: str, raise_error: bool = True) -> Optional[str]:
    """Resolves a runtime secret according to KEY_LOCATION."""
    if get_key_location() == KEY_LOCATION_LOCAL:
        return get_local_secret(key_name, raise_error=raise_error)

    secret_name = get_local_secret(SECRET_NAME_KEY, raise_error=False)
    if not secret_name:
        message = (
            f"{KEY_LOCATION_NAME}={KEY_LOCATION_AWS_SM} requires {SECRET_NAME_KEY} "
            f"in .env to identify the AWS Secrets Manager bundle."
        )
        logger.error(message)
        if raise_error:
            raise ValueError(message)
        return None

    value = get_aws_secret(key_name, secret_name)
    if not value:
        message = f"Secret '{key_name}' not found in AWS secret '{secret_name}'."
        logger.error(message)
        if raise_error:
            raise ValueError(message)
        return None
    return value


def get_database_url(raise_error: bool = True) -> Optional[str]:
    """Resolves DATABASE_URL through KEY_LOCATION."""
    return get_secret("DATABASE_URL", raise_error=raise_error)


def get_gemini_key_type() -> str:
    return gemini_cred.get_gemini_key_type(get_local_secret)


def load_gemini_service_account_credentials():
    return gemini_cred.load_gemini_service_account_credentials(get_local_secret, get_secret)


def resolve_gemini_project(credentials=None) -> str:
    return gemini_cred.resolve_gemini_project(get_local_secret, credentials)


def resolve_gemini_location() -> str:
    return gemini_cred.resolve_gemini_location(get_local_secret)


def get_keys_dict() -> dict:
    return _get_keys_dict(get_local_secret)
