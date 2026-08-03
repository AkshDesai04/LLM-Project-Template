"""
Environment configuration and secrets resolution module.
"""

from utils.env.constants import (
    ROUTED_SECRET_NAMES,
    KEY_LOCATION_NAME,
    KEY_LOCATION_LOCAL,
    KEY_LOCATION_AWS_SM,
    GEMINI_API_KEY_NAME,
    OPENAI_API_KEY_NAME,
    GEMINI_KEY_TYPE_API_KEY,
    GEMINI_KEY_TYPE_SERVICE_ACC_JSON,
)
from utils.env.aws_secrets import get_aws_secret
from utils.env.env_ops import (
    get_secret,
    get_local_secret,
    get_database_url,
    get_key_location,
    get_gemini_key_type,
    load_gemini_service_account_credentials,
    resolve_gemini_project,
    resolve_gemini_location,
    get_keys_dict,
)

__all__ = [
    "get_secret",
    "get_local_secret",
    "get_aws_secret",
    "get_database_url",
    "get_key_location",
    "get_gemini_key_type",
    "load_gemini_service_account_credentials",
    "resolve_gemini_project",
    "resolve_gemini_location",
    "get_keys_dict",
    "ROUTED_SECRET_NAMES",
    "KEY_LOCATION_NAME",
    "KEY_LOCATION_LOCAL",
    "KEY_LOCATION_AWS_SM",
    "GEMINI_API_KEY_NAME",
    "OPENAI_API_KEY_NAME",
    "GEMINI_KEY_TYPE_API_KEY",
    "GEMINI_KEY_TYPE_SERVICE_ACC_JSON",
]
