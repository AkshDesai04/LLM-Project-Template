import os
import json
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    class ClientError(Exception): pass

from dotenv import load_dotenv
from utils.logger import get_logger

logger = get_logger("EnvOps")

GEMINI_API_KEY_NAME = "GEMINI_KEY"
OPENAI_API_KEY_NAME = "OPEN_AI_KEY"

# Gemini auth selection. Stored in .env so the user can flip between the
# Developer API key and a Vertex AI service account without code changes.
GEMINI_AUTH_MODE_NAME = "GEMINI_AUTH_MODE"
GEMINI_SERVICE_ACCOUNT_FILE_NAME = "GEMINI_SERVICE_ACCOUNT_FILE"
GEMINI_SERVICE_ACCOUNT_JSON_NAME = "GEMINI_SERVICE_ACCOUNT_JSON"
GEMINI_PROJECT_NAME = "GEMINI_PROJECT"
GEMINI_LOCATION_NAME = "GEMINI_LOCATION"

GEMINI_AUTH_MODE_API_KEY = "api_key"
GEMINI_AUTH_MODE_SERVICE_ACCOUNT = "service_account"
VALID_GEMINI_AUTH_MODES = {
    GEMINI_AUTH_MODE_API_KEY,
    GEMINI_AUTH_MODE_SERVICE_ACCOUNT,
}

# Vertex AI cloud-platform scope required by the Generative AI APIs.
GEMINI_SERVICE_ACCOUNT_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
)
DEFAULT_GEMINI_LOCATION = "us-central1"

_secrets_manager_client = None
_aws_secrets_cache = {}
_api_keys_dict = None


def _get_secrets_manager_client():
    global _secrets_manager_client
    if _secrets_manager_client is not None:
        return _secrets_manager_client

    if not BOTO3_AVAILABLE:
        logger.error("boto3 is not installed. AWS Secrets Manager is unavailable.")
        raise ImportError("boto3 is not installed. AWS Secrets Manager is unavailable.")

    load_dotenv()
    region_name = os.getenv("AWS_REGION")
    if not region_name:
        logger.error("AWS_REGION environment variable not set.")
        raise ValueError("AWS_REGION environment variable must be set to use AWS Secrets Manager.")

    logger.info(f"Initializing Boto3 Secrets Manager client for region: {region_name}")
    _secrets_manager_client = boto3.client("secretsmanager", region_name=region_name)
    return _secrets_manager_client


def get_secret_dict(secret_name: str) -> dict:
    if secret_name in _aws_secrets_cache:
        return _aws_secrets_cache[secret_name]

    try:
        client = _get_secrets_manager_client()
        logger.info(f"Fetching secret '{secret_name}' from AWS Secrets Manager...")
        response = client.get_secret_value(SecretId=secret_name)
        secret_dict = json.loads(response["SecretString"])

        _aws_secrets_cache[secret_name] = secret_dict
        logger.info(f"Successfully fetched and cached secret '{secret_name}'.")
        return secret_dict
    except ClientError as e:
        logger.error(f"Failed to retrieve secret '{secret_name}' from AWS: {e}")
        _aws_secrets_cache[secret_name] = {}  # Cache failure to prevent retrying
        raise


def get_aws_secret(key_name: str, secret_name: str) -> str | None:
    try:
        secret_dict = get_secret_dict(secret_name)
        return secret_dict.get(key_name)
    except ClientError:
        return None


def get_local_secret(key_name: str, raise_error: bool = True) -> str | None:
    load_dotenv()
    value = os.getenv(key_name)
    if not value and raise_error:
        logger.error(f"Local environment variable '{key_name}' not set.")
        raise ValueError(f"Local environment variable '{key_name}' not set.")
    return value


def get_gemini_auth_mode() -> str:
    """
    Reads GEMINI_AUTH_MODE from .env.

    Accepted values: 'api_key' (Gemini Developer API) or 'service_account'
    (Vertex AI via a service-account JSON). Defaults to 'api_key' so existing
    setups keep working when the variable is absent.
    """
    raw = get_local_secret(GEMINI_AUTH_MODE_NAME, raise_error=False) or GEMINI_AUTH_MODE_API_KEY
    mode = raw.strip().lower()
    if mode not in VALID_GEMINI_AUTH_MODES:
        raise ValueError(
            f"Invalid {GEMINI_AUTH_MODE_NAME}='{raw}'. "
            f"Expected one of: {', '.join(sorted(VALID_GEMINI_AUTH_MODES))}."
        )
    return mode


def load_gemini_service_account_credentials():
    """
    Builds google.oauth2 service-account credentials for Vertex AI.

    Prefers GEMINI_SERVICE_ACCOUNT_FILE (path on disk). Falls back to
    GEMINI_SERVICE_ACCOUNT_JSON (inline JSON string) for environments that
    cannot mount a file.
    """
    try:
        from google.oauth2 import service_account
    except ImportError as e:
        raise ImportError(
            "google-auth is required for GEMINI_AUTH_MODE=service_account. "
            "Run `pip install google-auth`."
        ) from e

    file_path = get_local_secret(GEMINI_SERVICE_ACCOUNT_FILE_NAME, raise_error=False)
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

    inline_json = get_local_secret(GEMINI_SERVICE_ACCOUNT_JSON_NAME, raise_error=False)
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
        f"{GEMINI_AUTH_MODE_NAME}={GEMINI_AUTH_MODE_SERVICE_ACCOUNT} requires either "
        f"{GEMINI_SERVICE_ACCOUNT_FILE_NAME} (path) or "
        f"{GEMINI_SERVICE_ACCOUNT_JSON_NAME} (inline JSON) in .env."
    )


def resolve_gemini_project(credentials=None) -> str:
    """
    Resolves the GCP project for Vertex AI.

    Order: GEMINI_PROJECT, GOOGLE_CLOUD_PROJECT, then the project_id embedded
    in the service-account credentials when available.
    """
    project = (
        get_local_secret(GEMINI_PROJECT_NAME, raise_error=False)
        or get_local_secret("GOOGLE_CLOUD_PROJECT", raise_error=False)
        or getattr(credentials, "project_id", None)
    )
    if not project:
        raise ValueError(
            f"{GEMINI_AUTH_MODE_NAME}={GEMINI_AUTH_MODE_SERVICE_ACCOUNT} requires a GCP "
            f"project. Set {GEMINI_PROJECT_NAME} (or GOOGLE_CLOUD_PROJECT) in .env, "
            f"or use a service-account JSON that includes project_id."
        )
    return project.strip().strip("'\"")


def resolve_gemini_location() -> str:
    """
    Resolves the Vertex AI location.

    Order: GEMINI_LOCATION, GOOGLE_CLOUD_LOCATION, then us-central1.
    """
    location = (
        get_local_secret(GEMINI_LOCATION_NAME, raise_error=False)
        or get_local_secret("GOOGLE_CLOUD_LOCATION", raise_error=False)
        or DEFAULT_GEMINI_LOCATION
    )
    return location.strip().strip("'\"")


def get_keys_dict() -> dict:
    global _api_keys_dict
    if _api_keys_dict is not None:
        return _api_keys_dict

    try:
        secret_name = get_local_secret("SECRET_NAME")
        logger.info(f"Loading LLM API keys from AWS secret: '{secret_name}'")
        all_keys = get_secret_dict(secret_name)

        keys = {
            GEMINI_API_KEY_NAME: all_keys.get(GEMINI_API_KEY_NAME),
            OPENAI_API_KEY_NAME: all_keys.get(OPENAI_API_KEY_NAME),
        }
    except (ValueError, ClientError) as e:
        logger.error(f"Failed to load API keys from AWS. Ensure SECRET_NAME is set and secret is accessible. Error: {e}")
        keys = {GEMINI_API_KEY_NAME: None, OPENAI_API_KEY_NAME: None}

    _api_keys_dict = keys

    found = [k for k, v in keys.items() if v]
    if found:
        logger.info(f"Successfully configured API keys for: {', '.join(found)}")
    else:
        logger.warning("No LLM API keys were successfully configured from AWS.")

    return _api_keys_dict
