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

# Where runtime secrets are fetched from. LOCAL reads .env, AWS_SM reads the
# bundle named by SECRET_NAME out of AWS Secrets Manager.
KEY_LOCATION_NAME = "KEY_LOCATION"
KEY_LOCATION_LOCAL = "LOCAL"
KEY_LOCATION_AWS_SM = "AWS_SM"
VALID_KEY_LOCATIONS = (KEY_LOCATION_LOCAL, KEY_LOCATION_AWS_SM)

SECRET_NAME_KEY = "SECRET_NAME"

# Gemini credential selection: the Developer API key or a Vertex AI
# service-account JSON.
GEMINI_KEY_TYPE_NAME = "GEMINI_KEY_TYPE"
GEMINI_KEY_TYPE_API_KEY = "GEMINI_KEY"
GEMINI_KEY_TYPE_SERVICE_ACC_JSON = "SERVICE_ACC_JSON"
VALID_GEMINI_KEY_TYPES = (GEMINI_KEY_TYPE_API_KEY, GEMINI_KEY_TYPE_SERVICE_ACC_JSON)

GEMINI_SERVICE_ACCOUNT_FILE_NAME = "GEMINI_SERVICE_ACCOUNT_FILE"
GEMINI_SERVICE_ACCOUNT_JSON_NAME = "GEMINI_SERVICE_ACCOUNT_JSON"
GEMINI_PROJECT_NAME = "GEMINI_PROJECT"
GEMINI_LOCATION_NAME = "GEMINI_LOCATION"

# Vertex AI cloud-platform scope required by the Generative AI APIs.
GEMINI_SERVICE_ACCOUNT_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
)
DEFAULT_GEMINI_LOCATION = "us-central1"

# Credentials that follow KEY_LOCATION. Everything else (mode switches,
# project ids, regions) is deployment config and is always read from .env.
ROUTED_SECRET_NAMES = frozenset({
    "DATABASE_URL",
    GEMINI_API_KEY_NAME,
    GEMINI_SERVICE_ACCOUNT_JSON_NAME,
    OPENAI_API_KEY_NAME,
    "ANTHROPIC_KEY",
    "PERPLEXITY_KEY",
    "OLLAMA_URL",
    "OLLAMA_KEY",
    "VLLM_URL",
    "VLLM_KEY",
})

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

    region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region_name:
        load_dotenv()
        region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

    if not region_name:
        logger.error("AWS_REGION or AWS_DEFAULT_REGION environment variable not set.")
        raise ValueError("AWS_REGION or AWS_DEFAULT_REGION environment variable must be set to use AWS Secrets Manager.")

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


def get_key_location() -> str:
    """
    Reads KEY_LOCATION from .env.

    LOCAL resolves secrets through get_local_secret; AWS_SM resolves them
    through get_aws_secret against the bundle named by SECRET_NAME. Defaults
    to LOCAL so existing setups keep working.
    """
    raw = get_local_secret(KEY_LOCATION_NAME, raise_error=False) or KEY_LOCATION_LOCAL
    location = raw.strip().strip("'\"").upper()
    if location not in VALID_KEY_LOCATIONS:
        raise ValueError(
            f"Invalid {KEY_LOCATION_NAME}='{raw}'. "
            f"Expected one of: {', '.join(VALID_KEY_LOCATIONS)}."
        )
    return location


def get_secret(key_name: str, raise_error: bool = True) -> str | None:
    """
    Resolves a runtime secret according to KEY_LOCATION.

    This is the entry point providers should use for API keys, service URLs
    and the database URL, so that switching KEY_LOCATION moves every lookup
    at once. SECRET_NAME itself is always read locally, since it is a pointer
    to the bundle rather than a secret.
    """
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


def get_database_url(raise_error: bool = True) -> str | None:
    """Resolves DATABASE_URL through KEY_LOCATION."""
    return get_secret("DATABASE_URL", raise_error=raise_error)


def get_gemini_key_type() -> str:
    """
    Reads GEMINI_KEY_TYPE from .env.

    GEMINI_KEY uses the Gemini Developer API key; SERVICE_ACC_JSON uses a
    Vertex AI service-account JSON. Defaults to GEMINI_KEY.
    """
    raw = get_local_secret(GEMINI_KEY_TYPE_NAME, raise_error=False) or GEMINI_KEY_TYPE_API_KEY
    key_type = raw.strip().strip("'\"").upper()
    if key_type not in VALID_GEMINI_KEY_TYPES:
        raise ValueError(
            f"Invalid {GEMINI_KEY_TYPE_NAME}='{raw}'. "
            f"Expected one of: {', '.join(VALID_GEMINI_KEY_TYPES)}."
        )
    return key_type


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
            f"google-auth is required for {GEMINI_KEY_TYPE_NAME}="
            f"{GEMINI_KEY_TYPE_SERVICE_ACC_JSON}. Run `pip install google-auth`."
        ) from e

    # The path is deployment config, so it is always read locally; the JSON
    # body itself is a secret and follows KEY_LOCATION.
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

    inline_json = get_secret(GEMINI_SERVICE_ACCOUNT_JSON_NAME, raise_error=False)
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
            f"{GEMINI_KEY_TYPE_NAME}={GEMINI_KEY_TYPE_SERVICE_ACC_JSON} requires a GCP "
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
        secret_name = get_local_secret(SECRET_NAME_KEY)
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
