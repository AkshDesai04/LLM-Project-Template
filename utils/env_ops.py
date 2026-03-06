import os
import json
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from utils.logger import get_logger

logger = get_logger("EnvOps")

GEMINI_API_KEY_NAME = "GEMINI_KEY"
OPENAI_API_KEY_NAME = "OPEN_AI_KEY"

_secrets_manager_client = None
_aws_secrets_cache = {}
_api_keys_dict = None


def _get_secrets_manager_client():
    global _secrets_manager_client
    if _secrets_manager_client is not None:
        return _secrets_manager_client

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


def get_local_secret(key_name: str) -> str:
    load_dotenv()
    value = os.getenv(key_name)
    if not value:
        logger.error(f"Local environment variable '{key_name}' not set.")
        raise ValueError(f"Local environment variable '{key_name}' not set.")
    return value


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
