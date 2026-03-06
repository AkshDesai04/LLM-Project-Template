import os
import json

import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from utils.logger import get_logger

logger = get_logger("EnvOps")

_api_keys_dict = None


def get_local_secret(key_name: str) -> str:
    load_dotenv()
    value = os.getenv(key_name)

    if not value:
        logger.error(f"Environment variable '{key_name}' not set.")
        raise ValueError(f"Environment variable '{key_name}' not set.")
    return value


def get_aws_secret(key_name: str, secret_name: str):
    session = boto3.session.Session()

    # helper to safely get creds
    aws_access = get_local_secret("AWS_ACCESS_KEY_ID")
    aws_secret = get_local_secret("AWS_SECRET_ACCESS_KEY")
    region_name = get_local_secret("AWS_REGION")

    logger.info(f"Retrieving AWS secret: {key_name} from {secret_name} in {region_name}")

    if not aws_access or not aws_secret:
        logger.error("AWS Credentials not found in environment. Check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        raise ValueError("AWS Credentials not found in environment.")

    client = session.client(
        service_name="secretsmanager",
        region_name=region_name,
        aws_access_key_id=aws_access,
        aws_secret_access_key=aws_secret,
    )

    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret_dict = json.loads(response["SecretString"])
        secret_value = secret_dict.get(key_name, None)
        if secret_value:
            logger.info(f"Successfully retrieved AWS secret: {key_name}")
        else:
            logger.warning(f"AWS secret key '{key_name}' not found in secret store '{secret_name}'.")
        return secret_value
    except ClientError as e:
        logger.error(f"Failed to retrieve AWS secret from Secrets Manager: {e}")
        raise e


def get_keys_dict() -> dict:
    """
    Fetches API keys from AWS Secrets Manager and returns them as a dictionary.

    The keys are loaded only once and cached for subsequent calls.
    It looks for GEMINI_API_KEY and OPENAI_API_KEY within the 'llm_api_keys' secret.
    """
    global _api_keys_dict
    if _api_keys_dict is not None:
        return _api_keys_dict

    logger.info("First time call: Loading API keys from AWS Secrets Manager...")
    secret_name = "llm_api_keys"  # Assuming this is the secret name in AWS Secrets Manager

    # Note: This may fetch the same secret from AWS multiple times if not optimized.
    try:
        gemini_key = get_aws_secret("GEMINI_API_KEY", secret_name)
    except Exception as e:
        logger.warning(f"Could not get GEMINI_API_KEY from AWS secret '{secret_name}'. Error: {e}")
        gemini_key = None

    try:
        openai_key = get_aws_secret("OPENAI_API_KEY", secret_name)
    except Exception as e:
        logger.warning(f"Could not get OPENAI_API_KEY from AWS secret '{secret_name}'. Error: {e}")
        openai_key = None

    keys = {
        "GEMINI_API_KEY": gemini_key,
        "OPENAI_API_KEY": openai_key,
    }

    _api_keys_dict = keys

    # Log which keys were found
    found_keys = [key for key, value in keys.items() if value]
    if found_keys:
        logger.info(f"Successfully loaded from AWS: {', '.join(found_keys)}")
    else:
        logger.warning(f"Could not load any API keys from AWS secret '{secret_name}'.")

    return _api_keys_dict
