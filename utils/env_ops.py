import os
import json

import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from utils.logger import get_logger

logger = get_logger("EnvOps")


def get_local_secret(key_name: str) -> str:
    load_dotenv()
    value = os.getenv(key_name)

    if not value:
        logger.error(f"Environment variable '{key_name}' not set.")
        raise ValueError(f"Environment variable '{key_name}' not set.")
    return value


def get_aws_secret(key_name: str, secret_name: str = "RFP-New", region_name: str = "us-east-1"):
    logger.info(f"Retrieving AWS secret: {key_name} from {secret_name} in {region_name}")
    session = boto3.session.Session()

    # helper to safely get creds
    aws_access = get_local_secret("AWS_ACCESS_KEY_ID")
    aws_secret = get_local_secret("AWS_SECRET_ACCESS_KEY")

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