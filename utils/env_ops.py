import os
import json

import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError


def get_local_secret(key_name: str) -> str:
    load_dotenv()
    value = os.getenv(key_name)

    if not value:
        print(f"Warning: Environment variable '{key_name}' not set.")
        return ""
    return value


def get_aws_secret(key_name: str, secret_name: str = "RFP-New", region_name: str = "us-east-1"):
    session = boto3.session.Session()

    # helper to safely get creds
    aws_access = get_local_secret("AWS_ACCESS_KEY_ID")
    aws_secret = get_local_secret("AWS_SECRET_ACCESS_KEY")

    if not aws_access or not aws_secret:
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
        return secret_dict.get(key_name, None)
    except ClientError as e:
        print(f"Failed to retrieve AWS secret: {e}")
        raise e