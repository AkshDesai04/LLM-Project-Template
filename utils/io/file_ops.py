import os
import csv
from typing import List, Dict
from utils.logging import get_logger

logger = get_logger("FileOps")

# Top-level Constants
DEFAULT_ENCODING: str = "utf-8"
PROMPTS_DIR_NAME: str = os.path.join("core", "prompts")
PROMPT_FILE_EXTENSION: str = ".txt"


def read_file(file_path: str) -> str:
    logger.info(f"Reading text file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Text file not found at path: {file_path}")

    try:
        with open(file_path, 'r', encoding=DEFAULT_ENCODING) as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        raise IOError(f"Could not read text file '{file_path}': {e}") from e


def get_file(file_path: str) -> bytes:
    logger.info(f"Reading binary file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Binary file not found at path: {file_path}")

    try:
        with open(file_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading binary file {file_path}: {e}")
        raise IOError(f"Could not read binary file '{file_path}': {e}") from e


def read_prompt(prompt_title: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prompt_path = os.path.join(base_dir, PROMPTS_DIR_NAME, f"{prompt_title}{PROMPT_FILE_EXTENSION}")
    return read_file(prompt_path)


def read_csv(file_path: str) -> List[Dict[str, str]]:
    logger.info(f"Reading CSV file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"CSV file not found: {file_path}")
        raise FileNotFoundError(f"CSV file not found at path: {file_path}")

    try:
        with open(file_path, mode='r', encoding=DEFAULT_ENCODING) as f:
            reader = csv.DictReader(f)
            data = list(reader)
            logger.info(f"Successfully read {len(data)} rows from CSV.")
            return data
    except Exception as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}")
        raise IOError(f"Could not read CSV file '{file_path}': {e}") from e
