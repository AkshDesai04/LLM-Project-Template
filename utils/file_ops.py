import os
import csv
from typing import List, Dict
from utils.logger import get_logger

logger = get_logger("FileOps")


def read_file(file_path: str) -> str:
    logger.info(f"Reading text file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def get_file(file_path: str) -> bytes:
    logger.info(f"Reading binary file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        return f.read()


def get_prompt_paths(base_dir: str = "core/prompts") -> dict:
    """Scan prompt folders and return paths for system and user prompts."""
    logger.info(f"Scanning prompt paths in directory: {base_dir}")
    prompt_paths = {}
    
    # Calculate absolute path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_base_dir = os.path.join(project_root, base_dir)

    if not os.path.exists(full_base_dir):
        logger.error(f"Base directory '{full_base_dir}' does not exist.")
        return prompt_paths

    for folder in os.listdir(full_base_dir):
        folder_path = os.path.join(full_base_dir, folder)
        if os.path.isdir(folder_path):
            sys_file = os.path.join(folder_path, "sys.txt")
            user_file = os.path.join(folder_path, "user.txt")
            if os.path.exists(sys_file) and os.path.exists(user_file):
                prompt_paths[folder] = (sys_file, user_file)

    logger.info(f"Found {len(prompt_paths)} prompt folders.")
    return prompt_paths


def read_prompt(prompt_tite: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(base_dir, "core", "prompts", f"{prompt_tite}.txt")
    return read_file(prompt_path)


def read_csv(file_path: str) -> List[Dict[str, str]]:
    logger.info(f"Reading CSV file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"CSV file not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        logger.info(f"Successfully read {len(data)} rows from CSV.")
        return data