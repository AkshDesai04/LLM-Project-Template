import os
import csv
from typing import List, Dict


def read_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def get_file(file_path: str) -> bytes:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        return f.read()


def read_prompt(prompt_tite: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(base_dir, "core", "prompts", f"{prompt_tite}.txt")
    return read_file(prompt_path)


def read_csv(file_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)