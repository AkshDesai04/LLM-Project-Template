import os


def read_file(prompt_path: str) -> str:
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"File not found: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()