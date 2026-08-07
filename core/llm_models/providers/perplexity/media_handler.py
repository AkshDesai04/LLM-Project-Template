"""
Perplexity media processing utilities.
"""

from typing import Any
from utils.logging import get_logger

logger = get_logger("PerplexityMediaHandler")


def process_perplexity_media(file_bytes: bytes, mime_type: str) -> Any:
    """Processes media bytes for Perplexity."""
    logger.info(f"Perplexity: processing {mime_type} media.")
    if mime_type == 'text/plain':
        return file_bytes.decode('utf-8', errors='ignore')
    return f"[Media of type {mime_type} attached]"
