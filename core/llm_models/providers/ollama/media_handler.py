"""
Ollama media upload and preprocessing utilities.
"""

from typing import Any
from utils.logging import get_logger
from core.llm_models.utils.media_utils import extract_text_from_pdf_bytes, process_video_frames

logger = get_logger("OllamaMediaHandler")


def process_ollama_media(file_bytes: bytes, mime_type: str) -> Any:
    """Processes media bytes for Ollama."""
    try:
        if mime_type == 'application/pdf':
            logger.info("Processing PDF for Ollama (Local Extraction)...")
            return extract_text_from_pdf_bytes(file_bytes)
        elif mime_type.startswith('image/'):
            logger.info(f"Processing {mime_type} for Ollama Vision...")
            return file_bytes
        elif mime_type.startswith('video/'):
            logger.info("Ollama does not natively support video. Extracting frames...")
            return process_video_frames(file_bytes)
        else:
            logger.info(f"Treating {mime_type} as plain text...")
            return file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Ollama media processing failed: {e}")
        raise RuntimeError(f"Failed to process {mime_type} for Ollama: {e}")
