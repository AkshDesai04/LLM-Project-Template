"""
OpenAI media upload and preprocessing utilities.
"""

from typing import Any
from utils.logging import get_logger
from core.llm_models.utils.media_utils import (
    extract_text_from_pdf_bytes,
    process_video_frames,
    encode_image_base64
)

logger = get_logger("OpenAIMediaHandler")


def process_openai_media(file_bytes: bytes, mime_type: str) -> Any:
    """Processes media bytes into OpenAI compatible input blocks or base64 structures."""
    try:
        if mime_type == 'application/pdf':
            logger.info("Processing PDF for OpenAI (Local Extraction)...")
            return extract_text_from_pdf_bytes(file_bytes)
        elif mime_type.startswith('image/'):
            logger.info(f"Processing {mime_type} for OpenAI Vision...")
            return encode_image_base64(file_bytes, mime_type)
        elif mime_type.startswith('video/'):
            logger.info(f"Processing {mime_type} for OpenAI by extracting frames...")
            return process_video_frames(file_bytes)
        else:
            logger.info(f"Treating {mime_type} as plain text...")
            return file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"OpenAI media upload failed: {e}")
        raise RuntimeError(f"Failed to process {mime_type} for OpenAI: {e}")
