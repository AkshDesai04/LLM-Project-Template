"""
vLLM media upload and preprocessing utilities.
"""

from typing import Any
from utils.logging import get_logger
from core.llm_models.utils.media_utils import (
    extract_text_from_pdf_bytes,
    process_video_frames,
    encode_image_base64
)

logger = get_logger("VLLMMediaHandler")


def process_vllm_media(file_bytes: bytes, mime_type: str) -> Any:
    """Processes media bytes into vLLM compatible input structures."""
    try:
        if mime_type == 'application/pdf':
            logger.info("Processing PDF for vLLM (Local Extraction)...")
            return extract_text_from_pdf_bytes(file_bytes)
        elif mime_type.startswith('image/'):
            logger.info(f"Processing {mime_type} for a vLLM vision model...")
            return encode_image_base64(file_bytes, mime_type)
        elif mime_type.startswith('video/'):
            logger.info(f"Processing {mime_type} for vLLM by extracting frames...")
            return process_video_frames(file_bytes)
        else:
            logger.info(f"Treating {mime_type} as plain text...")
            return file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"vLLM media processing failed: {e}")
        raise RuntimeError(f"Failed to process {mime_type} for vLLM: {e}")
