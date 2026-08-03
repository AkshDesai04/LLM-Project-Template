"""
Anthropic media upload and preprocessing utilities.
"""

from typing import Any, List
from utils.logging import get_logger
from core.llm_models.utils.media_utils import (
    extract_text_from_pdf_bytes,
    process_video_frames,
    encode_image_base64
)

logger = get_logger("AnthropicMediaHandler")


def process_anthropic_media(file_bytes: bytes, mime_type: str) -> Any:
    """Processes media bytes into Anthropic compatible content blocks."""
    try:
        if mime_type == 'application/pdf':
            logger.info("Processing PDF for Anthropic (Local Extraction)...")
            return extract_text_from_pdf_bytes(file_bytes)
        elif mime_type.startswith('image/'):
            logger.info(f"Processing {mime_type} for Anthropic Vision...")
            return encode_image_base64(file_bytes, mime_type)
        elif mime_type.startswith('video/'):
            logger.info(f"Processing {mime_type} for Anthropic by extracting frames...")
            return process_video_frames(file_bytes)
        else:
            logger.info(f"Treating {mime_type} as plain text...")
            return file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Anthropic media processing failed: {e}")
        raise RuntimeError(f"Failed to process {mime_type} for Anthropic: {e}")


def to_content_blocks(prompt: str, files: List[Any]) -> List[dict]:
    """Converts the OpenAI-shaped payloads into Anthropic content blocks."""
    image_blocks = []
    text_attachments = []

    for item in files:
        if isinstance(item, str):
            text_attachments.append(item)
            continue
        if not isinstance(item, dict) or item.get("type") != "image_url":
            continue
        url = item.get("image_url", {}).get("url", "")
        if not url.startswith("data:"):
            continue
        header, _, data = url.partition(",")
        media_type = header.split(";")[0][len("data:"):]
        image_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        })

    full_prompt = prompt
    if text_attachments:
        full_prompt += "\n\n" + "\n\n".join(f"[Attached Content]:\n{t}" for t in text_attachments)

    return [{"type": "text", "text": full_prompt}] + image_blocks
