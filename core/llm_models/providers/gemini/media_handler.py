"""
Gemini media upload and preprocessing utilities.
"""

import time
from io import BytesIO
from typing import Any
from google.genai import types
from utils.logging import get_logger

logger = get_logger("GeminiMediaHandler")
FILE_PROCESSING_POLL_INTERVAL: float = 2.0


def upload_gemini_media(client: Any, uses_vertex: bool, file_bytes: bytes, mime_type: str) -> Any:
    """Uploads or inlines media for Gemini / Vertex AI client."""
    try:
        if uses_vertex:
            logger.info(f"Inlining {mime_type} as a Part for Vertex AI (Files API is unavailable under service-account auth).")
            return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

        logger.info(f"Uploading {mime_type} to Gemini...")
        file_obj = BytesIO(file_bytes)
        file_obj.seek(0)

        uploaded_file = client.files.upload(
            file=file_obj,
            config=types.UploadFileConfig(mime_type=mime_type)
        )

        while uploaded_file.state.name == "PROCESSING":
            logger.info(f"File {uploaded_file.name} is still processing...")
            time.sleep(FILE_PROCESSING_POLL_INTERVAL)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"File {uploaded_file.name} failed to process.")

        return uploaded_file
    except Exception as e:
        logger.error(f"Gemini upload failed: {e}")
        raise RuntimeError(f"Failed to upload {mime_type} to Gemini: {e}")
