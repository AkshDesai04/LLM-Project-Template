import base64
import os
import tempfile
from io import BytesIO
from typing import List
import cv2
import PyPDF2

from utils.logger import get_logger

logger = get_logger("MediaUtils")

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Helper to extract text from PDF bytes using PyPDF2."""
    try:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return f"[Error extracting text from PDF: {e}]"

def process_video_frames(video_bytes: bytes, frames_per_second: int = 1) -> List[dict]:
    """
    Extracts frames from video bytes, encodes them, and returns a list of dictionaries formatted for OpenAI.
    """
    MAX_FILE_SIZE = 512 * 1024 * 1024  # 512 MB limit
    if len(video_bytes) > MAX_FILE_SIZE:
        raise ValueError("Video file exceeds the maximum allowed size of 50MB.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_bytes)
    temp_file.close()

    base64_frames = []
    video = cv2.VideoCapture(temp_file.name)
    
    try:
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        # Prevent division by zero if fps is not detected
        fps = fps if fps > 0 else 30
        frame_interval = max(1, int(fps / frames_per_second))
        
        for frame_num in range(0, total_frames, frame_interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            success, frame = video.read()
            if not success:
                continue
            _, buffer = cv2.imencode(".jpeg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
    finally:
        video.release()
        os.unlink(temp_file.name)

    logger.info(f"Extracted {len(base64_frames)} frames from video.")

    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_frame}"
            }
        }
        for b64_frame in base64_frames
    ]

def encode_image_base64(image_bytes: bytes, mime_type: str) -> dict:
    """Encodes image bytes to base64 dict formatted for OpenAI."""
    b64_str = base64.b64encode(image_bytes).decode('utf-8')
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{b64_str}"
        }
    }
