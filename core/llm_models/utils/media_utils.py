import base64
import os
import tempfile
from io import BytesIO
from typing import List

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    PyPDF2 = None

from utils.logging import get_logger

logger = get_logger("MediaUtils")

MAX_VIDEO_FILE_SIZE: int = 512 * 1024 * 1024
DEFAULT_FRAMES_PER_SECOND: int = 1
DEFAULT_VIDEO_FPS_FALLBACK: int = 30
VIDEO_TEMP_SUFFIX: str = ".mp4"
JPEG_ENCODING_EXTENSION: str = ".jpeg"


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Helper to extract text from PDF bytes using PyPDF2."""
    if not pdf_bytes:
        raise ValueError("Provided PDF bytes array is empty.")
    if not PYPDF2_AVAILABLE:
        raise ImportError("PyPDF2 is not installed. Please install 'PyPDF2' to process PDF files.")
    try:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise RuntimeError(f"Failed to extract text from PDF document: {e}") from e


def process_video_frames(video_bytes: bytes, frames_per_second: int = DEFAULT_FRAMES_PER_SECOND) -> List[dict]:
    """Extracts frames from video bytes, encodes them, and returns a list of dictionaries formatted for OpenAI."""
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python (cv2) is not installed. Please install 'opencv-python' to process video files.")
    if len(video_bytes) > MAX_VIDEO_FILE_SIZE:
        max_mb = MAX_VIDEO_FILE_SIZE // (1024 * 1024)
        raise ValueError(f"Video file exceeds the maximum allowed size of {max_mb}MB.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=VIDEO_TEMP_SUFFIX)
    try:
        temp_file.write(video_bytes)
        temp_file.close()
    except Exception as e:
        logger.error(f"Failed to write temporary video file: {e}")
        raise RuntimeError(f"Could not create temporary video file for frame processing: {e}") from e

    base64_frames = []
    video = cv2.VideoCapture(temp_file.name)

    try:
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else DEFAULT_VIDEO_FPS_FALLBACK
        frame_interval = max(1, int(fps / frames_per_second))

        for frame_num in range(0, total_frames, frame_interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            success, frame = video.read()
            if not success:
                continue
            _, buffer = cv2.imencode(JPEG_ENCODING_EXTENSION, frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
    except Exception as e:
        logger.error(f"Video frame extraction failed: {e}")
        raise RuntimeError(f"Failed to extract video frames: {e}") from e
    finally:
        video.release()
        if os.path.exists(temp_file.name):
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
    if not image_bytes:
        raise ValueError("Provided image bytes array is empty.")
    try:
        b64_str = base64.b64encode(image_bytes).decode('utf-8')
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{b64_str}"
            }
        }
    except Exception as e:
        logger.error(f"Failed to encode image to base64 (mime_type={mime_type}): {e}")
        raise RuntimeError(f"Failed to encode image of type '{mime_type}' to base64: {e}") from e
