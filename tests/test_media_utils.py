"""
Unit tests for media utility functions (core.llm_models.utils.media_utils).
"""

from unittest.mock import MagicMock, patch
import pytest

from core.llm_models.utils.media_utils import (
    MAX_VIDEO_FILE_SIZE,
    encode_image_base64,
    extract_text_from_pdf_bytes,
    process_video_frames,
)


@pytest.mark.media
@pytest.mark.unit
def test_extract_text_from_pdf_bytes_empty():
    with pytest.raises(ValueError, match="PDF bytes array is empty"):
        extract_text_from_pdf_bytes(b"")


@pytest.mark.media
@pytest.mark.unit
def test_extract_text_from_pdf_bytes_missing_pypdf2():
    with patch("core.llm_models.utils.media_utils.PYPDF2_AVAILABLE", False):
        with pytest.raises(ImportError, match="PyPDF2 is not installed"):
            extract_text_from_pdf_bytes(b"%PDF-1.4")


@pytest.mark.media
@pytest.mark.unit
def test_extract_text_from_pdf_bytes_success():
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Page 1 Content"
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("PyPDF2.PdfReader", return_value=mock_reader):
        res = extract_text_from_pdf_bytes(b"%PDF-1.4 header")
        assert res == "Page 1 Content"


@pytest.mark.media
@pytest.mark.unit
def test_process_video_frames_missing_cv2():
    with patch("core.llm_models.utils.media_utils.CV2_AVAILABLE", False):
        with pytest.raises(ImportError, match="opencv-python"):
            process_video_frames(b"video_bytes")


@pytest.mark.media
@pytest.mark.unit
def test_process_video_frames_size_exceeded():
    large_bytes = b"0" * (MAX_VIDEO_FILE_SIZE + 1)
    with patch("core.llm_models.utils.media_utils.CV2_AVAILABLE", True):
        with pytest.raises(ValueError, match="exceeds the maximum allowed size"):
            process_video_frames(large_bytes)


@pytest.mark.media
@pytest.mark.unit
def test_process_video_frames_success():
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: 10 if prop == 7 else 30  # total_frames=10, fps=30
    mock_cap.read.return_value = (True, MagicMock())

    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.imencode.return_value = (True, b"jpeg_data")
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_POS_FRAMES = 1

    with patch("core.llm_models.utils.media_utils.CV2_AVAILABLE", True), patch(
        "core.llm_models.utils.media_utils.cv2", mock_cv2
    ):
        frames = process_video_frames(b"dummy_video_bytes", frames_per_second=1)
        assert isinstance(frames, list)
        assert len(frames) > 0
        assert frames[0]["type"] == "image_url"
        assert "data:image/jpeg;base64," in frames[0]["image_url"]["url"]


@pytest.mark.media
@pytest.mark.unit
def test_encode_image_base64():
    with pytest.raises(ValueError, match="image bytes array is empty"):
        encode_image_base64(b"", "image/png")

    res = encode_image_base64(b"png_data", "image/png")
    assert res["type"] == "image_url"
    assert "data:image/png;base64," in res["image_url"]["url"]
