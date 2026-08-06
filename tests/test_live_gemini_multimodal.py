"""
Live integration tests for Gemini multimodal file uploads (Image, Audio, Video, PDF) and combinations.
"""

import io
import wave
import pytest

from core.llm_models.router import ModelRouter
from core.modules.base import Base as BaseModule

# Synthetic minimal valid binary files
PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS"
    b"\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00"
    b"\x00IEND\xaeB`\x82"
)

JPEG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07"
    b"\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e"
    b"\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01"
    b"\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf"
    b"\x00\x7f\xff\xd9"
)

PDF_BYTES = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n00000000108 00000 n\n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n177\n%%EOF\n"
)


def create_wav_bytes() -> bytes:
    """Generates 0.5 sec of silence audio WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b'\x00\x00' * 4000)
    return buf.getvalue()


WAV_BYTES = create_wav_bytes()


def upload_and_generate(target_model: str, files_info: list, prompt: str) -> str:
    """Helper to upload files and generate Gemini response."""
    module = BaseModule(model=target_model, prompt=prompt)
    router = ModelRouter(module)

    uploaded_items = []
    for file_bytes, mime_type in files_info:
        uploaded_items.append(router.upload_media(file_bytes, mime_type))

    uploaded_arg = uploaded_items if len(uploaded_items) > 1 else uploaded_items[0]
    res = router.model_response(module, uploaded_file=uploaded_arg)
    return str(res)


@pytest.mark.live
@pytest.mark.gemini
def test_gemini_single_file_uploads():
    res_png = upload_and_generate("gemini/gemini-3.6-flash", [(PNG_BYTES, "image/png")], "Describe this image.")
    assert res_png is not None

    res_jpg = upload_and_generate("gemini/gemini-3.5-flash", [(JPEG_BYTES, "image/jpeg")], "What is in JPEG?")
    assert res_jpg is not None

    res_wav = upload_and_generate("gemini/gemini-2.5-flash", [(WAV_BYTES, "audio/wav")], "Transcribe or describe audio.")
    assert res_wav is not None

    res_pdf = upload_and_generate("gemini/gemini-2.5-pro", [(PDF_BYTES, "application/pdf")], "Summarize document.")
    assert res_pdf is not None


@pytest.mark.live
@pytest.mark.gemini
def test_gemini_dual_media_combos():
    res_img_aud = upload_and_generate(
        "gemini/gemini-3.6-flash",
        [(PNG_BYTES, "image/png"), (WAV_BYTES, "audio/wav")],
        "Analyze image and audio together."
    )
    assert res_img_aud is not None

    res_img_pdf = upload_and_generate(
        "gemini/gemini-3.5-flash",
        [(JPEG_BYTES, "image/jpeg"), (PDF_BYTES, "application/pdf")],
        "Compare image and PDF content."
    )
    assert res_img_pdf is not None


@pytest.mark.live
@pytest.mark.gemini
def test_gemini_quad_multi_media_combo():
    res_quad = upload_and_generate(
        "gemini/gemini-3.6-flash",
        [
            (PNG_BYTES, "image/png"),
            (WAV_BYTES, "audio/wav"),
            (PDF_BYTES, "application/pdf"),
        ],
        "Synthesize insights from image, audio, and PDF."
    )
    assert res_quad is not None
