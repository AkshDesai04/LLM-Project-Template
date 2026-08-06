"""
Live integration tests for OpenAI Multimodal model uploads (Image, Audio, PDF, and Quad Multi-Media).
"""

import base64
from typing import Dict, Any, List
import pytest

from core.llm_models.router import ModelRouter
from core.modules.base import Base as BaseModule

TARGET_MODEL = "openai/gpt-4o-mini"

# Synthetic test media byte payloads
DUMMY_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
DUMMY_JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
DUMMY_WAV_BYTES = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
DUMMY_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF"


def run_multimodal_test(prompt: str, uploaded_files: List[Any]) -> str:
    """Helper runner executing OpenAI multimodal prompt calls."""
    module = BaseModule(
        prompt=prompt,
        model=TARGET_MODEL,
        response_mime_type="text/plain",
    )
    router = ModelRouter(module)
    res = router.model_response(module, uploaded_file=uploaded_files)
    assert res is not None
    return res


@pytest.mark.live
@pytest.mark.openai
def test_openai_single_file_uploads():
    module = BaseModule(model=TARGET_MODEL)
    router = ModelRouter(module)

    # 1. PNG Image
    png_file = router.upload_media(DUMMY_PNG_BYTES, mime_type="image/png")
    res_png = run_multimodal_test("What is in PNG?", [png_file])
    assert len(res_png) > 0

    # 2. JPEG Image
    jpeg_file = router.upload_media(DUMMY_JPEG_BYTES, mime_type="image/jpeg")
    res_jpeg = run_multimodal_test("What is in JPEG?", [jpeg_file])
    assert len(res_jpeg) > 0

    # 3. Audio WAV
    wav_file = router.upload_media(DUMMY_WAV_BYTES, mime_type="audio/wav")
    res_wav = run_multimodal_test("Transcribe or describe audio.", [wav_file])
    assert len(res_wav) > 0

    # 4. PDF Document
    pdf_file = router.upload_media(DUMMY_PDF_BYTES, mime_type="application/pdf")
    res_pdf = run_multimodal_test("Summarize document.", [pdf_file])
    assert len(res_pdf) > 0


@pytest.mark.live
@pytest.mark.openai
def test_openai_dual_media_combos():
    module = BaseModule(model=TARGET_MODEL)
    router = ModelRouter(module)

    png_file = router.upload_media(DUMMY_PNG_BYTES, mime_type="image/png")
    wav_file = router.upload_media(DUMMY_WAV_BYTES, mime_type="audio/wav")
    res_dual_1 = run_multimodal_test("Compare image and audio content.", [png_file, wav_file])
    assert len(res_dual_1) > 0

    jpeg_file = router.upload_media(DUMMY_JPEG_BYTES, mime_type="image/jpeg")
    pdf_file = router.upload_media(DUMMY_PDF_BYTES, mime_type="application/pdf")
    res_dual_2 = run_multimodal_test("Analyze image and document together.", [jpeg_file, pdf_file])
    assert len(res_dual_2) > 0


@pytest.mark.live
@pytest.mark.openai
def test_openai_quad_multi_media_combo():
    module = BaseModule(model=TARGET_MODEL)
    router = ModelRouter(module)

    png_file = router.upload_media(DUMMY_PNG_BYTES, mime_type="image/png")
    wav_file = router.upload_media(DUMMY_WAV_BYTES, mime_type="audio/wav")
    pdf_file = router.upload_media(DUMMY_PDF_BYTES, mime_type="application/pdf")

    res_quad = run_multimodal_test(
        "Synthesize insights from the provided image, audio, and PDF document simultaneously.",
        [png_file, wav_file, pdf_file],
    )
    assert len(res_quad) > 0
