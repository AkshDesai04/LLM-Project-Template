# `core/llm_models/utils/`

Offline media decoding for providers that have no file-upload API.

Only Gemini (in API-key mode) can upload a file and let the service parse it. Every
other provider needs the bytes turned into something that fits in a chat message —
extracted text, or a base64 image. That conversion happens here, entirely locally, with
no network calls.

## Contents

- **`media_utils.py`** — three pure functions, described below.
- **`__init__.py`** — one-line placeholder, no re-exports.

Logger name: `MediaUtils`.

## Output shape

Two of the three functions return the **OpenAI content-block shape**:

```python
{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
```

This is the internal lingua franca. OpenAI, Perplexity and vLLM consume it directly.
`AnthropicProvider._to_content_blocks` translates it into Anthropic image blocks, and
`OllamaProvider` decodes the base64 back to raw bytes. If you add a provider, convert
*from* this shape rather than inventing another.

## Functions

### `extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str`

Extracts text with `PyPDF2.PdfReader`, concatenating every page separated by newlines,
and returns the stripped result. On failure, it logs the error and raises `RuntimeError`.

### `process_video_frames(video_bytes: bytes, frames_per_second: int = 1) -> List[dict]`

Samples frames from a video and returns them as a list of image content blocks.

Writes the bytes to a `NamedTemporaryFile` with suffix `.mp4`, opens it with
`cv2.VideoCapture`, seeks frame-by-frame at an interval of `fps / frames_per_second`,
JPEG-encodes each frame and base64-encodes it. The temp file is removed and the capture
released in a `finally` block, so both are cleaned up even on error.

Notes and caveats:

- Size cap is **512 MB** (`MAX_VIDEO_FILE_SIZE = 512 * 1024 * 1024`). Exceeding it raises
  `ValueError` stating the 512MB limit.
- If OpenCV cannot detect the frame rate it assumes **30 fps**.
- Unreadable frames are skipped silently.
- At the default 1 fps a ten-minute video yields ~600 images. Nothing here caps the
  frame count, so long videos will blow through context limits and cost. Raise
  `frames_per_second` only deliberately, and consider trimming the clip first.
- Requires `opencv-python-headless`, the heaviest dependency in the project. Importing
  this module fails without it, which is why providers import it lazily.

### `encode_image_base64(image_bytes: bytes, mime_type: str) -> dict`

Wraps image bytes into a single content block as a `data:` URL. Trivially thin, but
keeps the URL format in one place. The `mime_type` is used verbatim, so pass a real one
(`image/png`, `image/jpeg`) — it is not sniffed or validated.

## Who calls this

`upload_media` in `openai.py`, `anthropic.py`, `ollama.py` and `vllm.py`. Each dispatches
on mime type: `application/pdf` → text extraction, `image/*` → base64 encode,
`video/*` → frame extraction, anything else → `bytes.decode('utf-8', errors='ignore')`.

`gemini.py` only uses these under Vertex AI; in API-key mode it uploads to the Files API
instead. `perplexity.py` uses none of them — it returns a placeholder string for any
non-text media.

## Dependencies

`opencv-python-headless`, `numpy`, `PyPDF2`. All pinned in the root `requirements.txt`.
