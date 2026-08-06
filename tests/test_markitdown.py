"""
Unit tests for MarkItDown document conversion utility (utils.document.markitdown).
"""

from io import BytesIO
from unittest.mock import MagicMock, patch
import pytest

from utils.document import MarkItDownUtils, convert_to_markdown


@pytest.fixture
def mock_markitdown():
    with patch("utils.document.markitdown.MARKITDOWN_AVAILABLE", True), patch(
        "utils.document.markitdown.MarkItDown"
    ) as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.mark.document
@pytest.mark.unit
def test_markitdown_not_installed():
    with patch("utils.document.markitdown.MARKITDOWN_AVAILABLE", False):
        with pytest.raises(ImportError, match="markitdown is not installed"):
            MarkItDownUtils()


@pytest.mark.document
@pytest.mark.unit
def test_convert_general(mock_markitdown):
    mock_res = MagicMock()
    mock_res.text_content = "# Converted Content"
    mock_markitdown.convert.return_value = mock_res

    utils = MarkItDownUtils()
    res = utils.convert("doc.txt")
    assert res == "# Converted Content"
    mock_markitdown.convert.assert_called_once_with("doc.txt")


@pytest.mark.document
@pytest.mark.unit
def test_convert_local_success(mock_markitdown, tmp_path):
    f_path = tmp_path / "sample.pdf"
    f_path.write_bytes(b"%PDF-1.4")

    mock_res = MagicMock()
    mock_res.text_content = "# PDF Content"
    mock_markitdown.convert_local.return_value = mock_res

    utils = MarkItDownUtils()
    res = utils.convert_local(str(f_path))
    assert res == "# PDF Content"
    mock_markitdown.convert_local.assert_called_once_with(str(f_path))


@pytest.mark.document
@pytest.mark.unit
def test_convert_local_missing_file(mock_markitdown):
    utils = MarkItDownUtils()
    with pytest.raises(FileNotFoundError, match="Local file not found"):
        utils.convert_local("/missing/file.pdf")


@pytest.mark.document
@pytest.mark.unit
def test_convert_url_validation(mock_markitdown):
    utils = MarkItDownUtils()

    # Invalid scheme
    with pytest.raises(ValueError, match="Only HTTP/HTTPS protocols are allowed"):
        utils.convert_url("ftp://example.com/file.txt")

    # Missing hostname
    with pytest.raises(ValueError, match="no hostname found"):
        utils.convert_url("http://")

    # Blocked hostname (localhost)
    with pytest.raises(ValueError, match="Invalid or restricted URL hostname"):
        utils.convert_url("http://localhost/admin")

    # Private IP
    with pytest.raises(ValueError, match="Invalid or restricted IP address"):
        utils.convert_url("http://127.0.0.1/admin")


@pytest.mark.document
@pytest.mark.unit
def test_convert_url_success(mock_markitdown):
    mock_res = MagicMock()
    mock_res.text_content = "# Web Page"
    mock_markitdown.convert_url.return_value = mock_res

    utils = MarkItDownUtils()
    res = utils.convert_url("https://example.com/article")
    assert res == "# Web Page"
    mock_markitdown.convert_url.assert_called_once_with("https://example.com/article")


@pytest.mark.document
@pytest.mark.unit
def test_convert_stream(mock_markitdown):
    mock_res = MagicMock()
    mock_res.text_content = "# Stream Data"
    mock_markitdown.convert_stream.return_value = mock_res

    utils = MarkItDownUtils()
    stream = BytesIO(b"data")
    res = utils.convert_stream(stream, file_extension=".docx")
    assert res == "# Stream Data"
    mock_markitdown.convert_stream.assert_called_once_with(stream, file_extension=".docx")


@pytest.mark.document
@pytest.mark.unit
def test_convert_image_and_audio(mock_markitdown):
    mock_res = MagicMock()
    mock_res.text_content = "Media Content"
    mock_markitdown.convert.return_value = mock_res

    utils = MarkItDownUtils()
    assert utils.convert_image("image.jpg") == "Media Content"
    assert utils.convert_audio("audio.mp3") == "Media Content"


@pytest.mark.document
@pytest.mark.unit
def test_convert_to_markdown_helper(mock_markitdown):
    mock_res = MagicMock()
    mock_res.text_content = "# Helper Result"
    mock_markitdown.convert.return_value = mock_res

    res = convert_to_markdown("test.txt")
    assert res == "# Helper Result"
