"""
Unit tests for file I/O operations (utils.io.file_ops).
"""

from unittest.mock import patch
import pytest

from utils.io import (
    DEFAULT_ENCODING,
    PROMPT_FILE_EXTENSION,
    PROMPTS_DIR_NAME,
    get_file,
    read_csv,
    read_file,
    read_prompt,
)


@pytest.mark.io
@pytest.mark.unit
def test_read_file_success(tmp_path):
    f_path = tmp_path / "sample.txt"
    f_path.write_text("  Hello World!  \n", encoding=DEFAULT_ENCODING)

    content = read_file(str(f_path))
    assert content == "Hello World!"


@pytest.mark.io
@pytest.mark.unit
def test_read_file_not_found():
    with pytest.raises(FileNotFoundError, match="Text file not found"):
        read_file("/non/existent/path/file.txt")


@pytest.mark.io
@pytest.mark.unit
def test_read_file_io_error(tmp_path):
    f_path = tmp_path / "unreadable.txt"
    f_path.write_text("test")

    with patch("builtins.open", side_effect=PermissionError("Permission denied")):
        with pytest.raises(IOError, match="Could not read text file"):
            read_file(str(f_path))


@pytest.mark.io
@pytest.mark.unit
def test_get_file_success(tmp_path):
    f_path = tmp_path / "sample.bin"
    f_path.write_bytes(b"\x00\x01\x02\x03")

    data = get_file(str(f_path))
    assert data == b"\x00\x01\x02\x03"


@pytest.mark.io
@pytest.mark.unit
def test_get_file_not_found():
    with pytest.raises(FileNotFoundError, match="Binary file not found"):
        get_file("/non/existent/path/file.bin")


@pytest.mark.io
@pytest.mark.unit
def test_get_file_io_error(tmp_path):
    f_path = tmp_path / "unreadable.bin"
    f_path.write_bytes(b"data")

    with patch("builtins.open", side_effect=PermissionError("Permission denied")):
        with pytest.raises(IOError, match="Could not read binary file"):
            get_file(str(f_path))


@pytest.mark.io
@pytest.mark.unit
def test_read_prompt():
    content = read_prompt("test_prompt")
    assert isinstance(content, str)
    assert len(content) > 0


@pytest.mark.io
@pytest.mark.unit
def test_read_csv_success(tmp_path):
    f_path = tmp_path / "sample.csv"
    f_path.write_text("name,age\nAlice,30\nBob,25\n", encoding=DEFAULT_ENCODING)

    rows = read_csv(str(f_path))
    assert len(rows) == 2
    assert rows[0] == {"name": "Alice", "age": "30"}
    assert rows[1] == {"name": "Bob", "age": "25"}


@pytest.mark.io
@pytest.mark.unit
def test_read_csv_not_found():
    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        read_csv("/non/existent/data.csv")


@pytest.mark.io
@pytest.mark.unit
def test_read_csv_io_error(tmp_path):
    f_path = tmp_path / "broken.csv"
    f_path.write_text("a,b\n1,2")

    with patch("builtins.open", side_effect=PermissionError("Permission denied")):
        with pytest.raises(IOError, match="Could not read CSV file"):
            read_csv(str(f_path))
