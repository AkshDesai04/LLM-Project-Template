"""
Unit tests for logging utilities (utils.logging.logger & lambda_logger).
"""

import json
import logging
from unittest.mock import patch
import pytest

from utils.logging import (
    JsonFormatter,
    LambdaLogger,
    get_logger,
    get_logging_level,
    get_logging_mode,
)
from utils.logging.logger import (
    LOGGING_LEVEL_NAME,
    LOGGING_MODE_NAME,
)


@pytest.mark.logging
@pytest.mark.unit
def test_get_logging_mode_default():
    with patch("os.getenv", return_value=None):
        assert get_logging_mode() == "NORMAL"


@pytest.mark.logging
@pytest.mark.unit
def test_get_logging_mode_lambda():
    with patch("os.getenv", return_value="LAMBDA"):
        assert get_logging_mode() == "LAMBDA"


@pytest.mark.logging
@pytest.mark.unit
def test_get_logging_mode_invalid():
    with patch("os.getenv", return_value="INVALID_MODE"):
        with pytest.raises(ValueError, match="Invalid LOGGING_MODE"):
            get_logging_mode()


@pytest.mark.logging
@pytest.mark.unit
def test_get_logging_level_named():
    with patch("os.getenv", return_value="DEBUG"):
        assert get_logging_level() == logging.DEBUG

    with patch("os.getenv", return_value="WARNING"):
        assert get_logging_level() == logging.WARNING


@pytest.mark.logging
@pytest.mark.unit
def test_get_logging_level_numeric():
    with patch("os.getenv", return_value="30"):
        assert get_logging_level() == 30


@pytest.mark.logging
@pytest.mark.unit
def test_get_logging_level_invalid():
    with patch("os.getenv", return_value="SUPER_DEBUG"):
        with pytest.raises(ValueError, match="Invalid LOGGING_LEVEL"):
            get_logging_level()


@pytest.mark.logging
@pytest.mark.unit
def test_get_logger_normal_mode():
    with patch("utils.logging.logger.get_logging_mode", return_value="NORMAL"):
        logger = get_logger("TestNormalLogger", level=logging.INFO)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "TestNormalLogger"
        assert logger.level == logging.INFO


@pytest.mark.logging
@pytest.mark.unit
def test_get_logger_lambda_mode():
    with patch("utils.logging.logger.get_logging_mode", return_value="LAMBDA"):
        logger1 = get_logger("TestLambdaLogger", level=logging.INFO)
        assert isinstance(logger1, LambdaLogger)
        assert logger1.level == logging.INFO

        # Re-fetching returns cached instance with updated level
        logger2 = get_logger("TestLambdaLogger", level=logging.DEBUG)
        assert logger2 is logger1
        assert logger2.level == logging.DEBUG


@pytest.mark.logging
@pytest.mark.unit
def test_json_formatter():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="Test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Hello %s",
        args=("world",),
        exc_info=None,
    )
    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    assert parsed["name"] == "Test"
    assert parsed["levelname"] == "INFO"
    assert parsed["message"] == "Hello world"


@pytest.mark.logging
@pytest.mark.unit
def test_lambda_logger_output(capsys):
    logger = LambdaLogger("TestLambda", level=logging.INFO)
    assert logger.isEnabledFor(logging.INFO) is True
    assert logger.isEnabledFor(logging.DEBUG) is False

    # Debug ignored due to level
    logger.debug("Should not print")
    captured = capsys.readouterr()
    assert captured.out == ""

    # Info printed
    logger.info("Hello %s", "Lambda")
    captured = capsys.readouterr()
    parsed = json.loads(captured.out.strip())
    assert parsed["name"] == "TestLambda"
    assert parsed["levelname"] == "INFO"
    assert parsed["message"] == "Hello Lambda"

    # Exception printed with exc_info
    try:
        raise ValueError("Simulated error")
    except ValueError:
        logger.exception("An error occurred")

    captured = capsys.readouterr()
    parsed_err = json.loads(captured.out.strip())
    assert parsed_err["levelname"] == "ERROR"
    assert "Simulated error" in parsed_err["exc_info"]
