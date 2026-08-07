import logging
import sys
import os
from datetime import datetime
from typing import Optional, Union

from dotenv import load_dotenv

from utils.logging.lambda_logger import JsonFormatter, LambdaLogger

LOGGING_MODE_NAME = "LOGGING_MODE"
LOGGING_LEVEL_NAME = "LOGGING_LEVEL"

LOGGING_MODE_NORMAL = "NORMAL"
LOGGING_MODE_LAMBDA = "LAMBDA"
VALID_LOGGING_MODES = (LOGGING_MODE_NORMAL, LOGGING_MODE_LAMBDA)

DEFAULT_LOGGING_LEVEL = logging.INFO
LOG_DIR_NAME = "logs"
LOG_FILE_DATE_FORMAT = "%Y%m%d"
CONSOLE_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

_LEVEL_NAMES = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def get_logging_mode() -> str:
    """Reads LOGGING_MODE from the environment."""
    load_dotenv()
    raw = os.getenv(LOGGING_MODE_NAME) or LOGGING_MODE_NORMAL
    mode = raw.strip().strip("'\"").upper()
    if mode not in VALID_LOGGING_MODES:
        raise ValueError(
            f"Invalid {LOGGING_MODE_NAME}='{raw}'. "
            f"Expected one of: {', '.join(VALID_LOGGING_MODES)}."
        )
    return mode


def get_logging_level() -> int:
    """Reads LOGGING_LEVEL from the environment."""
    load_dotenv()
    raw = os.getenv(LOGGING_LEVEL_NAME)
    if not raw:
        return DEFAULT_LOGGING_LEVEL

    value = raw.strip().strip("'\"").upper()
    if value in _LEVEL_NAMES:
        return _LEVEL_NAMES[value]
    if value.isdigit():
        return int(value)

    raise ValueError(
        f"Invalid {LOGGING_LEVEL_NAME}='{raw}'. Expected one of: "
        f"{', '.join(sorted(set(_LEVEL_NAMES)))}, or a numeric level."
    )


_lambda_loggers: dict = {}


def get_logger(name: str, level: Optional[int] = None) -> Union[logging.Logger, LambdaLogger]:
    """Returns a logger honouring LOGGING_MODE and LOGGING_LEVEL from .env."""
    resolved_level = level if level is not None else get_logging_level()

    if get_logging_mode() == LOGGING_MODE_LAMBDA:
        existing = _lambda_loggers.get(name)
        if existing is None:
            existing = LambdaLogger(name, resolved_level)
            _lambda_loggers[name] = existing
        else:
            existing.setLevel(resolved_level)
        return existing

    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(CONSOLE_LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        try:
            log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", LOG_DIR_NAME))
            os.makedirs(log_dir, exist_ok=True)

            date_str = datetime.now().strftime(LOG_FILE_DATE_FORMAT)
            log_file = os.path.join(log_dir, f"logs_{date_str}.log")

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to initialize file logging: {e}", file=sys.stderr)

    return logger
