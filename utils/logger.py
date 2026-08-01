import logging
import sys
import json
import os
from datetime import datetime, timezone
from typing import Optional, Union

from dotenv import load_dotenv

# This module is imported by utils.env_ops, so it must read its own
# configuration straight from the environment to avoid a circular import.

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
    """
    Reads LOGGING_MODE from the environment.

    NORMAL uses the standard logging package with console and file handlers.
    LAMBDA prints to stdout instead, since Lambda has a read-only filesystem
    and CloudWatch already captures stdout. Defaults to NORMAL.
    """
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
    """
    Reads LOGGING_LEVEL from the environment, accepting either a level name
    (DEBUG, INFO, WARNING, ERROR, CRITICAL) or a numeric value.
    """
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


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "asctime": self.formatTime(record, self.datefmt),
            "name": record.name,
            "levelname": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


class LambdaLogger:
    """
    Print-based stand-in for logging.Logger used when LOGGING_MODE=LAMBDA.

    Emits one JSON object per line to stdout so CloudWatch keeps each record
    intact, and mirrors the parts of the Logger interface this project uses.
    """

    def __init__(self, name: str, level: int = DEFAULT_LOGGING_LEVEL):
        self.name = name
        self.level = level

    def setLevel(self, level: int) -> None:
        self.level = level

    def isEnabledFor(self, level: int) -> bool:
        return level >= self.level

    def _emit(self, levelname: str, levelno: int, message: str, args, exc_info=None) -> None:
        if levelno < self.level:
            return

        # Support the logging package's %-style lazy interpolation.
        if args:
            try:
                message = message % args
            except (TypeError, ValueError):
                message = f"{message} {args}"

        record = {
            "asctime": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "name": self.name,
            "levelname": levelname,
            "message": str(message),
        }

        if exc_info:
            import traceback
            if exc_info is True:
                exc_info = sys.exc_info()
            if exc_info and exc_info[0] is not None:
                record["exc_info"] = "".join(traceback.format_exception(*exc_info))

        print(json.dumps(record))

    def debug(self, message, *args, **kwargs):
        self._emit("DEBUG", logging.DEBUG, message, args, kwargs.get("exc_info"))

    def info(self, message, *args, **kwargs):
        self._emit("INFO", logging.INFO, message, args, kwargs.get("exc_info"))

    def warning(self, message, *args, **kwargs):
        self._emit("WARNING", logging.WARNING, message, args, kwargs.get("exc_info"))

    warn = warning

    def error(self, message, *args, **kwargs):
        self._emit("ERROR", logging.ERROR, message, args, kwargs.get("exc_info"))

    def critical(self, message, *args, **kwargs):
        self._emit("CRITICAL", logging.CRITICAL, message, args, kwargs.get("exc_info"))

    fatal = critical

    def exception(self, message, *args, **kwargs):
        kwargs.setdefault("exc_info", True)
        self._emit("ERROR", logging.ERROR, message, args, kwargs.get("exc_info"))

    def log(self, level, message, *args, **kwargs):
        self._emit(logging.getLevelName(level), level, message, args, kwargs.get("exc_info"))


_lambda_loggers: dict = {}


def get_logger(name: str, level: Optional[int] = None) -> Union[logging.Logger, LambdaLogger]:
    """
    Returns a logger honouring LOGGING_MODE and LOGGING_LEVEL from .env.

    An explicit `level` argument takes precedence over LOGGING_LEVEL.
    """
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

        # File Handler (JSON format)
        try:
            # Modified path to be relative to root
            log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", LOG_DIR_NAME))
            os.makedirs(log_dir, exist_ok=True)

            date_str = datetime.now().strftime(LOG_FILE_DATE_FORMAT)
            log_file = os.path.join(log_dir, f"logs_{date_str}.log")

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to initialize file logging: {e}", file=sys.stderr)

    return logger
