import sys
import json
import logging
from datetime import datetime, timezone

DEFAULT_LOGGING_LEVEL = logging.INFO


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
