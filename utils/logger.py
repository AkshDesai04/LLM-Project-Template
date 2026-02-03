import logging
import sys
import json
import os
from datetime import datetime


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


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File Handler (JSON format)
        try:
            # Modified path to be relative to root
            log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
            os.makedirs(log_dir, exist_ok=True)

            date_str = datetime.now().strftime("%Y%m%d")
            log_file = os.path.join(log_dir, f"rfp_ai_{date_str}.log")

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to initialize file logging: {e}", file=sys.stderr)

    return logger