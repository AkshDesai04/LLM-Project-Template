"""
Utils package - Modular infrastructure utilities.

Organized into domain-specific subpackages:
- utils.db: Database connectors and factory router
- utils.env: Secrets resolution and environment configuration
- utils.logging: Logging framework
- utils.io: File operations and I/O helpers
- utils.concurrency: Rate limiting and parallel execution
- utils.document: Document and media converters
"""

import sys

# Subpackage imports
from utils import concurrency, db, document, env, io, logging

# Re-exports for top-level convenience
from utils.concurrency import ThreadSafeRateLimiter, parallel_execute
from utils.db import (
    BaseDatabaseConnector,
    DatabaseRouter,
    MySQLConnector,
    PostgreSQLConnector,
    SQLiteConnector,
)
from utils.document import MarkItDownUtils, convert_to_markdown
from utils.env import (
    KEY_LOCATION_AWS_SM,
    KEY_LOCATION_LOCAL,
    KEY_LOCATION_NAME,
    ROUTED_SECRET_NAMES,
    get_aws_secret,
    get_database_url,
    get_key_location,
    get_local_secret,
    get_secret,
)
from utils.io import get_file, read_csv, read_file, read_prompt
from utils.logging import JsonFormatter, LambdaLogger, get_logger, get_logging_level, get_logging_mode

# Backward-compatibility module aliases for legacy import paths
sys.modules["utils.logger"] = logging.logger
sys.modules["utils.env_ops"] = env.env_ops
sys.modules["utils.file_ops"] = io.file_ops
sys.modules["utils.parallel_executor"] = concurrency.parallel_executor
sys.modules["utils.markitdown_utils"] = document.markitdown
sys.modules["utils.base_connector"] = db.base
sys.modules["utils.db_router"] = db.router
sys.modules["utils.mysql_connector"] = db.mysql.connector
sys.modules["utils.postgres_connector"] = db.postgres.connector
sys.modules["utils.sqlite_connector"] = db.sqlite.connector

__all__ = [
    # Subpackages
    "db",
    "env",
    "logging",
    "io",
    "concurrency",
    "document",
    # Connectors & Router
    "BaseDatabaseConnector",
    "DatabaseRouter",
    "MySQLConnector",
    "PostgreSQLConnector",
    "SQLiteConnector",
    # Logging
    "get_logger",
    "get_logging_mode",
    "get_logging_level",
    "JsonFormatter",
    "LambdaLogger",
    # Env & Secrets
    "get_secret",
    "get_local_secret",
    "get_aws_secret",
    "get_database_url",
    "get_key_location",
    "ROUTED_SECRET_NAMES",
    "KEY_LOCATION_NAME",
    "KEY_LOCATION_LOCAL",
    "KEY_LOCATION_AWS_SM",
    # File I/O
    "read_file",
    "get_file",
    "read_prompt",
    "read_csv",
    # Concurrency
    "parallel_execute",
    "ThreadSafeRateLimiter",
    # Document
    "MarkItDownUtils",
    "convert_to_markdown",
]
