"""
MySQL Database Connector.

Provides a robust MySQL connector using pymysql with support for connection URLs,
environment variable fallbacks, DictCursor row formatting, and transactions.
"""

import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

try:
    import pymysql
    import pymysql.cursors
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False
    pymysql = None

from utils.base_connector import BaseDatabaseConnector
from utils.env_ops import get_database_url, get_secret
from utils.logger import get_logger

logger = get_logger("MySQLConnector")


class MySQLConnector(BaseDatabaseConnector):
    """
    Connector for MySQL relational databases.

    Primary configuration uses a single connection URL (MYSQL_URL or DATABASE_URL).
    Individual connection attributes serve as secondary fallbacks.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        charset: str = "utf8mb4",
    ):
        """
        Initializes MySQL connector parameters.
        Prioritizes connection_string / MYSQL_URL / DATABASE_URL, falling back to individual parameters.
        """
        self.connection_string = (
            connection_string
            or get_secret("MYSQL_URL", raise_error=False)
            or get_database_url(raise_error=False)
        )
        
        parsed_host, parsed_port, parsed_user, parsed_pass, parsed_db = None, None, None, None, None
        if self.connection_string and self.connection_string.startswith("mysql"):
            try:
                parsed = urlparse(self.connection_string)
                parsed_host = parsed.hostname
                parsed_port = parsed.port
                parsed_user = unquote(parsed.username) if parsed.username else None
                parsed_pass = unquote(parsed.password) if parsed.password else None
                parsed_db = parsed.path.lstrip("/") if parsed.path else None
            except Exception as e:
                logger.warning(f"Could not parse MySQL connection_string '{self.connection_string}': {e}")

        self.host = host or parsed_host or get_secret("MYSQL_HOST", raise_error=False) or "localhost"
        self.port = port or parsed_port or int(get_secret("MYSQL_PORT", raise_error=False) or 3306)
        self.user = user or parsed_user or get_secret("MYSQL_USER", raise_error=False)
        self.password = password or parsed_pass or get_secret("MYSQL_PASSWORD", raise_error=False)
        self.database = database or parsed_db or get_secret("MYSQL_DB", raise_error=False)
        self.charset = charset
        self._connection = None
        self._in_transaction: bool = False

    def connect(self):
        """
        Establishes connection to MySQL database.

        Returns:
            pymysql.connections.Connection: Active connection instance.
        """
        if not PYMYSQL_AVAILABLE:
            message = "pymysql is not installed. Please install 'pymysql' to use MySQLConnector."
            logger.error(message)
            raise ImportError(message)

        if self._connection is None or not self._connection.open:
            logger.info(f"Connecting to MySQL database '{self.database}' at {self.host}:{self.port}...")
            kwargs = {
                "host": self.host,
                "port": self.port,
                "user": self.user,
                "password": self.password,
                "database": self.database,
                "charset": self.charset,
                "cursorclass": pymysql.cursors.DictCursor,
                "autocommit": False,
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            self._connection = pymysql.connect(**filtered_kwargs)
            logger.info("Successfully connected to MySQL database.")
        return self._connection

    def close(self) -> None:
        """Closes active MySQL connection."""
        if self._connection is not None and self._connection.open:
            logger.info("Closing MySQL connection...")
            self._connection.close()
            self._connection = None
            logger.info("MySQL connection closed.")

    def is_connected(self) -> bool:
        """
        Checks whether MySQL connection is active.

        Returns:
            bool: True if connected and responsive, False otherwise.
        """
        if not PYMYSQL_AVAILABLE or self._connection is None or not self._connection.open:
            return False
        try:
            self._connection.ping(reconnect=False)
            return True
        except Exception:
            return False

    def execute_query(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a SELECT query and returns rows as dictionaries.
        """
        conn = self.connect()
        params = params or ()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return list(rows)
        except Exception as e:
            logger.error(f"MySQL query execution failed: {e}")
            raise

    def execute_non_query(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> int:
        """
        Executes an INSERT, UPDATE, DELETE, or DDL statement.
        """
        conn = self.connect()
        params = params or ()
        try:
            with conn.cursor() as cursor:
                rowcount = cursor.execute(query, params)
            if not self._in_transaction:
                conn.commit()
            return rowcount
        except Exception as e:
            if not self._in_transaction:
                conn.rollback()
            logger.error(f"MySQL non-query execution failed: {e}")
            raise

    def execute_many(
        self, query: str, params_list: List[Union[Tuple[Any, ...], Dict[str, Any]]]
    ) -> int:
        """
        Executes a query repeatedly with batch parameters.
        """
        conn = self.connect()
        try:
            with conn.cursor() as cursor:
                rowcount = cursor.executemany(query, params_list)
            if not self._in_transaction:
                conn.commit()
            return rowcount
        except Exception as e:
            if not self._in_transaction:
                conn.rollback()
            logger.error(f"MySQL executemany failed: {e}")
            raise

    def fetch_one(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a query and returns the first row as a dictionary.
        """
        conn = self.connect()
        params = params or ()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
                return row if row else None
        except Exception as e:
            logger.error(f"MySQL fetch_one failed: {e}")
            raise

    @contextlib.contextmanager
    def transaction(self):
        """
        Context manager for MySQL transaction safety.
        """
        conn = self.connect()
        was_in_transaction = self._in_transaction
        self._in_transaction = True
        try:
            yield conn
            if not was_in_transaction:
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"MySQL transaction failed and rolled back: {e}")
            raise
        finally:
            self._in_transaction = was_in_transaction
