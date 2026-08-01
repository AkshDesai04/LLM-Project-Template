"""
PostgreSQL Database Connector.

Provides a robust PostgreSQL connector using psycopg2 with support for single connection URLs
and transactions.
"""

import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None

from utils.base_connector import BaseDatabaseConnector
from utils.env_ops import get_database_url, get_secret
from utils.logger import get_logger

logger = get_logger("PostgreSQLConnector")


class PostgreSQLConnector(BaseDatabaseConnector):
    """
    Connector for PostgreSQL relational databases using a single connection URL.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
    ):
        """
        Initializes PostgreSQL connector using a connection URL.
        Resolves from connection_string / POSTGRES_URL / DATABASE_URL env vars.
        """
        self.connection_string = (
            connection_string
            or get_secret("POSTGRES_URL", raise_error=False)
            or get_database_url(raise_error=False)
        )
        self._connection = None
        self._in_transaction: bool = False

    def connect(self):
        """
        Establishes connection to PostgreSQL database.

        Returns:
            psycopg2.connection: Active connection instance.
        """
        if not PSYCOPG2_AVAILABLE:
            message = "psycopg2 is not installed. Please install 'psycopg2-binary' to use PostgreSQLConnector."
            logger.error(message)
            raise ImportError(message)

        if self._connection is None or self._connection.closed != 0:
            if not self.connection_string:
                raise ValueError("No PostgreSQL connection URL provided (set POSTGRES_URL or DATABASE_URL).")
            logger.info("Connecting to PostgreSQL database via connection URL...")
            self._connection = psycopg2.connect(self.connection_string)
            logger.info("Successfully connected to PostgreSQL database.")
        return self._connection

    def close(self) -> None:
        """Closes active PostgreSQL connection."""
        if self._connection is not None and self._connection.closed == 0:
            logger.info("Closing PostgreSQL connection...")
            self._connection.close()
            self._connection = None
            logger.info("PostgreSQL connection closed.")

    def is_connected(self) -> bool:
        """
        Checks whether PostgreSQL connection is open and active.

        Returns:
            bool: True if connected and responsive, False otherwise.
        """
        if not PSYCOPG2_AVAILABLE or self._connection is None or self._connection.closed != 0:
            return False
        try:
            with self._connection.cursor() as cursor:
                cursor.execute("SELECT 1;")
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
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"PostgreSQL query execution failed: {e}")
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
                cursor.execute(query, params)
                rowcount = cursor.rowcount
            if not self._in_transaction:
                conn.commit()
            return rowcount
        except Exception as e:
            if not self._in_transaction:
                conn.rollback()
            logger.error(f"PostgreSQL non-query execution failed: {e}")
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
                cursor.executemany(query, params_list)
                rowcount = cursor.rowcount
            if not self._in_transaction:
                conn.commit()
            return rowcount
        except Exception as e:
            if not self._in_transaction:
                conn.rollback()
            logger.error(f"PostgreSQL executemany failed: {e}")
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
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"PostgreSQL fetch_one failed: {e}")
            raise

    @contextlib.contextmanager
    def transaction(self):
        """
        Context manager for PostgreSQL transaction management.
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
            logger.error(f"PostgreSQL transaction failed and rolled back: {e}")
            raise
        finally:
            self._in_transaction = was_in_transaction
