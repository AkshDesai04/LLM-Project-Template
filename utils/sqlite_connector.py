"""
SQLite Database Connector.

Provides a thread-safe, lightweight connector for local SQLite databases
or in-memory instances leveraging standard library sqlite3.
"""

import contextlib
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.base_connector import BaseDatabaseConnector
from utils.env_ops import get_database_url, get_secret
from utils.logger import get_logger

logger = get_logger("SQLiteConnector")


class SQLiteConnector(BaseDatabaseConnector):
    """
    Connector for SQLite relational databases.

    Attributes:
        db_path (str): File path to SQLite database or ':memory:'.
        timeout (float): Connection timeout in seconds.
    """

    def __init__(self, db_path: Optional[str] = None, timeout: float = 10.0):
        """
        Initializes the SQLite connector configuration.

        Args:
            db_path: Path to .db file or ':memory:'. If None, resolves from
                     SQLITE_DB_PATH or DATABASE_URL env vars.
            timeout: Command timeout in seconds.
        """
        resolved_path = (
            db_path
            or get_secret("SQLITE_DB_PATH", raise_error=False)
            or get_database_url(raise_error=False)
            or ":memory:"
        )
        if resolved_path.startswith("sqlite:///"):
            resolved_path = resolved_path[10:]
        elif resolved_path.startswith("sqlite://"):
            resolved_path = resolved_path[9:]

        self.db_path = resolved_path
        self.timeout = timeout
        self._connection: Optional[sqlite3.Connection] = None
        self._in_transaction: bool = False

    def connect(self) -> sqlite3.Connection:
        """
        Establishes connection to the SQLite database.

        Returns:
            sqlite3.Connection: Active SQLite connection.
        """
        if self._connection is None:
            logger.info(f"Connecting to SQLite database at '{self.db_path}'...")
            self._connection = sqlite3.connect(self.db_path, timeout=self.timeout)
            self._connection.row_factory = sqlite3.Row
            logger.info("Successfully connected to SQLite database.")
        return self._connection

    def close(self) -> None:
        """Closes the active SQLite database connection."""
        if self._connection is not None:
            logger.info("Closing SQLite database connection...")
            self._connection.close()
            self._connection = None
            logger.info("SQLite connection closed.")

    def is_connected(self) -> bool:
        """
        Checks whether the SQLite database connection is active.

        Returns:
            bool: True if connected and responsive, False otherwise.
        """
        if self._connection is None:
            return False
        try:
            self._connection.execute("SELECT 1;")
            return True
        except (sqlite3.Error, AttributeError):
            return False

    def execute_query(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a SELECT query and returns records as dictionaries.

        Args:
            query: SQL query.
            params: Query parameters.

        Returns:
            List[Dict[str, Any]]: List of dictionary rows.
        """
        conn = self.connect()
        params = params or ()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to execute SQLite query: {e}")
            raise

    def execute_non_query(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> int:
        """
        Executes an INSERT, UPDATE, DELETE, or DDL query.

        Args:
            query: SQL statement.
            params: Bound parameters.

        Returns:
            int: Number of affected rows.
        """
        conn = self.connect()
        params = params or ()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if not self._in_transaction:
                conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            if not self._in_transaction:
                conn.rollback()
            logger.error(f"Failed to execute SQLite non-query: {e}")
            raise

    def execute_many(
        self, query: str, params_list: List[Union[Tuple[Any, ...], Dict[str, Any]]]
    ) -> int:
        """
        Executes a query repeatedly with parameter sets.

        Args:
            query: SQL statement template.
            params_list: Sequence of parameters.

        Returns:
            int: Total affected rows count.
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            if not self._in_transaction:
                conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            if not self._in_transaction:
                conn.rollback()
            logger.error(f"Failed to execute SQLite executemany: {e}")
            raise

    def fetch_one(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a query and returns the first row as a dictionary.

        Args:
            query: SQL query.
            params: Parameters to bind.

        Returns:
            Optional[Dict[str, Any]]: Single dictionary row or None.
        """
        conn = self.connect()
        params = params or ()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch_one from SQLite: {e}")
            raise

    @contextlib.contextmanager
    def transaction(self):
        """
        Context manager for SQLite transaction safety.
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
            logger.error(f"SQLite transaction failed and rolled back: {e}")
            raise
        finally:
            self._in_transaction = was_in_transaction
