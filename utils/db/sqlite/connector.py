"""
SQLite Database Connector Implementation.
"""

import contextlib
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.db.base import BaseDatabaseConnector
from utils.db.sqlite.vector_handler import (
    create_sqlite_vector_table,
    insert_sqlite_vector,
    insert_sqlite_vectors,
    search_sqlite_vectors,
    delete_sqlite_vector,
)
from utils.env import get_database_url, get_secret
from utils.logging import get_logger

logger = get_logger("SQLiteConnector")


class SQLiteConnector(BaseDatabaseConnector):
    """Connector for SQLite relational databases with vector search support."""

    def __init__(self, db_path: Optional[str] = None, connection_string: Optional[str] = None, timeout: float = 10.0):
        resolved_path = (
            db_path
            or connection_string
            or get_secret("SQLITE_URL", raise_error=False)
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
        if self._connection is None:
            logger.info(f"Connecting to SQLite database at '{self.db_path}'...")
            self._connection = sqlite3.connect(self.db_path, timeout=self.timeout)
            self._connection.row_factory = sqlite3.Row
            logger.info("Successfully connected to SQLite database.")
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            logger.info("Closing SQLite database connection...")
            self._connection.close()
            self._connection = None
            logger.info("SQLite connection closed.")

    def is_connected(self) -> bool:
        if self._connection is None:
            return False
        try:
            self._connection.execute("SELECT 1;")
            return True
        except (sqlite3.Error, AttributeError):
            return False

    def execute_query(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Failed to execute SQLite query: {e}")
            raise

    def execute_non_query(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> int:
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            if not self._in_transaction:
                conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            if not self._in_transaction:
                conn.rollback()
            logger.error(f"Failed to execute SQLite non-query: {e}")
            raise

    def execute_many(self, query: str, params_list: List[Union[Tuple[Any, ...], Dict[str, Any]]]) -> int:
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

    def fetch_one(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch_one from SQLite: {e}")
            raise

    @contextlib.contextmanager
    def transaction(self):
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

    # Vector operations
    def create_vector_table(self, table_name: str, vector_dim: int, distance_metric: str = "cosine") -> None:
        create_sqlite_vector_table(self, table_name, vector_dim, distance_metric)

    def insert_vector(self, table_name: str, vector_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
        return insert_sqlite_vector(self, table_name, vector_id, vector, metadata)

    def insert_vectors(self, table_name: str, records: List[Dict[str, Any]]) -> int:
        return insert_sqlite_vectors(self, table_name, records)

    def vector_search(self, table_name: str, query_vector: List[float], top_k: int = 10, min_score: Optional[float] = None, distance_metric: str = "cosine") -> List[Dict[str, Any]]:
        return search_sqlite_vectors(self, table_name, query_vector, top_k=top_k, min_score=min_score, distance_metric=distance_metric)

    def delete_vector(self, table_name: str, vector_id: str) -> int:
        return delete_sqlite_vector(self, table_name, vector_id)
