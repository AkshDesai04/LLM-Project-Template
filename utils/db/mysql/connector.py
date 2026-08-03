"""
MySQL Database Connector Implementation.
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

from utils.db.base import BaseDatabaseConnector
from utils.db.mysql.vector_handler import (
    create_mysql_vector_table,
    insert_mysql_vector,
    insert_mysql_vectors,
    search_mysql_vectors,
    delete_mysql_vector,
)
from utils.env import get_database_url, get_secret
from utils.logging import get_logger

logger = get_logger("MySQLConnector")


class MySQLConnector(BaseDatabaseConnector):
    """Connector for MySQL relational databases configured via a single connection URL."""

    def __init__(self, connection_string: Optional[str] = None, charset: str = "utf8mb4"):
        self.connection_string = (
            connection_string
            or get_secret("MYSQL_URL", raise_error=False)
            or get_database_url(raise_error=False)
        )
        self.charset = charset
        self._connection = None
        self._in_transaction: bool = False

    def connect(self):
        if not PYMYSQL_AVAILABLE:
            message = "pymysql is not installed. Please install 'pymysql' to use MySQLConnector."
            logger.error(message)
            raise ImportError(message)

        if self._connection is None or not self._connection.open:
            if not self.connection_string:
                raise ValueError("No MySQL connection URL provided (set MYSQL_URL or DATABASE_URL).")

            logger.info("Connecting to MySQL database via connection URL...")
            parsed = urlparse(self.connection_string)
            host = parsed.hostname or "localhost"
            port = parsed.port or 3306
            user = unquote(parsed.username) if parsed.username else None
            password = unquote(parsed.password) if parsed.password else None
            database = parsed.path.lstrip("/") if parsed.path else None

            kwargs = {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "database": database,
                "charset": self.charset,
                "cursorclass": pymysql.cursors.DictCursor,
                "autocommit": False,
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            self._connection = pymysql.connect(**filtered_kwargs)
            logger.info("Successfully connected to MySQL database.")
        return self._connection

    def close(self) -> None:
        if self._connection is not None and self._connection.open:
            logger.info("Closing MySQL connection...")
            self._connection.close()
            self._connection = None
            logger.info("MySQL connection closed.")

    def is_connected(self) -> bool:
        if not PYMYSQL_AVAILABLE or self._connection is None or not self._connection.open:
            return False
        try:
            self._connection.ping(reconnect=False)
            return True
        except Exception:
            return False

    def execute_query(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        conn = self.connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                return list(cursor.fetchall())
        except Exception as e:
            logger.error(f"MySQL query execution failed: {e}")
            raise

    def execute_non_query(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> int:
        conn = self.connect()
        try:
            with conn.cursor() as cursor:
                rowcount = cursor.execute(query, params or ())
            if not self._in_transaction:
                conn.commit()
            return rowcount
        except Exception as e:
            if not self._in_transaction:
                conn.rollback()
            logger.error(f"MySQL non-query execution failed: {e}")
            raise

    def execute_many(self, query: str, params_list: List[Union[Tuple[Any, ...], Dict[str, Any]]]) -> int:
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

    def fetch_one(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                row = cursor.fetchone()
                return row if row else None
        except Exception as e:
            logger.error(f"MySQL fetch_one failed: {e}")
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
            logger.error(f"MySQL transaction failed and rolled back: {e}")
            raise
        finally:
            self._in_transaction = was_in_transaction

    # Vector operations
    def create_vector_table(self, table_name: str, vector_dim: int, distance_metric: str = "cosine") -> None:
        create_mysql_vector_table(self, table_name, vector_dim, distance_metric)

    def insert_vector(self, table_name: str, vector_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
        return insert_mysql_vector(self, table_name, vector_id, vector, metadata)

    def insert_vectors(self, table_name: str, records: List[Dict[str, Any]]) -> int:
        return insert_mysql_vectors(self, table_name, records)

    def vector_search(self, table_name: str, query_vector: List[float], top_k: int = 10, min_score: Optional[float] = None, distance_metric: str = "cosine") -> List[Dict[str, Any]]:
        return search_mysql_vectors(self, table_name, query_vector, top_k=top_k, min_score=min_score, distance_metric=distance_metric)

    def delete_vector(self, table_name: str, vector_id: str) -> int:
        return delete_mysql_vector(self, table_name, vector_id)
