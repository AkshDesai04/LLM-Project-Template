"""
OracleDB Database Connector.
"""

import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote

try:
    import oracledb
    ORACLEDB_AVAILABLE = True
except ImportError:
    ORACLEDB_AVAILABLE = False
    oracledb = None

from utils.db.base import BaseDatabaseConnector
from utils.env import get_database_url, get_secret
from utils.logging import get_logger

logger = get_logger("OracleDBConnector")


class OracleDBConnector(BaseDatabaseConnector):
    """Connector for Oracle Database instances configured via a single connection URL or DSN."""

    def __init__(self, connection_string: Optional[str] = None, dsn: Optional[str] = None):
        url = (
            connection_string
            or dsn
            or get_secret("ORACLE_URL", raise_error=False)
            or get_secret("ORACLE_DSN", raise_error=False)
            or get_database_url(raise_error=False)
        )
        parsed_user, parsed_pass, parsed_dsn = None, None, None
        if url:
            clean_url = url[9:] if url.startswith("oracle://") else url
            if "@" in clean_url:
                creds, target = clean_url.split("@", 1)
                if ":" in creds:
                    parsed_user, parsed_pass = creds.split(":", 1)
                    parsed_user = unquote(parsed_user)
                    parsed_pass = unquote(parsed_pass)
                else:
                    parsed_user = unquote(creds)
                parsed_dsn = target
            else:
                parsed_dsn = clean_url

        self.user = parsed_user
        self.password = parsed_pass
        self.dsn = parsed_dsn
        self.connection_string = url
        self._connection = None
        self._in_transaction: bool = False

    def connect(self):
        if not ORACLEDB_AVAILABLE:
            message = "oracledb is not installed. Please install 'oracledb' to use OracleDBConnector."
            logger.error(message)
            raise ImportError(message)

        if self._connection is None:
            if not self.dsn and not self.connection_string:
                raise ValueError("No Oracle connection URL or DSN provided (set ORACLE_URL, ORACLE_DSN, or DATABASE_URL).")
            logger.info("Connecting to Oracle Database via connection URL/DSN...")
            if self.user and self.password:
                self._connection = oracledb.connect(user=self.user, password=self.password, dsn=self.dsn)
            else:
                self._connection = oracledb.connect(dsn=self.dsn)
            logger.info("Successfully connected to Oracle Database.")
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            logger.info("Closing Oracle connection...")
            self._connection.close()
            self._connection = None
            logger.info("Oracle connection closed.")

    def is_connected(self) -> bool:
        if not ORACLEDB_AVAILABLE or self._connection is None:
            return False
        try:
            return self._connection.ping() is None
        except Exception:
            return False

    def execute_query(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        conn = self.connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                columns = [col[0].lower() for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"OracleDB query execution failed: {e}")
            raise

    def execute_non_query(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> int:
        conn = self.connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                rowcount = cursor.rowcount
            if not self._in_transaction:
                conn.commit()
            return rowcount
        except Exception as e:
            if not self._in_transaction:
                conn.rollback()
            logger.error(f"OracleDB non-query execution failed: {e}")
            raise

    def execute_many(self, query: str, params_list: List[Union[Tuple[Any, ...], Dict[str, Any]]]) -> int:
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
            logger.error(f"OracleDB executemany failed: {e}")
            raise

    def fetch_one(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                if not cursor.description:
                    return None
                columns = [col[0].lower() for col in cursor.description]
                row = cursor.fetchone()
                return dict(zip(columns, row)) if row else None
        except Exception as e:
            logger.error(f"OracleDB fetch_one failed: {e}")
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
            logger.error(f"OracleDB transaction failed and rolled back: {e}")
            raise
        finally:
            self._in_transaction = was_in_transaction
