"""
OracleDB Database Connector.

Provides a robust Oracle database connector using modern python-oracledb with support for
connection parameters, environment variable fallbacks, dictionary row mapping, and transactions.
"""

import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import oracledb
    ORACLEDB_AVAILABLE = True
except ImportError:
    ORACLEDB_AVAILABLE = False
    oracledb = None

from utils.base_connector import BaseDatabaseConnector
from utils.env_ops import get_secret
from utils.logger import get_logger

logger = get_logger("OracleDBConnector")


class OracleDBConnector(BaseDatabaseConnector):
    """
    Connector for Oracle Database instances.

    Attributes:
        user (str): Oracle DB username.
        password (str): Oracle DB password.
        dsn (str): Connection Data Source Name or EZConnect string.
        host (str): Host address.
        port (int): Port number (default: 1521).
        service_name (str): Oracle service name.
    """

    def __init__(
        self,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dsn: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        service_name: Optional[str] = None,
    ):
        """
        Initializes OracleDB connector parameters.
        Parameters fall back to environment variables via utils.env_ops if omitted.
        """
        self.user = user or get_secret("ORACLE_USER", raise_error=False)
        self.password = password or get_secret("ORACLE_PASSWORD", raise_error=False)
        self.host = host or get_secret("ORACLE_HOST", raise_error=False) or "localhost"
        self.port = port or int(get_secret("ORACLE_PORT", raise_error=False) or 1521)
        self.service_name = service_name or get_secret("ORACLE_SERVICE_NAME", raise_error=False)
        self.dsn = dsn or get_secret("ORACLE_DSN", raise_error=False)

        if not self.dsn and self.host and self.service_name:
            self.dsn = f"{self.host}:{self.port}/{self.service_name}"

        self._connection = None

    def connect(self):
        """
        Establishes connection to Oracle Database.

        Returns:
            oracledb.Connection: Active Oracle connection object.
        """
        if not ORACLEDB_AVAILABLE:
            message = "oracledb is not installed. Please install 'oracledb' to use OracleDBConnector."
            logger.error(message)
            raise ImportError(message)

        if self._connection is None:
            logger.info("Connecting to Oracle Database...")
            kwargs = {
                "user": self.user,
                "password": self.password,
                "dsn": self.dsn,
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            self._connection = oracledb.connect(**filtered_kwargs)
            logger.info("Successfully connected to Oracle Database.")
        return self._connection

    def close(self) -> None:
        """Closes active Oracle connection."""
        if self._connection is not None:
            logger.info("Closing Oracle connection...")
            self._connection.close()
            self._connection = None
            logger.info("Oracle connection closed.")

    def is_connected(self) -> bool:
        """
        Checks whether Oracle connection is healthy.

        Returns:
            bool: True if connected and responsive, False otherwise.
        """
        if not ORACLEDB_AVAILABLE or self._connection is None:
            return False
        try:
            return self._connection.ping() is None
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
                columns = [col[0].lower() for col in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"OracleDB query execution failed: {e}")
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
            conn.commit()
            return rowcount
        except Exception as e:
            conn.rollback()
            logger.error(f"OracleDB non-query execution failed: {e}")
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
            conn.commit()
            return rowcount
        except Exception as e:
            conn.rollback()
            logger.error(f"OracleDB executemany failed: {e}")
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
        """
        Context manager for OracleDB transaction safety.
        """
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"OracleDB transaction failed and rolled back: {e}")
            raise
