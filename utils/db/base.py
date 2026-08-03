"""
Base Abstract Class for Database Connectors.

Defines a common, interoperable interface across relational and document
databases.
"""

from abc import ABC, abstractmethod
import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.logging import get_logger

logger = get_logger("BaseDatabaseConnector")


class BaseDatabaseConnector(ABC):
    """
    Abstract Base Class for all database connectors.

    Provides standard method signatures for connecting, executing queries,
    fetching records, managing transactions, and closing connections cleanly.
    """

    @abstractmethod
    def connect(self) -> Any:
        """
        Establishes connection to the database.

        Returns:
            The native connection or client object.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the active database connection or client session."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Checks whether the database connection is currently active and healthy.

        Returns:
            bool: True if connected and responsive, False otherwise.
        """
        pass

    @abstractmethod
    def execute_query(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a read/fetch query (e.g. SELECT) and returns all results as dictionaries.

        Args:
            query: The SQL or query string to execute.
            params: Parameters to bind to the query.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing result rows.
        """
        pass

    @abstractmethod
    def execute_non_query(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> int:
        """
        Executes a write/mutation query (e.g. INSERT, UPDATE, DELETE, DDL).

        Args:
            query: The SQL or command string to execute.
            params: Parameters to bind to the statement.

        Returns:
            int: The count of affected rows/records.
        """
        pass

    @abstractmethod
    def execute_many(
        self, query: str, params_list: List[Union[Tuple[Any, ...], Dict[str, Any]]]
    ) -> int:
        """
        Executes a query repeatedly with a list of parameter sets.

        Args:
            query: The SQL query template.
            params_list: Sequence of parameter tuples or dictionaries.

        Returns:
            int: Total number of affected rows/records across all executions.
        """
        pass

    @abstractmethod
    def fetch_one(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a query and returns the first row as a dictionary.

        Args:
            query: The SQL query string.
            params: Parameters to bind to the query.

        Returns:
            Optional[Dict[str, Any]]: A dictionary representing the first result row, or None.
        """
        pass

    @abstractmethod
    @contextlib.contextmanager
    def transaction(self):
        """
        Context manager for executing operations inside a transaction block.
        Automatically commits changes on exit or rolls back on exception.
        """
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
