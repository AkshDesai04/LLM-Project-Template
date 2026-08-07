"""
Base Abstract Class for Database Connectors with Native Vector Search Support.
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
    fetching records, managing transactions, closing connections cleanly,
    and performing vector search operations.
    """

    @abstractmethod
    def connect(self) -> Any:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def execute_query(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def execute_non_query(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> int:
        pass

    @abstractmethod
    def execute_many(
        self, query: str, params_list: List[Union[Tuple[Any, ...], Dict[str, Any]]]
    ) -> int:
        pass

    @abstractmethod
    def fetch_one(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    @contextlib.contextmanager
    def transaction(self):
        pass

    # Vector Storage & Search Operations

    @abstractmethod
    def create_vector_table(self, table_name: str, vector_dim: int, distance_metric: str = "cosine") -> None:
        """Creates a table structured for storing vectors and metadata."""
        pass

    @abstractmethod
    def insert_vector(
        self, table_name: str, vector_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Inserts or replaces a single vector record."""
        pass

    @abstractmethod
    def insert_vectors(self, table_name: str, records: List[Dict[str, Any]]) -> int:
        """Batch inserts multiple vector records."""
        pass

    @abstractmethod
    def vector_search(
        self,
        table_name: str,
        query_vector: List[float],
        top_k: int = 10,
        min_score: Optional[float] = None,
        distance_metric: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """
        Executes vector similarity search using top_k and/or minimum score filtering.
        """
        pass

    @abstractmethod
    def delete_vector(self, table_name: str, vector_id: str) -> int:
        """Deletes a vector record by ID."""
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
