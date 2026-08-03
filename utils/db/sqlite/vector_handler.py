"""
SQLite Vector Handler implementation.
"""

from typing import Any, Dict, List, Optional
from utils.db.vector_utils import (
    generic_create_vector_table,
    generic_insert_vector,
    generic_insert_vectors,
    generic_vector_search,
    generic_delete_vector,
)


def create_sqlite_vector_table(connector: Any, table_name: str, vector_dim: int, distance_metric: str = "cosine") -> None:
    generic_create_vector_table(connector, table_name, vector_dim, distance_metric, vector_type="TEXT")


def insert_sqlite_vector(connector: Any, table_name: str, vector_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
    return generic_insert_vector(connector, table_name, vector_id, vector, metadata, param_style="?")


def insert_sqlite_vectors(connector: Any, table_name: str, records: List[Dict[str, Any]]) -> int:
    return generic_insert_vectors(connector, table_name, records, param_style="?")


def search_sqlite_vectors(
    connector: Any,
    table_name: str,
    query_vector: List[float],
    top_k: int = 10,
    min_score: Optional[float] = None,
    distance_metric: str = "cosine",
) -> List[Dict[str, Any]]:
    return generic_vector_search(connector, table_name, query_vector, top_k=top_k, min_score=min_score, distance_metric=distance_metric)


def delete_sqlite_vector(connector: Any, table_name: str, vector_id: str) -> int:
    return generic_delete_vector(connector, table_name, vector_id, param_style="?")
