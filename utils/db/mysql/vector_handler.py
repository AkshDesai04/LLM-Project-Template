"""
MySQL Vector Handler implementation.
"""

from typing import Any, Dict, List, Optional
from utils.db.vector_utils import (
    generic_create_vector_table,
    generic_insert_vector,
    generic_insert_vectors,
    generic_vector_search,
    generic_delete_vector,
)

MYSQL_UPSERT_SQL = """
INSERT INTO {table_name} (id, vector, metadata)
VALUES (%s, %s, %s)
ON DUPLICATE KEY UPDATE vector = VALUES(vector), metadata = VALUES(metadata);
"""


def create_mysql_vector_table(connector: Any, table_name: str, vector_dim: int, distance_metric: str = "cosine") -> None:
    generic_create_vector_table(connector, table_name, vector_dim, distance_metric, vector_type="JSON")


def insert_mysql_vector(connector: Any, table_name: str, vector_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
    sql = MYSQL_UPSERT_SQL.format(table_name=table_name)
    return generic_insert_vector(connector, table_name, vector_id, vector, metadata, param_style="%s", upsert_sql=sql)


def insert_mysql_vectors(connector: Any, table_name: str, records: List[Dict[str, Any]]) -> int:
    sql = MYSQL_UPSERT_SQL.format(table_name=table_name)
    return generic_insert_vectors(connector, table_name, records, param_style="%s", upsert_sql=sql)


def search_mysql_vectors(
    connector: Any,
    table_name: str,
    query_vector: List[float],
    top_k: int = 10,
    min_score: Optional[float] = None,
    distance_metric: str = "cosine",
) -> List[Dict[str, Any]]:
    return generic_vector_search(connector, table_name, query_vector, top_k=top_k, min_score=min_score, distance_metric=distance_metric)


def delete_mysql_vector(connector: Any, table_name: str, vector_id: str) -> int:
    return generic_delete_vector(connector, table_name, vector_id, param_style="%s")
