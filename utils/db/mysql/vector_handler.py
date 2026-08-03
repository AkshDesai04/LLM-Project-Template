"""
MySQL Vector Handler implementation (supporting JSON vector storage & ranking).
"""

import json
from typing import Any, Dict, List, Optional
from utils.db.vector_utils import (
    serialize_vector,
    deserialize_vector,
    filter_and_rank_vectors,
)
from utils.logging import get_logger

logger = get_logger("MySQLVectorHandler")


def create_mysql_vector_table(connector: Any, table_name: str, vector_dim: int, distance_metric: str = "cosine") -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id VARCHAR(255) PRIMARY KEY,
        vector JSON NOT NULL,
        metadata JSON
    );
    """
    connector.execute_non_query(ddl)
    logger.info(f"MySQL vector table '{table_name}' created.")


def insert_mysql_vector(connector: Any, table_name: str, vector_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
    query = f"""
    INSERT INTO {table_name} (id, vector, metadata)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE vector = VALUES(vector), metadata = VALUES(metadata);
    """
    vec_json = serialize_vector(vector)
    meta_json = json.dumps(metadata) if metadata else None
    return connector.execute_non_query(query, (vector_id, vec_json, meta_json))


def insert_mysql_vectors(connector: Any, table_name: str, records: List[Dict[str, Any]]) -> int:
    query = f"""
    INSERT INTO {table_name} (id, vector, metadata)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE vector = VALUES(vector), metadata = VALUES(metadata);
    """
    params_list = []
    for r in records:
        meta_json = json.dumps(r.get("metadata")) if r.get("metadata") else None
        vec_json = serialize_vector(r.get("vector", []))
        params_list.append((str(r["id"]), vec_json, meta_json))
    return connector.execute_many(query, params_list)


def search_mysql_vectors(
    connector: Any,
    table_name: str,
    query_vector: List[float],
    top_k: int = 10,
    min_score: Optional[float] = None,
    distance_metric: str = "cosine",
) -> List[Dict[str, Any]]:
    rows = connector.execute_query(f"SELECT id, vector, metadata FROM {table_name};")
    records = []
    for r in rows:
        meta = r.get("metadata")
        if isinstance(meta, str):
            try: meta = json.loads(meta)
            except Exception: pass
        records.append({
            "id": r["id"],
            "vector": deserialize_vector(r["vector"]),
            "metadata": meta,
        })
    return filter_and_rank_vectors(records, query_vector, top_k=top_k, min_score=min_score, distance_metric=distance_metric)


def delete_mysql_vector(connector: Any, table_name: str, vector_id: str) -> int:
    return connector.execute_non_query(f"DELETE FROM {table_name} WHERE id = %s;", (vector_id,))
