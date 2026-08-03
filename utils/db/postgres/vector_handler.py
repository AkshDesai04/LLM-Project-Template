"""
PostgreSQL Vector Handler implementation (supporting pgvector and fallback math).
"""

import json
from typing import Any, Dict, List, Optional
from utils.db.vector_utils import (
    serialize_vector,
    deserialize_vector,
    generic_insert_vector,
    generic_insert_vectors,
    generic_vector_search,
    generic_delete_vector,
)
from utils.logging import get_logger

logger = get_logger("PostgreSQLVectorHandler")

POSTGRES_UPSERT_SQL = """
INSERT INTO {table_name} (id, vector, metadata)
VALUES (%s, %s, %s)
ON CONFLICT (id) DO UPDATE SET vector = EXCLUDED.vector, metadata = EXCLUDED.metadata;
"""


def create_postgres_vector_table(connector: Any, table_name: str, vector_dim: int, distance_metric: str = "cosine") -> None:
    try:
        connector.execute_non_query("CREATE EXTENSION IF NOT EXISTS vector;")
        vector_type = f"vector({vector_dim})"
    except Exception as e:
        logger.warning(f"pgvector extension not enabled on Postgres; using TEXT storage: {e}")
        vector_type = "TEXT"

    ddl = f"CREATE TABLE IF NOT EXISTS {table_name} (id VARCHAR(255) PRIMARY KEY, vector {vector_type} NOT NULL, metadata JSONB);"
    connector.execute_non_query(ddl)
    logger.info(f"PostgreSQL vector table '{table_name}' created.")


def insert_postgres_vector(connector: Any, table_name: str, vector_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
    sql = POSTGRES_UPSERT_SQL.format(table_name=table_name)
    return generic_insert_vector(connector, table_name, vector_id, vector, metadata, param_style="%s", upsert_sql=sql)


def insert_postgres_vectors(connector: Any, table_name: str, records: List[Dict[str, Any]]) -> int:
    sql = POSTGRES_UPSERT_SQL.format(table_name=table_name)
    return generic_insert_vectors(connector, table_name, records, param_style="%s", upsert_sql=sql)


def search_postgres_vectors(
    connector: Any,
    table_name: str,
    query_vector: List[float],
    top_k: int = 10,
    min_score: Optional[float] = None,
    distance_metric: str = "cosine",
) -> List[Dict[str, Any]]:
    try:
        vec_str = serialize_vector(query_vector)
        op = "<=>" if distance_metric == "cosine" else ("<->" if distance_metric in ("l2", "euclidean") else "<#>")
        score_expr = f"1 - (vector {op} %s::vector)" if distance_metric == "cosine" else f"1 / (1 + (vector {op} %s::vector))"

        sql = f"SELECT id, vector, metadata, ({score_expr}) AS score FROM {table_name}"
        params = [vec_str]

        if min_score is not None:
            sql += f" WHERE ({score_expr}) >= %s"
            params.append(min_score)

        sql += " ORDER BY score DESC LIMIT %s;"
        params.append(top_k)

        rows = connector.execute_query(sql, tuple(params))
        results = []
        for r in rows:
            meta = r.get("metadata")
            if isinstance(meta, str):
                try: meta = json.loads(meta)
                except Exception: pass
            results.append({
                "id": r["id"],
                "vector": deserialize_vector(r["vector"]),
                "metadata": meta,
                "score": float(r["score"]),
            })
        return results
    except Exception as e:
        logger.warning(f"Native pgvector search failed ({e}); falling back to generic python vector search...")
        return generic_vector_search(connector, table_name, query_vector, top_k=top_k, min_score=min_score, distance_metric=distance_metric)


def delete_postgres_vector(connector: Any, table_name: str, vector_id: str) -> int:
    return generic_delete_vector(connector, table_name, vector_id, param_style="%s")
