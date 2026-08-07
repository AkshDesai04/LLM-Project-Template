"""
Vector math, indexing, and generic CRUD operations for database vector search.
"""

import json
import math
from typing import Any, Dict, List, Optional


def serialize_vector(vector: List[float]) -> str:
    return json.dumps([float(x) for x in vector])


def deserialize_vector(val: Any) -> List[float]:
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except Exception:
            pass
    return []


def dot_product(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


def norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    n1, n2 = norm(v1), norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot_product(v1, v2) / (n1 * n2)))


def l2_distance(v1: List[float], v2: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def compute_similarity_score(v1: List[float], v2: List[float], metric: str = "cosine") -> float:
    metric = metric.lower()
    if metric == "cosine":
        return (cosine_similarity(v1, v2) + 1.0) / 2.0
    elif metric in ("l2", "euclidean"):
        return 1.0 / (1.0 + l2_distance(v1, v2))
    elif metric == "dot":
        return dot_product(v1, v2)
    return (cosine_similarity(v1, v2) + 1.0) / 2.0


def filter_and_rank_vectors(
    records: List[Dict[str, Any]],
    query_vector: List[float],
    top_k: int = 10,
    min_score: Optional[float] = None,
    distance_metric: str = "cosine",
) -> List[Dict[str, Any]]:
    results = []
    for r in records:
        v = r.get("vector", [])
        if not v:
            continue
        score = compute_similarity_score(query_vector, v, metric=distance_metric)
        if min_score is not None and score < min_score:
            continue
        rec = dict(r)
        rec["score"] = score
        results.append(rec)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# Generic DB Vector CRUD Operations

def generic_create_vector_table(connector: Any, table_name: str, vector_dim: int, distance_metric: str = "cosine", vector_type: str = "TEXT") -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id VARCHAR(255) PRIMARY KEY,
        vector {vector_type} NOT NULL,
        metadata TEXT,
        vector_dim INTEGER DEFAULT {vector_dim},
        distance_metric VARCHAR(50) DEFAULT '{distance_metric}'
    );
    """
    connector.execute_non_query(ddl)


def generic_insert_vector(connector: Any, table_name: str, vector_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None, param_style: str = "?", upsert_sql: Optional[str] = None) -> int:
    ph = param_style
    query = upsert_sql or f"INSERT OR REPLACE INTO {table_name} (id, vector, metadata) VALUES ({ph}, {ph}, {ph});"
    meta_str = json.dumps(metadata) if metadata else None
    vec_str = serialize_vector(vector)
    return connector.execute_non_query(query, (vector_id, vec_str, meta_str))


def generic_insert_vectors(connector: Any, table_name: str, records: List[Dict[str, Any]], param_style: str = "?", upsert_sql: Optional[str] = None) -> int:
    ph = param_style
    query = upsert_sql or f"INSERT OR REPLACE INTO {table_name} (id, vector, metadata) VALUES ({ph}, {ph}, {ph});"
    params_list = []
    for r in records:
        meta_str = json.dumps(r.get("metadata")) if r.get("metadata") else None
        vec_str = serialize_vector(r.get("vector", []))
        params_list.append((str(r["id"]), vec_str, meta_str))
    return connector.execute_many(query, params_list)


def generic_vector_search(connector: Any, table_name: str, query_vector: List[float], top_k: int = 10, min_score: Optional[float] = None, distance_metric: str = "cosine") -> List[Dict[str, Any]]:
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


def generic_delete_vector(connector: Any, table_name: str, vector_id: str, param_style: str = "?") -> int:
    return connector.execute_non_query(f"DELETE FROM {table_name} WHERE id = {param_style};", (vector_id,))
