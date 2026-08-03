"""
Vector math and utility functions for database vector search.
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple


def serialize_vector(vector: List[float]) -> str:
    """Serializes a float list vector into a JSON string representation."""
    return json.dumps([float(x) for x in vector])


def deserialize_vector(val: Any) -> List[float]:
    """Deserializes a JSON string, tuple, or list into a float list vector."""
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
    """Computes the dot product of two vectors."""
    return sum(a * b for a, b in zip(v1, v2))


def norm(v: List[float]) -> float:
    """Computes the Euclidean norm (magnitude) of a vector."""
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two vectors (range: [-1.0, 1.0], normalized [0, 1])."""
    n1, n2 = norm(v1), norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    dot = dot_product(v1, v2)
    sim = dot / (n1 * n2)
    return max(-1.0, min(1.0, sim))


def l2_distance(v1: List[float], v2: List[float]) -> float:
    """Computes Euclidean (L2) distance between two vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def compute_similarity_score(v1: List[float], v2: List[float], metric: str = "cosine") -> float:
    """
    Computes a normalized similarity score (0.0 to 1.0) based on metric.

    Supported metrics:
    - 'cosine': Cosine similarity mapped to [0, 1] via (sim + 1) / 2
    - 'l2' / 'euclidean': 1.0 / (1.0 + l2_distance)
    - 'dot': Dot product
    """
    metric = metric.lower()
    if metric == "cosine":
        raw_sim = cosine_similarity(v1, v2)
        return (raw_sim + 1.0) / 2.0
    elif metric in ("l2", "euclidean"):
        dist = l2_distance(v1, v2)
        return 1.0 / (1.0 + dist)
    elif metric == "dot":
        return dot_product(v1, v2)
    else:
        raw_sim = cosine_similarity(v1, v2)
        return (raw_sim + 1.0) / 2.0


def filter_and_rank_vectors(
    records: List[Dict[str, Any]],
    query_vector: List[float],
    top_k: int = 10,
    min_score: Optional[float] = None,
    distance_metric: str = "cosine",
) -> List[Dict[str, Any]]:
    """
    Filters records by minimum score threshold and returns top_k ranked items.
    """
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
