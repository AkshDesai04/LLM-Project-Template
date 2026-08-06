"""
Unit tests for vector math and database vector utility helpers (utils.db.vector_utils).
"""

from unittest.mock import MagicMock
import pytest

from utils.db.vector_utils import (
    compute_similarity_score,
    cosine_similarity,
    deserialize_vector,
    dot_product,
    filter_and_rank_vectors,
    generic_create_vector_table,
    generic_delete_vector,
    generic_insert_vector,
    generic_insert_vectors,
    generic_vector_search,
    l2_distance,
    norm,
    serialize_vector,
)


@pytest.mark.db
@pytest.mark.unit
def test_vector_serialization():
    vec = [1.0, 2.5, 3.0]
    s = serialize_vector(vec)
    assert s == "[1.0, 2.5, 3.0]"

    # Deserialization from list
    assert deserialize_vector([1, 2, 3]) == [1.0, 2.0, 3.0]

    # Deserialization from json string
    assert deserialize_vector("[1.0, 2.0]") == [1.0, 2.0]

    # Deserialization invalid
    assert deserialize_vector("invalid json") == []
    assert deserialize_vector(None) == []


@pytest.mark.db
@pytest.mark.unit
def test_vector_math():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    v3 = [1.0, 1.0]

    assert dot_product(v1, v2) == 0.0
    assert dot_product(v1, v3) == 1.0

    assert norm([3.0, 4.0]) == 5.0
    assert cosine_similarity(v1, v2) == 0.0
    assert cosine_similarity(v1, v1) == 1.0
    assert cosine_similarity([0.0, 0.0], v1) == 0.0

    assert l2_distance([0.0, 0.0], [3.0, 4.0]) == 5.0


@pytest.mark.db
@pytest.mark.unit
def test_compute_similarity_score():
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]

    # Cosine score normalized to [0, 1]
    assert compute_similarity_score(v1, v2, metric="cosine") == 1.0

    # L2 distance score: 1 / (1 + 0) = 1.0
    assert compute_similarity_score(v1, v2, metric="l2") == 1.0

    # Dot product score
    assert compute_similarity_score(v1, v2, metric="dot") == 1.0


@pytest.mark.db
@pytest.mark.unit
def test_filter_and_rank_vectors():
    records = [
        {"id": "a", "vector": [1.0, 0.0]},
        {"id": "b", "vector": [0.8, 0.2]},
        {"id": "c", "vector": [0.0, 1.0]},
        {"id": "d", "vector": []},  # Should be skipped
    ]

    # Top 2
    res = filter_and_rank_vectors(records, [1.0, 0.0], top_k=2, distance_metric="cosine")
    assert len(res) == 2
    assert res[0]["id"] == "a"
    assert res[1]["id"] == "b"

    # min_score filter
    res_min = filter_and_rank_vectors(records, [1.0, 0.0], top_k=10, min_score=0.9, distance_metric="cosine")
    assert len(res_min) == 2  # a and b (both >= 0.9 normalized score)


@pytest.mark.db
@pytest.mark.unit
def test_generic_crud_helpers():
    mock_conn = MagicMock()

    # Create table
    generic_create_vector_table(mock_conn, "vec_tbl", vector_dim=128)
    mock_conn.execute_non_query.assert_called_once()

    # Insert single
    mock_conn.reset_mock()
    generic_insert_vector(mock_conn, "vec_tbl", "id_1", [1.0, 2.0], {"meta": "val"})
    mock_conn.execute_non_query.assert_called_once()

    # Insert batch
    mock_conn.reset_mock()
    batch = [{"id": "id_2", "vector": [0.1, 0.2], "metadata": {"x": 1}}]
    generic_insert_vectors(mock_conn, "vec_tbl", batch)
    mock_conn.execute_many.assert_called_once()

    # Search
    mock_conn.reset_mock()
    mock_conn.execute_query.return_value = [
        {"id": "id_1", "vector": "[1.0, 0.0]", "metadata": '{"key": "val"}'}
    ]
    results = generic_vector_search(mock_conn, "vec_tbl", [1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "id_1"

    # Delete
    mock_conn.reset_mock()
    generic_delete_vector(mock_conn, "vec_tbl", "id_1")
    mock_conn.execute_non_query.assert_called_once()
