# `utils/db/` Database Connectors & Vector Handlers

This package provides a unified abstract interface (`BaseDatabaseConnector`), a factory router (`DatabaseRouter`), and dialect-specific implementations for **SQLite**, **PostgreSQL**, and **MySQL** with native & fallback vector similarity search support.

---

## Core Principles & Architecture

1. **200 Lines Limit:** Every file in this package strictly stays under 200 lines.
2. **Absolute Imports:** All internal module imports use absolute paths (`from utils.db.base import BaseDatabaseConnector`).
3. **Unified Secret Management:** Credentials resolve dynamically via `utils.env` (honoring `.env` or AWS Secrets Manager).

---

## Package Directory Structure

| Module / Directory | Class / Functions | Description |
|---|---|---|
| [`base.py`](file:///e:/Code/Personal/LLM-Project-Template/utils/db/base.py) | `BaseDatabaseConnector` | Abstract Base Class defining standard query execution, transaction management, and vector CRUD signatures. |
| [`router.py`](file:///e:/Code/Personal/LLM-Project-Template/utils/db/router.py) | `DatabaseRouter` | Unified factory entry point resolving dialect aliases (`postgres`, `postgresql`, `pg`, `mysql`, `sqlite`) or connection URIs. |
| [`helpers.py`](file:///e:/Code/Personal/LLM-Project-Template/utils/db/helpers.py) | `rows_to_dicts`, `execute_sql_query`, `execute_sql_non_query`, `execute_sql_many` | Low-level SQL helper routines with normalized `rowcount` reporting across drivers. |
| [`vector_utils.py`](file:///e:/Code/Personal/LLM-Project-Template/utils/db/vector_utils.py) | `dot_product`, `cosine_similarity`, `l2_distance`, `generic_vector_search`, etc. | Offline vector math, similarity scoring, ranking algorithms, and generic fallback table CRUD operations. |
| `postgres/` | `PostgreSQLConnector`, `PostgreSQLVectorHandler` | PostgreSQL connector with native `pgvector` support, division-by-zero handling, and automatic transaction rollback on fallback. |
| `mysql/` | `MySQLConnector`, `MySQLVectorHandler` | MySQL connector with connection URI query parameter parsing (`parse_qs`) and JSON-based vector storage fallback. |
| `sqlite/` | `SQLiteConnector`, `SQLiteVectorHandler` | SQLite connector with shared in-memory database support (`file::memory:?cache=shared`) and text-encoded vector storage fallback. |

---

## Supported Methods (`BaseDatabaseConnector`)

- **Connection Lifecycle**: `connect()`, `close()`, `is_connected()`, `__enter__()`, `__exit__()`
- **Query Execution**:
  - `execute_query(query, params=None)` -> `List[Dict[str, Any]]`
  - `fetch_one(query, params=None)` -> `Optional[Dict[str, Any]]`
  - `execute_non_query(query, params=None)` -> `int`
  - `execute_many(query, params_list)` -> `int`
- **Transactions**: `transaction()` (context manager committing on success and rolling back on exception)
- **Vector Operations**:
  - `create_vector_table(table_name, vector_dim, distance_metric="cosine")`
  - `insert_vector(table_name, vector_id, vector, metadata=None)`
  - `insert_vectors(table_name, records)`
  - `vector_search(table_name, query_vector, top_k=10, min_score=None, distance_metric="cosine")`
  - `delete_vector(table_name, vector_id)`

---

## Connection Routing Usage Example

```python
from utils.db import DatabaseRouter

# Route by dialect string or connection URI
connector = DatabaseRouter.get_connector("postgresql://user:pass@localhost:5432/testdb")

with connector as db:
    rows = db.execute_query("SELECT * FROM categories WHERE id = %s;", (1,))
    db.create_vector_table("embeddings", vector_dim=1536)
    db.insert_vector("embeddings", "doc_1", [0.1] * 1536, {"title": "Doc 1"})
    results = db.vector_search("embeddings", [0.1] * 1536, top_k=5)
```
