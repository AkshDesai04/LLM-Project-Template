"""
Unit tests for Database Connector Utilities & DatabaseRouter.

Verifies exports, single connection URL resolution, SQLite functional operations,
vector search (top_k and min_score filtering), context managers, and DatabaseRouter.
"""

import pytest

from utils import (
    BaseDatabaseConnector,
    DatabaseRouter,
    MySQLConnector,
    PostgreSQLConnector,
    SQLiteConnector,
)


@pytest.mark.db
@pytest.mark.unit
def test_exports():
    """Verify all database connectors export correctly from utils."""
    assert issubclass(SQLiteConnector, BaseDatabaseConnector)
    assert issubclass(PostgreSQLConnector, BaseDatabaseConnector)
    assert issubclass(MySQLConnector, BaseDatabaseConnector)


@pytest.mark.db
@pytest.mark.unit
def test_sqlite_functional():
    """Test full functional lifecycle of SQLiteConnector using an in-memory database."""
    with SQLiteConnector(db_path=":memory:") as db:
        assert db.is_connected() is True

        create_table_sql = """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL
        );
        """
        db.execute_non_query(create_table_sql)

        insert_sql = "INSERT INTO users (name, email) VALUES (?, ?);"
        affected = db.execute_non_query(insert_sql, ("Alice", "alice@example.com"))
        assert affected == 1

        users_batch = [
            ("Bob", "bob@example.com"),
            ("Charlie", "charlie@example.com"),
        ]
        batch_affected = db.execute_many(insert_sql, users_batch)
        assert batch_affected == 2

        rows = db.execute_query("SELECT * FROM users ORDER BY id ASC;")
        assert len(rows) == 3
        assert rows[0] == {"id": 1, "name": "Alice", "email": "alice@example.com"}

        row = db.fetch_one("SELECT * FROM users WHERE email = ?;", ("charlie@example.com",))
        assert row is not None
        assert row["name"] == "Charlie"

        with db.transaction():
            db.execute_non_query("INSERT INTO users (name, email) VALUES (?, ?);", ("David", "david@example.com"))

        david = db.fetch_one("SELECT * FROM users WHERE name = ?;", ("David",))
        assert david is not None

        try:
            with db.transaction():
                db.execute_non_query("INSERT INTO users (name, email) VALUES (?, ?);", ("Eve", "eve@example.com"))
                raise ValueError("Simulated transaction error")
        except ValueError:
            pass

        eve = db.fetch_one("SELECT * FROM users WHERE name = ?;", ("Eve",))
        assert eve is None

    assert db.is_connected() is False


@pytest.mark.db
@pytest.mark.unit
def test_sqlite_vector_search():
    """Test vector table creation, insertion, top_k, min_score filtering, and deletion."""
    with SQLiteConnector(db_path=":memory:") as db:
        db.create_vector_table("item_embeddings", vector_dim=3, distance_metric="cosine")

        # Single insert
        db.insert_vector("item_embeddings", "vec_1", [1.0, 0.0, 0.0], {"category": "A"})

        # Batch insert
        batch = [
            {"id": "vec_2", "vector": [0.9, 0.1, 0.0], "metadata": {"category": "A"}},
            {"id": "vec_3", "vector": [0.0, 1.0, 0.0], "metadata": {"category": "B"}},
            {"id": "vec_4", "vector": [-1.0, 0.0, 0.0], "metadata": {"category": "C"}},
        ]
        db.insert_vectors("item_embeddings", batch)

        # Query vector close to vec_1 and vec_2
        query_vec = [1.0, 0.0, 0.0]

        # Top-k search
        results_top2 = db.vector_search("item_embeddings", query_vec, top_k=2, distance_metric="cosine")
        assert len(results_top2) == 2
        assert results_top2[0]["id"] == "vec_1"
        assert results_top2[0]["score"] == 1.0
        assert results_top2[1]["id"] == "vec_2"

        # min_score filtering
        results_min_score = db.vector_search("item_embeddings", query_vec, top_k=10, min_score=0.9, distance_metric="cosine")
        assert len(results_min_score) == 2
        assert {r["id"] for r in results_min_score} == {"vec_1", "vec_2"}

        # Delete vector
        deleted_count = db.delete_vector("item_embeddings", "vec_1")
        assert deleted_count == 1

        results_after_delete = db.vector_search("item_embeddings", query_vec, top_k=10)
        assert len(results_after_delete) == 3
        assert "vec_1" not in {r["id"] for r in results_after_delete}


@pytest.mark.db
@pytest.mark.unit
def test_postgres_url_configuration():
    """Verify PostgreSQLConnector connection URL resolution."""
    url_conn = PostgreSQLConnector(connection_string="postgresql://user:pass@localhost:5432/db")
    assert url_conn.connection_string == "postgresql://user:pass@localhost:5432/db"
    assert url_conn.is_connected() is False


@pytest.mark.db
@pytest.mark.unit
def test_mysql_url_configuration():
    """Verify MySQLConnector connection URL resolution."""
    url_conn = MySQLConnector(connection_string="mysql://myuser:mypass@127.0.0.1:3306/mydb")
    assert url_conn.connection_string == "mysql://myuser:mypass@127.0.0.1:3306/mydb"
    assert url_conn.is_connected() is False


@pytest.mark.db
@pytest.mark.unit
def test_database_router():
    """Verify DatabaseRouter resolves connectors dynamically by single connection URL and dialect."""
    sqlite_conn = DatabaseRouter.get_connector("sqlite", db_path=":memory:")
    assert isinstance(sqlite_conn, SQLiteConnector)

    pg_conn = DatabaseRouter.get_connector("postgresql://user:pass@localhost:5432/db")
    assert isinstance(pg_conn, PostgreSQLConnector)

    mysql_conn = DatabaseRouter.get_connector("mysql://root:pass@localhost:3306/mydb")
    assert isinstance(mysql_conn, MySQLConnector)


if __name__ == "__main__":
    test_exports()
    test_sqlite_functional()
    test_sqlite_vector_search()
    test_postgres_url_configuration()
    test_mysql_url_configuration()
    test_database_router()
    print("All database connector and vector search tests passed successfully!")
