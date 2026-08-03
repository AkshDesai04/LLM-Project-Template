"""
Unit tests for Database Connector Utilities & DatabaseRouter.

Verifies exports, single connection URL resolution, SQLite functional operations,
context managers, DatabaseRouter factory routing, and fallback behavior for optional drivers.
"""

from utils import (
    BaseDatabaseConnector,
    DatabaseRouter,
    MySQLConnector,
    PostgreSQLConnector,
    SQLiteConnector,
)


def test_exports():
    """Verify all database connectors export correctly from utils."""
    assert issubclass(SQLiteConnector, BaseDatabaseConnector)
    assert issubclass(PostgreSQLConnector, BaseDatabaseConnector)
    assert issubclass(MySQLConnector, BaseDatabaseConnector)


def test_sqlite_functional():
    """Test full functional lifecycle of SQLiteConnector using an in-memory database."""
    with SQLiteConnector(db_path=":memory:") as db:
        assert db.is_connected() is True

        # DDL Execution
        create_table_sql = """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL
        );
        """
        db.execute_non_query(create_table_sql)

        # Single insert
        insert_sql = "INSERT INTO users (name, email) VALUES (?, ?);"
        affected = db.execute_non_query(insert_sql, ("Alice", "alice@example.com"))
        assert affected == 1

        # Batch insert
        users_batch = [
            ("Bob", "bob@example.com"),
            ("Charlie", "charlie@example.com"),
        ]
        batch_affected = db.execute_many(insert_sql, users_batch)
        assert batch_affected == 2

        # Query all rows
        rows = db.execute_query("SELECT * FROM users ORDER BY id ASC;")
        assert len(rows) == 3
        assert rows[0] == {"id": 1, "name": "Alice", "email": "alice@example.com"}
        assert rows[1] == {"id": 2, "name": "Bob", "email": "bob@example.com"}

        # Fetch one
        row = db.fetch_one("SELECT * FROM users WHERE email = ?;", ("charlie@example.com",))
        assert row is not None
        assert row["name"] == "Charlie"

        # Test transaction context manager commit
        with db.transaction():
            db.execute_non_query("INSERT INTO users (name, email) VALUES (?, ?);", ("David", "david@example.com"))

        david = db.fetch_one("SELECT * FROM users WHERE name = ?;", ("David",))
        assert david is not None

        # Test transaction rollback on exception
        try:
            with db.transaction():
                db.execute_non_query("INSERT INTO users (name, email) VALUES (?, ?);", ("Eve", "eve@example.com"))
                raise ValueError("Simulated transaction error")
        except ValueError:
            pass

        eve = db.fetch_one("SELECT * FROM users WHERE name = ?;", ("Eve",))
        assert eve is None

    # Verify closed connection
    assert db.is_connected() is False


def test_postgres_url_configuration():
    """Verify PostgreSQLConnector connection URL resolution."""
    url_conn = PostgreSQLConnector(connection_string="postgresql://user:pass@localhost:5432/db")
    assert url_conn.connection_string == "postgresql://user:pass@localhost:5432/db"
    assert url_conn.is_connected() is False


def test_mysql_url_configuration():
    """Verify MySQLConnector connection URL resolution."""
    url_conn = MySQLConnector(connection_string="mysql://myuser:mypass@127.0.0.1:3306/mydb")
    assert url_conn.connection_string == "mysql://myuser:mypass@127.0.0.1:3306/mydb"
    assert url_conn.is_connected() is False


def test_database_router():
    """Verify DatabaseRouter resolves connectors dynamically by single connection URL and dialect."""
    # SQLite routing
    sqlite_conn = DatabaseRouter.get_connector("sqlite", db_path=":memory:")
    assert isinstance(sqlite_conn, SQLiteConnector)

    # Postgres single URL routing
    pg_conn = DatabaseRouter.get_connector("postgresql://user:pass@localhost:5432/db")
    assert isinstance(pg_conn, PostgreSQLConnector)
    assert pg_conn.connection_string == "postgresql://user:pass@localhost:5432/db"

    # MySQL single URL routing
    mysql_conn = DatabaseRouter.get_connector("mysql://root:pass@localhost:3306/mydb")
    assert isinstance(mysql_conn, MySQLConnector)


if __name__ == "__main__":
    test_exports()
    test_sqlite_functional()
    test_postgres_url_configuration()
    test_mysql_url_configuration()
    test_database_router()
    print("All database connector and URL router tests passed successfully!")
