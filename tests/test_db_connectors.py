"""
Unit tests for Database Connector Utilities.

Verifies exports, SQLite functional operations, context managers,
fallback behavior for optional database drivers, and BaseDatabaseConnector inheritance.
"""

from utils import (
    BaseDatabaseConnector,
    MongoDBConnector,
    MySQLConnector,
    OracleDBConnector,
    PostgreSQLConnector,
    SQLiteConnector,
)


def test_exports():
    """Verify all database connectors export correctly from utils."""
    assert issubclass(SQLiteConnector, BaseDatabaseConnector)
    assert issubclass(PostgreSQLConnector, BaseDatabaseConnector)
    assert issubclass(MySQLConnector, BaseDatabaseConnector)
    assert issubclass(MongoDBConnector, BaseDatabaseConnector)
    assert issubclass(OracleDBConnector, BaseDatabaseConnector)


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


def test_postgres_initialization():
    """Verify PostgreSQLConnector initialization and fallback attributes."""
    connector = PostgreSQLConnector(
        host="localhost",
        port=5432,
        user="test_user",
        password="test_password",
        dbname="test_db",
    )
    assert connector.host == "localhost"
    assert connector.port == 5432
    assert connector.user == "test_user"
    assert connector.dbname == "test_db"
    assert connector.is_connected() is False


def test_mysql_initialization():
    """Verify MySQLConnector initialization and fallback attributes."""
    connector = MySQLConnector(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="root_password",
        database="mysql_test",
    )
    assert connector.host == "127.0.0.1"
    assert connector.port == 3306
    assert connector.database == "mysql_test"
    assert connector.is_connected() is False


def test_mongo_initialization():
    """Verify MongoDBConnector initialization and fallback attributes."""
    connector = MongoDBConnector(
        uri="mongodb://localhost:27017",
        database="mongo_test",
    )
    assert connector.uri == "mongodb://localhost:27017"
    assert connector.database_name == "mongo_test"
    assert connector.is_connected() is False


def test_oracle_initialization():
    """Verify OracleDBConnector initialization and fallback attributes."""
    connector = OracleDBConnector(
        user="system",
        password="oracle_password",
        dsn="localhost:1521/ORCLCDB",
    )
    assert connector.user == "system"
    assert connector.dsn == "localhost:1521/ORCLCDB"
    assert connector.is_connected() is False


if __name__ == "__main__":
    test_exports()
    test_sqlite_functional()
    test_postgres_initialization()
    test_mysql_initialization()
    test_mongo_initialization()
    test_oracle_initialization()
    print("All database connector tests passed successfully!")
