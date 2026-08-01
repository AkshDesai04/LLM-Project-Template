"""
Unit tests for Database Connector Utilities & DatabaseRouter.

Verifies exports, single connection URL resolution, SQLite functional operations,
context managers, DatabaseRouter factory routing, and fallback behavior for optional drivers.
"""

from utils import (
    BaseDatabaseConnector,
    DatabaseRouter,
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


def test_postgres_single_url_and_fallbacks():
    """Verify PostgreSQLConnector single connection URL parsing and fallback attributes."""
    url_conn = PostgreSQLConnector(connection_string="postgresql://user:pass@localhost:5432/db")
    assert url_conn.connection_string == "postgresql://user:pass@localhost:5432/db"

    fallback_conn = PostgreSQLConnector(
        host="localhost",
        port=5432,
        user="test_user",
        password="test_password",
        dbname="test_db",
    )
    assert fallback_conn.host == "localhost"
    assert fallback_conn.port == 5432
    assert fallback_conn.user == "test_user"
    assert fallback_conn.dbname == "test_db"


def test_mysql_single_url_and_fallbacks():
    """Verify MySQLConnector single connection URL parsing and fallback attributes."""
    url_conn = MySQLConnector(connection_string="mysql://myuser:mypass@127.0.0.1:3306/mydb")
    assert url_conn.host == "127.0.0.1"
    assert url_conn.port == 3306
    assert url_conn.user == "myuser"
    assert url_conn.password == "mypass"
    assert url_conn.database == "mydb"

    fallback_conn = MySQLConnector(
        host="localhost",
        port=3306,
        user="root",
        password="root_password",
        database="mysql_test",
    )
    assert fallback_conn.host == "localhost"
    assert fallback_conn.port == 3306


def test_mongo_single_url_and_fallbacks():
    """Verify MongoDBConnector single connection URI parsing and fallback attributes."""
    url_conn = MongoDBConnector(uri="mongodb://admin:secret@localhost:27017/db", database="test_db")
    assert url_conn.uri == "mongodb://admin:secret@localhost:27017/db"
    assert url_conn.database_name == "test_db"


def test_oracle_single_url_and_fallbacks():
    """Verify OracleDBConnector single connection URL parsing and fallback attributes."""
    url_conn = OracleDBConnector(connection_string="oracle://sysuser:syspass@localhost:1521/ORCLCDB")
    assert url_conn.user == "sysuser"
    assert url_conn.password == "syspass"
    assert url_conn.dsn == "localhost:1521/ORCLCDB"


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
    assert mysql_conn.database == "mydb"

    # MongoDB single URI routing
    mongo_conn = DatabaseRouter.get_connector("mongodb://localhost:27017")
    assert isinstance(mongo_conn, MongoDBConnector)

    # Oracle single URL routing
    oracle_conn = DatabaseRouter.get_connector("oracle://system:oracle@localhost:1521/ORCLCDB")
    assert isinstance(oracle_conn, OracleDBConnector)
    assert oracle_conn.user == "system"


if __name__ == "__main__":
    test_exports()
    test_sqlite_functional()
    test_postgres_single_url_and_fallbacks()
    test_mysql_single_url_and_fallbacks()
    test_mongo_single_url_and_fallbacks()
    test_oracle_single_url_and_fallbacks()
    test_database_router()
    print("All database connector and URL router tests passed successfully!")
