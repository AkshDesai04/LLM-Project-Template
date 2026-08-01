"""
Database Router & Factory.

Provides a unified factory entry point (DatabaseRouter) for instantiating
database connectors by name, dialect, or single connection string URL.
All connector credentials seamlessly resolve via utils.env_ops (honouring KEY_LOCATION
to fetch credentials from .env or AWS Secrets Manager).
"""

from typing import Any, Dict, Optional

from utils.base_connector import BaseDatabaseConnector
from utils.env_ops import get_database_url
from utils.logger import get_logger
from utils.mongo_connector import MongoDBConnector
from utils.mysql_connector import MySQLConnector
from utils.oracle_connector import OracleDBConnector
from utils.postgres_connector import PostgreSQLConnector
from utils.sqlite_connector import SQLiteConnector

logger = get_logger("DatabaseRouter")

# Dialect Constants
DIALECT_SQLITE = "sqlite"
DIALECT_POSTGRES = "postgres"
DIALECT_MYSQL = "mysql"
DIALECT_MONGO = "mongo"
DIALECT_ORACLE = "oracle"

DIALECT_ALIASES = {
    "sqlite": DIALECT_SQLITE,
    "sqlite3": DIALECT_SQLITE,
    "postgres": DIALECT_POSTGRES,
    "postgresql": DIALECT_POSTGRES,
    "pg": DIALECT_POSTGRES,
    "mysql": DIALECT_MYSQL,
    "mariadb": DIALECT_MYSQL,
    "mongo": DIALECT_MONGO,
    "mongodb": DIALECT_MONGO,
    "oracle": DIALECT_ORACLE,
    "oracledb": DIALECT_ORACLE,
}


class DatabaseRouter:
    """
    Unified router and factory for Database Connectors.

    Supports building connectors based on explicit dialect name, alias,
    or connection URI scheme while fetching secrets through utils.env_ops.
    """

    @staticmethod
    def resolve_dialect(db_identifier: str) -> str:
        """
        Resolves dialect string or connection URI scheme to a standard dialect key.

        Args:
            db_identifier: Dialect name (e.g. 'postgres', 'sqlite') or URI (e.g. 'postgresql://...').

        Returns:
            str: Canonical dialect key ('sqlite', 'postgres', 'mysql', 'mongo', 'oracle').
        """
        clean = db_identifier.strip().lower()

        # Extract scheme if a URI is provided
        if "://" in clean:
            scheme = clean.split("://", 1)[0]
            if "+" in scheme:
                scheme = scheme.split("+", 1)[0]
            clean = scheme

        if clean in DIALECT_ALIASES:
            return DIALECT_ALIASES[clean]

        raise ValueError(
            f"Unsupported database dialect or URI scheme: '{db_identifier}'. "
            f"Supported dialects: {', '.join(sorted(set(DIALECT_ALIASES.values())))}"
        )

    @classmethod
    def get_connector(
        cls,
        db_type_or_url: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseDatabaseConnector:
        """
        Factory method to instantiate the requested database connector.

        Args:
            db_type_or_url: Dialect name (e.g. 'postgres', 'sqlite', 'mysql', 'mongo', 'oracle')
                            or single connection string URL. If None, defaults to DATABASE_URL or 'sqlite'.
            **kwargs: Direct overrides for connector initialization parameters.

        Returns:
            BaseDatabaseConnector: Initialized database connector instance.
        """
        identifier = db_type_or_url or get_database_url(raise_error=False) or DIALECT_SQLITE
        dialect = cls.resolve_dialect(identifier)

        logger.info(f"Routing database request to connector for dialect: '{dialect}'")

        if dialect == DIALECT_SQLITE:
            db_path = kwargs.pop("db_path", None)
            connection_string = kwargs.pop("connection_string", None)
            if not db_path and not connection_string and ("/" in identifier or "\\" in identifier or identifier.endswith(".db")):
                db_path = identifier
            return SQLiteConnector(db_path=db_path, connection_string=connection_string, **kwargs)

        if dialect == DIALECT_POSTGRES:
            connection_string = kwargs.pop("connection_string", None)
            if not connection_string and "://" in identifier:
                connection_string = identifier
            return PostgreSQLConnector(connection_string=connection_string, **kwargs)

        if dialect == DIALECT_MYSQL:
            connection_string = kwargs.pop("connection_string", None)
            if not connection_string and "://" in identifier:
                connection_string = identifier
            return MySQLConnector(connection_string=connection_string, **kwargs)

        if dialect == DIALECT_MONGO:
            uri = kwargs.pop("uri", None)
            connection_string = kwargs.pop("connection_string", None)
            if not uri and not connection_string and "://" in identifier:
                uri = identifier
            return MongoDBConnector(uri=uri, connection_string=connection_string, **kwargs)

        if dialect == DIALECT_ORACLE:
            connection_string = kwargs.pop("connection_string", None)
            dsn = kwargs.pop("dsn", None)
            if not connection_string and not dsn and ("://" in identifier or "/" in identifier or ":" in identifier):
                connection_string = identifier
            return OracleDBConnector(connection_string=connection_string, dsn=dsn, **kwargs)

        raise ValueError(f"Unable to route database connector for dialect '{dialect}'.")
