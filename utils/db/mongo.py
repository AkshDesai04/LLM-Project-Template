"""
MongoDB Database Connector.

Provides a robust MongoDB connector using pymongo configured via single connection URIs,
with support for document CRUD operations, aggregation pipelines, transactions, and an interoperable query facade.
"""

import contextlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pymongo
    import pymongo.errors
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    pymongo = None

from .base import BaseDatabaseConnector
from utils.env import get_database_url, get_secret
from utils.logging import get_logger

logger = get_logger("MongoDBConnector")


class MongoDBConnector(BaseDatabaseConnector):
    """
    Connector for MongoDB document databases configured via a single connection URI.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        connection_string: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        Initializes MongoDB connector using a connection URI.
        Resolves from uri / connection_string / MONGODB_URI / MONGODB_URL / DATABASE_URL env vars.
        """
        self.database_name = database or get_secret("MONGODB_DB", raise_error=False) or "test"
        resolved_uri = (
            uri
            or connection_string
            or get_secret("MONGODB_URI", raise_error=False)
            or get_secret("MONGODB_URL", raise_error=False)
            or get_database_url(raise_error=False)
            or "mongodb://localhost:27017"
        )

        self.uri = resolved_uri
        self._client = None
        self._db = None

    def connect(self):
        """
        Establishes connection client to MongoDB server.

        Returns:
            pymongo.database.Database: Connected database instance.
        """
        if not PYMONGO_AVAILABLE:
            message = "pymongo is not installed. Please install 'pymongo' to use MongoDBConnector."
            logger.error(message)
            raise ImportError(message)

        if self._client is None:
            logger.info("Connecting to MongoDB instance via URI...")
            self._client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self._db = self._client[self.database_name]
            logger.info(f"Successfully connected to MongoDB database '{self.database_name}'.")
        return self._db

    def close(self) -> None:
        """Closes active MongoDB client connection."""
        if self._client is not None:
            logger.info("Closing MongoDB connection...")
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed.")

    def is_connected(self) -> bool:
        """
        Checks whether MongoDB instance is active and reachable via ping command.

        Returns:
            bool: True if connected and responsive, False otherwise.
        """
        if not PYMONGO_AVAILABLE or self._client is None:
            return False
        try:
            self._client.admin.command("ping")
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Native Document CRUD Operations
    # -------------------------------------------------------------------------

    def find_documents(
        self,
        collection_name: str,
        filter_query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves matching documents from a collection.
        """
        db = self.connect()
        filter_query = filter_query or {}
        try:
            cursor = db[collection_name].find(filter_query, projection)
            if limit > 0:
                cursor = cursor.limit(limit)
            return [self._format_doc(doc) for doc in cursor]
        except Exception as e:
            logger.error(f"MongoDB find_documents failed: {e}")
            raise

    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """
        Inserts a single document into a collection and returns string ID.
        """
        db = self.connect()
        try:
            result = db[collection_name].insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"MongoDB insert_document failed: {e}")
            raise

    def insert_many_documents(
        self, collection_name: str, documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Inserts multiple documents into a collection.
        """
        db = self.connect()
        try:
            result = db[collection_name].insert_many(documents)
            return [str(_id) for _id in result.inserted_ids]
        except Exception as e:
            logger.error(f"MongoDB insert_many_documents failed: {e}")
            raise

    def update_documents(
        self,
        collection_name: str,
        filter_query: Dict[str, Any],
        update_data: Dict[str, Any],
        many: bool = True,
    ) -> int:
        """
        Updates document(s) in a collection matching filter.
        """
        db = self.connect()
        update_payload = update_data if "$set" in update_data or "$inc" in update_data else {"$set": update_data}
        try:
            if many:
                result = db[collection_name].update_many(filter_query, update_payload)
            else:
                result = db[collection_name].update_one(filter_query, update_payload)
            return result.modified_count
        except Exception as e:
            logger.error(f"MongoDB update_documents failed: {e}")
            raise

    def delete_documents(
        self, collection_name: str, filter_query: Dict[str, Any], many: bool = True
    ) -> int:
        """
        Deletes document(s) matching filter query.
        """
        db = self.connect()
        try:
            if many:
                result = db[collection_name].delete_many(filter_query)
            else:
                result = db[collection_name].delete_one(filter_query)
            return result.deleted_count
        except Exception as e:
            logger.error(f"MongoDB delete_documents failed: {e}")
            raise

    def count_documents(
        self, collection_name: str, filter_query: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Counts documents in collection matching filter.
        """
        db = self.connect()
        filter_query = filter_query or {}
        try:
            return db[collection_name].count_documents(filter_query)
        except Exception as e:
            logger.error(f"MongoDB count_documents failed: {e}")
            raise

    def aggregate(
        self, collection_name: str, pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Executes an aggregation pipeline.
        """
        db = self.connect()
        try:
            results = db[collection_name].aggregate(pipeline)
            return [self._format_doc(doc) for doc in results]
        except Exception as e:
            logger.error(f"MongoDB aggregate failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # BaseDatabaseConnector Interoperable Facade Implementation
    # -------------------------------------------------------------------------

    def execute_query(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Interoperable query method. Expects query to be a collection name, or JSON string.
        If params contains a dict, uses it as filter query.
        """
        filter_dict = params if isinstance(params, dict) else {}
        collection_name = self._extract_collection(query)
        return self.find_documents(collection_name, filter_query=filter_dict)

    def execute_non_query(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> int:
        """
        Interoperable write method. If params is dict, inserts into collection.
        """
        collection_name = self._extract_collection(query)
        if isinstance(params, dict):
            self.insert_document(collection_name, params)
            return 1
        return 0

    def execute_many(
        self, query: str, params_list: List[Union[Tuple[Any, ...], Dict[str, Any]]]
    ) -> int:
        """
        Interoperable batch insert.
        """
        collection_name = self._extract_collection(query)
        docs = [p for p in params_list if isinstance(p, dict)]
        if docs:
            inserted = self.insert_many_documents(collection_name, docs)
            return len(inserted)
        return 0

    def fetch_one(
        self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Interoperable fetch_one.
        """
        results = self.execute_query(query, params)
        return results[0] if results else None

    @contextlib.contextmanager
    def transaction(self):
        """
        Context manager for MongoDB session transactions.
        Requires a replica set deployment.
        """
        self.connect()
        if self._client is None:
            raise RuntimeError("MongoDB client not initialized.")
        with self._client.start_session() as session:
            with session.start_transaction():
                try:
                    yield session
                except Exception as e:
                    logger.error(f"MongoDB transaction failed: {e}")
                    raise

    @staticmethod
    def _format_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Converts BSON ObjectId into string representation for serialization."""
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc

    @staticmethod
    def _extract_collection(query: str) -> str:
        """Helper to extract collection name from query string or JSON payload."""
        cleaned = query.strip()
        if cleaned.startswith("{"):
            try:
                payload = json.loads(cleaned)
                return payload.get("collection", "default")
            except Exception:
                pass
        return cleaned if cleaned else "default"
