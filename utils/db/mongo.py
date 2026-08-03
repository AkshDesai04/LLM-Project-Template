"""
MongoDB Database Connector.
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

from utils.db.base import BaseDatabaseConnector
from utils.env import get_database_url, get_secret
from utils.logging import get_logger

logger = get_logger("MongoDBConnector")


class MongoDBConnector(BaseDatabaseConnector):
    """Connector for MongoDB document databases configured via a single connection URI."""

    def __init__(self, uri: Optional[str] = None, connection_string: Optional[str] = None, database: Optional[str] = None):
        self.database_name = database or get_secret("MONGODB_DB", raise_error=False) or "test"
        self.uri = (
            uri
            or connection_string
            or get_secret("MONGODB_URI", raise_error=False)
            or get_secret("MONGODB_URL", raise_error=False)
            or get_database_url(raise_error=False)
            or "mongodb://localhost:27017"
        )
        self._client = None
        self._db = None

    def connect(self):
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
        if self._client is not None:
            logger.info("Closing MongoDB connection...")
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed.")

    def is_connected(self) -> bool:
        if not PYMONGO_AVAILABLE or self._client is None:
            return False
        try:
            self._client.admin.command("ping")
            return True
        except Exception:
            return False

    def find_documents(self, collection_name: str, filter_query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, Any]] = None, limit: int = 0) -> List[Dict[str, Any]]:
        db = self.connect()
        try:
            cursor = db[collection_name].find(filter_query or {}, projection)
            if limit > 0:
                cursor = cursor.limit(limit)
            return [self._format_doc(doc) for doc in cursor]
        except Exception as e:
            logger.error(f"MongoDB find_documents failed: {e}")
            raise

    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        db = self.connect()
        try:
            return str(db[collection_name].insert_one(document).inserted_id)
        except Exception as e:
            logger.error(f"MongoDB insert_document failed: {e}")
            raise

    def insert_many_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        db = self.connect()
        try:
            return [str(_id) for _id in db[collection_name].insert_many(documents).inserted_ids]
        except Exception as e:
            logger.error(f"MongoDB insert_many_documents failed: {e}")
            raise

    def update_documents(self, collection_name: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], many: bool = True) -> int:
        db = self.connect()
        payload = update_data if "$set" in update_data or "$inc" in update_data else {"$set": update_data}
        try:
            res = db[collection_name].update_many(filter_query, payload) if many else db[collection_name].update_one(filter_query, payload)
            return res.modified_count
        except Exception as e:
            logger.error(f"MongoDB update_documents failed: {e}")
            raise

    def delete_documents(self, collection_name: str, filter_query: Dict[str, Any], many: bool = True) -> int:
        db = self.connect()
        try:
            res = db[collection_name].delete_many(filter_query) if many else db[collection_name].delete_one(filter_query)
            return res.deleted_count
        except Exception as e:
            logger.error(f"MongoDB delete_documents failed: {e}")
            raise

    def count_documents(self, collection_name: str, filter_query: Optional[Dict[str, Any]] = None) -> int:
        db = self.connect()
        try:
            return db[collection_name].count_documents(filter_query or {})
        except Exception as e:
            logger.error(f"MongoDB count_documents failed: {e}")
            raise

    def aggregate(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        db = self.connect()
        try:
            return [self._format_doc(doc) for doc in db[collection_name].aggregate(pipeline)]
        except Exception as e:
            logger.error(f"MongoDB aggregate failed: {e}")
            raise

    def execute_query(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        filter_dict = params if isinstance(params, dict) else {}
        return self.find_documents(self._extract_collection(query), filter_query=filter_dict)

    def execute_non_query(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> int:
        col = self._extract_collection(query)
        if isinstance(params, dict):
            self.insert_document(col, params)
            return 1
        return 0

    def execute_many(self, query: str, params_list: List[Union[Tuple[Any, ...], Dict[str, Any]]]) -> int:
        col = self._extract_collection(query)
        docs = [p for p in params_list if isinstance(p, dict)]
        return len(self.insert_many_documents(col, docs)) if docs else 0

    def fetch_one(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        res = self.execute_query(query, params)
        return res[0] if res else None

    @contextlib.contextmanager
    def transaction(self):
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
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc

    @staticmethod
    def _extract_collection(query: str) -> str:
        cleaned = query.strip()
        if cleaned.startswith("{"):
            try:
                return json.loads(cleaned).get("collection", "default")
            except Exception:
                pass
        return cleaned if cleaned else "default"
