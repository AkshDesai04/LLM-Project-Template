"""
Helper functions for database connectors.
"""

from typing import Any, Dict, List, Optional, Tuple, Union


def rows_to_dicts(cursor, rows: List[Any]) -> List[Dict[str, Any]]:
    """Converts raw database cursor rows into a list of dictionaries."""
    if not rows or not cursor.description:
        return []
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def execute_sql_query(connection: Any, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Generic query execution returning a list of dictionaries."""
    params = params or ()
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        if hasattr(cursor, 'fetchall'):
            rows = cursor.fetchall()
            if hasattr(cursor, 'description') and cursor.description:
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
            return [dict(row) for row in rows] if rows and isinstance(rows[0], dict) else []
    return []


def execute_sql_non_query(connection: Any, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None, in_transaction: bool = False) -> int:
    """Generic non-query execution returning row count."""
    params = params or ()
    cursor = connection.cursor()
    try:
        cursor.execute(query, params)
        rowcount = getattr(cursor, 'rowcount', 0)
        if not in_transaction and hasattr(connection, 'commit'):
            connection.commit()
        return max(rowcount, 0)
    except Exception:
        if not in_transaction and hasattr(connection, 'rollback'):
            connection.rollback()
        raise
    finally:
        if hasattr(cursor, 'close'):
            cursor.close()


def execute_sql_many(connection: Any, query: str, params_list: List[Union[Tuple[Any, ...], Dict[str, Any]]], in_transaction: bool = False) -> int:
    """Generic executemany execution returning row count."""
    cursor = connection.cursor()
    try:
        if hasattr(cursor, 'executemany'):
            cursor.executemany(query, params_list)
            total = getattr(cursor, 'rowcount', len(params_list))
        else:
            total = 0
            for params in params_list:
                cursor.execute(query, params)
                total += getattr(cursor, 'rowcount', 1)
        if not in_transaction and hasattr(connection, 'commit'):
            connection.commit()
        return max(total, 0)
    except Exception:
        if not in_transaction and hasattr(connection, 'rollback'):
            connection.rollback()
        raise
    finally:
        if hasattr(cursor, 'close'):
            cursor.close()
