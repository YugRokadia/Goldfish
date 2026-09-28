from __future__ import annotations

import sqlite3

from app.context.clustering import build_context


def get_memory_context(
    connection: sqlite3.Connection,
    memory_id: int,
    limit: int = 8,
) -> dict:
    """
    Return the context surrounding a single memory.

    This is intentionally a thin service layer so the retrieval/API layers
    don't need to know how clustering works internally.
    """
    return build_context(
        connection=connection,
        memory_id=memory_id,
        limit=limit,
    )