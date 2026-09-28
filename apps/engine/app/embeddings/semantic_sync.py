from __future__ import annotations

import sqlite3

from app.embeddings.index import get_semantic_index
from app.embeddings.model import embed_text
from app.storage.database import (
    CURRENT_EMBEDDING_VERSION,
    EMBEDDING_STATUS_FAILED,
    EMBEDDING_STATUS_PENDING,
    EMBEDDING_STATUS_SYNCED,
)


def _mark_embedding_pending(
    connection: sqlite3.Connection,
    memory_id: int,
) -> None:
    connection.execute(
        """
        UPDATE memories
        SET embedding_status = ?,
            embedding_error = NULL
        WHERE id = ?
        """,
        (EMBEDDING_STATUS_PENDING, memory_id),
    )
    connection.commit()


def _mark_embedding_synced(
    connection: sqlite3.Connection,
    memory_id: int,
) -> None:
    connection.execute(
        """
        UPDATE memories
        SET embedding_status = ?,
            embedding_version = ?,
            embedding_updated_at = strftime('%s', 'now'),
            embedding_error = NULL
        WHERE id = ?
        """,
        (
            EMBEDDING_STATUS_SYNCED,
            CURRENT_EMBEDDING_VERSION,
            memory_id,
        ),
    )
    connection.commit()


def _mark_embedding_failed(
    connection: sqlite3.Connection,
    memory_id: int,
    error: Exception,
) -> None:
    connection.execute(
        """
        UPDATE memories
        SET embedding_status = ?,
            embedding_error = ?
        WHERE id = ?
        """,
        (
            EMBEDDING_STATUS_FAILED,
            str(error)[:2000],
            memory_id,
        ),
    )
    connection.commit()


def sync_memory_embedding(
    connection: sqlite3.Connection,
    memory_id: int,
) -> None:
    """
    Create or replace the FAISS embedding for one active memory.

    The database tracks the embedding lifecycle so that failed
    embeddings can be detected and retried later.
    """

    memory = connection.execute(
        """
        SELECT id, title, content
        FROM memories
        WHERE id = ?
          AND is_deleted = 0
        """,
        (memory_id,),
    ).fetchone()

    if memory is None:
        raise ValueError(
            f"Active memory not found: {memory_id}"
        )

    _mark_embedding_pending(connection, memory_id)

    try:
        text = f"{memory['title']}\n{memory['content']}"

        # Generate the embedding before touching FAISS.
        embedding = embed_text(text)

        semantic_index = get_semantic_index()

        # Remove the old vector so updates do not create duplicates.
        semantic_index.remove([memory_id])

        try:
            semantic_index.add(
                embedding.reshape(1, -1),
                [memory_id],
            )

            # Persist the new index before marking the DB row as synced.
            semantic_index.save()

        except Exception:
            # If FAISS mutation/save fails, reload the persisted index
            # so the in-memory index is not left in a partially modified
            # state.
            semantic_index.load()
            raise

        _mark_embedding_synced(connection, memory_id)

    except Exception as error:
        _mark_embedding_failed(
            connection,
            memory_id,
            error,
        )
        raise


def remove_memory_embedding(
    memory_id: int,
) -> None:
    """
    Remove a memory's vector from FAISS.

    This function deliberately does not modify SQLite state. The caller
    is responsible for marking the memory deleted and tracking the
    embedding state in the database.
    """

    semantic_index = get_semantic_index()

    semantic_index.remove([memory_id])
    semantic_index.save()