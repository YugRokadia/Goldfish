from __future__ import annotations

import re
import sqlite3
import time
from typing import Any

from app.config import DATABASE_PATH


# ---------------------------------------------------------------------------
# Embedding state
# ---------------------------------------------------------------------------

CURRENT_EMBEDDING_VERSION = 1

EMBEDDING_STATUS_PENDING = "pending"
EMBEDDING_STATUS_SYNCED = "synced"
EMBEDDING_STATUS_FAILED = "failed"
EMBEDDING_STATUS_DELETED = "deleted"


# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    source_type TEXT NOT NULL,
    content_type TEXT NOT NULL,

    title TEXT NOT NULL,
    path TEXT NOT NULL,

    content TEXT NOT NULL DEFAULT "",

    size_bytes INTEGER NOT NULL DEFAULT 0,
    created_at REAL,
    modified_at REAL,
    accessed_at REAL,

    indexed_at REAL NOT NULL,

    file_hash TEXT,

    is_deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at REAL,

    metadata_json TEXT NOT NULL DEFAULT '{}',

    embedding_status TEXT NOT NULL DEFAULT 'pending',
    embedding_version INTEGER,
    embedding_updated_at REAL,
    embedding_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_source_type
ON memories(source_type);

CREATE INDEX IF NOT EXISTS idx_memories_modified_at
ON memories(modified_at);

CREATE INDEX IF NOT EXISTS idx_memories_accessed_at
ON memories(accessed_at);

CREATE INDEX IF NOT EXISTS idx_memories_file_hash
ON memories(file_hash);

CREATE INDEX IF NOT EXISTS idx_memories_deleted
ON memories(is_deleted);

CREATE INDEX IF NOT EXISTS idx_memories_deleted_at
ON memories(deleted_at);

CREATE INDEX IF NOT EXISTS idx_memories_embedding_status
ON memories(embedding_status);

CREATE INDEX IF NOT EXISTS idx_memories_embedding_version
ON memories(embedding_version);

CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_active_path
ON memories(path, source_type)
WHERE is_deleted = 0;

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(
    title,
    content,
    content='memories',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS memories_ai
AFTER INSERT ON memories
BEGIN
    INSERT INTO memories_fts(rowid, title, content)
    VALUES (new.id, new.title, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad
AFTER DELETE ON memories
BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, content)
    VALUES ('delete', old.id, old.title, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au
AFTER UPDATE ON memories
BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, content)
    VALUES ('delete', old.id, old.title, old.content);

    INSERT INTO memories_fts(rowid, title, content)
    VALUES (new.id, new.title, new.content);
END;
"""


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_connection() -> sqlite3.Connection:
    """
    Open a SQLite connection.

    The caller owns the connection and is responsible for closing it.
    """

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    return connection


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _get_memory_columns(
    connection: sqlite3.Connection,
) -> set[str]:
    """
    Return the existing columns in the memories table.
    """

    rows = connection.execute(
        "PRAGMA table_info(memories)"
    ).fetchall()

    return {
        str(row["name"])
        for row in rows
    }


def _memories_table_exists(
    connection: sqlite3.Connection,
) -> bool:
    """
    Check whether the memories table already exists.
    """

    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
        AND name = 'memories'
        """
    ).fetchone()

    return row is not None


def _add_embedding_columns(
    connection: sqlite3.Connection,
) -> bool:
    """
    Add embedding synchronization columns to an existing database.

    Existing memories are marked as synced because RecallX already has
    an established FAISS index for the current embedding model.
    """

    columns = _get_memory_columns(connection)

    added = False

    if "embedding_status" not in columns:
        connection.execute(
            """
            ALTER TABLE memories
            ADD COLUMN embedding_status TEXT
            NOT NULL DEFAULT 'synced'
            """
        )
        added = True

    if "embedding_version" not in columns:
        connection.execute(
            """
            ALTER TABLE memories
            ADD COLUMN embedding_version INTEGER
            """
        )
        added = True

    if "embedding_updated_at" not in columns:
        connection.execute(
            """
            ALTER TABLE memories
            ADD COLUMN embedding_updated_at REAL
            """
        )
        added = True

    if "embedding_error" not in columns:
        connection.execute(
            """
            ALTER TABLE memories
            ADD COLUMN embedding_error TEXT
            """
        )
        added = True

    if added:
        now = time.time()

        connection.execute(
            """
            UPDATE memories
            SET
                embedding_status = ?,
                embedding_version = ?,
                embedding_updated_at = ?,
                embedding_error = NULL
            WHERE embedding_status IS NULL
            """,
            (
                EMBEDDING_STATUS_SYNCED,
                CURRENT_EMBEDDING_VERSION,
                now,
            ),
        )

    return added


def _migrate_existing_database(
    connection: sqlite3.Connection,
) -> bool:
    """
    Migrate existing RecallX database schemas.

    Supports:

    1. Original schema with path UNIQUE.
    2. Current versioned-memory schema.
    3. Adding embedding synchronization state.

    Existing memories are preserved.
    """

    columns = _get_memory_columns(
        connection,
    )

    if not columns:
        return False

    migration_performed = False

    # ------------------------------------------------------------------
    # Original database migration
    # ------------------------------------------------------------------

    required_columns = {
        "is_deleted",
        "deleted_at",
    }

    if not required_columns.issubset(columns):
        print(
            "RecallX: migrating memories database "
            "to versioned file-memory schema..."
        )

        connection.execute(
            "PRAGMA foreign_keys = OFF"
        )

        try:
            # Remove FTS triggers before rebuilding the memories table.
            connection.execute(
                "DROP TRIGGER IF EXISTS memories_ai"
            )

            connection.execute(
                "DROP TRIGGER IF EXISTS memories_ad"
            )

            connection.execute(
                "DROP TRIGGER IF EXISTS memories_au"
            )

            # Recreate the FTS table after rebuilding memories.
            connection.execute(
                "DROP TABLE IF EXISTS memories_fts"
            )

            connection.execute(
                """
                CREATE TABLE memories_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    source_type TEXT NOT NULL,
                    content_type TEXT NOT NULL,

                    title TEXT NOT NULL,
                    path TEXT NOT NULL,

                    content TEXT NOT NULL DEFAULT "",

                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    created_at REAL,
                    modified_at REAL,
                    accessed_at REAL,

                    indexed_at REAL NOT NULL,

                    file_hash TEXT,

                    is_deleted INTEGER NOT NULL DEFAULT 0,
                    deleted_at REAL,

                    metadata_json TEXT NOT NULL DEFAULT '{}',

                    embedding_status TEXT NOT NULL DEFAULT 'synced',
                    embedding_version INTEGER,
                    embedding_updated_at REAL,
                    embedding_error TEXT
                )
                """
            )

            # Preserve all existing memories.
            #
            # Existing records are treated as active and synced because
            # they belong to the already-built RecallX semantic index.
            connection.execute(
                """
                INSERT INTO memories_new (
                    id,
                    source_type,
                    content_type,
                    title,
                    path,
                    content,
                    size_bytes,
                    created_at,
                    modified_at,
                    accessed_at,
                    indexed_at,
                    file_hash,
                    is_deleted,
                    deleted_at,
                    metadata_json,
                    embedding_status,
                    embedding_version,
                    embedding_updated_at,
                    embedding_error
                )
                SELECT
                    id,
                    source_type,
                    content_type,
                    title,
                    path,
                    content,
                    size_bytes,
                    created_at,
                    modified_at,
                    accessed_at,
                    indexed_at,
                    file_hash,
                    0,
                    NULL,
                    metadata_json,
                    ?,
                    ?,
                    ?,
                    NULL
                FROM memories
                """,
                (
                    EMBEDDING_STATUS_SYNCED,
                    CURRENT_EMBEDDING_VERSION,
                    time.time(),
                ),
            )

            connection.execute(
                "DROP TABLE memories"
            )

            connection.execute(
                """
                ALTER TABLE memories_new
                RENAME TO memories
                """
            )

            connection.commit()

            print(
                "RecallX: database migration complete."
            )

            migration_performed = True

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.execute(
                "PRAGMA foreign_keys = ON"
            )

    # ------------------------------------------------------------------
    # Add embedding state to an already versioned database.
    # ------------------------------------------------------------------

    if not migration_performed:
        if _add_embedding_columns(connection):
            migration_performed = True

    return migration_performed


# ---------------------------------------------------------------------------
# FTS helpers
# ---------------------------------------------------------------------------


def _get_fts_count(
    connection: sqlite3.Connection,
) -> int:
    """
    Return the number of rows currently represented in the FTS table.
    """

    row = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM memories_fts
        """
    ).fetchone()

    return int(row["count"])


def _get_memory_row_count(
    connection: sqlite3.Connection,
) -> int:
    """
    Return the number of rows in the memories table.
    """

    row = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM memories
        """
    ).fetchone()

    return int(row["count"])


def _rebuild_fts(
    connection: sqlite3.Connection,
) -> None:
    """
    Rebuild the FTS5 external-content index.

    This is intentionally NOT called on every startup.
    """

    print(
        "RecallX: rebuilding FTS index..."
    )

    connection.execute(
        """
        INSERT INTO memories_fts(memories_fts)
        VALUES ('rebuild')
        """
    )

    connection.commit()

    print(
        "RecallX: FTS index rebuilt."
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_database() -> None:
    """
    Create or migrate the RecallX database.

    FTS is rebuilt only when necessary.
    """

    connection = get_connection()

    try:
        migration_performed = False

        if _memories_table_exists(connection):
            migration_performed = _migrate_existing_database(
                connection,
            )

        # Create the current schema, indexes, FTS table,
        # and triggers if they do not already exist.
        connection.executescript(
            SCHEMA
        )

        memory_count = _get_memory_row_count(
            connection,
        )

        fts_count = _get_fts_count(
            connection,
        )

        # Rebuild FTS only when:
        #
        # 1. The database was just migrated, or
        # 2. FTS is out of sync with the memories table.
        if (
            migration_performed
            or memory_count != fts_count
        ):
            _rebuild_fts(
                connection,
            )

        connection.commit()

    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Memory lookup
# ---------------------------------------------------------------------------


def get_memory_by_path(
    connection: sqlite3.Connection,
    path: str,
    source_type: str = "filesystem",
) -> sqlite3.Row | None:
    """
    Return the currently active memory for a path.

    Historical deleted memories are ignored.
    """

    return connection.execute(
        """
        SELECT
            id,
            source_type,
            content_type,
            title,
            path,
            content,
            size_bytes,
            created_at,
            modified_at,
            accessed_at,
            indexed_at,
            file_hash,
            is_deleted,
            deleted_at,
            metadata_json,
            embedding_status,
            embedding_version,
            embedding_updated_at,
            embedding_error
        FROM memories
        WHERE path = ?
          AND source_type = ?
          AND is_deleted = 0
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            path,
            source_type,
        ),
    ).fetchone()


def get_active_memory_paths(
    connection: sqlite3.Connection,
    source_type: str = "filesystem",
) -> list[sqlite3.Row]:
    """
    Return all currently active memories for a source.

    Historical deleted memories are excluded.
    """

    return connection.execute(
        """
        SELECT
            id,
            path,
            file_hash,
            modified_at,
            embedding_status,
            embedding_version,
            embedding_updated_at,
            embedding_error
        FROM memories
        WHERE source_type = ?
          AND is_deleted = 0
        """,
        (
            source_type,
        ),
    ).fetchall()


def get_deleted_memory_by_path_and_hash(
    connection: sqlite3.Connection,
    path: str,
    file_hash: str,
    source_type: str = "filesystem",
) -> sqlite3.Row | None:
    """
    Find a historical deleted memory representing the exact same
    file content.

    This allows an identical file downloaded again at the same path
    to be restored instead of creating a duplicate memory.
    """

    return connection.execute(
        """
        SELECT
            id,
            source_type,
            content_type,
            title,
            path,
            content,
            size_bytes,
            created_at,
            modified_at,
            accessed_at,
            indexed_at,
            file_hash,
            is_deleted,
            deleted_at,
            metadata_json,
            embedding_status,
            embedding_version,
            embedding_updated_at,
            embedding_error
        FROM memories
        WHERE path = ?
          AND source_type = ?
          AND file_hash = ?
          AND is_deleted = 1
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            path,
            source_type,
            file_hash,
        ),
    ).fetchone()


# ---------------------------------------------------------------------------
# Embedding synchronization state
# ---------------------------------------------------------------------------


def mark_embedding_pending(
    connection: sqlite3.Connection,
    memory_id: int,
) -> None:
    """
    Mark a memory as requiring semantic synchronization.

    The previous embedding version is intentionally preserved so we
    can tell that the currently stored vector may be stale.
    """

    connection.execute(
        """
        UPDATE memories
        SET
            embedding_status = ?,
            embedding_error = NULL
        WHERE id = ?
          AND is_deleted = 0
        """,
        (
            EMBEDDING_STATUS_PENDING,
            memory_id,
        ),
    )


def mark_embedding_synced(
    connection: sqlite3.Connection,
    memory_id: int,
    *,
    embedding_version: int = CURRENT_EMBEDDING_VERSION,
    embedding_updated_at: float | None = None,
) -> None:
    """
    Mark a memory as successfully synchronized with FAISS.
    """

    if embedding_updated_at is None:
        embedding_updated_at = time.time()

    connection.execute(
        """
        UPDATE memories
        SET
            embedding_status = ?,
            embedding_version = ?,
            embedding_updated_at = ?,
            embedding_error = NULL
        WHERE id = ?
          AND is_deleted = 0
        """,
        (
            EMBEDDING_STATUS_SYNCED,
            embedding_version,
            embedding_updated_at,
            memory_id,
        ),
    )


def mark_embedding_failed(
    connection: sqlite3.Connection,
    memory_id: int,
    error: str,
) -> None:
    """
    Mark a memory as failed during semantic synchronization.

    The record remains eligible for reconciliation.
    """

    connection.execute(
        """
        UPDATE memories
        SET
            embedding_status = ?,
            embedding_error = ?
        WHERE id = ?
          AND is_deleted = 0
        """,
        (
            EMBEDDING_STATUS_FAILED,
            error,
            memory_id,
        ),
    )


def mark_embedding_deleted(
    connection: sqlite3.Connection,
    memory_id: int,
) -> None:
    """
    Mark a deleted memory as having no active semantic embedding.
    """

    connection.execute(
        """
        UPDATE memories
        SET
            embedding_status = ?,
            embedding_updated_at = ?,
            embedding_error = NULL
        WHERE id = ?
        """,
        (
            EMBEDDING_STATUS_DELETED,
            time.time(),
            memory_id,
        ),
    )


def get_memories_needing_embedding_sync(
    connection: sqlite3.Connection,
    *,
    limit: int = 100,
) -> list[sqlite3.Row]:
    """
    Return active memories whose embeddings are missing, stale,
    pending, or failed.

    This is the core query used by semantic reconciliation.
    """

    return connection.execute(
        """
        SELECT
            id,
            source_type,
            content_type,
            title,
            path,
            content,
            file_hash,
            is_deleted,
            embedding_status,
            embedding_version,
            embedding_updated_at,
            embedding_error
        FROM memories
        WHERE is_deleted = 0
          AND (
              embedding_status IS NULL
              OR embedding_status != ?
              OR embedding_version IS NULL
              OR embedding_version != ?
          )
        ORDER BY
            CASE
                WHEN embedding_status = ? THEN 0
                WHEN embedding_status = ? THEN 1
                ELSE 2
            END,
            id ASC
        LIMIT ?
        """,
        (
            EMBEDDING_STATUS_SYNCED,
            CURRENT_EMBEDDING_VERSION,
            EMBEDDING_STATUS_PENDING,
            EMBEDDING_STATUS_FAILED,
            limit,
        ),
    ).fetchall()


# ---------------------------------------------------------------------------
# Deletion / restoration
# ---------------------------------------------------------------------------


def mark_memory_deleted(
    connection: sqlite3.Connection,
    memory_id: int,
    deleted_at: float,
) -> None:
    """
    Mark a memory as deleted without removing it.

    The memory's content remains available for historical
    search/context.

    Its active semantic embedding is marked as deleted and will
    therefore no longer be considered valid.
    """

    connection.execute(
        """
        UPDATE memories
        SET
            is_deleted = 1,
            deleted_at = ?,
            embedding_status = ?,
            embedding_error = NULL
        WHERE id = ?
        """,
        (
            deleted_at,
            EMBEDDING_STATUS_DELETED,
            memory_id,
        ),
    )


def restore_memory(
    connection: sqlite3.Connection,
    memory_id: int,
    *,
    modified_at: float | None,
    accessed_at: float | None,
    indexed_at: float,
) -> None:
    """
    Restore a previously deleted memory.

    The memory must be re-embedded because it is becoming active again.
    """

    connection.execute(
        """
        UPDATE memories
        SET
            is_deleted = 0,
            deleted_at = NULL,
            modified_at = ?,
            accessed_at = ?,
            indexed_at = ?,
            embedding_status = ?,
            embedding_error = NULL
        WHERE id = ?
        """,
        (
            modified_at,
            accessed_at,
            indexed_at,
            EMBEDDING_STATUS_PENDING,
            memory_id,
        ),
    )


# ---------------------------------------------------------------------------
# Memory upsert
# ---------------------------------------------------------------------------


def upsert_memory(
    connection: sqlite3.Connection,
    *,
    source_type: str,
    content_type: str,
    title: str,
    path: str,
    content: str,
    size_bytes: int,
    created_at: float | None,
    modified_at: float | None,
    accessed_at: float | None,
    indexed_at: float,
    file_hash: str | None,
    metadata_json: str = "{}",
) -> None:
    """
    Insert or update the active memory for a path.

    Every new or changed memory is marked as embedding-pending.

    Historical deleted memories are never overwritten.
    """

    existing = connection.execute(
        """
        SELECT id
        FROM memories
        WHERE path = ?
          AND source_type = ?
          AND is_deleted = 0
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            path,
            source_type,
        ),
    ).fetchone()

    if existing is not None:
        connection.execute(
            """
            UPDATE memories
            SET
                content_type = ?,
                title = ?,
                content = ?,
                size_bytes = ?,
                created_at = ?,
                modified_at = ?,
                accessed_at = ?,
                indexed_at = ?,
                file_hash = ?,
                is_deleted = 0,
                deleted_at = NULL,
                metadata_json = ?,
                embedding_status = ?,
                embedding_error = NULL
            WHERE id = ?
            """,
            (
                content_type,
                title,
                content,
                size_bytes,
                created_at,
                modified_at,
                accessed_at,
                indexed_at,
                file_hash,
                metadata_json,
                EMBEDDING_STATUS_PENDING,
                existing["id"],
            ),
        )

        return

    connection.execute(
        """
        INSERT INTO memories (
            source_type,
            content_type,
            title,
            path,
            content,
            size_bytes,
            created_at,
            modified_at,
            accessed_at,
            indexed_at,
            file_hash,
            is_deleted,
            deleted_at,
            metadata_json,
            embedding_status,
            embedding_version,
            embedding_updated_at,
            embedding_error
        )
        VALUES (
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            0,
            NULL,
            ?,
            ?,
            NULL,
            NULL,
            NULL
        )
        """,
        (
            source_type,
            content_type,
            title,
            path,
            content,
            size_bytes,
            created_at,
            modified_at,
            accessed_at,
            indexed_at,
            file_hash,
            metadata_json,
            EMBEDDING_STATUS_PENDING,
        ),
    )


# ---------------------------------------------------------------------------
# FTS search
# ---------------------------------------------------------------------------


def _build_fts_query(query: str) -> str:
    """
    Convert a user search into a safe FTS5 prefix query.

    Examples:
        "res" -> "res"*
        "aws networking" -> "aws"* AND "networking"*
        "GLIRE report" -> "GLIRE"* AND "report"*
    """

    tokens = re.findall(
        r"[A-Za-z0-9]+",
        query,
    )

    if not tokens:
        return ""

    return " AND ".join(
        f'"{token}"*'
        for token in tokens
    )


def search_memories(
    connection: sqlite3.Connection,
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Search memories using SQLite FTS5 with improved relevance.

    Ranking priorities:

    1. Active memories before deleted memories.
    2. Exact title match.
    3. Title containing the query.
    4. Path containing the query.
    5. FTS5 BM25 relevance.
    6. More recently modified files as a tie-breaker.

    Deleted memories remain searchable because they are retained
    as historical memories.
    """

    query = query.strip()

    if not query:
        return []

    fts_query = _build_fts_query(query)

    if not fts_query:
        return []

    rows = connection.execute(
        """
        SELECT
            memories.id,
            memories.source_type,
            memories.content_type,
            memories.title,
            memories.path,
            memories.modified_at,
            memories.accessed_at,
            memories.is_deleted,
            memories.deleted_at,
            memories.metadata_json,

            bm25(
                memories_fts,
                8.0,
                1.0
            ) AS rank

        FROM memories_fts

        JOIN memories
            ON memories.id = memories_fts.rowid

        WHERE memories_fts MATCH ?

        ORDER BY
            memories.is_deleted ASC,

            CASE
                WHEN lower(memories.title) = lower(?)
                THEN 0
                ELSE 1
            END ASC,

            CASE
                WHEN instr(
                    lower(memories.title),
                    lower(?)
                ) > 0
                THEN 0
                ELSE 1
            END ASC,

            CASE
                WHEN instr(
                    lower(memories.path),
                    lower(?)
                ) > 0
                THEN 0
                ELSE 1
            END ASC,

            rank ASC,

            memories.modified_at DESC

        LIMIT ?
        """,
        (
            fts_query,
            query,
            query,
            query,
            limit,
        ),
    ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def get_memory_count(
    connection: sqlite3.Connection,
) -> int:
    """
    Return the total number of stored memories.

    This includes historical deleted memories.
    """

    row = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM memories
        """
    ).fetchone()

    return int(row["count"])


def get_active_memory_count(
    connection: sqlite3.Connection,
) -> int:
    """
    Return the number of currently active memories.
    """

    row = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM memories
        WHERE is_deleted = 0
        """
    ).fetchone()

    return int(row["count"])


def get_deleted_memory_count(
    connection: sqlite3.Connection,
) -> int:
    """
    Return the number of historical deleted memories.
    """

    row = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM memories
        WHERE is_deleted = 1
        """
    ).fetchone()

    return int(row["count"])