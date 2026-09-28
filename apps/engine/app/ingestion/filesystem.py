from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import DATABASE_PATH
from app.extraction.file_extractor import (
    SUPPORTED_EXTENSIONS,
    build_metadata_context,
    extract_text,
    get_indexing_mode,
    iter_text_chunks,
)
from app.storage.database import (
    get_deleted_memory_by_path_and_hash,
    get_memory_by_path,
    get_active_memory_paths,
    mark_memory_deleted,
    restore_memory,
    upsert_memory,
)


# ---------------------------------------------------------------------------
# Filesystem configuration
# ---------------------------------------------------------------------------

SKIP_DIRECTORY_NAMES = {
    "AppData",
    "node_modules",
    ".git",
    ".svn",
    ".hg",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    "site-packages",
    "cache",
    "caches",
    "temp",
    "tmp",
}


# Maximum amount of content retained for a chunked file while the database
# still uses one memory row per file.
MAX_CHUNKED_CONTENT_CHARS = 500_000


# Read files in bounded blocks when calculating SHA-256.
HASH_BLOCK_SIZE = 1024 * 1024


ProgressCallback = Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Filesystem roots
# ---------------------------------------------------------------------------


def get_index_roots() -> list[Path]:
    """
    Return the user directories RecallX currently indexes.
    """

    home = Path.home()

    candidates = [
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
        home / "Pictures",
        home / "Videos",
    ]

    return [
        path
        for path in candidates
        if path.exists() and path.is_dir()
    ]


def should_skip_directory(path: Path) -> bool:
    """
    Return True when a directory is intentionally excluded.
    """

    return any(
        part in SKIP_DIRECTORY_NAMES
        for part in path.parts
    )


def should_skip_file(path: Path) -> bool:
    """
    Skip temporary files such as Microsoft Office lock files.
    """

    return path.name.startswith("~$")


def iter_files(root: Path):
    """
    Yield filesystem files beneath one indexing root.
    """

    for current_root, directories, filenames in os.walk(root):
        current_path = Path(current_root)

        directories[:] = [
            directory
            for directory in directories
            if directory not in SKIP_DIRECTORY_NAMES
        ]

        if should_skip_directory(current_path):
            directories[:] = []
            continue

        for filename in filenames:
            path = current_path / filename

            if should_skip_file(path):
                continue

            yield path


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def calculate_file_hash(path: Path) -> str:
    """
    Calculate the SHA-256 hash of a file.

    The entire file is never loaded into memory.
    """

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(HASH_BLOCK_SIZE)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


def _extract_chunked_content(path: Path) -> str:
    """
    Extract a bounded amount of content from a larger file.

    The complete large file is never assembled in memory.
    """

    chunks: list[str] = []
    total_chars = 0

    for chunk in iter_text_chunks(path):
        remaining = (
            MAX_CHUNKED_CONTENT_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        bounded_chunk = chunk[:remaining]

        if bounded_chunk.strip():
            chunks.append(bounded_chunk)
            total_chars += len(bounded_chunk)

    return "\n\n".join(chunks)


def _build_metadata_json(
    path: Path,
    indexing_mode: str,
    *,
    extraction: str,
    content_fallback: bool = False,
) -> str:
    """
    Build structured metadata for a filesystem memory.

    Core file attributes such as path, size, timestamps, and SHA-256
    are stored in dedicated SQLite columns and are intentionally not
    duplicated here.
    """

    return json.dumps(
        {
            "source": "filesystem",
            "indexing_mode": indexing_mode,
            "extension": path.suffix.lower(),
            "file_name": path.name,
            "parent_directory": str(path.parent),
            "extraction": extraction,
            "content_fallback": content_fallback,
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Individual file indexing
# ---------------------------------------------------------------------------


def index_file(
    connection: sqlite3.Connection,
    path: Path,
) -> tuple[str, str, int | None]:
    """
    Incrementally index one file.

    Returns:

        ("indexed", mode, memory_id)
            New file.

        ("updated", mode, memory_id)
            Existing file whose content changed.

        ("unchanged", "none", memory_id)
            Existing file with identical content.

        ("restored", mode, memory_id)
            Previously deleted memory whose exact content returned.

        ("skipped", "skipped", None)
            Unsupported or unavailable file.

        ("failed", "failed", None)
            File could not be processed.
    """

    try:
        # ---------------------------------------------------------------
        # Validate file
        # ---------------------------------------------------------------

        if not path.is_file():
            return "skipped", "skipped", None

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            return "skipped", "skipped", None

        stat = path.stat()

        path_string = str(path)

        # ---------------------------------------------------------------
        # Look for currently active memory
        # ---------------------------------------------------------------

        existing_memory = get_memory_by_path(
            connection,
            path_string,
        )

        current_hash: str | None = None

        # ---------------------------------------------------------------
        # Existing active file
        # ---------------------------------------------------------------

        if existing_memory is not None:
            old_hash = existing_memory["file_hash"]

            # Content hash is the source of truth.
            current_hash = calculate_file_hash(path)

            if old_hash and current_hash == old_hash:
                return (
                    "unchanged",
                    "none",
                    int(existing_memory["id"]),
                )

        # ---------------------------------------------------------------
        # New path or changed existing file
        # ---------------------------------------------------------------

        if existing_memory is None:
            # We need the hash to determine whether this file is a
            # restoration of an old deleted memory.
            current_hash = calculate_file_hash(path)

            deleted_memory = get_deleted_memory_by_path_and_hash(
                connection,
                path_string,
                current_hash,
            )

            if deleted_memory is not None:
                indexing_mode = get_indexing_mode(
                    path,
                    stat.st_size,
                )

                restore_memory(
                    connection,
                    deleted_memory["id"],
                    modified_at=stat.st_mtime,
                    accessed_at=stat.st_atime,
                    indexed_at=time.time(),
                )

                # Update the metadata/content so the restored memory
                # represents the current file.
                if indexing_mode == "metadata_only":
                    content = build_metadata_context(
                        path,
                        stat.st_size,
                    )

                    extraction = "metadata"
                    content_fallback = False

                elif indexing_mode == "chunked":
                    content = _extract_chunked_content(path)

                    if not content.strip():
                        content = build_metadata_context(
                            path,
                            stat.st_size,
                        )

                        extraction = "metadata"
                        content_fallback = True
                    else:
                        extraction = "text_chunked"
                        content_fallback = False

                else:
                    content = extract_text(path)

                    # Empty files are valid files.
                    # Store metadata instead of treating them as failures.
                    if not content.strip():
                        content = build_metadata_context(
                            path,
                            stat.st_size,
                        )

                        extraction = "metadata"
                        content_fallback = True
                    else:
                        extraction = "text"
                        content_fallback = False

                upsert_memory(
                    connection,
                    source_type="filesystem",
                    content_type=extension.lstrip("."),
                    title=path.name,
                    path=path_string,
                    content=content,
                    size_bytes=stat.st_size,
                    created_at=stat.st_ctime,
                    modified_at=stat.st_mtime,
                    accessed_at=stat.st_atime,
                    indexed_at=time.time(),
                    file_hash=current_hash,
                    metadata_json=_build_metadata_json(
                        path,
                        indexing_mode,
                        extraction=extraction,
                        content_fallback=content_fallback,
                    ),
                )

                return (
                    "restored",
                    indexing_mode,
                    int(deleted_memory["id"]),
                )

            action = "indexed"

        else:
            action = "updated"

            if current_hash is None:
                current_hash = calculate_file_hash(path)

        # ---------------------------------------------------------------
        # Determine extraction mode
        # ---------------------------------------------------------------

        indexing_mode = get_indexing_mode(
            path,
            stat.st_size,
        )

        # ---------------------------------------------------------------
        # Extract content
        # ---------------------------------------------------------------

        if indexing_mode == "metadata_only":
            content = build_metadata_context(
                path,
                stat.st_size,
            )

            extraction = "metadata"
            content_fallback = False

        elif indexing_mode == "chunked":
            content = _extract_chunked_content(path)

            if not content.strip():
                content = build_metadata_context(
                    path,
                    stat.st_size,
                )

                extraction = "metadata"
                content_fallback = True
            else:
                extraction = "text_chunked"
                content_fallback = False

        else:
            content = extract_text(path)

            # -----------------------------------------------------------
            # IMPORTANT:
            #
            # An empty file is not an indexing error.
            #
            # Store useful metadata instead of returning:
            # ("failed", "full")
            #
            # This also prevents the watcher from pointlessly retrying
            # an empty file three times.
            # -----------------------------------------------------------

            if not content.strip():
                print(
                    f"RecallX: file contains no extractable text; "
                    f"using metadata fallback: {path}"
                )

                content = build_metadata_context(
                    path,
                    stat.st_size,
                )

                indexing_mode = "metadata_only"
                extraction = "metadata"
                content_fallback = True

            else:
                extraction = "text"
                content_fallback = False

        # ---------------------------------------------------------------
        # Store memory
        # ---------------------------------------------------------------

        if current_hash is None:
            current_hash = calculate_file_hash(path)

        upsert_memory(
            connection,
            source_type="filesystem",
            content_type=extension.lstrip("."),
            title=path.name,
            path=path_string,
            content=content,
            size_bytes=stat.st_size,
            created_at=stat.st_ctime,
            modified_at=stat.st_mtime,
            accessed_at=stat.st_atime,
            indexed_at=time.time(),
            file_hash=current_hash,
            metadata_json=_build_metadata_json(
                path,
                indexing_mode,
                extraction=extraction,
                content_fallback=content_fallback,
            ),
        )

        # Retrieve the memory ID assigned by SQLite.
        memory = get_memory_by_path(
            connection,
            path_string,
        )

        if memory is None:
            raise RuntimeError(
                f"Memory was not found after indexing: {path_string}"
            )

        memory_id = int(memory["id"])

        return action, indexing_mode, memory_id

    except MemoryError:
        # ---------------------------------------------------------------
        # Memory safety fallback
        # ---------------------------------------------------------------

        print(
            f"RecallX: memory error while indexing {path}. "
            "Falling back to metadata-only."
        )

        try:
            stat = path.stat()

            current_hash = calculate_file_hash(path)

            existing_memory = get_memory_by_path(
                connection,
                str(path),
            )

            action = (
                "updated"
                if existing_memory is not None
                else "indexed"
            )

            content = build_metadata_context(
                path,
                stat.st_size,
            )

            upsert_memory(
                connection,
                source_type="filesystem",
                content_type=path.suffix.lower().lstrip("."),
                title=path.name,
                path=str(path),
                content=content,
                size_bytes=stat.st_size,
                created_at=stat.st_ctime,
                modified_at=stat.st_mtime,
                accessed_at=stat.st_atime,
                indexed_at=time.time(),
                file_hash=current_hash,
                metadata_json=_build_metadata_json(
                    path,
                    "metadata_only",
                    extraction="metadata",
                    content_fallback=True,
                ),
            )

            memory = get_memory_by_path(
                connection,
                str(path),
            )

            if memory is None:
                raise RuntimeError(
                    f"Memory was not found after metadata fallback: {path}"
                )

            memory_id = int(memory["id"])

            return action, "metadata_only", memory_id

        except Exception as fallback_error:
            print(
                f"RecallX: metadata fallback failed for "
                f"{path}: "
                f"{type(fallback_error).__name__}: "
                f"{fallback_error}"
            )

            traceback.print_exc()

            return "failed", "failed", None

    except Exception as error:
        # ---------------------------------------------------------------
        # DO NOT HIDE THE REAL ERROR
        #
        # This is especially important for watcher indexing because
        # index_file() is called from a background worker.
        # ---------------------------------------------------------------

        print(
            f"RecallX: failed to process {path} "
            f"({type(error).__name__}: {error})"
        )

        traceback.print_exc()

        return "failed", "failed", None


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def _send_progress(
    progress_callback: ProgressCallback | None,
    *,
    scanned: int,
    indexed: int,
    updated: int,
    unchanged: int,
    restored: int,
    deleted: int,
    skipped: int,
    failed: int,
    full_count: int,
    chunked_count: int,
    metadata_only_count: int,
    current_file: str,
) -> None:
    """
    Send current indexing state to the API layer.
    """

    if progress_callback is None:
        return

    progress_callback(
        {
            "scanned": scanned,
            "indexed": indexed,
            "updated": updated,
            "unchanged": unchanged,
            "restored": restored,
            "deleted": deleted,
            "skipped": skipped,
            "failed": failed,
            "full": full_count,
            "chunked": chunked_count,
            "metadata_only": metadata_only_count,
            "current_file": current_file,
        }
    )


# ---------------------------------------------------------------------------
# Deleted-file reconciliation
# ---------------------------------------------------------------------------


def _path_belongs_to_index_roots(
    path: Path,
    roots: list[Path],
) -> bool:
    """
    Determine whether a stored path belongs to one of RecallX's
    currently indexed roots.
    """

    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue

    return False


def _should_reconcile_path(
    path: Path,
) -> bool:
    """
    Determine whether a stored memory should participate in
    deletion reconciliation.

    Unsupported files are not considered because RecallX never
    indexed them in the first place.
    """

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False

    if should_skip_file(path):
        return False

    if should_skip_directory(path):
        return False

    return True


def reconcile_deleted_files(
    connection: sqlite3.Connection,
    roots: list[Path],
    seen_paths: set[str],
) -> int:
    """
    Mark active memories as deleted when their files no longer exist.

    Historical memories are retained.

    Only files belonging to RecallX's currently indexed roots and
    supported file types are reconciled.
    """

    active_paths = get_active_memory_paths(
        connection,
        source_type="filesystem",
    )

    deleted_count = 0

    now = time.time()

    for memory in active_paths:
        path_string = str(memory["path"])
        path = Path(path_string)

        if not _path_belongs_to_index_roots(
            path,
            roots,
        ):
            continue

        if not _should_reconcile_path(path):
            continue

        # We saw this path during the current scan.
        if path_string in seen_paths:
            continue

        # The path was previously indexed but no longer exists.
        if path.exists():
            continue

        mark_memory_deleted(
            connection,
            int(memory["id"]),
            deleted_at=now,
        )

        deleted_count += 1

    return deleted_count


# ---------------------------------------------------------------------------
# Full filesystem indexing
# ---------------------------------------------------------------------------


def index_filesystem(
    connection: sqlite3.Connection,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """
    Incrementally index the filesystem using one SQLite connection.

    Content identity:
        SHA-256 hash

    Metadata:
        modified_at

    Actions:
        indexed
        updated
        unchanged
        restored
        deleted
        skipped
        failed

    Existing files are NOT re-extracted when their content hash
    matches the stored hash.
    """

    roots = get_index_roots()

    scanned = 0
    indexed = 0
    updated = 0
    unchanged = 0
    restored = 0
    deleted = 0
    skipped = 0
    failed = 0

    full_count = 0
    chunked_count = 0
    metadata_only_count = 0

    seen_paths: set[str] = set()

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Scan current filesystem
    # ---------------------------------------------------------------

    for root in roots:
        print(
            f"RecallX: indexing root: {root}"
        )

        for path in iter_files(root):
            scanned += 1

            path_string = str(path)

            # Record every encountered filesystem path.
            seen_paths.add(path_string)

            # -------------------------------------------------------
            # Tell API what file is currently being processed.
            # -------------------------------------------------------

            _send_progress(
                progress_callback,
                scanned=scanned,
                indexed=indexed,
                updated=updated,
                unchanged=unchanged,
                restored=restored,
                deleted=deleted,
                skipped=skipped,
                failed=failed,
                full_count=full_count,
                chunked_count=chunked_count,
                metadata_only_count=metadata_only_count,
                current_file=path_string,
            )

            # -------------------------------------------------------
            # Process file
            # -------------------------------------------------------

            action, indexing_mode, _memory_id = index_file(
                connection,
                path,
            )

            if action == "indexed":
                indexed += 1

            elif action == "updated":
                updated += 1

            elif action == "unchanged":
                unchanged += 1

            elif action == "restored":
                restored += 1

            elif action == "skipped":
                skipped += 1

            else:
                failed += 1

            # -------------------------------------------------------
            # Track extraction mode
            # -------------------------------------------------------

            if indexing_mode == "full":
                full_count += 1

            elif indexing_mode == "chunked":
                chunked_count += 1

            elif indexing_mode == "metadata_only":
                metadata_only_count += 1

            # -------------------------------------------------------
            # Periodic commit/progress
            # -------------------------------------------------------

            if scanned % 100 == 0:
                connection.commit()

                print(
                    "RecallX: progress "
                    f"Scanned={scanned} "
                    f"Indexed={indexed} "
                    f"Updated={updated} "
                    f"Unchanged={unchanged} "
                    f"Restored={restored} "
                    f"Skipped={skipped} "
                    f"Failed={failed}"
                )

                _send_progress(
                    progress_callback,
                    scanned=scanned,
                    indexed=indexed,
                    updated=updated,
                    unchanged=unchanged,
                    restored=restored,
                    deleted=deleted,
                    skipped=skipped,
                    failed=failed,
                    full_count=full_count,
                    chunked_count=chunked_count,
                    metadata_only_count=metadata_only_count,
                    current_file=path_string,
                )

            if scanned % 25 == 0:
                time.sleep(0.01)

    # ---------------------------------------------------------------
    # Reconcile deleted files AFTER the complete scan.
    #
    # We cannot mark a file deleted while scanning because we don't
    # know whether we simply haven't reached it yet.
    # ---------------------------------------------------------------

    deleted = reconcile_deleted_files(
        connection,
        roots,
        seen_paths,
    )

    # ---------------------------------------------------------------
    # Final commit
    # ---------------------------------------------------------------

    connection.commit()

    _send_progress(
        progress_callback,
        scanned=scanned,
        indexed=indexed,
        updated=updated,
        unchanged=unchanged,
        restored=restored,
        deleted=deleted,
        skipped=skipped,
        failed=failed,
        full_count=full_count,
        chunked_count=chunked_count,
        metadata_only_count=metadata_only_count,
        current_file="",
    )

    return {
        "roots": len(roots),
        "scanned": scanned,
        "indexed": indexed,
        "updated": updated,
        "unchanged": unchanged,
        "restored": restored,
        "deleted": deleted,
        "skipped": skipped,
        "failed": failed,
        "full": full_count,
        "chunked": chunked_count,
        "metadata_only": metadata_only_count,
    }