from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from app.embeddings.semantic_sync import (
    remove_memory_embedding,
    sync_memory_embedding,
)
from app.extraction.file_extractor import SUPPORTED_EXTENSIONS
from app.ingestion.filesystem import index_file
from app.storage.database import (
    CURRENT_EMBEDDING_VERSION,
    EMBEDDING_STATUS_FAILED,
    EMBEDDING_STATUS_PENDING,
    EMBEDDING_STATUS_SYNCED,
    get_connection,
    get_memory_by_path,
    mark_memory_deleted,
)


# ---------------------------------------------------------------------------
# Watched roots
# ---------------------------------------------------------------------------

WATCH_ROOTS = [
    Path.home() / "Desktop",
    Path.home() / "Documents",
    Path.home() / "Downloads",
    Path.home() / "Videos",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEBOUNCE_SECONDS = 0.75
FILE_READY_DELAY_SECONDS = 0.5

MAX_INDEX_RETRIES = 3
INDEX_RETRY_DELAY_SECONDS = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_path(
    path: str | Path,
) -> str:
    return str(
        Path(path).resolve()
    )


def is_supported_file(
    path: Path,
) -> bool:
    return (
        path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Filesystem event handler
# ---------------------------------------------------------------------------


class RecallXEventHandler(
    FileSystemEventHandler
):
    """
    Receives Windows filesystem events.

    The handler does not perform indexing itself.
    It only places events into the worker queue.
    """

    def __init__(
        self,
        event_queue: queue.Queue[
            tuple[str, str]
        ],
    ) -> None:
        super().__init__()

        self.event_queue = event_queue

    def _enqueue(
        self,
        path: str,
        event_type: str,
    ) -> None:
        self.event_queue.put(
            (
                normalize_path(path),
                event_type,
            )
        )

    def on_created(
        self,
        event: FileSystemEvent,
    ) -> None:
        if event.is_directory:
            return

        self._enqueue(
            event.src_path,
            "created",
        )

    def on_modified(
        self,
        event: FileSystemEvent,
    ) -> None:
        if event.is_directory:
            return

        self._enqueue(
            event.src_path,
            "modified",
        )

    def on_deleted(
        self,
        event: FileSystemEvent,
    ) -> None:
        if event.is_directory:
            return

        self._enqueue(
            event.src_path,
            "deleted",
        )

    def on_moved(
        self,
        event: FileSystemEvent,
    ) -> None:
        if event.is_directory:
            return

        # Treat a move as:
        #
        # old path -> deleted
        # new path -> created

        self._enqueue(
            event.src_path,
            "deleted",
        )

        self._enqueue(
            event.dest_path,
            "created",
        )


# ---------------------------------------------------------------------------
# RecallX watcher
# ---------------------------------------------------------------------------


class RecallXWatcher:
    """
    One filesystem observer and one indexing worker for RecallX.

    Architecture:

        Windows filesystem
                ↓
          watchdog observer
                ↓
            event queue
                ↓
             single worker
                ↓
           filesystem.py
                ↓
              SQLite
                ↓
        semantic_sync.py
                ↓
             MiniLM
                ↓
              FAISS

    The database is the source of truth for whether an embedding
    is synchronized with a memory.
    """

    def __init__(
        self,
        progress_callback: Callable[
            [dict[str, Any]],
            None,
        ],
    ) -> None:
        self.progress_callback = progress_callback

        self.observer: Observer | None = None

        self.connection = None

        self.event_queue: queue.Queue[
            tuple[str, str]
        ] = queue.Queue()

        self.handler: RecallXEventHandler | None = None

        self.worker_thread: threading.Thread | None = None

        self.stop_event = threading.Event()

        self._last_event: dict[
            tuple[str, str],
            float,
        ] = {}

        self._running = False

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return

        self.connection = get_connection()

        self.handler = RecallXEventHandler(
            self.event_queue,
        )

        self.observer = Observer()

        scheduled_roots = 0

        for root in WATCH_ROOTS:
            root = root.resolve()

            if not root.exists():
                print(
                    "RecallX watcher: root does not exist: "
                    f"{root}"
                )
                continue

            if not root.is_dir():
                continue

            self.observer.schedule(
                self.handler,
                str(root),
                recursive=True,
            )

            scheduled_roots += 1

            print(
                "RecallX watcher: watching "
                f"{root}"
            )

        if scheduled_roots == 0:
            self.connection.close()

            self.connection = None
            self.handler = None
            self.observer = None

            raise RuntimeError(
                "RecallX watcher: no valid filesystem roots."
            )

        self.stop_event.clear()

        self.observer.start()

        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="RecallXWatcherWorker",
            daemon=True,
        )

        self.worker_thread.start()

        self._running = True

        self.progress_callback(
            {
                "status": "watching",
                "current_file": "",
                "error": "",
            }
        )

        print(
            "RecallX watcher: started with "
            f"{scheduled_roots} roots."
        )

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                path, event_type = (
                    self.event_queue.get(
                        timeout=0.5,
                    )
                )

            except queue.Empty:
                continue

            try:
                self._process_event(
                    path,
                    event_type,
                )

            except Exception as error:
                self.progress_callback(
                    {
                        "status": "watching",
                        "failed": 1,
                        "current_file": path,
                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    }
                )

                print(
                    "RecallX watcher: failed to process "
                    f"{path}: "
                    f"{type(error).__name__}: {error}"
                )

            finally:
                self.event_queue.task_done()

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def _process_event(
        self,
        path: str,
        event_type: str,
    ) -> None:
        key = (
            path,
            event_type,
        )

        now = time.monotonic()

        previous = self._last_event.get(
            key,
        )

        if (
            previous is not None
            and now - previous
            < DEBOUNCE_SECONDS
        ):
            return

        self._last_event[key] = now

        file_path = Path(path)

        # --------------------------------------------------------------
        # Deleted
        # --------------------------------------------------------------

        if event_type == "deleted":
            self._handle_deleted(
                path,
            )
            return

        # --------------------------------------------------------------
        # File disappeared before processing
        # --------------------------------------------------------------

        if not file_path.exists():
            self._handle_deleted(
                path,
            )
            return

        # --------------------------------------------------------------
        # Unsupported file
        # --------------------------------------------------------------

        if not is_supported_file(
            file_path,
        ):
            return

        # --------------------------------------------------------------
        # Give the application time to finish writing
        # --------------------------------------------------------------

        if event_type == "created":
            time.sleep(
                FILE_READY_DELAY_SECONDS
            )

        # --------------------------------------------------------------
        # Created / modified
        # --------------------------------------------------------------

        self._handle_index(
            file_path,
        )

    # ------------------------------------------------------------------
    # Embedding state
    # ------------------------------------------------------------------

    def _embedding_needs_sync(
        self,
        memory_id: int,
    ) -> bool:
        """
        Determine whether a memory needs semantic synchronization.

        This is deliberately checked independently of the filesystem
        action. A file can be unchanged while its embedding is still
        pending or failed from an earlier attempt.
        """

        if self.connection is None:
            return False

        memory = self.connection.execute(
            """
            SELECT
                is_deleted,
                embedding_status,
                embedding_version
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()

        if memory is None:
            return False

        if bool(memory["is_deleted"]):
            return False

        status = memory["embedding_status"]
        version = memory["embedding_version"]

        return (
            status != EMBEDDING_STATUS_SYNCED
            or version != CURRENT_EMBEDDING_VERSION
            or status in {
                EMBEDDING_STATUS_PENDING,
                EMBEDDING_STATUS_FAILED,
            }
        )

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _handle_index(
        self,
        file_path: Path,
    ) -> None:
        if self.connection is None:
            return

        last_action = "failed"
        last_mode = "failed"

        for attempt in range(
            1,
            MAX_INDEX_RETRIES + 1,
        ):
            # The file may have disappeared between the
            # filesystem event and this worker processing it.
            if not file_path.exists():
                self._handle_deleted(
                    str(file_path),
                )
                return

            try:
                # filesystem.py returns:
                #
                #     (action, indexing_mode, memory_id)

                action, indexing_mode, memory_id = index_file(
                    self.connection,
                    file_path,
                )

                last_action = action
                last_mode = indexing_mode

                # ------------------------------------------------------
                # Filesystem indexing failed
                # ------------------------------------------------------

                if action == "failed":
                    self.connection.rollback()

                    if attempt < MAX_INDEX_RETRIES:
                        print(
                            "RecallX watcher: indexing failed for "
                            f"{file_path}; retrying "
                            f"({attempt}/{MAX_INDEX_RETRIES})..."
                        )

                        time.sleep(
                            INDEX_RETRY_DELAY_SECONDS
                        )

                    continue

                # ------------------------------------------------------
                # Commit filesystem/database changes
                # ------------------------------------------------------

                self.connection.commit()

                # ------------------------------------------------------
                # Semantic synchronization
                # ------------------------------------------------------

                if memory_id is not None:
                    needs_sync = (
                        action in {
                            "indexed",
                            "updated",
                            "restored",
                        }
                        or self._embedding_needs_sync(
                            memory_id,
                        )
                    )

                    if needs_sync:
                        print(
                            "RecallX watcher: generating semantic "
                            f"embedding for memory {memory_id}: "
                            f"{file_path}"
                        )

                        sync_memory_embedding(
                            self.connection,
                            memory_id,
                        )

                        print(
                            "RecallX watcher: semantic index "
                            f"updated for memory {memory_id}"
                        )

                self._report_index_result(
                    file_path,
                    action,
                    indexing_mode,
                )

                return

            except Exception as error:
                # Important:
                #
                # sync_memory_embedding() records its own FAILED
                # embedding state before re-raising. We therefore
                # rollback only the caller's uncommitted transaction.
                self.connection.rollback()

                print(
                    "RecallX watcher: exception while indexing "
                    f"{file_path}: "
                    f"{type(error).__name__}: {error}"
                )

                if attempt < MAX_INDEX_RETRIES:
                    print(
                        "RecallX watcher: retrying "
                        f"{file_path} "
                        f"({attempt}/{MAX_INDEX_RETRIES})..."
                    )

                    time.sleep(
                        INDEX_RETRY_DELAY_SECONDS
                    )

                else:
                    self._report_failure(
                        file_path,
                        error,
                    )

                    return

        # --------------------------------------------------------------
        # All retries failed
        # --------------------------------------------------------------

        self.progress_callback(
            {
                "status": "watching",
                "failed": 1,
                "current_file": str(file_path),
                "error": (
                    "index_file returned failed after "
                    f"{MAX_INDEX_RETRIES} attempts"
                ),
            }
        )

        print(
            "RecallX watcher: failed to index "
            f"{file_path} after "
            f"{MAX_INDEX_RETRIES} attempts "
            f"(action={last_action}, "
            f"mode={last_mode})"
        )

    # ------------------------------------------------------------------
    # Report successful index
    # ------------------------------------------------------------------

    def _report_index_result(
        self,
        file_path: Path,
        action: str,
        indexing_mode: str,
    ) -> None:
        progress: dict[str, Any] = {
            "status": "watching",
            "current_file": str(file_path),
            "error": "",
        }

        if action == "indexed":
            progress["indexed"] = 1

        elif action == "updated":
            progress["updated"] = 1

        elif action == "unchanged":
            progress["unchanged"] = 1

        elif action == "restored":
            progress["restored"] = 1

        elif action == "skipped":
            progress["skipped"] = 1

        elif action == "failed":
            progress["failed"] = 1

        if indexing_mode == "full":
            progress["full"] = 1

        elif indexing_mode == "chunked":
            progress["chunked"] = 1

        elif indexing_mode == "metadata_only":
            progress["metadata_only"] = 1

        self.progress_callback(
            progress,
        )

        print(
            "RecallX watcher: "
            f"{action or 'processed'} "
            f"{file_path}"
        )

    # ------------------------------------------------------------------
    # Report unexpected exception
    # ------------------------------------------------------------------

    def _report_failure(
        self,
        file_path: Path,
        error: Exception,
    ) -> None:
        error_message = (
            f"{type(error).__name__}: {error}"
        )

        self.progress_callback(
            {
                "status": "watching",
                "failed": 1,
                "current_file": str(file_path),
                "error": error_message,
            }
        )

        print(
            "RecallX watcher: exception while indexing "
            f"{file_path}: {error_message}"
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _handle_deleted(
        self,
        path: str,
    ) -> None:
        if self.connection is None:
            return

        memory = get_memory_by_path(
            self.connection,
            path,
            source_type="filesystem",
        )

        if memory is None:
            return

        memory_id = int(memory["id"])

        mark_memory_deleted(
            self.connection,
            memory_id,
            deleted_at=time.time(),
        )

        self.connection.commit()

        # Remove the corresponding semantic vector.
        #
        # Retry because the SQLite state is already marked deleted.
        # A later reconciliation pass can clean up any vector that
        # could not be removed here.
        print(
            "RecallX watcher: removing semantic embedding "
            f"for memory {memory_id}"
        )

        last_error: Exception | None = None

        for attempt in range(
            1,
            MAX_INDEX_RETRIES + 1,
        ):
            try:
                remove_memory_embedding(
                    memory_id,
                )

                last_error = None
                break

            except Exception as error:
                last_error = error

                print(
                    "RecallX watcher: failed to remove semantic "
                    f"embedding for memory {memory_id}; "
                    f"retrying ({attempt}/{MAX_INDEX_RETRIES}): "
                    f"{type(error).__name__}: {error}"
                )

                if attempt < MAX_INDEX_RETRIES:
                    time.sleep(
                        INDEX_RETRY_DELAY_SECONDS
                    )

        if last_error is not None:
            self.progress_callback(
                {
                    "status": "watching",
                    "failed": 1,
                    "deleted": 1,
                    "current_file": path,
                    "error": (
                        "Failed to remove semantic embedding "
                        f"for memory {memory_id}: "
                        f"{type(last_error).__name__}: "
                        f"{last_error}"
                    ),
                }
            )

            print(
                "RecallX watcher: semantic embedding could not "
                f"be removed for deleted memory {memory_id}"
            )

            return

        self.progress_callback(
            {
                "status": "watching",
                "deleted": 1,
                "current_file": path,
                "error": "",
            }
        )

        print(
            "RecallX watcher: deleted "
            f"{path}"
        )

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self) -> None:
        if not self._running:
            return

        print(
            "RecallX watcher: stopping..."
        )

        self.stop_event.set()

        # Stop receiving new events.
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()

        # Give the worker time to finish the current operation.
        if self.worker_thread is not None:
            self.worker_thread.join(
                timeout=5,
            )

        if self.connection is not None:
            self.connection.close()

        self.observer = None
        self.handler = None
        self.worker_thread = None
        self.connection = None

        self._running = False

        print(
            "RecallX watcher: stopped."
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running