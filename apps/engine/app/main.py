from __future__ import annotations

import asyncio
import re
import threading
import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.ingestion.filesystem import index_filesystem
from app.ingestion.watcher import RecallXWatcher
from app.retrieval.search import search as hybrid_search
from app.storage.database import (
    get_connection,
    get_memory_count,
    initialize_database,
)

from app.integrations.gmail import (
    connect_gmail,
    disconnect_gmail,
    get_gmail_profile,
    is_gmail_connected,
    search_gmail,
)


# ---------------------------------------------------------------------------
# Global watcher
# ---------------------------------------------------------------------------

watcher: RecallXWatcher | None = None


# ---------------------------------------------------------------------------
# Indexing progress
# ---------------------------------------------------------------------------

index_status_lock = threading.Lock()

index_progress: dict[str, Any] = {
    "status": "idle",
    "scanned": 0,
    "indexed": 0,
    "updated": 0,
    "unchanged": 0,
    "restored": 0,
    "deleted": 0,
    "skipped": 0,
    "failed": 0,
    "full": 0,
    "chunked": 0,
    "metadata_only": 0,
    "total_memories": 0,
    "current_file": "",
    "error": "",
}


def update_index_progress(progress: dict[str, Any]) -> None:
    """
    Update indexing progress.

    Full filesystem scans provide cumulative values.
    Watcher events provide individual increments.
    """

    counter_fields = {
        "scanned",
        "indexed",
        "updated",
        "unchanged",
        "restored",
        "deleted",
        "skipped",
        "failed",
        "full",
        "chunked",
        "metadata_only",
    }

    with index_status_lock:
        for key, value in progress.items():
            if key in counter_fields:
                index_progress[key] += int(value)
            else:
                index_progress[key] = value


def get_index_progress() -> dict[str, Any]:
    with index_status_lock:
        return dict(index_progress)


def reset_index_progress() -> None:
    with index_status_lock:
        index_progress.update(
            {
                "status": "idle",
                "scanned": 0,
                "indexed": 0,
                "updated": 0,
                "unchanged": 0,
                "restored": 0,
                "deleted": 0,
                "skipped": 0,
                "failed": 0,
                "full": 0,
                "chunked": 0,
                "metadata_only": 0,
                "current_file": "",
                "error": "",
            }
        )


# ---------------------------------------------------------------------------
# Initial filesystem scan
# ---------------------------------------------------------------------------


def database_has_memories() -> bool:
    """
    Return True if RecallX already has at least one memory.

    This prevents a full filesystem scan on every application startup.
    """

    connection = get_connection()

    try:
        return get_memory_count(connection) > 0
    finally:
        connection.close()


def run_initial_scan() -> None:
    """
    Perform the first full filesystem scan.

    This runs only when the database does not contain any memories.
    """

    reset_index_progress()

    with index_status_lock:
        index_progress["status"] = "indexing"

    print("RecallX: first run detected.")
    print("RecallX: starting initial filesystem scan...")

    connection = get_connection()

    try:
        result = index_filesystem(
            connection=connection,
            progress_callback=update_index_progress,
        )

        connection.commit()

        with index_status_lock:
            index_progress["status"] = "complete"
            index_progress["current_file"] = ""
            index_progress["error"] = ""

        print(
            "RecallX: initial filesystem indexing complete. "
            f"Scanned={result.get('scanned', 0)} "
            f"Indexed={result.get('indexed', 0)} "
            f"Updated={result.get('updated', 0)} "
            f"Unchanged={result.get('unchanged', 0)} "
            f"Restored={result.get('restored', 0)} "
            f"Deleted={result.get('deleted', 0)} "
            f"Skipped={result.get('skipped', 0)} "
            f"Failed={result.get('failed', 0)}"
        )

    except Exception as error:
        with index_status_lock:
            index_progress["status"] = "error"
            index_progress["error"] = f"{type(error).__name__}: {error}"

        print(
            "RecallX: initial filesystem indexing failed: "
            f"{type(error).__name__}: {error}"
        )

        traceback.print_exc()

        raise

    finally:
        connection.close()


# ---------------------------------------------------------------------------
# FastAPI lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global watcher

    # ---------------------------------------------------------------
    # 1. Initialize database
    # ---------------------------------------------------------------

    initialize_database()

    print("RecallX: database ready.")

    # ---------------------------------------------------------------
    # 2. First-run detection
    # ---------------------------------------------------------------

    try:
        has_memories = database_has_memories()

        if not has_memories:
            # First-ever run.
            #
            # Run the complete filesystem scan BEFORE starting
            # the watcher so that the database has a consistent
            # initial state.
            await asyncio.to_thread(run_initial_scan)

        else:
            print(
                "RecallX: existing memory database detected. "
                "Skipping initial filesystem scan."
            )

        # -----------------------------------------------------------
        # 3. Start filesystem watcher
        # -----------------------------------------------------------

        watcher = RecallXWatcher(update_index_progress)
        watcher.start()

        print("RecallX: filesystem watcher ready.")

        yield

    except Exception as error:
        with index_status_lock:
            index_progress["status"] = "error"
            index_progress["error"] = f"{type(error).__name__}: {error}"

        print(
            "RecallX: startup failed: "
            f"{type(error).__name__}: {error}"
        )

        traceback.print_exc()

        raise

    finally:
        # -----------------------------------------------------------
        # 4. Stop watcher on application shutdown
        # -----------------------------------------------------------

        if watcher is not None:
            watcher.stop()
            watcher = None

        print("RecallX: filesystem watcher stopped.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RecallX Engine",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

# RecallX's Tauri frontend runs from a different WebView origin than
# the local FastAPI server. Allow the local frontend to call the engine.
#
# This is appropriate for the current local-only development setup.
# We can tighten this before production packaging.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Index status
# ---------------------------------------------------------------------------


@app.get("/index/status")
def index_status() -> dict[str, Any]:
    progress = get_index_progress()

    connection = get_connection()

    try:
        progress["total_memories"] = get_memory_count(connection)
    finally:
        connection.close()

    progress["watcher_running"] = (
        watcher is not None and watcher.running
    )

    return progress


# ---------------------------------------------------------------------------
# Gmail
# ---------------------------------------------------------------------------


@app.get("/gmail/status")
def gmail_status() -> dict[str, Any]:
    """
    Return the current Gmail connection status.
    """

    connected = is_gmail_connected()

    if not connected:
        return {
            "connected": False,
            "email": "",
        }

    try:
        profile = get_gmail_profile()

        return {
            "connected": True,
            "email": profile.get("email", ""),
        }

    except Exception as error:
        return {
            "connected": False,
            "email": "",
            "error": f"{type(error).__name__}: {error}",
        }


@app.post("/gmail/connect")
async def gmail_connect() -> dict[str, Any]:
    """
    Start the Gmail OAuth flow.

    The user's browser opens for Google authorization.
    """

    try:
        return await asyncio.to_thread(
            connect_gmail,
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Gmail connection failed: {error}",
        ) from error


@app.post("/gmail/disconnect")
def gmail_disconnect() -> dict[str, Any]:
    """
    Disconnect Gmail by removing locally stored credentials.
    """

    return disconnect_gmail()


@app.get("/gmail/search")
async def gmail_search(
    q: str = Query(
        ...,
        min_length=1,
        description="Gmail search query",
    ),
    limit: int = Query(
        10,
        ge=1,
        le=20,
        description="Maximum Gmail results",
    ),
) -> dict[str, Any]:
    """
    Search Gmail live.

    This does NOT synchronize the mailbox into RecallX's database.
    Gmail is queried only when this endpoint is explicitly called.
    """

    query = q.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Gmail search query cannot be empty.",
        )

    if not is_gmail_connected():
        raise HTTPException(
            status_code=401,
            detail="Gmail is not connected.",
        )

    try:
        results = await asyncio.to_thread(
            search_gmail,
            query,
            limit,
        )

        return {
            "query": query,
            "count": len(results),
            "results": results,
        }

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Gmail search failed: {error}",
        ) from error


# ---------------------------------------------------------------------------
# Search source routing
# ---------------------------------------------------------------------------

EMAIL_INTENT_WORDS = {
    "email",
    "emails",
    "e-mail",
    "e-mails",
    "mail",
    "mails",
    "gmail",
    "message",
    "messages",
}


EMAIL_INTENT_PHRASES = (
    "email from",
    "emails from",
    "mail from",
    "mails from",
    "message from",
    "messages from",
    "email about",
    "emails about",
    "mail about",
    "mails about",
    "message about",
    "messages about",
    "email regarding",
    "emails regarding",
    "mail regarding",
    "mails regarding",
    "message regarding",
    "messages regarding",
    "email subject",
    "emails subject",
    "mail subject",
    "mails subject",
    "message subject",
    "messages subject",
    "sent me",
    "sent by",
    "received from",
    "received an email",
    "received a mail",
)


def is_email_query(query: str) -> bool:
    """
    Decide whether the user explicitly wants email/message search.

    Ordinary local queries such as:
        kpmg resume
        aws networking
        python project

    stay local.

    Email-oriented queries such as:
        email from KPMG
        gmail internship
        messages from Microsoft
        mail about internship
        from:someone@gmail.com

    search Gmail as well.
    """

    normalized = query.strip().lower()

    if not normalized:
        return False

    # Explicit Gmail-style search operators.
    if re.search(
        r"(^|\s)(from|to|cc|bcc|subject|after|before):",
        normalized,
    ):
        return True

    # Natural-language email phrases.
    if any(
        phrase in normalized
        for phrase in EMAIL_INTENT_PHRASES
    ):
        return True

    # Explicit email/mail/message words.
    tokens = set(
        re.findall(
            r"[a-z0-9@._'-]+",
            normalized,
        )
    )

    return bool(tokens & EMAIL_INTENT_WORDS)


def build_gmail_query(query: str) -> str:
    """
    Convert RecallX-style natural language into a cleaner Gmail query.

    Examples:

        email from KPMG
            -> from:KPMG

        email to recruiter internship
            -> to:recruiter internship

        email subject internship
            -> subject:internship

        email about internship

        gmail internship
            -> internship

    Existing Gmail operators such as from:, subject:, after:
    are preserved.
    """

    cleaned = query.strip()

    # ---------------------------------------------------------------
    # Natural-language "from"
    # ---------------------------------------------------------------

    match = re.search(
        r"\b(?:email|emails|e-mail|e-mails|mail|mails|"
        r"message|messages)\s+from\s+([^\s]+)",
        cleaned,
        flags=re.IGNORECASE,
    )

    if match:
        sender = match.group(1).strip(".,;")
        cleaned = (
            cleaned[: match.start()]
            + f"from:{sender}"
            + cleaned[match.end() :]
        )

    # ---------------------------------------------------------------
    # Natural-language "to"
    # ---------------------------------------------------------------

    match = re.search(
        r"\b(?:email|emails|e-mail|e-mails|mail|mails|"
        r"message|messages)\s+to\s+([^\s]+)",
        cleaned,
        flags=re.IGNORECASE,
    )

    if match:
        recipient = match.group(1).strip(".,;")
        cleaned = (
            cleaned[: match.start()]
            + f"to:{recipient}"
            + cleaned[match.end() :]
        )

    # ---------------------------------------------------------------
    # Natural-language "subject"
    # ---------------------------------------------------------------

    match = re.search(
        r"\b(?:email|emails|e-mail|e-mails|mail|mails|"
        r"message|messages)\s+subject\s+([^\s]+)",
        cleaned,
        flags=re.IGNORECASE,
    )

    if match:
        subject = match.group(1).strip(".,;")
        cleaned = (
            cleaned[: match.start()]
            + f"subject:{subject}"
            + cleaned[match.end() :]
        )

    # ---------------------------------------------------------------
    # Remove routing words.
    # ---------------------------------------------------------------

    cleaned = re.sub(
        r"\b(?:email|emails|e-mail|e-mails|mail|mails|"
        r"gmail|message|messages)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()

    return cleaned or query.strip()


def normalize_gmail_result(
    result: dict[str, Any],
    rank: float,
) -> dict[str, Any]:
    """
    Convert a Gmail result into the same shape used by the
    RecallX frontend for local search results.
    """

    message_id = str(
        result.get("id")
        or result.get("message_id")
        or ""
    )

    title = (
        result.get("title")
        or result.get("subject")
        or "(No subject)"
    )

    sender = result.get("sender") or ""
    recipient = result.get("recipient") or ""
    date = result.get("date") or ""
    snippet = result.get("snippet") or ""
    thread_id = result.get("thread_id") or message_id

    path = (
        result.get("path")
        or f"gmail:{message_id}"
    )

    url = result.get("url")

    metadata = {
        "source": "gmail",
        "provider": "gmail",
        "message_id": message_id,
        "thread_id": thread_id,
        "sender": sender,
        "recipient": recipient,
        "date": date,
        "url": url,
    }

    metadata_parts: list[str] = []

    if sender:
        metadata_parts.append(f"From: {sender}")

    if recipient:
        metadata_parts.append(f"To: {recipient}")

    if date:
        metadata_parts.append(f"Date: {date}")

    content_parts: list[str] = []

    if metadata_parts:
        content_parts.append(
            " | ".join(metadata_parts)
        )

    if snippet:
        content_parts.append(snippet)

    content = "\n".join(content_parts)

    return {
        "id": f"gmail:{message_id}",
        "source_type": "gmail",
        "content_type": "email",
        "title": title,
        "path": path,
        "content": content,
        "modified_at": None,
        "accessed_at": None,
        "is_deleted": 0,
        "deleted_at": None,
        "metadata_json": metadata,
        "rank": rank,
        "lexical_rank": None,
        "semantic_rank": None,
        "temporal_rank": None,
        "context": None,
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@app.get("/search")
async def search(
    q: str = Query(
        ...,
        min_length=1,
        description="Search query",
    ),
    limit: int = Query(
        20,
        ge=1,
        le=100,
        description="Maximum number of results",
    ),
) -> dict[str, Any]:
    """
    Main RecallX search endpoint.

    Every query searches the local RecallX memory.

    Explicit email/message queries are routed directly to Gmail.
    Ordinary queries use local RecallX memory only.

    Examples:

        "kpmg resume"
            -> local only

        "aws networking"
            -> local only

        "my python project"
            -> local only

        "email from KPMG"
            -> Gmail only

        "gmail internship"
            -> Gmail only

        "messages from Microsoft"
            -> Gmail only

        "subject: internship"
            -> Gmail only

        "from:someone@gmail.com"
            -> Gmail only
    """

    query = q.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    # ---------------------------------------------------------------
    # 1. Decide the search source before doing any expensive retrieval.
    # ---------------------------------------------------------------

    # Explicit email queries should go directly to Gmail.
    # There is no reason to run SQLite/FTS5 + FAISS retrieval first
    # because Gmail results are the only results returned for these queries.
    email_query = is_email_query(query)

    local_payload: list[dict[str, Any]] = []

    if not email_query:
        local_results = await asyncio.to_thread(
            hybrid_search,
            query=query,
            limit=limit,
        )

        for result in local_results:
            local_payload.append(
                {
                    "id": result.memory_id,
                    "source_type": result.source_type,
                    "content_type": "",
                    "title": result.title,
                    "path": result.path,
                    "content": result.content,
                    "modified_at": None,
                    "accessed_at": None,
                    "is_deleted": 0,
                    "deleted_at": None,
                    "metadata_json": None,
                    "rank": result.score,
                    "lexical_rank": result.lexical_rank,
                    "semantic_rank": result.semantic_rank,
                    "temporal_rank": result.temporal_rank,
                    "context": result.context,
                }
            )

    # ---------------------------------------------------------------
    # 2. Search Gmail only when the query explicitly targets email.
    # ---------------------------------------------------------------

    gmail_connected = False
    gmail_results: list[dict[str, Any]] = []
    gmail_error: str | None = None

    if email_query:
        gmail_connected = is_gmail_connected()

        if gmail_connected:
            try:
                # Pass the natural-language query through unchanged.
                # search_gmail() now performs the canonical Gmail query
                # translation inside the integration layer.
                raw_gmail_results = await asyncio.to_thread(
                    search_gmail,
                    query,
                    min(limit, 10),
                )

                gmail_results = [
                    normalize_gmail_result(
                        result,
                        rank=max(
                            0.0,
                            1.0 - (index * 0.01),
                        ),
                    )
                    for index, result in enumerate(
                        raw_gmail_results
                    )
                ]

            except Exception as error:
                gmail_error = (
                    f"{type(error).__name__}: {error}"
                )

        else:
            gmail_error = (
                "Gmail is not connected."
            )

    # ---------------------------------------------------------------
    # 3. Combine results.
    # ---------------------------------------------------------------
    #
    # For an email query, Gmail results are placed first because
    # the user explicitly asked for email/message information.
    #
    # For ordinary queries, only local results are returned.
    #
    # A true cross-source ranking system can be added later once
    # Outlook is also integrated.

    if email_query:
        # An explicit email query should return Gmail results only.
        # This prevents unrelated local files from displacing the emails
        # the user actually asked for.
        combined_results = gmail_results
    else:
        combined_results = local_payload

    combined_results = combined_results[:limit]

    response: dict[str, Any] = {
        "query": query,
        "count": len(combined_results),
        "results": combined_results,
        "sources": {
            "local": True,
            "gmail": (
                email_query
                and gmail_connected
            ),
        },
    }

    if gmail_error is not None:
        response["gmail_error"] = gmail_error

    return response