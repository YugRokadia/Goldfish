from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_SIMILARITY = 0.20
MAX_RELATED_MEMORIES = 12

# Weight used when combining different relationship signals.
FILENAME_WEIGHT = 0.40
DIRECTORY_WEIGHT = 0.25
CONTENT_WEIGHT = 0.35


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RelatedMemory:
    memory_id: int
    title: str
    path: str | None
    source_type: str
    score: float
    reasons: list[str]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
    "you",
    "your",
}


def _tokenize(text: str) -> set[str]:
    """
    Convert text into normalized tokens.

    This deliberately stays lightweight. Context clustering should not
    require another embedding model or expensive NLP pipeline.
    """
    if not text:
        return set()

    tokens = re.findall(
        r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}",
        text.lower(),
    )

    return {
        token
        for token in tokens
        if token not in _STOP_WORDS
        and len(token) >= 2
    }


def _filename_tokens(
    path: str | None,
    title: str | None,
) -> set[str]:
    """
    Extract meaningful tokens from a filename/title.

    File extensions are intentionally preserved because they can provide
    useful context about the type of artifact.
    """
    parts: list[str] = []

    if title:
        parts.append(title)

    if path:
        try:
            file_name = Path(path).name
        except (OSError, ValueError):
            file_name = ""

        if file_name:
            parts.append(file_name)

    return _tokenize(" ".join(parts))


# Generic filesystem components that should not contribute to directory
# similarity. They are common across unrelated files and create noise.
_GENERIC_PATH_COMPONENTS = {
    "users",
    "user",
    "home",
    "desktop",
    "documents",
    "downloads",
    "pictures",
    "videos",
    "music",
    "public",
    "onedrive",
    "appdata",
    "local",
    "locallow",
    "roaming",
}


def _directory_tokens(path: str | None) -> set[str]:
    """
    Extract meaningful directory names from a filesystem path.

    Generic Windows/user-home components are intentionally ignored.

    The Windows ``Users/<username>`` portion is treated as the user's
    home-directory boundary and is excluded entirely.

    Example:

        C:/Users/Yug Rokadia/Documents/Projects/GLIRE/paper.pdf

    produces directory tokens roughly equivalent to:

        projects, glire

    while preserving meaningful project/folder names.
    """
    if not path:
        return set()

    try:
        parent = Path(path).parent

        components: list[str] = []
        skip_home_directory = False

        for part in parent.parts:
            normalized = part.strip().lower()

            if not normalized:
                continue

            # Ignore Windows drive roots such as "C:\\"
            if len(normalized) == 2 and normalized[1] == ":":
                continue

            # Ignore Windows root separators.
            if normalized in {"\\", "/"}:
                continue

            # Ignore the Windows Users directory and the immediately
            # following component, which is the username/home directory.
            #
            # Example:
            #   C:\\Users\\Yug Rokadia\\Documents\\RecallX
            #        ^^^^^^^^^^^^^^^^^ ignored
            if normalized == "users":
                skip_home_directory = True
                continue

            if skip_home_directory:
                skip_home_directory = False
                continue

            # Ignore generic user/system path components.
            if normalized in _GENERIC_PATH_COMPONENTS:
                continue

            components.append(part)

        return _tokenize(" ".join(components))

    except (OSError, ValueError):
        return set()


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def _jaccard_similarity(
    left: set[str],
    right: set[str],
) -> float:
    """
    Jaccard similarity between two token sets.
    """
    if not left or not right:
        return 0.0

    intersection = len(left & right)
    union = len(left | right)

    if union == 0:
        return 0.0

    return intersection / union


def _cosine_similarity_from_tokens(
    left: set[str],
    right: set[str],
) -> float:
    """
    Lightweight cosine similarity using token presence.

    This is intentionally separate from FAISS. The context engine should
    remain useful even when semantic embeddings are unavailable.
    """
    if not left or not right:
        return 0.0

    vocabulary = left | right

    if not vocabulary:
        return 0.0

    left_vector = [
        1.0 if token in left else 0.0
        for token in vocabulary
    ]

    right_vector = [
        1.0 if token in right else 0.0
        for token in vocabulary
    ]

    dot = sum(
        a * b
        for a, b in zip(left_vector, right_vector)
    )

    left_norm = math.sqrt(
        sum(value * value for value in left_vector)
    )

    right_norm = math.sqrt(
        sum(value * value for value in right_vector)
    )

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot / (left_norm * right_norm)


def _shared_tokens(
    left: set[str],
    right: set[str],
) -> set[str]:
    return left & right


# ---------------------------------------------------------------------------
# Memory loading
# ---------------------------------------------------------------------------


def _get_memory(
    connection: sqlite3.Connection,
    memory_id: int,
) -> sqlite3.Row | None:
    """
    Fetch one active memory.
    """
    return connection.execute(
        """
        SELECT
            id,
            title,
            path,
            content,
            source_type,
            created_at,
            modified_at,
            accessed_at,
            metadata_json
        FROM memories
        WHERE id = ?
          AND is_deleted = 0
        """,
        (memory_id,),
    ).fetchone()


def _get_candidate_memories(
    connection: sqlite3.Connection,
    memory_id: int,
) -> list[sqlite3.Row]:
    """
    Fetch active memories that can potentially belong to the same context.

    We intentionally avoid loading the entire content of every memory when
    possible. The initial candidate pool is narrowed using source type and
    path relationships.
    """
    target = _get_memory(
        connection,
        memory_id,
    )

    if target is None:
        return []

    path = target["path"]
    source_type = target["source_type"]

    parent_directory = None

    if path:
        try:
            parent_directory = str(
                Path(path).parent
            )
        except (OSError, ValueError):
            parent_directory = None

    candidates: list[sqlite3.Row] = []

    if parent_directory:
        rows = connection.execute(
            """
            SELECT
                id,
                title,
                path,
                content,
                source_type,
                created_at,
                modified_at,
                accessed_at,
                metadata_json
            FROM memories
            WHERE is_deleted = 0
              AND id != ?
              AND (
                    path LIKE ?
                    OR source_type = ?
              )
            LIMIT 500
            """,
            (
                memory_id,
                f"{parent_directory}%",
                source_type,
            ),
        ).fetchall()

        candidates.extend(rows)

    # If the directory produced very few candidates, also look for the same
    # source type. This helps when related files are spread across folders.
    if len(candidates) < 100:
        rows = connection.execute(
            """
            SELECT
                id,
                title,
                path,
                content,
                source_type,
                created_at,
                modified_at,
                accessed_at,
                metadata_json
            FROM memories
            WHERE is_deleted = 0
              AND id != ?
              AND source_type = ?
            LIMIT 500
            """,
            (
                memory_id,
                source_type,
            ),
        ).fetchall()

        existing_ids = {
            row["id"]
            for row in candidates
        }

        for row in rows:
            if row["id"] not in existing_ids:
                candidates.append(row)

    return candidates


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _parse_metadata(
    metadata_json: str | None,
) -> dict:
    if not metadata_json:
        return {}

    try:
        value = json.loads(
            metadata_json
        )
    except (
        json.JSONDecodeError,
        TypeError,
    ):
        return {}

    return (
        value
        if isinstance(value, dict)
        else {}
    )


def _get_extension(
    memory: sqlite3.Row,
) -> str:
    metadata = _parse_metadata(
        memory["metadata_json"]
    )

    extension = metadata.get(
        "extension"
    )

    if isinstance(extension, str):
        return extension.lower()

    path = memory["path"]

    if path:
        try:
            return Path(path).suffix.lower()
        except (OSError, ValueError):
            pass

    return ""


# ---------------------------------------------------------------------------
# Relationship scoring
# ---------------------------------------------------------------------------


def _score_relationship(
    target: sqlite3.Row,
    candidate: sqlite3.Row,
) -> tuple[float, list[str]]:
    """
    Calculate how strongly two memories appear to belong to the same context.

    Signals:

    1. Filename/title similarity
    2. Directory similarity
    3. Content similarity
    4. Same source type
    5. Shared extension

    The first three are the primary context signals.
    """
    target_filename = _filename_tokens(
        target["path"],
        target["title"],
    )

    candidate_filename = _filename_tokens(
        candidate["path"],
        candidate["title"],
    )

    target_directory = _directory_tokens(
        target["path"]
    )

    candidate_directory = _directory_tokens(
        candidate["path"]
    )

    target_content = _tokenize(
        target["content"] or ""
    )

    candidate_content = _tokenize(
        candidate["content"] or ""
    )

    filename_similarity = _jaccard_similarity(
        target_filename,
        candidate_filename,
    )

    directory_similarity = _jaccard_similarity(
        target_directory,
        candidate_directory,
    )

    content_similarity = _cosine_similarity_from_tokens(
        target_content,
        candidate_content,
    )

    score = (
        FILENAME_WEIGHT
        * filename_similarity
        + DIRECTORY_WEIGHT
        * directory_similarity
        + CONTENT_WEIGHT
        * content_similarity
    )

    reasons: list[str] = []

    shared_filename = _shared_tokens(
        target_filename,
        candidate_filename,
    )

    if shared_filename:
        important = sorted(
            shared_filename
        )[:5]

        reasons.append(
            "shared filename terms: "
            + ", ".join(important)
        )

    shared_directory = _shared_tokens(
        target_directory,
        candidate_directory,
    )

    if shared_directory:
        important = sorted(
            shared_directory
        )[:5]

        reasons.append(
            "shared directory: "
            + ", ".join(important)
        )

    if content_similarity >= 0.25:
        reasons.append(
            "similar content"
        )

    if target["source_type"] == candidate["source_type"]:
        score += 0.05

        reasons.append(
            "same source type"
        )

    target_extension = _get_extension(
        target
    )

    candidate_extension = _get_extension(
        candidate
    )

    if (
        target_extension
        and target_extension == candidate_extension
    ):
        score += 0.02

        reasons.append(
            f"same file type: {target_extension}"
        )

    return min(
        score,
        1.0,
    ), reasons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_related_memories(
    connection: sqlite3.Connection,
    memory_id: int,
    limit: int = MAX_RELATED_MEMORIES,
    min_similarity: float = MIN_SIMILARITY,
) -> list[RelatedMemory]:
    """
    Find memories that are likely related to the supplied memory.

    This is the first version of RecallX's context engine.

    It does NOT modify the database.

    Example:

        related = find_related_memories(
            connection,
            memory_id,
        )

        for item in related:
            print(
                item.title,
                item.score,
                item.reasons,
            )
    """
    target = _get_memory(
        connection,
        memory_id,
    )

    if target is None:
        return []

    candidates = _get_candidate_memories(
        connection,
        memory_id,
    )

    results: list[RelatedMemory] = []

    for candidate in candidates:
        score, reasons = _score_relationship(
            target,
            candidate,
        )

        if score < min_similarity:
            continue

        results.append(
            RelatedMemory(
                memory_id=int(
                    candidate["id"]
                ),
                title=candidate["title"] or "",
                path=candidate["path"],
                source_type=(
                    candidate["source_type"]
                    or "unknown"
                ),
                score=round(
                    score,
                    4,
                ),
                reasons=reasons,
            )
        )

    results.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    return results[:limit]


def find_context(
    connection: sqlite3.Connection,
    memory_id: int,
    limit: int = MAX_RELATED_MEMORIES,
) -> list[int]:
    """
    Return only the IDs of memories related to the supplied memory.

    This is useful for the retrieval layer because it avoids coupling search
    results to the RelatedMemory data structure.
    """
    related = find_related_memories(
        connection,
        memory_id,
        limit=limit,
    )

    return [
        item.memory_id
        for item in related
    ]


def build_context(
    connection: sqlite3.Connection,
    memory_id: int,
    limit: int = MAX_RELATED_MEMORIES,
) -> dict:
    """
    Build a serializable context object around one memory.

    This will later become the input structure for:
        - grouped search results
        - temporal context
        - browser history
        - Gmail/Outlook
        - local LLM context
    """
    target = _get_memory(
        connection,
        memory_id,
    )

    if target is None:
        raise ValueError(
            f"Memory not found: {memory_id}"
        )

    related = find_related_memories(
        connection,
        memory_id,
        limit=limit,
    )

    return {
        "memory_id": int(
            target["id"]
        ),
        "title": target["title"] or "",
        "path": target["path"],
        "source_type": (
            target["source_type"]
            or "unknown"
        ),
        "related": [
            {
                "memory_id": item.memory_id,
                "title": item.title,
                "path": item.path,
                "source_type": item.source_type,
                "score": item.score,
                "reasons": item.reasons,
            }
            for item in related
        ],
    }


# ---------------------------------------------------------------------------
# Batch clustering
# ---------------------------------------------------------------------------


def cluster_memories(
    connection: sqlite3.Connection,
    memory_ids: list[int],
    min_similarity: float = MIN_SIMILARITY,
) -> dict[int, list[int]]:
    """
    Build an in-memory relationship graph.

    The result looks like:

        {
            101: [102, 103],
            102: [101],
            103: [101],
        }

    This does not persist context IDs yet.

    Persistence will be added later once the relationship model is stable.
    """
    graph: dict[int, list[int]] = {
        memory_id: []
        for memory_id in memory_ids
    }

    allowed_ids = set(
        memory_ids
    )

    for memory_id in memory_ids:
        related = find_related_memories(
            connection,
            memory_id,
            limit=MAX_RELATED_MEMORIES,
            min_similarity=min_similarity,
        )

        graph[memory_id] = [
            item.memory_id
            for item in related
            if item.memory_id in allowed_ids
        ]

    return graph


def connected_components(
    graph: dict[int, list[int]],
) -> list[list[int]]:
    """
    Convert a relationship graph into connected context groups.

    Example:

        1 <-> 2 <-> 3

    becomes:

        [[1, 2, 3]]
    """
    visited: set[int] = set()
    components: list[list[int]] = []

    for node in graph:
        if node in visited:
            continue

        component: list[int] = []
        stack = [node]

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            component.append(current)

            for neighbour in graph.get(
                current,
                [],
            ):
                if neighbour not in visited:
                    stack.append(neighbour)

        components.append(
            sorted(component)
        )

    return components