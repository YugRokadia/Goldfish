from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import log1p

from app.context.service import get_memory_context
from app.embeddings.index import get_semantic_index
from app.embeddings.model import embed_text
from app.storage.database import get_connection, search_memories


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    memory_id: int
    title: str
    path: str | None
    content: str
    source_type: str
    score: float
    lexical_rank: int | None
    semantic_rank: int | None
    temporal_rank: int | None
    context: list[dict] | None = None


# ---------------------------------------------------------------------------
# Query understanding
# ---------------------------------------------------------------------------


TEMPORAL_PATTERNS = [
    (r"\blast month\b", "last month"),
    (r"\bprevious month\b", "last month"),
    (r"\bthis month\b", "this month"),
    (r"\blast week\b", "last week"),
    (r"\bprevious week\b", "last week"),
    (r"\bthis week\b", "this week"),
    (r"\byesterday\b", "yesterday"),
    (r"\btoday\b", "today"),
]


FILLER_PATTERNS = [
    r"\bi did\b",
    r"\bi worked on\b",
    r"\bi was working on\b",
    r"\bi worked with\b",
    r"\bi made\b",
    r"\bi created\b",
    r"\bi used\b",
    r"\bi wrote\b",
    r"\bi built\b",
    r"\bi was doing\b",
    r"\bthat i did\b",
    r"\bthat i worked on\b",
    r"\bthe stuff\b",
    r"\bstuff about\b",
    r"\bthings about\b",
]


CONTEXT_PATTERNS = [
    r"\bdocument\b",
    r"\bfile\b",
    r"\bthing\b",
    r"\bstuff\b",
    r"\bproject\b",
    r"\bnotes?\b",
    r"\bcode\b",
]


# Conversational words that rarely help filesystem retrieval.
# Meaningful nouns such as "project" and "code" are deliberately preserved.
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "show",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "you",
    "your",
}


@dataclass
class QueryIntent:
    retrieval_query: str
    temporal_label: str | None
    start_timestamp: float | None
    end_timestamp: float | None


def _day_start(value: datetime) -> datetime:
    return value.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )


def _month_start(value: datetime) -> datetime:
    return value.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )


def _next_month_start(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(
            year=value.year + 1,
            month=1,
            day=1,
        )

    return value.replace(
        month=value.month + 1,
        day=1,
    )


def _temporal_range(
    label: str,
    now: datetime,
) -> tuple[datetime, datetime]:
    today = _day_start(now)

    if label == "today":
        return today, today + timedelta(days=1)

    if label == "yesterday":
        return today - timedelta(days=1), today

    if label == "this week":
        start = today - timedelta(days=today.weekday())
        return start, start + timedelta(days=7)

    if label == "last week":
        end = today - timedelta(days=today.weekday())
        return end - timedelta(days=7), end

    if label == "this month":
        start = _month_start(now)
        return start, _next_month_start(now)

    if label == "last month":
        end = _month_start(now)

        if end.month == 1:
            start = end.replace(
                year=end.year - 1,
                month=12,
            )
        else:
            start = end.replace(
                month=end.month - 1,
            )

        return start, end

    raise ValueError(
        f"Unknown temporal label: {label}"
    )


def _clean_query(query: str) -> str:
    """
    Normalize a natural-language query without destroying useful search terms.

    Conversational filler is removed, but meaningful nouns such as
    "project", "code", and file names are deliberately preserved.
    """
    cleaned = query.lower().strip()

    for pattern in FILLER_PATTERNS:
        cleaned = re.sub(
            pattern,
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    tokens = _unique_tokens(cleaned)

    meaningful = [
        token
        for token in tokens
        if token not in QUERY_STOPWORDS
    ]

    return " ".join(meaningful).strip()


def _query_tokens(query: str) -> list[str]:
    return [
        token
        for token in _unique_tokens(query)
        if token not in QUERY_STOPWORDS
    ]


def _query_mode(query: str) -> str:
    """
    Select retrieval strategy from query shape.

    short  : 1-2 terms, filename/title-first.
    medium : 3-4 terms, lexical-first hybrid.
    long   : 5+ terms, natural-language hybrid.
    """
    token_count = len(_query_tokens(query))

    if token_count <= 2:
        return "short"

    if token_count <= 4:
        return "medium"

    return "long"


def _parse_query(query: str) -> QueryIntent:
    retrieval_query = query.strip()
    temporal_label: str | None = None

    for pattern, label in TEMPORAL_PATTERNS:
        if re.search(
            pattern,
            retrieval_query,
            flags=re.IGNORECASE,
        ):
            retrieval_query = re.sub(
                pattern,
                " ",
                retrieval_query,
                flags=re.IGNORECASE,
            )

            temporal_label = label
            break

    retrieval_query = _clean_query(
        retrieval_query
    )

    if temporal_label is None:
        return QueryIntent(
            retrieval_query=retrieval_query,
            temporal_label=None,
            start_timestamp=None,
            end_timestamp=None,
        )

    now = datetime.now().astimezone()

    start, end = _temporal_range(
        temporal_label,
        now,
    )

    return QueryIntent(
        retrieval_query=retrieval_query,
        temporal_label=temporal_label,
        start_timestamp=start.timestamp(),
        end_timestamp=end.timestamp(),
    )


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


def _get_memories(
    connection: sqlite3.Connection,
    memory_ids: list[int],
) -> dict[int, dict]:
    if not memory_ids:
        return {}

    placeholders = ",".join(
        "?" for _ in memory_ids
    )

    rows = connection.execute(
        f"""
        SELECT
            id,
            title,
            path,
            content,
            source_type,
            created_at,
            modified_at,
            accessed_at,
            size_bytes,
            metadata_json,
            file_hash,
            embedding_status,
            embedding_version
        FROM memories
        WHERE is_deleted = 0
          AND id IN ({placeholders})
        """,
        memory_ids,
    ).fetchall()

    return {
        int(row["id"]): dict(row)
        for row in rows
    }


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    return re.findall(
        r"[a-zA-Z0-9_]+",
        text.lower(),
    )


def _unique_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for token in _tokens(text):
        if token not in seen:
            seen.add(token)
            result.append(token)

    return result


# ---------------------------------------------------------------------------
# Exact lexical relevance
# ---------------------------------------------------------------------------


def _lexical_match_score(
    query: str,
    memory: dict,
) -> float:
    """
    Precision-first lexical signal.

    Title/filename and path evidence dominate content frequency. This avoids
    a large document winning simply because it mentions a query term many
    times.
    """
    query_tokens = _query_tokens(query)

    if not query_tokens:
        return 0.0

    title = str(memory.get("title") or "").lower()
    path = str(memory.get("path") or "").lower()
    content = str(memory.get("content") or "").lower()

    title_matches = sum(
        token in title
        for token in query_tokens
    )

    path_matches = sum(
        token in path
        for token in query_tokens
    )

    content_matches = sum(
        token in content
        for token in query_tokens
    )

    total = len(query_tokens)

    title_coverage = title_matches / total
    path_coverage = path_matches / total
    content_coverage = content_matches / total

    score = (
        0.55 * title_coverage
        + 0.25 * path_coverage
        + 0.15 * content_coverage
    )

    if title_coverage == 1.0:
        score += 0.20

    if path_coverage == 1.0:
        score += 0.08

    normalized_query = " ".join(query_tokens)

    if normalized_query and normalized_query in title:
        score += 0.20

    return min(score, 1.0)


def _title_path_boost(
    query: str,
    title: str | None,
    path: str | None,
) -> float:
    query_tokens = _query_tokens(query)

    if not query_tokens:
        return 0.0

    title_text = (title or "").lower()
    path_text = (path or "").lower()

    title_matches = sum(
        token in title_text
        for token in query_tokens
    )

    path_matches = sum(
        token in path_text
        for token in query_tokens
    )

    title_coverage = title_matches / len(query_tokens)
    path_coverage = path_matches / len(query_tokens)

    boost = 0.0

    if title_coverage == 1.0:
        boost += 0.55
    else:
        boost += 0.30 * title_coverage

    if path_coverage == 1.0:
        boost += 0.25
    else:
        boost += 0.12 * path_coverage

    normalized_query = " ".join(query_tokens)

    if normalized_query and normalized_query in title_text:
        boost += 0.25

    return min(boost, 0.80)


# ---------------------------------------------------------------------------
# Metadata relevance
# ---------------------------------------------------------------------------


def _parse_metadata(memory: dict) -> dict:
    raw = memory.get("metadata_json")

    if not raw:
        return {}

    if isinstance(raw, dict):
        return raw

    try:
        parsed = json.loads(raw)

        if isinstance(parsed, dict):
            return parsed

    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        pass

    return {}


def _metadata_boost(
    query: str,
    memory: dict,
) -> float:
    """
    Use the structured filesystem metadata introduced during metadata
    cleanup.

    This is intentionally a small signal. Metadata should improve
    ranking, not overpower actual lexical/semantic relevance.
    """
    metadata = _parse_metadata(memory)

    if not metadata:
        return 0.0

    query_tokens = _unique_tokens(query)

    if not query_tokens:
        return 0.0

    extension = str(
        metadata.get("extension") or ""
    ).lower()

    filename = str(
        metadata.get("file_name") or ""
    ).lower()

    parent = str(
        metadata.get("parent_directory") or ""
    ).lower()

    metadata_text = " ".join(
        [
            extension,
            filename,
            parent,
        ]
    )

    matches = sum(
        token in metadata_text
        for token in query_tokens
    )

    return min(
        matches * 0.02,
        0.10,
    )


# ---------------------------------------------------------------------------
# Temporal relevance
# ---------------------------------------------------------------------------


def _temporal_match(
    memory: dict,
    start_timestamp: float | None,
    end_timestamp: float | None,
) -> bool:
    if (
        start_timestamp is None
        or end_timestamp is None
    ):
        return True

    timestamps = [
        memory.get("modified_at"),
        memory.get("accessed_at"),
        memory.get("created_at"),
    ]

    return any(
        timestamp is not None
        and start_timestamp <= timestamp < end_timestamp
        for timestamp in timestamps
    )


def _best_timestamp(
    memory: dict,
) -> float | None:
    timestamps = [
        memory.get("modified_at"),
        memory.get("accessed_at"),
        memory.get("created_at"),
    ]

    valid = [
        float(timestamp)
        for timestamp in timestamps
        if timestamp is not None
    ]

    if not valid:
        return None

    return max(valid)


def _temporal_score(
    memory: dict,
    intent: QueryIntent,
) -> float:
    timestamp = _best_timestamp(memory)

    if timestamp is None:
        return 0.0

    now = datetime.now().astimezone().timestamp()

    age_seconds = max(
        now - timestamp,
        0.0,
    )

    age_days = age_seconds / 86400.0

    # Explicit temporal query:
    # being inside the requested period is much more important.
    if intent.temporal_label is not None:
        if _temporal_match(
            memory,
            intent.start_timestamp,
            intent.end_timestamp,
        ):
            return 1.0

        return 0.0

    # Generic queries get a small recency preference.
    #
    # The logarithmic decay prevents very recent files from completely
    # dominating older but highly relevant documents.
    return 1.0 / (
        1.0 + log1p(age_days)
    )


# ---------------------------------------------------------------------------
# Source relevance
# ---------------------------------------------------------------------------


def _source_boost(
    memory: dict,
) -> float:
    """
    Small source-aware signal.

    Filesystem remains the primary source currently, but keeping this
    generic makes the ranking ready for Gmail, Outlook and browser
    memories later.
    """
    source_type = str(
        memory.get("source_type") or ""
    ).lower()

    if source_type == "filesystem":
        return 0.02

    if source_type in {
        "email",
        "gmail",
        "outlook",
    }:
        return 0.02

    if source_type in {
        "browser",
        "browser_history",
    }:
        return 0.01

    return 0.0


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


RRF_K = 60.0


def _rrf_score(
    lexical_rank: int | None,
    semantic_rank: int | None,
) -> float:
    score = 0.0

    if lexical_rank is not None:
        score += 1.0 / (
            RRF_K + lexical_rank
        )

    if semantic_rank is not None:
        score += 1.0 / (
            RRF_K + semantic_rank
        )

    return score


def _normalized_rrf(
    rrf_score: float,
) -> float:
    """
    Convert RRF to a stable 0..1 signal.

    Two rank-1 results are the theoretical maximum for our two retrieval
    channels. Scaling against that maximum prevents RRF from overpowering
    direct title/path evidence.
    """
    if rrf_score <= 0:
        return 0.0

    maximum = 2.0 / (
        RRF_K + 1.0
    )

    return min(
        rrf_score / maximum,
        1.0,
    )


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def _build_candidate_ids(
    lexical_rank: dict[int, int],
    semantic_similarity: dict[int, float],
    memories: dict[int, dict],
    query: str,
    mode: str,
) -> set[int]:
    """
    Build a precision-first candidate pool.

    Short queries never admit semantic-only candidates. Medium queries remain
    lexical-first and admit semantic candidates only when lexical evidence is
    weak. Long queries use both retrieval channels.
    """
    candidate_ids = set(lexical_rank)

    if not lexical_rank or mode == "short":
        return candidate_ids

    query_tokens = _query_tokens(query)
    strong_lexical = False

    for memory_id, rank in lexical_rank.items():
        if rank > 25:
            break

        memory = memories.get(memory_id)

        if not memory or not query_tokens:
            continue

        title = str(
            memory.get("title") or ""
        ).lower()

        path = str(
            memory.get("path") or ""
        ).lower()

        if (
            all(token in title for token in query_tokens)
            or all(token in path for token in query_tokens)
        ):
            strong_lexical = True
            break

    if strong_lexical:
        return candidate_ids

    threshold = (
        0.50
        if mode == "medium"
        else 0.42
    )

    for memory_id, similarity in semantic_similarity.items():
        if similarity >= threshold:
            candidate_ids.add(memory_id)

    return candidate_ids


# ---------------------------------------------------------------------------
# Context enrichment
# ---------------------------------------------------------------------------


def _get_context_by_memory(
    connection: sqlite3.Connection,
    memory_ids: list[int],
    limit: int = 8,
) -> dict[int, list[dict]]:
    """
    Build related-memory context for selected search results.

    Context is intentionally best-effort. If clustering fails for one
    memory, that memory simply receives an empty context rather than
    breaking the entire search request.
    """
    context_by_memory: dict[int, list[dict]] = {}

    for memory_id in memory_ids:
        try:
            context_data = get_memory_context(
                connection,
                memory_id,
                limit=limit,
            )

            related = context_data.get(
                "related",
                [],
            )

            if isinstance(related, list):
                context_by_memory[memory_id] = related
            else:
                context_by_memory[memory_id] = []

        except Exception:
            context_by_memory[memory_id] = []

    return context_by_memory


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search(
    query: str,
    limit: int = 10,
    lexical_limit: int = 75,
    semantic_limit: int = 75,
) -> list[SearchResult]:
    """
    Adaptive RecallX retrieval.

    SHORT (1-2 meaningful terms)
        Filename/title/path first. Semantic retrieval is skipped unless
        lexical search has no results.

    MEDIUM (3-4 meaningful terms)
        Lexical-first hybrid. Semantic retrieval is used only when lexical
        evidence is weak.

    LONG (5+ meaningful terms)
        Full hybrid retrieval for natural-language queries.

    Context clustering is deliberately not performed on this critical path.
    """
    query = query.strip()

    if not query or limit <= 0:
        return []

    intent = _parse_query(query)
    retrieval_query = intent.retrieval_query

    if not retrieval_query:
        return []

    mode = _query_mode(
        retrieval_query
    )

    connection = get_connection()

    try:
        # -------------------------------------------------------------
        # 1. Lexical retrieval FIRST
        # -------------------------------------------------------------
        lexical_rows = search_memories(
            connection,
            retrieval_query,
            limit=lexical_limit,
        )

        lexical_rank: dict[int, int] = {}

        for rank, row in enumerate(
            lexical_rows,
            start=1,
        ):
            lexical_rank[int(row["id"])] = rank

        lexical_memories = _get_memories(
            connection,
            list(lexical_rank),
        )

        # -------------------------------------------------------------
        # 2. Decide whether semantic retrieval is necessary
        # -------------------------------------------------------------
        use_semantic = mode == "long"

        if mode == "medium":
            query_tokens = _query_tokens(
                retrieval_query
            )

            strong_lexical = False

            for memory_id, rank in lexical_rank.items():
                if rank > 25:
                    break

                memory = lexical_memories.get(
                    memory_id
                )

                if not memory or not query_tokens:
                    continue

                title = str(
                    memory.get("title") or ""
                ).lower()

                path = str(
                    memory.get("path") or ""
                ).lower()

                if (
                    all(
                        token in title
                        for token in query_tokens
                    )
                    or all(
                        token in path
                        for token in query_tokens
                    )
                ):
                    strong_lexical = True
                    break

            use_semantic = not strong_lexical

        # Short queries behave like Spotlight/file search. Semantic fallback
        # is used only when lexical retrieval has nothing to offer.
        if mode == "short":
            use_semantic = not bool(
                lexical_rank
            )

        semantic_rank: dict[int, int] = {}
        semantic_similarity: dict[int, float] = {}

        if use_semantic:
            query_embedding = embed_text(
                retrieval_query
            )

            semantic_rows = get_semantic_index().search(
                query_embedding,
                semantic_limit,
            )

            for rank, (
                memory_id,
                similarity,
            ) in enumerate(
                semantic_rows,
                start=1,
            ):
                memory_id = int(memory_id)

                if memory_id < 0:
                    continue

                similarity = float(
                    similarity
                )

                semantic_rank[memory_id] = rank
                semantic_similarity[memory_id] = similarity

        # -------------------------------------------------------------
        # 3. Candidate pool
        # -------------------------------------------------------------
        candidate_ids = _build_candidate_ids(
            lexical_rank,
            semantic_similarity,
            lexical_memories,
            retrieval_query,
            mode,
        )

        if not candidate_ids:
            return []

        memories = _get_memories(
            connection,
            list(candidate_ids),
        )

        candidate_ids &= set(memories)

        if not candidate_ids:
            return []

        # -------------------------------------------------------------
        # 4. Temporal filtering
        # -------------------------------------------------------------
        if intent.temporal_label is not None:
            temporal_candidates = {
                memory_id
                for memory_id in candidate_ids
                if _temporal_match(
                    memories[memory_id],
                    intent.start_timestamp,
                    intent.end_timestamp,
                )
            }

            if temporal_candidates:
                candidate_ids = temporal_candidates

        if not candidate_ids:
            return []

        # -------------------------------------------------------------
        # 5. Precision-first scoring
        # -------------------------------------------------------------
        scores: dict[int, float] = {}
        temporal_scores: dict[int, float] = {}

        for memory_id in candidate_ids:
            memory = memories[memory_id]

            rrf = _normalized_rrf(
                _rrf_score(
                    lexical_rank.get(
                        memory_id
                    ),
                    semantic_rank.get(
                        memory_id
                    ),
                )
            )

            semantic_similarity_value = max(
                semantic_similarity.get(
                    memory_id,
                    0.0,
                ),
                0.0,
            )

            lexical_score = _lexical_match_score(
                retrieval_query,
                memory,
            )

            title_path_score = _title_path_boost(
                retrieval_query,
                memory.get("title"),
                memory.get("path"),
            )

            metadata_score = _metadata_boost(
                retrieval_query,
                memory,
            )

            temporal_score = _temporal_score(
                memory,
                intent,
            )

            if mode == "short":
                score = (
                    0.25 * rrf
                    + 0.35 * lexical_score
                    + 0.30 * title_path_score
                    + 0.05 * metadata_score
                    + 0.05 * temporal_score
                )

            elif mode == "medium":
                score = (
                    0.20 * rrf
                    + 0.32 * lexical_score
                    + 0.28 * title_path_score
                    + 0.12 * semantic_similarity_value
                    + 0.04 * metadata_score
                    + 0.04 * temporal_score
                )

            else:
                score = (
                    0.18 * rrf
                    + 0.28 * lexical_score
                    + 0.24 * title_path_score
                    + 0.20 * semantic_similarity_value
                    + 0.05 * metadata_score
                    + 0.05 * temporal_score
                )

            if intent.temporal_label is not None:
                score += 0.10 * temporal_score

            score += _source_boost(
                memory
            )

            scores[memory_id] = score
            temporal_scores[memory_id] = (
                temporal_score
            )

        # -------------------------------------------------------------
        # 6. Final ranking
        # -------------------------------------------------------------
        ranked_ids = sorted(
            candidate_ids,
            key=lambda memory_id: (
                scores[memory_id],

                _lexical_match_score(
                    retrieval_query,
                    memories[memory_id],
                ),

                _title_path_boost(
                    retrieval_query,
                    memories[memory_id].get(
                        "title"
                    ),
                    memories[memory_id].get(
                        "path"
                    ),
                ),

                semantic_similarity.get(
                    memory_id,
                    0.0,
                ),

                -lexical_rank.get(
                    memory_id,
                    10_000,
                ),
            ),
            reverse=True,
        )[:limit]

        # -------------------------------------------------------------
        # 7. Temporal ranking
        # -------------------------------------------------------------
        temporal_ranked_ids = sorted(
            candidate_ids,
            key=lambda memory_id: (
                temporal_scores.get(
                    memory_id,
                    0.0,
                ),
                -memory_id,
            ),
            reverse=True,
        )

        temporal_rank = {
            memory_id: rank
            for rank, memory_id in enumerate(
                temporal_ranked_ids,
                start=1,
            )
        }

        # -------------------------------------------------------------
        # 8. Results
        # -------------------------------------------------------------
        # Context is intentionally empty here. Search should never wait for
        # clustering. A separate context request can populate it later.
        return [
            SearchResult(
                memory_id=memory_id,
                title=memories[memory_id]["title"],
                path=memories[memory_id]["path"],
                content=memories[memory_id]["content"],
                source_type=memories[memory_id]["source_type"],
                score=scores[memory_id],
                lexical_rank=lexical_rank.get(
                    memory_id
                ),
                semantic_rank=semantic_rank.get(
                    memory_id
                ),
                temporal_rank=temporal_rank.get(
                    memory_id
                ),
                context=[],
            )
            for memory_id in ranked_ids
        ]

    finally:
        connection.close()