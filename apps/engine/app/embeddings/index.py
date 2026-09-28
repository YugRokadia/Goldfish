from __future__ import annotations

import faiss
import numpy as np

from app.config import INDEX_DIR


EMBEDDING_DIMENSION = 384
INDEX_PATH = INDEX_DIR / "semantic.faiss"


class SemanticIndex:
    """
    Persistent FAISS index for RecallX semantic memories.

    Each FAISS vector is associated directly with a RecallX
    SQLite memory ID.
    """

    def __init__(self) -> None:
        self.index: faiss.IndexIDMap2 | None = None
        self._load_or_create()

    def _create_index(self) -> faiss.IndexIDMap2:
        """
        Create an exact cosine-similarity index.

        Embeddings are normalized in model.py, so inner product
        is equivalent to cosine similarity.
        """
        base_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        return faiss.IndexIDMap2(base_index)

    def _load_or_create(self) -> None:
        """
        Load the existing persistent index or create a new one.
        """
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        if INDEX_PATH.exists():
            index = faiss.read_index(str(INDEX_PATH))

            if index.d != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Invalid semantic index dimension: "
                    f"{index.d}. Expected {EMBEDDING_DIMENSION}."
                )

            if not isinstance(index, faiss.IndexIDMap2):
                raise ValueError(
                    "Semantic index must be a FAISS IndexIDMap2."
                )

            self.index = index
        else:
            self.index = self._create_index()

    def save(self) -> None:
        """
        Persist the current FAISS index to disk.
        """
        if self.index is None:
            raise RuntimeError(
                "Semantic index has not been initialized."
            )

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))

    def add(
        self,
        embeddings: np.ndarray,
        memory_ids: list[int],
    ) -> None:
        """
        Add embeddings associated with RecallX memory IDs.
        """
        if self.index is None:
            raise RuntimeError(
                "Semantic index has not been initialized."
            )

        embeddings = self._prepare_embeddings(embeddings)

        if len(embeddings) != len(memory_ids):
            raise ValueError(
                "Number of embeddings must match number of memory IDs."
            )

        if not memory_ids:
            return

        ids = np.asarray(memory_ids, dtype=np.int64)

        self.index.add_with_ids(embeddings, ids)

    def remove(self, memory_ids: list[int]) -> None:
        """
        Remove memories from the semantic index.
        """
        if self.index is None:
            raise RuntimeError(
                "Semantic index has not been initialized."
            )

        if not memory_ids:
            return

        ids = np.asarray(memory_ids, dtype=np.int64)
        self.index.remove_ids(ids)

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 20,
    ) -> list[tuple[int, float]]:
        """
        Search for semantically similar memories.

        Returns:
            [(memory_id, similarity_score), ...]
        """
        if self.index is None:
            raise RuntimeError(
                "Semantic index has not been initialized."
            )

        if self.index.ntotal == 0:
            return []

        if limit <= 0:
            return []

        query = self._prepare_query(query_embedding)

        scores, ids = self.index.search(query, limit)

        results: list[tuple[int, float]] = []

        for memory_id, score in zip(ids[0], scores[0]):
            if memory_id == -1:
                continue

            results.append(
                (
                    int(memory_id),
                    float(score),
                )
            )

        return results

    def count(self) -> int:
        """
        Return the number of vectors currently stored.
        """
        if self.index is None:
            return 0

        return self.index.ntotal

    @staticmethod
    def _prepare_embeddings(
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and normalize a batch of embeddings.
        """
        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2-dimensional array."
            )

        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected embeddings with dimension "
                f"{EMBEDDING_DIMENSION}, "
                f"got {embeddings.shape[1]}."
            )

        faiss.normalize_L2(embeddings)

        return np.ascontiguousarray(embeddings)

    @staticmethod
    def _prepare_query(
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and normalize a single query embedding.
        """
        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2 or query.shape != (
            1,
            EMBEDDING_DIMENSION,
        ):
            raise ValueError(
                f"Expected query embedding with shape "
                f"(1, {EMBEDDING_DIMENSION}), "
                f"got {query.shape}."
            )

        faiss.normalize_L2(query)

        return np.ascontiguousarray(query)


_index: SemanticIndex | None = None


def get_semantic_index() -> SemanticIndex:
    """
    Return the shared RecallX semantic index.
    """
    global _index

    if _index is None:
        _index = SemanticIndex()

    return _index