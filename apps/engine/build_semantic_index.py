from __future__ import annotations

import time

from app.embeddings.index import get_semantic_index
from app.embeddings.model import embed_texts
from app.storage.database import get_connection


BATCH_SIZE = 64


def main() -> None:
    start = time.time()

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT id, title, content
            FROM memories
            WHERE is_deleted = 0
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()

    print(f"Found {len(rows)} active memories.")

    if not rows:
        print("No active memories found.")
        return

    semantic_index = get_semantic_index()

    # Start with a completely fresh index.
    semantic_index.index = semantic_index._create_index()

    total = len(rows)

    for start_index in range(0, total, BATCH_SIZE):
        batch = rows[start_index:start_index + BATCH_SIZE]

        texts = [
            f"{row['title']}\n{row['content']}"
            for row in batch
        ]

        memory_ids = [
            int(row["id"])
            for row in batch
        ]

        embeddings = embed_texts(texts)

        semantic_index.add(
            embeddings,
            memory_ids,
        )

        processed = min(
            start_index + BATCH_SIZE,
            total,
        )

        print(
            f"Embedded {processed}/{total} "
            f"({processed / total * 100:.1f}%)"
        )

    semantic_index.save()

    elapsed = time.time() - start

    print()
    print("Semantic index built successfully.")
    print(f"Vectors: {semantic_index.count()}")
    print(f"Time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()