from app.embeddings.index import get_semantic_index
from app.embeddings.model import embed_text
from app.storage.database import get_connection


def search(query: str, limit: int = 10) -> None:
    index = get_semantic_index()

    query_embedding = embed_text(query)

    results = index.search(
        query_embedding,
        limit=limit,
    )

    print(f"\nQuery: {query}")
    print("-" * 80)

    if not results:
        print("No semantic results.")
        return

    connection = get_connection()

    try:
        for memory_id, score in results:
            row = connection.execute(
                """
                SELECT
                    id,
                    title,
                    path,
                    is_deleted
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()

            if row is None:
                continue

            print(f"\nScore: {score:.4f}")
            print(f"ID:    {row['id']}")
            print(f"Title: {row['title']}")
            print(f"Path:  {row['path']}")
            print(f"Deleted: {bool(row['is_deleted'])}")
    finally:
        connection.close()


if __name__ == "__main__":
    search("AWS networking")
    search("machine learning")
    search("cybersecurity")