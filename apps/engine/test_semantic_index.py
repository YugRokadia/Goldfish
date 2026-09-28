import numpy as np

from app.embeddings.index import SemanticIndex
from app.embeddings.model import embed_texts


texts = [
    "AWS VPC networking and internet gateway configuration",
    "React frontend development with TypeScript",
    "Docker container deployment on AWS EC2",
    "Python machine learning and data analysis",
]

memory_ids = [101, 102, 103, 104]

print("Generating embeddings...")

embeddings = embed_texts(texts)

print("Embedding shape:", embeddings.shape)

index = SemanticIndex()

# Start this test with a fresh in-memory index rather than
# touching the real persistent index.
index.index = index._create_index()

index.add(embeddings, memory_ids)

print("Vectors in FAISS:", index.count())

query = "AWS networking"

query_embedding = embed_texts([query])

results = index.search(query_embedding, limit=4)

print("\nSemantic search results:")

for memory_id, score in results:
    title = dict(zip(memory_ids, texts))[memory_id]
    print(f"{memory_id}: {score:.4f} -> {title}")

assert index.count() == 4
assert len(results) > 0
assert results[0][0] in (101, 103)

print("\nSemantic index test PASSED.")