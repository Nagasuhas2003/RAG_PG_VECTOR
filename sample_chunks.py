# Sample document chunks (10 paragraphs, ~4-5 lines each)
CHUNKS = [
    """PostgreSQL is an advanced open-source relational database system \
with over 35 years of active development. It supports complex queries, \
foreign keys, triggers, views, and transactional integrity. \
Extensibility is a core feature — users can define custom data types, \
operators, and index methods.""",

    """pgvector enables vector similarity search directly in PostgreSQL. \
It stores embeddings as a native `vector` type and supports exact and \
approximate nearest neighbor search. Index types include HNSW and IVFFlat \
for scaling to millions of vectors. Distance metrics: L2, inner product, cosine.""",

    """pgvectorscale builds on pgvector with StreamingDiskANN, a graph-based \
ANN index inspired by Microsoft's DiskANN. It achieves 28x lower latency \
and 16x higher throughput than Pinecone at 99% recall on 50M vectors. \
Statistical Binary Quantization (SBQ) compresses vectors 32x with minimal recall loss.""",

    """StreamingDiskANN uses a hierarchical navigable graph on disk, \
keeping only quantized vectors in memory. During search, it streams \
neighbors from SSD, enabling billion-scale indexes on modest RAM. \
Build-time params: `num_neighbors`, `search_list_size`, `storage_layout`.""",

    """Label-based filtered search lets you combine vector similarity \
with metadata filters (e.g., `labels && ARRAY[1,3]`) at index time. \
Based on Filtered DiskANN research, it avoids post-filtering overhead. \
Labels must be `smallint[]`; arbitrary WHERE clauses still use post-filtering.""",

    """Query-time tuning: `diskann.query_search_list_size` (default 100) \
controls candidate pool; `diskann.query_rescore` (default 50) re-ranks \
top candidates with full-precision vectors. Increase for higher recall, \
decrease for latency. Use `SET LOCAL` for per-transaction overrides.""",

    """Parallel index builds accelerate construction on large tables. \
Enabled automatically when vectors ≥ 65,536, no labels, SBQ layout. \
Control with `diskann.parallel_flush_interval`, \
`diskann.min_vectors_for_parallel_build`, `diskann.force_parallel_workers`.""",

    """Timescale Cloud offers managed pgvector + pgvectorscale with \
vector-optimized compute. Private beta includes automatic index tuning, \
tiered storage, and streaming backups. Self-hosted users can run the \
Docker image: `timescale/pgvectorscale:latest-pg17`.""",

    """Typical RAG pipeline: chunk documents → embed with text-embedding-3-small \
or Cohere v3 → store in `document_chunk` table → create diskann index → \
query with `embedding <=> $query_vector` + metadata filters. \
pgvectorscale handles 100M+ vectors on a single node.""",

    """Best practices: set `maintenance_work_mem = '2GB'` for index builds; \
use `memory_optimized` storage (SBQ) for most workloads; \
monitor `diskann.query_rescore` impact on p95 latency; \
partition by time or tenant for multi-tenant apps."""
]

# SQL schema
SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector CASCADE;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

CREATE TABLE document_chunk_test (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(1536),  -- adjust dimensions to your model
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- StreamingDiskANN index with label filtering support
CREATE INDEX document_chunk_embedding_idx ON document_chunk
USING diskann (embedding vector_cosine_ops, (metadata->>'label_ids')::smallint[])
WITH (storage_layout = 'memory_optimized', num_neighbors = 50);
"""

# Insert statement (parameterized)
INSERT_SQL = """
INSERT INTO document_chunk (document_id, chunk_index, content, metadata, embedding)
VALUES (%s, %s, %s, %s, %s);
"""

# Example Python usage with psycopg2 + sentence-transformers
EXAMPLE_PYTHON = '''
import psycopg2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384 dims
# or: "intfloat/e5-base-v2" (768 dims), "BAAI/bge-large-en-v1.5" (1024 dims)

conn = psycopg2.connect("postgresql://user:pass@localhost:5432/db")
cur = conn.cursor()

doc_id = 1
for i, chunk in enumerate(CHUNKS):
    embedding = model.encode(chunk).tolist()
    metadata = {"label_ids": [1, 2], "source": "pgvectorscale_docs"}
    cur.execute(INSERT_SQL, (doc_id, i, chunk, psycopg2.extras.Json(metadata), embedding))

conn.commit()
cur.close()
conn.close()
'''

if __name__ == "__main__":
    print("CHUNKS:")
    for i, c in enumerate(CHUNKS):
        print(f"\n--- Chunk {i} ---\n{c}")
    print("\n\nSCHEMA_SQL:")
    print(SCHEMA_SQL)
    print("\nINSERT_SQL:")
    print(INSERT_SQL)
    print("\nEXAMPLE_PYTHON:")
    print(EXAMPLE_PYTHON)