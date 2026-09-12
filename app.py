import os
import hashlib
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import execute_values,Json
from pgvector.psycopg2 import register_vector
from google import genai
from google.genai import types

load_dotenv()

# Configure Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    # This line is crucial for pgvector to work with psycopg2
    register_vector(conn)
    return conn


def setup_database(conn):
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(3072),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    print("✅ Database table 'document_chunks' is ready.")


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Splits text into overlapping chunks, preferring sentence boundaries."""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Look backward for a sentence boundary
            boundary = -1
            for i in range(end, max(start, end - 200), -1):
                if text[i] in '.!?\n':
                    boundary = i + 1
                    break
            if boundary != -1:
                end = boundary
        
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap if end - chunk_overlap > start else end

    return chunks

import time

def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT", max_retries: int = 3) -> List[float]:
    """Generates an embedding for a single string using Gemini, with retries."""
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model="gemini-embedding-2",
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            return result.embeddings[0].values
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt
            print(f"⚠️  Embedding failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

def ingest_document(conn, text_content: str, doc_name: str = "sample_doc"):
    doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:12]

    print(f"📄 Chunking document '{doc_name}'...")
    chunks = chunk_text(text_content)
    print(f"   Created {len(chunks)} chunks.")

    print("🧠 Generating embeddings (this may take a moment)...")
    rows_to_insert = []
    for i, chunk in enumerate(chunks):
        vec = get_embedding(chunk, task_type="RETRIEVAL_DOCUMENT")
        metadata = {"source": doc_name, "char_length": len(chunk)}
        rows_to_insert.append((doc_id, i, chunk, vec, Json(metadata)))

    print("💾 Inserting into PostgreSQL...")
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO document_chunks
                (document_id, chunk_index, content, embedding, metadata)
            VALUES %s
            """,
            rows_to_insert,
            template="(%s, %s, %s, %s, %s)",
        )
    conn.commit()
    print(f"✅ Successfully ingested {len(chunks)} chunks from '{doc_name}'.")

# --- RUN THE INGESTION ---
SAMPLE_DOCUMENT = """
PostgreSQL is a powerful, open-source object-relational database system that has earned a strong reputation for reliability, feature robustness, and performance over more than three decades of active development. It runs on all major operating systems and is used by organizations of every size, from solo developers to Fortune 500 companies.
The pgvector extension brings vector similarity search directly into PostgreSQL by introducing a native VECTOR data type. This means you can store embeddings alongside your regular relational data without needing a separate vector database, keeping your architecture simpler and your data consistent.
Embeddings are dense numerical representations of text, images, or audio that capture semantic meaning in a high-dimensional space. Two pieces of content that mean similar things will have embeddings that are close together, which is what makes semantic search possible.

Chunking is the process of splitting a long document into smaller, coherent pieces before embedding. Good chunking respects sentence and paragraph boundaries so each chunk contains a complete idea, which dramatically improves retrieval quality compared to blindly cutting every N characters.

Overlap between chunks is a common technique to avoid losing context at the boundaries. A typical setup uses a chunk size of around 1000 characters with an overlap of 100 to 200 characters, so ideas that span a boundary still appear intact in at least one chunk.

The Gemini embedding model produces vectors with 3072 dimensions by default, and it supports task-specific embedding modes. Using RETRIEVAL_DOCUMENT when embedding chunks and RETRIEVAL_QUERY when embedding a search query tells the model to optimize the vectors for asymmetric retrieval.

Cosine distance is the most common metric for comparing text embeddings because it measures the angle between vectors rather than their magnitude. In pgvector, the <=> operator computes cosine distance, where a smaller value means greater similarity.

Approximate nearest neighbor indexes like HNSW and IVFFlat make vector search fast at scale by avoiding a full scan of every row. HNSW generally offers better recall and query speed, while IVFFlat uses less memory and builds faster on large datasets.

The HNSW index in pgvector is controlled by parameters such as m, which sets the number of connections per node, and ef_construction, which controls the size of the candidate list during index building. Higher values improve recall at the cost of memory and build time.

Query-time tuning uses the hnsw.ef_search parameter, which determines how many candidates the index considers during a search. Raising it improves recall but slows queries, so it is usually tuned per workload rather than set globally.

The document_chunks table stores each chunk along with its embedding, a reference to the parent document, the chunk index, and any metadata you want to filter on later. Keeping metadata in a JSONB column gives you flexibility without schema changes.

Batch insertion with psycopg2's execute_values is significantly faster than inserting rows one at a time, because it sends multiple rows in a single round trip to the database. This matters when ingesting thousands of chunks from a large corpus.

Registering the vector type with register_vector(conn) is what allows psycopg2 to adapt Python lists into PostgreSQL VECTOR values automatically. Without it, inserting an embedding list raises an adaptation error and you have to cast manually.

Retrieval-augmented generation, or RAG, combines a language model with a retrieval system so the model can answer questions using your own data. The retriever fetches the most relevant chunks, and those chunks are inserted into the prompt as context before the model generates an answer.

Hybrid search blends vector similarity with traditional keyword search using PostgreSQL's full-text search capabilities. This often outperforms pure vector search on queries that contain rare terms, names, or exact identifiers that embeddings tend to blur.

Metadata filtering lets you narrow a vector search to a subset of rows before or after similarity ranking. Pre-filtering is faster when the filter is highly selective, while post-filtering is simpler but can miss good results if the filter is too aggressive.

Reciprocal rank fusion is a simple and effective way to merge results from multiple retrieval methods, such as vector search and keyword search. It combines ranked lists by summing reciprocal ranks, so documents that appear near the top of either list rise to the surface.

Evaluation is essential for any retrieval system, and metrics like recall@k, mean reciprocal rank, and normalized discounted cumulative gain tell you how well your pipeline surfaces relevant chunks. Without measurement, tuning chunk size and index parameters becomes guesswork.

Common failure modes in vector search include embedding the wrong task type, using mismatched distance metrics between index and query, and chunking so aggressively that each piece loses its context. Each of these silently degrades recall in ways that are hard to spot without evaluation.

Production considerations include monitoring index size, vacuuming regularly to avoid bloat, backing up embeddings alongside your relational data, and planning for model upgrades that change the embedding dimension. A well-designed pipeline treats embeddings as first-class data with the same care you give to any other column.
"""
def semantic_search(conn, query: str, limit: int = 3):
    qvec = get_embedding(query, task_type="RETRIEVAL_QUERY")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (qvec, qvec, limit))
        return cur.fetchall()

if __name__ == "__main__":
    conn = get_db_connection()

    setup_database(conn)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        existing = cur.fetchone()[0]

    if existing == 0:
        ingest_document(conn, SAMPLE_DOCUMENT, doc_name="postgres_intro.txt")
    else:
        print(f"ℹ️  Skipping ingestion, {existing} chunks already present.")

    results = semantic_search(conn, "How do I store vectors in Postgres?", limit=5)
    for content, score in results:
        print(f"\nScore: {score:.4f}\n{content[:200]}...")

    conn.close()

    



