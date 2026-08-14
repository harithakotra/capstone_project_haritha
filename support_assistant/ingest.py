"""
ingest.py

Ingestion + embedding stage of the RAG pipeline.

Responsibilities:
  1. Load the 8 policy documents from docs/doc_01.txt ... docs/doc_08.txt
  2. Chunk each document (one chunk per document, since each doc is a single
     short policy paragraph - no further splitting is needed given their length)
  3. Embed each chunk locally with sentence-transformers (all-MiniLM-L6-v2)
  4. Store the embeddings + chunk text + metadata in a persistent ChromaDB
     collection called "zepto_policies"

This module is imported by graph.py, which calls get_collection() to run
retrieval queries against the same persisted collection. Running this file
directly (`python ingest.py`) (re)builds the collection from scratch.
"""

import os
import glob

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # lazily-loaded singleton so we only load the model once per process


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def load_documents() -> list[dict]:
    """Load each doc_XX.txt file as a single chunk.

    Returns a list of {"id": "doc_01", "text": "..."} dicts, sorted by doc id.
    """
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt")))
    chunks = []
    for path in paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        chunks.append({"id": doc_id, "text": text})
    return chunks


def build_collection(persist: bool = True) -> chromadb.api.models.Collection.Collection:
    """(Re)build the ChromaDB collection from the docs/ directory.

    If persist=True, uses a PersistentClient so embeddings survive across
    process restarts (used by the FastAPI app). If persist=False, uses an
    in-memory client (useful for quick tests).
    """
    if persist:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
    else:
        client = chromadb.EphemeralClient()

    # Drop and recreate so re-running ingestion is idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    chunks = load_documents()
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": doc_id} for doc_id in ids],
    )
    return collection


def get_collection(persist: bool = True):
    """Get the existing collection, building it first if it doesn't exist yet."""
    if persist:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
    else:
        client = chromadb.EphemeralClient()
    try:
        collection = client.get_collection(COLLECTION_NAME)
        # Sanity check: make sure it's actually populated
        if collection.count() == 0:
            raise ValueError("empty collection")
        return collection
    except Exception:
        return build_collection(persist=persist)


if __name__ == "__main__":
    collection = build_collection(persist=True)
    print(f"Built collection '{COLLECTION_NAME}' with {collection.count()} chunks.")
    for c in load_documents():
        print(f"  - {c['id']}: {c['text'][:60]}...")
