# retriever.py
import warnings
import os
warnings.filterwarnings("ignore")

from pathlib import Path
from langchain_community.vectorstores import FAISS

RELEVANCE_THRESHOLD = 0.48

# ── Lazy embedding model ───────────────────────────────────────────────────────
# Do NOT instantiate at module level — if the env var is missing the whole
# lambda crashes before it can even serve the homepage.

_EMBEDDING_MODEL = None

def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        token = (
            os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HF_API_TOKEN")
        )
        _EMBEDDING_MODEL = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=token,
        )
    return _EMBEDDING_MODEL


# ── Lazy vector store loader ───────────────────────────────────────────────────

_BOOK_VECTORS = None

VECTOR_STORES = {
    "Ancient & Early Medieval India": "vector_store/upinder",
    "India After Gandhi": "vector_store/gandhi",
}

def _get_book_vectors():
    global _BOOK_VECTORS
    if _BOOK_VECTORS is None:
        emb = _get_embedding_model()
        _BOOK_VECTORS = {}
        for book, path in VECTOR_STORES.items():
            if Path(path).exists():
                _BOOK_VECTORS[book] = FAISS.load_local(
                    path, emb, allow_dangerous_deserialization=True
                )
                print(f"[OK] Loaded vector store: {book}")
            else:
                print(f"[MISSING] Vector store missing: {book}")
    return _BOOK_VECTORS


def get_raw_max_similarity(query: str) -> float:
    """
    Return the highest similarity score across all books WITHOUT threshold
    filtering. Used by the pipeline so it never sees a false 0.0 for legit
    queries that narrowly miss the retriever cutoff.
    """
    best = 0.0
    for store in _get_book_vectors().values():
        hits = store.similarity_search_with_score(query, k=1)
        if hits:
            raw_score = hits[0][1]
            sim = 1 / (1 + float(raw_score))
            best = max(best, sim)
    return float(best)


def retrieve(query: str, top_k=10):
    """
    Retrieve relevant documents from all books.
    Only returns documents above the relevance threshold.
    """
    results = {}
    for book, store in _get_book_vectors().items():
        docs_and_scores = store.similarity_search_with_score(query, k=top_k)
        docs = []
        for doc, score in docs_and_scores:
            similarity = 1 / (1 + score)
            if similarity >= RELEVANCE_THRESHOLD:
                doc.metadata["score"] = float(score)
                doc.metadata["similarity"] = float(similarity)
                doc.metadata["book"] = book
                docs.append(doc)
        results[book] = docs
    return results
