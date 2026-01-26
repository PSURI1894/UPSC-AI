# retriever.py
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_STORES = {
    "Ancient & Early Medieval India": "vector_store/upinder",
    "India After Gandhi": "vector_store/gandhi"
}

BOOK_VECTORS = {}

for book, path in VECTOR_STORES.items():
    if Path(path).exists():
        BOOK_VECTORS[book] = FAISS.load_local(
            path,
            EMBEDDING_MODEL,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Loaded vector store: {book}")
    else:
        print(f"❌ Vector store missing: {book}")


# More lenient threshold - accept more documents
RELEVANCE_THRESHOLD = 0.48  # Lowered from 0.50 to be more inclusive


def retrieve(query: str, top_k=10):  # Get more candidates
    """
    Retrieve relevant documents from all books.
    Only returns documents above the relevance threshold.
    """
    results = {}
    for book, store in BOOK_VECTORS.items():
        docs_and_scores = store.similarity_search_with_score(query, k=top_k)
        docs = []
        for doc, score in docs_and_scores:
            # Convert to similarity score
            similarity = 1 / (1 + score)
            
            # Only include documents above threshold
            if similarity >= RELEVANCE_THRESHOLD:
                doc.metadata["score"] = score
                doc.metadata["similarity"] = similarity
                doc.metadata["book"] = book
                docs.append(doc)
        
        results[book] = docs
    
    return results