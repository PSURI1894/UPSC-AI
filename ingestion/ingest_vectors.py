"""
ingest_vectors.py

Production-grade ingestion pipeline for UPSC History AI.
- Loads chapter-wise text files
- Converts to LangChain Documents
- Chunks intelligently
- Stores embeddings in FAISS vector stores (book-wise)
"""

from pathlib import Path
from typing import Dict, List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==========================
# CONFIGURATION
# ==========================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

BASE_DATA_DIR = Path("data")
VECTOR_STORE_DIR = Path("vector_store")

BOOKS = {
    "Ancient & Early Medieval India": {
        "data_dir": BASE_DATA_DIR / "history-of-ancient-and-early-medieval-india",
        "store_dir": VECTOR_STORE_DIR / "upinder"
    },
    "India After Gandhi": {
        "data_dir": BASE_DATA_DIR / "india-after-gandhi",
        "store_dir": VECTOR_STORE_DIR / "gandhi"
    }
}


# ==========================
# INITIALIZE EMBEDDINGS
# ==========================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)


# ==========================
# STEP 1: LOAD CHAPTER FILES
# ==========================

def load_chapters(data_dir: Path) -> Dict[str, str]:
    """
    Loads chapter_XX.txt files into memory.
    Returns: {chapter_name: text}
    """
    chapters = {}

    if not data_dir.exists():
        raise FileNotFoundError(f"❌ Data directory not found: {data_dir}")

    for file in sorted(data_dir.glob("chapter_*.txt")):
        chapters[file.stem] = file.read_text(encoding="utf-8")

    if not chapters:
        raise ValueError(f"❌ No chapter files found in {data_dir}")

    print(f"📚 Loaded {len(chapters)} chapters from {data_dir}")
    return chapters


# ==========================
# STEP 2: BUILD DOCUMENTS
# ==========================

def build_documents(chapters: Dict[str, str], book_name: str) -> List[Document]:
    """
    Converts chapter text into LangChain Document objects.
    """
    documents = []

    for chapter_id, text in chapters.items():
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "book": book_name,
                    "chapter": chapter_id
                }
            )
        )

    return documents


# ==========================
# STEP 3: CHUNK DOCUMENTS
# ==========================

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into overlapping semantic chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunked_docs = []

    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                )
            )

    print(f"✂️ Created {len(chunked_docs)} chunks")
    return chunked_docs


# ==========================
# STEP 4: STORE IN FAISS
# ==========================

def store_vectors(documents: List[Document], store_dir: Path):
    """
    Embeds documents and saves FAISS index to disk.
    """
    store_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    vectorstore.save_local(str(store_dir))
    print(f"✅ Vector store saved at {store_dir}")


# ==========================
# MAIN PIPELINE
# ==========================

def ingest_book(book_name: str, data_dir: Path, store_dir: Path):
    print(f"\n🚀 Ingesting: {book_name}")

    chapters = load_chapters(data_dir)
    documents = build_documents(chapters, book_name)
    chunked_docs = chunk_documents(documents)
    store_vectors(chunked_docs, store_dir)

    print(f"🎯 Completed ingestion for: {book_name}")


if __name__ == "__main__":
    for book_name, config in BOOKS.items():
        ingest_book(
            book_name=book_name,
            data_dir=config["data_dir"],
            store_dir=config["store_dir"]
        )

    print("\n🏁 All books ingested successfully.")
