from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel

from rag.retriever import retrieve
from rag.comparator import evaluate_books
from rag.answer_generator import generate_answer

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR.parent / "static" / "index.html"


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def serve_home():
    """Serve the main HTML page"""
    if not INDEX_FILE.exists():
        return JSONResponse(
            {"error": "index.html not found"},
            status_code=404
        )
    return FileResponse(INDEX_FILE)


@app.post("/ask")
def ask_question(payload: QuestionRequest):
    """
    Main endpoint to process questions.
    Returns answer with comparative analysis.
    """
    question = payload.question.strip()
    
    if not question:
        return JSONResponse(
            {"error": "Question cannot be empty"},
            status_code=400
        )

    try:
        
        contexts = retrieve(question, top_k=5)
        
        
        best_book, analysis_data = evaluate_books(contexts)
        
        
        answer = generate_answer(
            question, 
            contexts, 
            best_book, 
            analysis_data.get("text", "")
        )

        return {
            "success": True,
            "question": question,
            "best_book": best_book,
            "analysis": analysis_data.get("structured", {}),  # For cards
            "comparison": analysis_data.get("text", ""),  # For text display
            "answer": answer,
            "books_analyzed": list(contexts.keys())
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {
                "success": False,
                "error": f"Error processing question: {str(e)}"
            },
            status_code=500
        )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    from rag.retriever import BOOK_VECTORS
    
    loaded_books = list(BOOK_VECTORS.keys())
    
    return {
        "status": "healthy",
        "books_loaded": loaded_books,
        "total_books": len(loaded_books)
    }