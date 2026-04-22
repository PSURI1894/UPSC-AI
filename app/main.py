import sys
from pathlib import Path

# Make project root importable when running from app/ or project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
from rag.retriever import retrieve, get_raw_max_similarity


def _to_python(obj):
    """Recursively convert numpy scalars/arrays to Python native types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
from rag.comparator import evaluate_books
from rag.answer_generator import generate_answer
from pipeline.soc_monitor import SOCMonitor

app = FastAPI(title="UPSC-AI + Adversarially Resilient Detection Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR.parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Pipeline: single global SOC monitor instance ─────────────────────────────
_soc_monitor = SOCMonitor(alpha=0.10)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def serve_home():
    if not INDEX_FILE.exists():
        return JSONResponse({"error": "index.html not found"}, status_code=404)
    return FileResponse(INDEX_FILE)


@app.post("/ask")
def ask_question(payload: QuestionRequest):
    """
    Main Q&A endpoint — now guarded by the adversarially resilient pipeline.

    Pipeline steps (mirroring main_pipeline.py):
        1. Retrieve documents (FAISS similarity search)
        2. Run SOC monitor:
            a. Adversarial query detection (patterns + score anomaly)
            b. Conformal scoring (split-CP on similarity)
            c. Streaming drift detection (ADWIN + Page-Hinkley)
        3. Generate answer (blocked if EVASION_LOCKED)
        4. Return answer + pipeline_status in response
    """
    question = payload.question.strip()

    if not question:
        return JSONResponse({"error": "Question cannot be empty"}, status_code=400)

    try:
        # ── Step 1: Retrieve ──────────────────────────────────────────
        contexts = retrieve(question, top_k=5)

        # Raw max similarity (no threshold filter) — avoids false 0.0 on
        # legit questions that narrowly miss the retriever cutoff
        max_similarity = get_raw_max_similarity(question)

        all_scores = []
        for docs in contexts.values():
            for doc in docs:
                sim = doc.metadata.get("similarity", 0.0)
                all_scores.append(float(sim))

        # ── Step 2: SOC Pipeline ──────────────────────────────────────
        alert = _soc_monitor.process_query(
            query=question,
            max_similarity=max_similarity,
            all_scores=all_scores,
        )

        pipeline_status = {
            "state": alert.state.value,
            "signals": alert.signals,
            "severity": alert.severity,
            "action": alert.action,
            "adversarial": {
                "is_adversarial": alert.adversarial_result.is_adversarial,
                "attack_types": alert.adversarial_result.attack_types,
                "confidence": alert.adversarial_result.confidence,
            },
            "conformal": {
                "max_similarity": alert.conformal_result.max_similarity,
                "p_value": alert.conformal_result.p_value,
                "is_anomalous": alert.conformal_result.is_anomalous,
            },
            "drift": alert.drift_signals,
        }

        # ── Step 3: Block if EVASION_LOCKED ──────────────────────────
        from pipeline.soc_monitor import SOCState
        if alert.state == SOCState.EVASION_LOCKED:
            return JSONResponse(
                {
                    "success": False,
                    "blocked": True,
                    "reason": "Query blocked by adversarial detection pipeline.",
                    "attack_types": alert.adversarial_result.attack_types,
                    "pipeline_status": pipeline_status,
                },
                status_code=403,
            )

        # ── Step 4: Evaluate + Generate answer ───────────────────────
        best_book, analysis_data = evaluate_books(contexts)

        answer = generate_answer(
            question,
            contexts,
            best_book,
            analysis_data.get("text", ""),
        )

        return _to_python({
            "success": True,
            "question": question,
            "best_book": best_book,
            "analysis": analysis_data.get("structured", {}),
            "comparison": analysis_data.get("text", ""),
            "answer": answer,
            "books_analyzed": list(contexts.keys()),
            "pipeline_status": pipeline_status,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"success": False, "error": f"Error processing question: {str(e)}"},
            status_code=500,
        )


@app.get("/health")
def health_check():
    from rag.retriever import BOOK_VECTORS
    return {
        "status": "healthy",
        "books_loaded": list(BOOK_VECTORS.keys()),
        "total_books": len(BOOK_VECTORS),
        "pipeline_state": _soc_monitor.state.value,
    }


# ── Pipeline-specific endpoints ───────────────────────────────────────────────

@app.get("/dashboard")
def serve_dashboard():
    """Serve the live SOC pipeline dashboard."""
    f = BASE_DIR.parent / "static" / "dashboard.html"
    return FileResponse(f)


@app.get("/pipeline/status")
def pipeline_status():
    """Live SOC monitor status — call this to watch the pipeline in real time."""
    return _soc_monitor.get_status()


@app.get("/pipeline/report")
def pipeline_report():
    """Full SOC report as plain text (for terminal display)."""
    return {"report": _soc_monitor.get_full_report()}


@app.post("/pipeline/reset")
def pipeline_reset():
    """Reset the SOC state machine (keeps conformal calibration)."""
    _soc_monitor.reset()
    return {"message": "Pipeline state reset to STABLE."}
