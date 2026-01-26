# comparator.py
def evaluate_books(contexts: dict):
    """
    Evaluate which book is most relevant based on similarity scores.
    Returns None only if genuinely no relevant documents exist.
    """
    # Check if any book has relevant documents
    has_relevant_docs = any(len(docs) > 0 for docs in contexts.values())
    
    if not has_relevant_docs:
        return None, {
            "structured": {"error": "No relevant information found in any book"},
            "text": "No relevant documents found. This question may be outside the scope of the available books."
        }
    
    best_book = None
    best_score = -1
    book_analysis = {}

    for book, docs in contexts.items():
        if not docs:
            book_analysis[book] = {
                "avg_score": 0,
                "max_score": 0,
                "num_docs": 0,
                "top_3_avg": 0,
                "status": "No relevant documents found"
            }
            continue
        
        # Use similarity scores (higher is better)
        similarities = [doc.metadata.get("similarity", 0) for doc in docs]
        top_3_similarities = similarities[:min(3, len(similarities))]
        
        avg_score = sum(top_3_similarities) / len(top_3_similarities) if top_3_similarities else 0
        max_score = max(similarities) if similarities else 0
        
        book_analysis[book] = {
            "avg_score": avg_score,
            "max_score": max_score,
            "num_docs": len(docs),
            "top_3_avg": avg_score,
            "all_scores": similarities[:5]
        }
        
        if avg_score > best_score:
            best_score = avg_score
            best_book = book

    # ADJUSTED: Lower minimum threshold - if we have ANY documents, proceed
    # Only reject if best score is extremely low (< 0.45)
    if best_score < 0.45:
        return None, {
            "structured": {"error": "Relevance scores too low"},
            "text": "The available documents do not contain sufficient relevant information to answer this question."
        }

    # Create structured analysis for frontend
    structured_analysis = _create_structured_analysis(book_analysis, best_book, best_score)
    
    # Also create text analysis for the answer
    text_analysis = _create_text_analysis(book_analysis, best_book, best_score)
    
    return best_book, {
        "structured": structured_analysis,
        "text": text_analysis
    }


def _create_structured_analysis(book_analysis: dict, best_book: str, best_score: float):
    """
    Create structured analysis for frontend cards.
    """
    structured = {}
    
    sorted_books = sorted(
        book_analysis.items(), 
        key=lambda x: x[1]["top_3_avg"], 
        reverse=True
    )
    
    for rank, (book, metrics) in enumerate(sorted_books, 1):
        status_emoji = "✅" if book == best_book else "📖"
        
        if metrics.get("status"):
            summary = metrics["status"]
        else:
            summary = f"""{status_emoji} Rank #{rank}

📊 Relevance Score: {metrics['top_3_avg']:.4f}
🎯 Max Score: {metrics['max_score']:.4f}
📚 Documents Found: {metrics['num_docs']}

Top Similarity Scores:
{', '.join([f"{s:.3f}" for s in metrics.get('all_scores', [])[:5]])}"""
        
        if book == best_book:
            summary = f"🏆 SELECTED (Best Match)\n\n{summary}"
        
        structured[book] = summary
    
    return structured


def _create_text_analysis(book_analysis: dict, best_book: str, best_score: float):
    """
    Create detailed text analysis.
    """
    analysis_lines = ["=" * 60]
    analysis_lines.append("📊 COMPARATIVE ANALYSIS OF BOOKS")
    analysis_lines.append("=" * 60)
    
    sorted_books = sorted(
        book_analysis.items(), 
        key=lambda x: x[1]["top_3_avg"], 
        reverse=True
    )
    
    for rank, (book, metrics) in enumerate(sorted_books, 1):
        status = "✅ SELECTED" if book == best_book else "  "
        analysis_lines.append(f"\n{status} Rank #{rank}: {book}")
        analysis_lines.append("-" * 60)
        analysis_lines.append(f"  • Average Relevance Score: {metrics['top_3_avg']:.4f}")
        analysis_lines.append(f"  • Maximum Relevance Score: {metrics['max_score']:.4f}")
        analysis_lines.append(f"  • Documents Retrieved: {metrics['num_docs']}")
        
        if metrics.get('all_scores'):
            scores_str = ", ".join([f"{s:.3f}" for s in metrics['all_scores']])
            analysis_lines.append(f"  • Top 5 Scores: [{scores_str}]")
    
    analysis_lines.append("\n" + "=" * 60)
    analysis_lines.append("📌 CONCLUSION")
    analysis_lines.append("=" * 60)
    
    if best_book:
        analysis_lines.append(
            f"Selected '{best_book}' as the most relevant source "
            f"with an average relevance score of {best_score:.4f}."
        )
        
        if len(sorted_books) > 1:
            second_best_score = sorted_books[1][1]["top_3_avg"]
            diff = best_score - second_best_score
            diff_percent = (diff / best_score * 100) if best_score > 0 else 0
            analysis_lines.append(
                f"This book scored {diff_percent:.1f}% higher than the next best option."
            )
    
    analysis_lines.append("=" * 60)
    
    return "\n".join(analysis_lines)