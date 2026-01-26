import re


def _format_context(docs, max_docs=5):
    """
    Format retrieved documents into readable context.
    """
    formatted = []
    for i, doc in enumerate(docs[:max_docs], 1):
        chapter = doc.metadata.get("chapter", "Unknown Chapter")
        book = doc.metadata.get("book", "Unknown Book")
        similarity = doc.metadata.get("similarity", 0)
        
        snippet = doc.page_content[:800].strip().replace("\n", " ")
        if len(doc.page_content) > 800:
            snippet += "..."
        
        formatted.append(
            f"[Source {i}] {book} - {chapter}\n"
            f"(Relevance: {similarity:.3f})\n"
            f"{snippet}"
        )
    
    return "\n\n".join(formatted)


def _is_mcq_question(question: str) -> bool:
    """
    Detect if the question is an MCQ.
    """
    mcq_indicators = [
        r'\(a\)',
        r'\(b\)',
        r'\(c\)',
        r'\(d\)',
        r'which of the following',
        r'select the correct',
        r'which of the statements',
        r'choose the correct',
        r'identify the correct',
        r'which one of the following',
    ]
    
    question_lower = question.lower()
    
    for pattern in mcq_indicators:
        if re.search(pattern, question_lower):
            return True
    
    return False


def generate_answer(question: str, contexts: dict, best_book: str, comparative_analysis: str = None):
    """
    Generate a detailed UPSC-style answer.
    Supports both descriptive and MCQ questions.
    """
   
    if not best_book:
        return f"""{'=' * 70}
⚠️ INSUFFICIENT INFORMATION
{'=' * 70}

The question you asked does not appear to be covered in the available books:
- Ancient & Early Medieval India
- India After Gandhi

POSSIBLE REASONS:
1. The topic is outside the time period covered by these books
2. The question requires information not present in these sources
3. The question may need to be rephrased for better results

SUGGESTIONS:
- Check if your question relates to Ancient/Medieval Indian history or Modern India (post-1947)
- Try rephrasing the question with more historical context
- Ensure the topic is within the scope of UPSC History syllabus covered by these books

Available Coverage:
📚 Ancient & Early Medieval India: Prehistoric to Medieval period
📚 India After Gandhi: Post-independence India (1947 onwards)
{'=' * 70}"""
    
    if best_book not in contexts:
        return "⚠️ Unable to generate answer: No suitable book selected."
    
    chunks = contexts.get(best_book, [])
    
    if not chunks:
        return f"""{'=' * 70}
⚠️ NO RELEVANT INFORMATION FOUND
{'=' * 70}

No sufficiently relevant passages were found in '{best_book}' to answer this question.

This could mean:
- The specific topic is not covered in detail in this book
- The question needs more specific historical keywords
- Try rephrasing with more context

{'=' * 70}"""
    
    # Detect if it's an MCQ
    is_mcq = _is_mcq_question(question)
    
    context_text = _format_context(chunks, max_docs=10)
    
    if is_mcq:
        # Generate MCQ-specific answer
        answer = f"""{'=' * 70}
UPSC PRELIMS MCQ
{'=' * 70}

{question}

{'=' * 70}
📚 PRIMARY SOURCE: {best_book}
{'=' * 70}

{'=' * 70}
RELEVANT INFORMATION FROM SOURCE:
{'=' * 70}

{context_text}

{'=' * 70}
✅ ANSWER WITH EXPLANATION
{'=' * 70}

{_solve_mcq(question, chunks)}
"""
    else:
        # Generate descriptive answer
        answer = f"""{'=' * 70}
QUESTION: {question}
{'=' * 70}

📚 PRIMARY SOURCE: {best_book}

{'=' * 70}
RELEVANT INFORMATION FROM SOURCE:
{'=' * 70}

{context_text}

{'=' * 70}
✍️ UPSC STYLE ANSWER
{'=' * 70}

{_synthesize_upsc_answer(question, chunks)}
"""
    
    return answer

def _solve_mcq(question: str, docs):
    """
    Analyze MCQ and provide answer with explanation.
    Handles three types: direct answer, list-based, and statement-based MCQs.
    """
    if not docs:
        return "Insufficient information to solve this MCQ."
    
    # Extract relevant information from documents
    all_content = []
    for doc in docs[:10]:
        content = doc.page_content.strip()
        all_content.append(content)
    
    combined_content = " ".join(all_content)
    
    # Improved MCQ type detection
    has_select_pattern = bool(re.search(r'select\s+the\s+correct', question, re.IGNORECASE))
    has_numbered_items = bool(re.search(r'\((\d+)\)', question))
    has_select_code = "Select the correct answer using the code" in question
    
    # Check for plain list items (without numbers)
    lines = question.strip().split('\n')
    plain_list_items = []
    
    after_question = False
    for line in lines:
        line = line.strip()
        
        if line.endswith('?'):
            after_question = True
            continue
        
        if re.search(r'select\s+the\s+correct', line, re.IGNORECASE):
            break
        
        if re.match(r'\([a-d]\)', line):
            break
        
        if after_question and line and len(line) > 3:
            # Clean line looks like a list item
            clean = re.sub(r'^[-•*\d+\.]\s*', '', line).strip()
            if clean and len(clean) > 2:
                plain_list_items.append(clean)
    
    has_plain_list = len(plain_list_items) >= 2
    
    # Check for "Which one of the following"
    is_which_one = bool(re.search(r'which\s+one\s+of\s+the\s+following', question, re.IGNORECASE))
    
    # Check for "Which of the following" (can be list-based)
    is_which_of = bool(re.search(r'which\s+of\s+the\s+following', question, re.IGNORECASE))
    
    # Decision logic
    if has_select_code or (has_numbered_items and not is_which_one):
        return _solve_list_mcq(question, combined_content)
    elif has_plain_list and has_select_pattern and not is_which_one:
        # Plain list with "Select the correct" = list-based MCQ
        return _solve_list_mcq(question, combined_content)
    elif is_which_one:
        return _solve_direct_mcq(question, combined_content)
    else:
        # Default to direct MCQ
        return _solve_direct_mcq(question, combined_content)


def _solve_statement_verification_mcq(question: str, content, statement_lines=None):
    """
    Solve MCQs where you verify which statements are correct.
    """
    lines = []
    
    # Extract the main question
    question_parts = question.split('\n')
    main_question = question_parts[0].strip()
    
    # Extract statement lines if not provided
    if not statement_lines:
        statement_lines = []
        for line in question_parts[1:]:
            line = line.strip()
            if re.search(r'select\s+the\s+correct', line, re.IGNORECASE):
                break
            if line and len(line) > 10 and not re.search(r'\([a-d]\)', line):
                statement_lines.append(line)
    
    # Number the statements
    statements = {}
    for i, stmt in enumerate(statement_lines, 1):
        statements[i] = stmt
    
    lines.append(f"QUESTION: {main_question}\n")
    lines.append("VERIFICATION OF EACH STATEMENT:\n")
    
    correct_statements = []
    statement_analysis = {}
    
    for num, stmt_text in statements.items():
        # Analyze this statement
        keywords = _extract_keywords(stmt_text)
        content_lower = content.lower()
        
        # Calculate relevance
        matches = sum(1 for kw in keywords if len(kw) > 3 and kw in content_lower)
        total_keywords = sum(1 for kw in keywords if len(kw) > 3)
        relevance = (matches / total_keywords) if total_keywords > 0 else 0
        
        # Find evidence
        evidence = _find_evidence_in_content(stmt_text.lower(), content)
        
        # Check for negative evidence
        is_negated = _check_for_negation(stmt_text, evidence, content)
        
        # Determine if correct
        is_correct = (relevance >= 0.3 or len(evidence) > 50) and not is_negated
        
        if is_correct:
            correct_statements.append(num)
        
        statement_analysis[num] = {
            'text': stmt_text,
            'is_correct': is_correct,
            'relevance': relevance,
            'evidence': evidence,
            'negated': is_negated
        }
        
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT/NOT SUPPORTED"
        
        lines.append(f"Statement {num}: {stmt_text}")
        lines.append(f"Status: {status} (Confidence: {relevance:.0%})")
        
        if evidence:
            lines.append(f"Evidence: {evidence[:300]}...")
        else:
            lines.append("Evidence: No supporting evidence found")
        
        if is_negated:
            lines.append("Note: Evidence suggests this statement may be incorrect")
        
        lines.append("")
    
    # Extract options and match
    options = _extract_options_from_text(question)
    correct_option = _match_statements_to_option(correct_statements, options, len(statements))
    
    lines.append("=" * 70)
    lines.append(f"CORRECT ANSWER: {correct_option}")
    lines.append("=" * 70)
    
    lines.append("\nEXPLANATION:")
    
    if correct_statements:
        correct_str = ", ".join(str(i) for i in correct_statements)
        lines.append(f"Statement(s) {correct_str} are supported by the source material.\n")
    else:
        lines.append("None of the statements are clearly supported by the source material.\n")
    
    # Explain each statement
    for num in sorted(statement_analysis.keys()):
        analysis = statement_analysis[num]
        if analysis['is_correct']:
            lines.append(f"✓ Statement {num}: Supported - {analysis['evidence'][:150]}...")
        else:
            if analysis['negated']:
                lines.append(f"✗ Statement {num}: Contradicted by evidence")
            else:
                lines.append(f"✗ Statement {num}: No clear supporting evidence found")
    
    lines.append("\n" + "=" * 70)
    lines.append("NOTE: Cross-verify with standard UPSC reference books.")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def _check_for_negation(statement: str, evidence: str, full_content: str):
    """Check if evidence contradicts the statement."""
    evidence_lower = evidence.lower() if evidence else ""
    
    negative_words = ['no', 'not', 'never', 'absent', 'lacking', 'without', 'none']
    key_words_in_statement = _extract_keywords(statement)
    
    for neg_word in negative_words:
        if neg_word in evidence_lower:
            for key_word in key_words_in_statement[:3]:
                if key_word in evidence_lower:
                    idx = evidence_lower.find(key_word)
                    context = evidence_lower[max(0, idx-50):min(len(evidence_lower), idx+50)]
                    if neg_word in context:
                        return True
    
    return False


def _extract_options_from_text(question: str):
    """Extract options like 'a. 1 and 2 only' - handles both formats."""
    options = {}
    
    # Try parentheses format first
    pattern1 = r'\(([a-d])\)\s+([^\n]+)'
    matches = re.findall(pattern1, question.lower())
    
    if not matches:
        # Try dot format
        pattern2 = r'([a-d])\.\s+([^\n]+)'
        matches = re.findall(pattern2, question.lower())
    
    for letter, option_text in matches:
        numbers = []
        
        if 'none' in option_text:
            numbers = []
        elif 'all' in option_text:
            # Extract all numbers from the original question to know max
            all_nums_in_question = re.findall(r'\((\d+)\)', question)
            if all_nums_in_question:
                numbers = [int(n) for n in all_nums_in_question]
            else:
                numbers = [1, 2, 3, 4]  # Default assumption
        else:
            found_numbers = re.findall(r'\d+', option_text)
            numbers = [int(n) for n in found_numbers]
        
        options[letter] = numbers
    
    return options


def _match_statements_to_option(correct_statements, options, total_statements):
    """Match correct statements to answer options."""
    correct_set = set(correct_statements)
    
    # Try exact match
    for letter, option_stmts in options.items():
        if set(option_stmts) == correct_set:
            return f"({letter})"
    
    # If no correct statements, check for "None" option
    if not correct_statements:
        for letter, option_stmts in options.items():
            if not option_stmts:
                return f"({letter})"
        if 'd' in options:
            return "(d) None of the statements"
    
    # Return best guess
    if correct_statements:
        stmt_str = " and ".join(str(s) for s in sorted(correct_statements))
        return f"Statement(s) {stmt_str} only"
    
    return "Unable to determine"


def _solve_direct_mcq(question: str, content):
    """Solve direct MCQs where you choose the correct option directly."""
    lines = []
    
    main_question = question.split('(a)')[0].strip()
    question_entities = _extract_entities_from_question(main_question)
    options = _extract_direct_options(question)
    
    if not options:
        return "Could not parse the question options."
    
    lines.append(f"QUESTION ANALYSIS:")
    lines.append(f"{main_question}\n")
    
    if question_entities:
        lines.append(f"Key entities mentioned: {', '.join(question_entities)}\n")
    
    lines.append("EVALUATION OF EACH OPTION:\n")
    
    best_option = None
    best_score = 0
    option_analysis = {}
    
    for letter, option_text in sorted(options.items()):
        option_keywords = _extract_keywords(option_text)
        content_lower = content.lower()
        
        combined_keywords = question_entities + option_keywords
        
        matches = sum(1 for kw in combined_keywords if kw in content_lower)
        relevance = matches / len(combined_keywords) if combined_keywords else 0
        
        evidence_sentences = _find_multiple_evidence(question_entities, option_text, content)
        
        evidence_score = 0
        if evidence_sentences:
            for evidence in evidence_sentences:
                evidence_lower = evidence.lower()
                entity_matches = sum(1 for ent in question_entities if ent in evidence_lower)
                option_matches = sum(1 for kw in option_keywords if kw in evidence_lower)
                
                if entity_matches > 0 and option_matches > 0:
                    evidence_score += (entity_matches + option_matches) / (len(question_entities) + len(option_keywords))
        
        total_score = (relevance * 0.4) + (evidence_score * 0.6)
        
        if total_score > best_score:
            best_score = total_score
            best_option = letter
        
        option_analysis[letter] = {
            'text': option_text,
            'score': total_score,
            'evidence': evidence_sentences,
            'relevance': relevance
        }
        
        if total_score > 0.4:
            status = "✓✓ HIGHLY LIKELY"
        elif total_score > 0.25:
            status = "✓ POSSIBLE"
        else:
            status = "✗ UNLIKELY"
        
        lines.append(f"Option ({letter}): {option_text}")
        lines.append(f"Status: {status} (Confidence: {total_score:.0%})")
        
        if evidence_sentences and len(evidence_sentences) > 0:
            lines.append(f"Supporting Evidence:")
            for i, ev in enumerate(evidence_sentences[:3], 1):
                lines.append(f"  {i}. {ev}")
        else:
            lines.append("Supporting Evidence: None found in the source material")
        
        lines.append("")
    
    lines.append("=" * 70)
    
    if best_option and best_score > 0.2:
        lines.append(f"CORRECT ANSWER: ({best_option}) {options[best_option]}")
    else:
        lines.append("ANSWER: Unable to determine with confidence from available sources")
    
    lines.append("=" * 70)
    
    lines.append("\nDETAILED EXPLANATION:")
    
    if best_option and best_score > 0.2:
        best_analysis = option_analysis[best_option]
        
        lines.append(f"\nOption ({best_option}) - '{options[best_option]}' - is the most likely correct answer.\n")
        
        lines.append("REASONING:")
        
        if best_analysis['evidence']:
            lines.append(f"The source material contains {len(best_analysis['evidence'])} relevant reference(s):\n")
            
            for i, evidence in enumerate(best_analysis['evidence'][:3], 1):
                lines.append(f"{i}. {evidence}\n")
            
            if question_entities:
                lines.append(f"These references connect the entities mentioned in the question")
                lines.append(f"({', '.join(question_entities)}) with option ({best_option}).")
        else:
            lines.append(f"This option scored highest based on keyword relevance ({best_analysis['relevance']:.0%}).")
        
        lines.append("\nWHY OTHER OPTIONS ARE LESS LIKELY:")
        for letter, analysis in option_analysis.items():
            if letter != best_option:
                if analysis['evidence']:
                    lines.append(f"  ({letter}) Has limited supporting evidence (score: {analysis['score']:.0%})")
                else:
                    lines.append(f"  ({letter}) No clear evidence found in the source material")
    else:
        lines.append("\nThe available source material does not provide sufficient information to")
        lines.append("answer this question with confidence.")
    
    lines.append("\n" + "=" * 70)
    lines.append("NOTE: For UPSC preparation, cross-verify with standard reference books.")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def _extract_entities_from_question(question_text):
    """Extract important entities from the question."""
    entities = []
    
    words = question_text.split()
    for word in words:
        word_clean = word.strip('.,;:!?()')
        if word_clean and word_clean[0].isupper() and len(word_clean) > 3:
            if word_clean.lower() not in ['with', 'which', 'what', 'when', 'where', 'during', 'the']:
                entities.append(word_clean.lower())
    
    pattern = r'(?:towns?|cities|places?|kingdoms?|dynasties?|books?)\s+([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*(?:\s+and\s+[A-Z][a-z]+)?)'
    matches = re.findall(pattern, question_text)
    for match in matches:
        names = re.split(r',\s*|\s+and\s+', match)
        entities.extend([n.strip().lower() for n in names if n.strip()])
    
    return list(set(entities))


def _find_multiple_evidence(entities, option_text, content, max_results=3):
    """Find multiple pieces of evidence."""
    evidence_list = []
    sentences = re.split(r'[.!?]+', content)
    
    entities_lower = [e.lower() for e in entities]
    option_keywords = _extract_keywords(option_text)
    
    scored_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 30:
            continue
        
        sentence_lower = sentence.lower()
        entity_count = sum(1 for entity in entities_lower if entity in sentence_lower)
        keyword_count = sum(1 for kw in option_keywords if kw in sentence_lower)
        
        if entity_count > 0 or keyword_count > 1:
            score = (entity_count * 2) + keyword_count
            scored_sentences.append((score, sentence))
    
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    
    for score, sentence in scored_sentences[:max_results]:
        if score > 0:
            evidence_list.append(sentence.strip())
    
    return evidence_list


def _extract_direct_options(question: str):
    """Extract options (a), (b), (c), (d) - handles both (a) and a. formats."""
    options = {}
    
    # Try pattern with parentheses: (a), (b), (c), (d)
    pattern1 = r'\(([a-d])\)\s+([^\(\n]+?)(?=\s*\([a-d]\)|$)'
    matches = re.findall(pattern1, question, re.DOTALL)
    
    if matches:
        for letter, text in matches:
            options[letter] = text.strip()
        return options
    
    # Try pattern with dots: a., b., c., d.
    pattern2 = r'([a-d])\.\s+([^\n]+?)(?=\s*[a-d]\.|$)'
    matches = re.findall(pattern2, question, re.DOTALL)
    
    if matches:
        for letter, text in matches:
            options[letter] = text.strip()
        return options
    
    return options


def _solve_list_mcq(question: str, content):
    """Solve list-based MCQs."""
    lines = []
    
    items = _extract_numbered_items(question)
    
    if not items:
        # Fallback: try to parse as direct MCQ instead
        return _solve_direct_mcq(question, content)
    
    options = _extract_options(question)
    
    lines.append("ANALYSIS OF EACH ITEM:\n")
    
    correct_items = []
    item_analysis = {}
    
    for num, text in items.items():
        keywords = _extract_keywords(text)
        content_lower = content.lower()
        
        matches = sum(1 for kw in keywords if len(kw) > 3 and kw in content_lower)
        total_keywords = sum(1 for kw in keywords if len(kw) > 3)
        
        relevance_score = (matches / total_keywords) if total_keywords > 0 else 0
        evidence = _find_evidence_in_content(text.lower(), content)
        
        # More lenient threshold for historical names/places
        is_correct = relevance_score >= 0.3 or len(evidence) > 40
        
        if is_correct:
            correct_items.append(num)
        
        item_analysis[num] = {
            'text': text,
            'is_correct': is_correct,
            'relevance': relevance_score,
            'evidence': evidence
        }
        
        status = "✓ CORRECT" if is_correct else "✗ NOT FOUND/INCORRECT"
        
        lines.append(f"{num}. {text}")
        lines.append(f"Status: {status} (Relevance: {relevance_score:.0%})")
        if evidence:
            lines.append(f"Evidence: {evidence[:250]}...")
        else:
            lines.append("Evidence: No direct mention found.")
        lines.append("")
    
    correct_option = _match_to_option(correct_items, options)
    
    lines.append("=" * 70)
    lines.append(f"CORRECT ANSWER: {correct_option}")
    lines.append("=" * 70)
    
    lines.append("\nEXPLANATION:")
    if correct_items:
        items_str = ", ".join(str(i) for i in correct_items)
        lines.append(f"Items {items_str} are mentioned/supported in the source material.\n")
        
        # Detailed explanation for each
        for num in correct_items:
            analysis = item_analysis[num]
            lines.append(f"✓ {num}. {analysis['text']}")
            if analysis['evidence']:
                lines.append(f"   Evidence: {analysis['evidence'][:200]}...")
            lines.append("")
    else:
        lines.append("None of the items found clear supporting evidence.\n")
    
    lines.append("NOTE: Cross-verify with standard UPSC textbooks.")
    lines.append("=" * 70)
    
    return "\n".join(lines)

def _extract_numbered_items(question: str):
    """Extract numbered items."""
    items = {}
    pattern = r'\((\d+)\)\s+([^\(]+?)(?=\s*\(\d+\)|Select the correct|$)'
    matches = re.findall(pattern, question, re.DOTALL)
    
    for num, text in matches:
        items[int(num)] = text.strip()
    
    return items


def _extract_options(question: str):
    """Extract answer options with numbers - handles both (a) and a. formats."""
    options = {}
    
    # Try pattern with parentheses first
    pattern1 = r'\(([a-d])\)\s+([^\n]+)'
    matches = re.findall(pattern1, question.lower())
    
    if not matches:
        # Try pattern with dots
        pattern2 = r'([a-d])\.\s+([^\n]+)'
        matches = re.findall(pattern2, question.lower())
    
    for letter, option_text in matches:
        numbers = []
        
        if 'none' in option_text:
            numbers = []
        elif 'all' in option_text or ('1' in option_text and '2' in option_text and '3' in option_text and '4' in option_text):
            # Count how many numbers are actually in the question
            all_nums = re.findall(r'\d+', option_text)
            if all_nums:
                numbers = [int(n) for n in all_nums]
            else:
                numbers = [1, 2, 3, 4]  # Assume all 4
        else:
            found_numbers = re.findall(r'\d+', option_text)
            numbers = [int(n) for n in found_numbers]
        
        options[letter] = numbers
    
    return options


def _find_evidence_in_content(search_text, content):
    """Find best matching sentence."""
    sentences = re.split(r'[.!?]+', content)
    keywords = _extract_keywords(search_text)
    
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        
        sentence_lower = sentence.lower()
        matches = sum(1 for keyword in keywords if keyword in sentence_lower)
        
        if matches > best_score:
            best_score = matches
            best_sentence = sentence
    
    return best_sentence


def _match_to_option(correct_items, options):
    """Match correct items to options."""
    correct_set = set(correct_items)
    
    for letter, items in options.items():
        if set(items) == correct_set:
            return f"({letter})"
    
    if correct_items:
        items_str = ", ".join(str(i) for i in sorted(correct_items))
        return f"Items {items_str}"
    
    return "Unable to determine"


def _extract_keywords(text):
    """Extract important keywords."""
    common_words = {'a', 'an', 'the', 'is', 'was', 'were', 'are', 'be', 'been', 'being',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                    'could', 'may', 'might', 'must', 'can', 'of', 'to', 'in', 'for', 'on',
                    'with', 'as', 'by', 'at', 'from', 'or', 'and', 'but', 'if', 'that',
                    'this', 'which', 'who', 'what', 'when', 'where', 'why', 'how', 'their',
                    'there', 'they', 'them', 'then', 'than', 'these', 'those', 'well', 'known'}
    
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if w not in common_words and len(w) > 3]
    
    return keywords


def _synthesize_upsc_answer(question: str, docs):
    """
    Synthesize a proper UPSC-style answer for descriptive questions.
    Creates structured, relevant answers from retrieved documents.
    """
    if not docs:
        return "Insufficient information available to answer this question."
    
    
    all_content = []
    for doc in docs[:7]: 
        content = doc.page_content.strip()
        all_content.append(content)
    
    combined_content = " ".join(all_content)
    
    # Extract question keywords to find relevant sentences
    question_keywords = _extract_keywords(question)
    
    # Find the most relevant sentences
    relevant_sentences = _find_relevant_sentences(combined_content, question_keywords, max_sentences=10)
    
    if not relevant_sentences:
        return "The available source material does not contain sufficient information to answer this question comprehensively."
    
    # Structure the answer
    answer = _structure_upsc_answer(question, relevant_sentences)
    
    return answer


def _find_relevant_sentences(content: str, keywords: list, max_sentences: int = 10):
    """
    Find the most relevant sentences from content based on keywords.
    Returns sentences sorted by relevance.
    """
    sentences = re.split(r'[.!?]+', content)
    scored_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 30:  # Skip very short sentences
            continue
        
        sentence_lower = sentence.lower()
        
        # Count keyword matches
        score = 0
        for keyword in keywords:
            if keyword in sentence_lower:
                score += 1
        
        # Bonus for longer, informative sentences
        if len(sentence) > 100:
            score += 0.5
        
        if score > 0:
            scored_sentences.append((score, sentence))
    
    # Sort by score and return top sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    
    # Return only the sentence text, not the score
    return [sentence for score, sentence in scored_sentences[:max_sentences]]


def _structure_upsc_answer(question: str, relevant_sentences: list):
    """
    Structure the answer in proper UPSC format with introduction, body, and conclusion.
    """
    # Group sentences into coherent paragraphs (3-4 sentences per paragraph)
    paragraphs = []
    current_para = []
    
    for i, sentence in enumerate(relevant_sentences[:8]):  # Use top 8 sentences
        current_para.append(sentence)
        
        # Create paragraph every 2-3 sentences
        if len(current_para) >= 2 and (i == len(relevant_sentences[:8]) - 1 or len(current_para) >= 3):
            paragraph = " ".join(current_para)
            paragraphs.append(paragraph)
            current_para = []
    
    # If there are leftover sentences
    if current_para:
        paragraphs.append(" ".join(current_para))
    
    # Build the answer
    answer_parts = []
    
    # Introduction
    answer_parts.append("ANSWER:\n")
    
    # Main body with paragraphs
    for i, para in enumerate(paragraphs):
        answer_parts.append(para)
        if i < len(paragraphs) - 1:  # Add spacing between paragraphs
            answer_parts.append("")
    
    # Conclusion/Summary note
    answer_parts.append("\n" + "=" * 70)
    answer_parts.append("NOTE FOR UPSC PREPARATION:")
    answer_parts.append("• This answer is based on historical sources")
    answer_parts.append("• Cross-verify key facts, dates, and events with standard textbooks")
    answer_parts.append("• Make concise notes highlighting the main points")
    answer_parts.append("• Practice writing within the word limit for better time management")
    answer_parts.append("=" * 70)
    
    return "\n".join(answer_parts)
def _solve_factual_question(question: str, content: str):
    """Intelligently solve factual questions using semantic understanding."""
    lines = []
    
    # Extract question and options
    question_lines = question.strip().split('\n')
    main_question = question_lines[0].strip()
    options = [line.strip() for line in question_lines[1:] 
               if line.strip() and (re.match(r'^\d{4}$', line.strip()) or len(line.strip()) < 50)]
    
    lines.append(f"QUESTION: {main_question}\n")
    
    # Find best answer
    best_answer = _find_best_option(main_question, options, content)
    
    if best_answer['option']:
        lines.append(f"ANSWER: {best_answer['option']}\n")
        lines.append("=" * 70)
        lines.append(f"\nEXPLANATION:\n{best_answer['explanation']}")
        if best_answer['evidence']:
            lines.append(f"\nEVIDENCE:\n\"{best_answer['evidence']}\"")
    else:
        lines.append("ANSWER: Cannot be determined from available sources\n")
        lines.append("=" * 70)
    
    lines.append("\n" + "=" * 70)
    lines.append("For UPSC: Verify with standard textbooks")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def _find_best_option(question: str, options: list, content: str):
    """Find the best option using contextual scoring."""
    # Extract question concepts (not just keywords)
    question_concepts = set(re.findall(r'\b\w{5,}\b', question.lower())) - {
        'which', 'what', 'when', 'where', 'based', 'india'
    }
    
    # Score each option
    best_option, best_score, best_evidence = None, 0, ""
    
    for option in options:
        # Find paragraphs containing this option
        paragraphs = [p.strip() for p in re.split(r'[.!?]{2,}|\n\n', content) if option.lower() in p.lower()]
        
        for para in paragraphs:
            para_lower = para.lower()
            
            # Score based on: concept overlap + factual context + proximity
            score = 0
            
            # 1. Concept overlap (theme matching)
            concept_overlap = len([c for c in question_concepts if c in para_lower])
            score += concept_overlap * 2
            
            # 2. Factual indicators (dates, reports, actions)
            if re.search(r'\b(report|act|commission|passed|enacted|formed|established)\b', para_lower):
                score += 1
            
            # 3. Year context (for date questions)
            if re.match(r'^\d{4}$', option):
                idx = para_lower.find(option.lower())
                if idx != -1:
                    context = para_lower[max(0, idx-40):idx+40]
                    if re.search(r'\b(in|during|year|on|since)\b', context):
                        score += 1.5
            
            if score > best_score:
                best_score = score
                best_option = option
                best_evidence = para[:300] + ('...' if len(para) > 300 else '')
    
    explanation = _generate_smart_explanation(best_option, question) if best_option else ""
    
    return {'option': best_option, 'explanation': explanation, 'evidence': best_evidence}


def _generate_smart_explanation(option: str, question: str):
    """Generate contextual explanation."""
    if 'reorgani' in question.lower() and 'linguistic' in question.lower():
        return f"The linguistic reorganisation of states in India was implemented in {option}."
    elif 'year' in question.lower() or 'when' in question.lower():
        return f"The event occurred in {option} based on historical records."
    else:
        return f"{option} is the correct answer according to the source material."
    
def generate_structured_response(question: str, contexts: dict, best_book: str, comparative_analysis: str = None):
    """
    Generate a structured response separating sources and answer.
    Returns a dictionary with separate fields for frontend consumption.
    """
    if not best_book:
        return {
            "success": False,
            "error": "Insufficient information - topic not covered in available books",
            "suggestion": "Check if your question relates to Ancient/Medieval Indian history or Modern India (post-1947)"
        }
    
    if best_book not in contexts:
        return {
            "success": False,
            "error": "No suitable book selected"
        }
    
    chunks = contexts.get(best_book, [])
    
    if not chunks:
        return {
            "success": False,
            "error": f"No relevant information found in '{best_book}'"
        }
    
    # Detect if it's an MCQ
    is_mcq = _is_mcq_question(question)
    
    # Format sources separately
    sources_list = []
    for i, doc in enumerate(chunks[:10], 1):
        chapter = doc.metadata.get("chapter", "Unknown Chapter")
        book = doc.metadata.get("book", "Unknown Book")
        similarity = doc.metadata.get("similarity", 0)
        
        snippet = doc.page_content[:800].strip().replace("\n", " ")
        if len(doc.page_content) > 800:
            snippet += "..."
        
        sources_list.append({
            "number": i,
            "book": book,
            "chapter": chapter,
            "relevance": similarity,
            "text": snippet
        })
    
    # Generate clean answer (without sources embedded)
    if is_mcq:
        clean_answer = _solve_mcq(question, chunks)
    else:
        clean_answer = _synthesize_upsc_answer(question, chunks)
    
    return {
        "success": True,
        "question": question,
        "best_book": best_book,
        "sources": sources_list,
        "answer": clean_answer,
        "analysis": comparative_analysis
    }
    