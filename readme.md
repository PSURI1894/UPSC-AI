# UPSC History QA System - RAG-based Question Answering

A Retrieval-Augmented Generation (RAG) system designed specifically for UPSC History preparation. This system intelligently answers both descriptive and MCQ questions from two comprehensive history books using semantic search and context-aware processing.

## 📚 Features

### Question Types Supported
1. **Descriptive Questions** - Long-form UPSC mains style answers
2. **MCQ Questions**
   - Direct answer MCQs (choose one option)
   - List-based MCQs (select correct items from a numbered list)
   - Statement verification MCQs (verify which statements are correct)
   - Factual questions (specific dates, names, events)

### Key Capabilities
- ✅ Intelligent semantic search (not keyword-based)
- ✅ Source-grounded answers (no hallucination)
- ✅ Comparative analysis across multiple books
- ✅ UPSC-style structured answers
- ✅ Evidence-based explanations with source citations
- ✅ Automatic question type detection

## 📖 Supported Books

1. **Ancient & Early Medieval India** - Upinder Singh
   - Coverage: Prehistoric to Medieval period
   
2. **India After Gandhi** - Ramachandra Guha
   - Coverage: Post-independence India (1947 onwards)

## 🏗️ Architecture
```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retriever     │ ← Semantic search using FAISS
│  (retriever.py) │ ← Similarity threshold: 0.48
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Comparator    │ ← Analyzes all books
│ (comparator.py) │ ← Selects best source
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Answer Generator │ ← Question type detection
│(answer_gen.py)  │ ← Context-aware answering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Answer   │ ← UPSC-style structured
│  + Evidence     │ ← Source citations
└─────────────────┘
```

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Required Dependencies
```txt
fastapi
uvicorn
langchain
langchain-community
faiss-cpu
sentence-transformers
pydantic
```

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd upsc-history-qa

# Install dependencies
pip install -r requirements.txt

# Ensure vector stores exist
# Place your FAISS vector stores in:
# - vector_store/upinder/
# - vector_store/gandhi/

# Run the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📁 Project Structure
```
upsc-history-qa/
│
├── main.py                 # FastAPI application
├── rag/
│   ├── retriever.py       # Document retrieval with FAISS
│   ├── comparator.py      # Book comparison & selection
│   └── answer_generator.py # Answer generation logic
│
├── vector_store/
│   ├── upinder/           # FAISS index for Ancient India
│   └── gandhi/            # FAISS index for Modern India
│
├── static/
│   └── index.html         # Frontend interface
│
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

### Relevance Threshold
```python
# In retriever.py
RELEVANCE_THRESHOLD = 0.48  # Adjust for precision/recall balance
```

### Embedding Model
```python
# In retriever.py
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Top-K Documents
```python
# In retriever.py
def retrieve(query: str, top_k=10):  # Adjust number of documents
```

## 📝 API Endpoints

### `POST /ask`
Submit a question and get an answer.

**Request:**
```json
{
  "question": "How did the Harappan civilization decline?"
}
```

**Response:**
```json
{
  "success": true,
  "question": "How did the Harappan civilization decline?",
  "best_book": "Ancient & Early Medieval India",
  "analysis": {
    "structured": {
      "Ancient & Early Medieval India": "✅ Rank #1...",
      "India After Gandhi": "📖 Rank #2..."
    }
  },
  "answer": "Full UPSC-style answer with sources...",
  "books_analyzed": ["Ancient & Early Medieval India", "India After Gandhi"]
}
```

### `GET /health`
Check system health and loaded books.

**Response:**
```json
{
  "status": "healthy",
  "books_loaded": ["Ancient & Early Medieval India", "India After Gandhi"],
  "total_books": 2
}
```

### `GET /books`
Get list of available books.

**Response:**
```json
{
  "books": [
    {
      "name": "Ancient & Early Medieval India",
      "documents": 1500
    },
    {
      "name": "India After Gandhi",
      "documents": 1200
    }
  ],
  "total": 2
}
```

## 🎯 Key Features Explained

### 1. Intelligent Semantic Search
- Uses sentence transformers for embedding
- Similarity threshold filtering (0.48)
- Context-aware retrieval (not keyword matching)

### 2. Comparative Analysis
- Analyzes all available books
- Selects best source based on relevance scores
- Shows transparency in book selection

### 3. Question Type Detection
- Factual questions (dates, names)
- Direct MCQs (choose one option)
- List-based MCQs (select from numbered items)
- Statement verification MCQs
- Descriptive questions

### 4. Answer Quality Features
- Source citations for every claim
- Evidence extraction from documents
- UPSC-style structured formatting
- No hallucinations (source-grounded)

## 🛠️ How It Works

### Step 1: Document Retrieval
```python
# Retrieves top-k most relevant documents from FAISS
# Converts distance scores to similarity scores
# Filters by relevance threshold
```

### Step 2: Book Selection
```python
# Compares average similarity scores across books
# Selects book with highest relevance
# Generates comparative analysis
```

### Step 3: Answer Generation
```python
# Detects question type automatically
# Extracts relevant sentences from documents
# Structures answer in UPSC format
# Adds evidence and citations
```

## ⚙️ Customization

### Adjust Answer Length
```python
# In answer_generator.py
def _find_relevant_sentences(content, keywords, max_sentences=10):
    # Increase max_sentences for longer answers
```

### Change Scoring Logic
```python
# In answer_generator.py
def _find_best_option(question, options, content):
    # Modify scoring weights:
    score += concept_overlap * 2  # Theme matching
    score += factual_context * 1   # Factual indicators
```

### Modify Relevance Threshold
```python
# In retriever.py
RELEVANCE_THRESHOLD = 0.48  # Lower = more results, Higher = more precise
```

## 🐛 Troubleshooting

### No answers found
- Check if vector stores are properly loaded
- Lower `RELEVANCE_THRESHOLD` in `retriever.py`
- Increase `top_k` in retrieval

### Wrong answers
- Increase number of retrieved documents
- Adjust scoring weights in answer generation
- Check if question is within book coverage

### Slow performance
- Reduce `top_k` in retrieval
- Use smaller embedding model
- Optimize FAISS index

## 📊 Performance Metrics

- **Retrieval Time:** ~200-500ms
- **Answer Generation:** ~1-2s
- **Total Response Time:** ~2-3s
- **Accuracy:** Depends on source coverage

## 🔒 Limitations

1. **Knowledge Cutoff:** Limited to content in the two books
2. **No External Knowledge:** Cannot answer questions outside book scope
3. **Fact Verification:** Cannot fact-check across multiple sources
4. **Complex Reasoning:** Limited multi-hop reasoning capability

## 🚦 Future Enhancements

- [ ] Add more UPSC history books
- [ ] Implement multi-hop reasoning
- [ ] Add answer caching for common questions
- [ ] Support for image-based questions
- [ ] Integration with more embedding models
- [ ] Answer quality scoring
- [ ] User feedback mechanism

## 📄 License

This project is for educational purposes only.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

## 🙏 Acknowledgments

- **Books:** Upinder Singh, Ramachandra Guha
- **Frameworks:** LangChain, FastAPI, FAISS
- **Models:** Sentence Transformers

---

**Built for UPSC aspirants** | **Source-grounded answers** | **No hallucinations**