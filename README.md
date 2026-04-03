# 🧠 DocuMind AI

### Retrieval-Augmented PDF Intelligence System

DocuMind AI is a Retrieval-Augmented Generation (RAG) system that enables users to **query, understand, and interact with PDF documents using natural language**.

It combines **semantic search, vector indexing, and LLM-based reasoning** to deliver accurate, context-grounded responses — with optional text-to-speech output.

---

## 🧠 What This System Actually Does

Most “chat with PDF” tools:

* Dump entire documents into prompts
* Produce hallucinated answers
* Lack grounding and traceability

DocuMind AI solves this by implementing a **true RAG pipeline**:

* Retrieve only relevant content
* Constrain LLM to that context
* Return **grounded, explainable answers**

This is not just chat — it’s **controlled document intelligence**.

---

## ⚙️ Core Features

### 📄 Semantic Document Understanding

* Splits PDFs into structured chunks
* Converts text into embeddings
* Enables meaning-based search (not keyword matching)

---

### 🔍 Vector Search with FAISS

* Embeddings stored in FAISS index
* Fast similarity search for relevant chunks
* Retrieves **top-k context** per query

---

### 🤖 Context-Grounded LLM Responses

* Uses Groq (LLaMA 3.x) for generation
* Strict prompt design:

  * Only uses retrieved context
  * No hallucination fallback (“Not found in this page”)
* Supports:

  * Q&A
  * Summarization
  * Explanation

---

### 💬 Conversational Memory

* Maintains chat history
* Improves continuity in multi-turn interactions

---

### 🔊 Text-to-Speech (TTS)

* Converts responses into audio
* Uses Google TTS for real-time playback

---

### 📚 Source Attribution

* Returns page references for answers
* Improves trust and interpretability

---

## 🏗️ System Architecture

PDF Input
   ↓
Text Extraction
   ↓
Chunking (Fixed-size segments)
   ↓
Embedding (Sentence Transformers)
   ↓
FAISS Vector Index
   ↓
User Query
   ↓
Query Embedding
   ↓
Top-K Retrieval
   ↓
LLM (Context-Constrained Generation)
   ↓
Answer + Source Pages + Optional TTS
```

---

## 🛠️ Tech Stack

| Layer           | Technologies                               |
| --------------- | ------------------------------------------ |
| Backend         | Flask                                      |
| Embeddings      | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector DB       | FAISS                                      |
| LLM             | Groq API (LLaMA 3.x)                       |
| TTS             | gTTS                                       |
| Data Processing | NumPy                                      |

---

## 📂 Project Structure

documind/
├── app.py                # Core Flask app + RAG pipeline
├── templates/            # Frontend UI
├── static/
└── requirements.txt
```

---

## 🔬 Example Workflow

1. User uploads/processes a PDF
2. Document is chunked and embedded
3. FAISS builds vector index
4. User asks a question
5. System retrieves top relevant chunks
6. LLM generates answer using ONLY that context
7. Output includes:

   * Answer
   * Source pages
   * Optional audio

---

## ⚠️ Limitations (Real Talk)

* In-memory FAISS index (not persistent)
* Fixed chunk size (may miss optimal boundaries)
* No reranking layer (retrieval can be improved)
* Model download required on first run

---

## 🔮 Future Improvements

* Persistent vector DB (Chroma / Pinecone)
* Hybrid retrieval (BM25 + embeddings)
* Reranking models (cross-encoder)
* Streaming responses
* Multi-document support
* Scalable deployment

---

## 🚀 Why This Project Matters

This project demonstrates:

* End-to-end **RAG system design**
* Semantic search implementation
* LLM grounding techniques
* Prompt engineering for hallucination control
* Real-world AI system integration

This is the difference between:

> “I used an API”
> and
> “I built an AI system that retrieves, reasons, and responds”

---

## 👨‍💻 Author

**Gautam Jangir**
