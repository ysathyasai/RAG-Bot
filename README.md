# 🤖 RAG Chatbot — AI Knowledge Assistant

A full-stack **Retrieval-Augmented Generation (RAG)** chatbot built from scratch. Upload your documents, ask questions, and get AI-powered answers grounded in your knowledge base — with source citations.

## ✨ Features

- 📄 **Document Ingestion** — Upload PDFs, DOCX, TXT, and Markdown files
- 📝 **Text Paste** — Paste raw text directly into the knowledge base
- 🔍 **Semantic Search** — ChromaDB + HuggingFace embeddings (`all-MiniLM-L6-v2`)
- 🤖 **LLM Generation** — Groq API (llama-3.3-70b-versatile) with streaming
- 📎 **Source Citations** — Every answer shows which documents and pages it used
- 💬 **Conversation History** — Multi-turn chat with context
- 🎨 **Premium UI** — Dark glassmorphism, animated particles, responsive

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python |
| Vector DB | ChromaDB (local, persistent) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Groq API (llama-3.3-70b) |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Frontend | HTML + CSS + Vanilla JS |

## 🚀 Quick Start

### 1. Get a Free Groq API Key
Sign up at [console.groq.com](https://console.groq.com) — it's free and takes ~1 minute.

### 2. Set Your API Key
```bash
cd backend
copy .env.example .env
# Edit .env and replace "your_groq_api_key_here" with your actual key
```

### 3. Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Start the Backend
```bash
cd backend
python main.py
# Or: uvicorn main:app --reload --port 8000
```

### 5. Open the Frontend
Open `frontend/index.html` in your browser. That's it!

## 📁 Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py              # FastAPI app & all endpoints
│   ├── rag_engine.py        # RAG pipeline + Groq LLM
│   ├── vector_store.py      # ChromaDB vector store wrapper
│   ├── document_processor.py # Document loading & chunking
│   ├── requirements.txt
│   ├── .env.example         # API key template
│   └── chroma_db/           # Auto-created: persisted vectors
└── frontend/
    ├── index.html           # Chat UI
    ├── style.css            # Dark-mode styles & animations
    └── app.js               # Chat logic & streaming
```

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check + stats |
| `POST` | `/upload` | Upload a document file |
| `POST` | `/ingest-text` | Ingest raw text |
| `POST` | `/chat` | Send a message (streaming or standard) |
| `GET` | `/documents` | List all documents |
| `DELETE` | `/documents/{doc_id}` | Delete a document |
| `DELETE` | `/clear` | Clear all documents |
| `POST` | `/retrieve` | Debug: get raw retrieved chunks |

## 💡 How RAG Works

```
1. UPLOAD:  Document → Extract text → Split into chunks → Embed → Store in ChromaDB
2. QUERY:   User question → Embed → Similarity search → Top-5 chunks
3. GENERATE: Chunks + Question → LLM prompt → Streamed answer with citations
```

## ⚙️ Configuration

Edit these constants in the source files to customize behavior:

| File | Constant | Default | Description |
|---|---|---|---|
| `document_processor.py` | `CHUNK_SIZE` | `600` | Characters per chunk |
| `document_processor.py` | `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `rag_engine.py` | `TOP_K_CHUNKS` | `5` | Retrieved chunks per query |
| `rag_engine.py` | `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `rag_engine.py` | `TEMPERATURE` | `0.2` | LLM temperature |
| `vector_store.py` | `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
