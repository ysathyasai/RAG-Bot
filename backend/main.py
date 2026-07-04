"""
main.py — FastAPI application for the RAG chatbot.
Provides endpoints for document upload, chat, and management.
"""

import os
import logging
import tempfile
import json
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from vector_store import VectorStore
from document_processor import DocumentProcessor
from rag_engine import RAGEngine

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Global State ─────────────────────────────────────────────────────────────
vector_store: Optional[VectorStore] = None
doc_processor: Optional[DocumentProcessor] = None
rag_engine: Optional[RAGEngine] = None

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".md"}
MAX_FILE_SIZE_MB = 20


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, doc_processor, rag_engine
    logger.info("Starting RAG Chatbot backend…")
    vector_store = VectorStore()
    doc_processor = DocumentProcessor()
    rag_engine = RAGEngine(vector_store)
    logger.info("All components initialized. Ready to serve.")
    yield
    logger.info("Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot — upload docs, ask questions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    stream: bool = False


class TextIngestRequest(BaseModel):
    text: str
    source_name: str = "Pasted Text"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    chunk_count = vector_store.get_document_count() if vector_store else 0
    groq_key_set = bool(os.getenv("GROQ_API_KEY"))
    return {
        "status": "ok",
        "chunk_count": chunk_count,
        "groq_api_key_set": groq_key_set,
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, TXT, MD) and ingest it into ChromaDB.
    Returns the generated doc_id and chunk count.
    """
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file content and check size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB.",
        )

    # Save to temp file and process
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc_id, chunks = doc_processor.process_file(tmp_path, file.filename)
        if not chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted from the file.")
        vector_store.add_documents(chunks)
        logger.info(f"Ingested '{file.filename}': {len(chunks)} chunks, doc_id={doc_id}")
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_count": len(chunks),
            "file_size_mb": round(size_mb, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/ingest-text")
async def ingest_text(req: TextIngestRequest):
    """Ingest raw pasted text content into the knowledge base."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
    try:
        doc_id, chunks = doc_processor.process_text(req.text, req.source_name)
        vector_store.add_documents(chunks)
        return {
            "success": True,
            "doc_id": doc_id,
            "source_name": req.source_name,
            "chunk_count": len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Send a message and receive a RAG-powered answer.
    Set stream=true for streaming response (SSE).
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if req.stream:
        async def event_stream():
            async for chunk in rag_engine.chat_stream(req.message, req.history):
                data = json.dumps({"type": "content", "data": chunk})
                yield f"data: {data}\n\n"
            # Send sources after streaming
            sources = rag_engine.get_relevant_chunks(req.message)
            data = json.dumps({"type": "sources", "data": sources})
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    result = rag_engine.chat(req.message, req.history)
    return result


@app.get("/documents")
async def list_documents():
    """List all ingested documents with metadata."""
    docs = vector_store.list_documents()
    return {"documents": docs, "total": len(docs)}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a specific document and all its chunks from the vector store."""
    success = vector_store.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete document.")
    return {"success": True, "doc_id": doc_id}


@app.delete("/clear")
async def clear_all_documents():
    """Delete ALL documents from the knowledge base."""
    success = vector_store.clear_all()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear knowledge base.")
    return {"success": True, "message": "Knowledge base cleared."}


@app.post("/retrieve")
async def retrieve_chunks(req: ChatRequest):
    """Debug endpoint: return the raw retrieved chunks for a query without LLM generation."""
    sources = rag_engine.get_relevant_chunks(req.message)
    return {"query": req.message, "chunks": sources}


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend"), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
