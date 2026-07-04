"""
rag_engine.py — Core RAG pipeline: retrieval + LLM generation.
Uses Groq API (llama-3.3-70b-versatile) for fast, high-quality responses.
"""

import os
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from groq import Groq
from langchain.schema import Document
from vector_store import VectorStore

logger = logging.getLogger(__name__)

# ─── RAG Settings ─────────────────────────────────────────────────────────────
TOP_K_CHUNKS = 5
MAX_CONTEXT_LENGTH = 4000   # chars of retrieved context to include
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions strictly based on the provided document context.

Guidelines:
- Answer ONLY using information from the provided context.
- If the context doesn't contain enough information, say so clearly.
- Be concise but thorough. Use bullet points or numbered lists when appropriate.
- Always cite your sources by mentioning the document name and page number when available.
- Never make up facts. If uncertain, say you're uncertain.
- Format code or technical content using markdown code blocks.
"""


class RAGEngine:
    """
    Orchestrates the full RAG pipeline:
    1. Retrieve relevant chunks from VectorStore
    2. Build a grounded prompt
    3. Generate a response via Groq LLM
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("GROQ_API_KEY not set — LLM calls will fail.")
        self.client = Groq(api_key=api_key)

    # ── Public Methods ─────────────────────────────────────────────────────────

    def chat(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline (non-streaming).
        Returns answer text + retrieved source documents.
        """
        # 1. Retrieve
        retrieved = self.vector_store.similarity_search_with_score(query, k=TOP_K_CHUNKS)

        # 2. Build context
        context, sources = self._build_context(retrieved)

        # 3. Build messages
        messages = self._build_messages(query, context, conversation_history)

        # 4. Generate
        if not os.getenv("GROQ_API_KEY"):
            return {
                "answer": (
                    "⚠️ **No GROQ_API_KEY set.**\n\n"
                    "Please create a `.env` file in the `backend/` directory with:\n"
                    "```\nGROQ_API_KEY=your_key_here\n```\n"
                    "Get a free key at https://console.groq.com"
                ),
                "sources": sources,
                "model": LLM_MODEL,
            }

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"❌ LLM error: {str(e)}"

        return {
            "answer": answer,
            "sources": sources,
            "model": LLM_MODEL,
        }

    async def chat_stream(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming RAG pipeline — yields text chunks as they arrive.
        Also yields a special [SOURCES] marker at the end.
        """
        # 1. Retrieve
        retrieved = self.vector_store.similarity_search_with_score(query, k=TOP_K_CHUNKS)

        # 2. Build context
        context, sources = self._build_context(retrieved)

        # 3. Build messages
        messages = self._build_messages(query, context, conversation_history)

        # 4. Check API key
        if not os.getenv("GROQ_API_KEY"):
            yield (
                "⚠️ **No GROQ_API_KEY set.**\n\n"
                "Please create a `.env` file in the `backend/` directory with:\n"
                "```\nGROQ_API_KEY=your_key_here\n```\n"
                "Get a free key at https://console.groq.com"
            )
            return

        try:
            stream = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"Streaming LLM failed: {e}")
            yield f"\n\n❌ LLM error: {str(e)}"

    def get_relevant_chunks(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks without generation — useful for debugging."""
        retrieved = self.vector_store.similarity_search_with_score(query, k=TOP_K_CHUNKS)
        _, sources = self._build_context(retrieved)
        return sources

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _build_context(
        self, retrieved: List[tuple[Document, float]]
    ) -> tuple[str, List[Dict]]:
        """Build a context string and sources list from retrieved chunks."""
        context_parts = []
        sources = []
        seen_content = set()

        for doc, score in retrieved:
            # Deduplicate near-identical chunks
            content_key = doc.page_content[:100]
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            meta = doc.metadata
            source_label = (
                f"[{meta.get('filename', 'Unknown')} — Page {meta.get('page', '?')}]"
            )
            context_parts.append(f"{source_label}\n{doc.page_content}")
            sources.append(
                {
                    "filename": meta.get("filename", "Unknown"),
                    "page": meta.get("page", 1),
                    "doc_id": meta.get("doc_id", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "relevance_score": round(float(score), 3),
                    "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
            )

        context = "\n\n---\n\n".join(context_parts)
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[:MAX_CONTEXT_LENGTH] + "\n\n[Context truncated]"

        return context, sources

    def _build_messages(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        """Construct the message list for the LLM."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history (last 6 turns to stay within token budget)
        if history:
            messages.extend(history[-6:])

        # Add the grounded user query
        user_content = (
            f"Context from documents:\n\n{context}\n\n"
            f"---\n\nQuestion: {query}"
            if context.strip()
            else f"Question: {query}\n\n(No relevant documents found in the knowledge base.)"
        )
        messages.append({"role": "user", "content": user_content})
        return messages
