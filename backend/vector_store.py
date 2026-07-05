"""
vector_store.py — ChromaDB vector store wrapper for the RAG chatbot.
Manages document collections, embeddings, and similarity search.
"""

import os
import threading
import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain_community.embeddings import FastEmbedEmbeddings

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_EMBEDDING_LOCK = threading.Lock()
_EMBEDDING_MODEL_INSTANCE: Optional[FastEmbedEmbeddings] = None


def _get_shared_embedder() -> FastEmbedEmbeddings:
    global _EMBEDDING_MODEL_INSTANCE
    if _EMBEDDING_MODEL_INSTANCE is None:
        with _EMBEDDING_LOCK:
            if _EMBEDDING_MODEL_INSTANCE is None:
                logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
                _EMBEDDING_MODEL_INSTANCE = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    return _EMBEDDING_MODEL_INSTANCE


class VectorStore:
    """
    Wrapper around ChromaDB + HuggingFace Embeddings.
    Handles document storage and semantic retrieval.
    """

    def __init__(self):
        self.embeddings = None

        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorStore initialized successfully.")

    def _ensure_embeddings(self) -> FastEmbedEmbeddings:
        if self.embeddings is None:
            self.embeddings = _get_shared_embedder()
        return self.embeddings

    # ── Public Methods ─────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add chunked documents to the vector store. Returns list of IDs."""
        if not documents:
            return []

        ids = [str(doc.metadata.get("doc_id", "")) + f"_{i}" for i, doc in enumerate(documents)]
        self.collection.add(
            ids=ids,
            documents=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )
        logger.info(f"Added {len(documents)} chunks to ChromaDB.")
        return ids

    def similarity_search(
        self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """Return top-k most semantically relevant document chunks."""
        try:
            embeddings = self._ensure_embeddings()
            query_embedding = embeddings.embed_query(query)
            result = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"],
            )
            docs = []
            for doc_text, metadata in zip(result["documents"][0], result["metadatas"][0]):
                docs.append(Document(page_content=doc_text, metadata=metadata))
            return docs
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[tuple[Document, float]]:
        """Return top-k chunks along with their relevance scores."""
        try:
            embeddings = self._ensure_embeddings()
            query_embedding = embeddings.embed_query(query)
            result = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            docs_with_scores = []
            for doc_text, metadata, distance in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            ):
                docs_with_scores.append((Document(page_content=doc_text, metadata=metadata), float(distance)))
            return docs_with_scores
        except Exception as e:
            logger.error(f"Scored similarity search failed: {e}")
            return []

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return a deduplicated list of ingested document metadata."""
        try:
            data = self.collection.get(include=["documents", "metadatas"])
            seen = {}
            for meta in data["metadatas"]:
                doc_id = meta.get("doc_id", "unknown")
                if doc_id not in seen:
                    seen[doc_id] = {
                        "doc_id": doc_id,
                        "filename": meta.get("filename", "unknown"),
                        "file_type": meta.get("file_type", "unknown"),
                        "total_chunks": 0,
                    }
                seen[doc_id]["total_chunks"] += 1
            return list(seen.values())
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks belonging to a specific document."""
        try:
            data = self.collection.get(include=["ids", "documents", "metadatas"])
            ids_to_delete = [
                data["ids"][i]
                for i, meta in enumerate(data["metadatas"])
                if meta.get("doc_id") == doc_id
            ]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id={doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def clear_all(self) -> bool:
        """Delete the entire collection and recreate it empty."""
        try:
            self.chroma_client.delete_collection(COLLECTION_NAME)
            self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)
            logger.info("Cleared all documents from ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def get_document_count(self) -> int:
        """Return total number of chunks stored."""
        try:
            return self.collection.count()
        except Exception:
            return 0