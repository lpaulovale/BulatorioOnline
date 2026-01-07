"""
ChromaDB Vector Store for Drug Bulletins

Provides semantic search capabilities for drug information using embeddings.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for drug bulletin embeddings.
    
    Stores chunked drug bulletin text with metadata for semantic search.
    """
    
    COLLECTION_NAME = "drug_bulletins"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            persist_path: Path to persist ChromaDB data
        """
        settings = get_settings()
        self.persist_path = persist_path or settings.chroma_persist_path
        
        # Ensure directory exists
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self._client = chromadb.PersistentClient(
            path=self.persist_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(
            f"Vector store initialized with {self._collection.count()} documents"
        )
    
    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period > self.CHUNK_SIZE // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - self.CHUNK_OVERLAP
        
        return [c for c in chunks if c]
    
    def add_document(
        self,
        drug_id: str,
        drug_name: str,
        text_content: str,
        metadata: Optional[dict] = None
    ) -> int:
        """
        Add a drug bulletin to the vector store.
        
        Args:
            drug_id: Unique drug identifier
            drug_name: Name of the drug
            text_content: Full text content of the bulletin
            metadata: Additional metadata to store
            
        Returns:
            Number of chunks added
        """
        if not text_content:
            logger.warning(f"Empty content for drug {drug_id}, skipping")
            return 0
        
        chunks = self._chunk_text(text_content)
        
        # Prepare data for ChromaDB
        ids = [f"{drug_id}_chunk_{i}" for i in range(len(chunks))]
        documents = chunks
        metadatas = [
            {
                "drug_id": drug_id,
                "drug_name": drug_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {})
            }
            for i in range(len(chunks))
        ]
        
        # Upsert to handle updates
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks for drug {drug_name}")
        return len(chunks)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        drug_filter: Optional[str] = None
    ) -> list[dict]:
        """
        Search for relevant drug information.
        
        Args:
            query: Search query
            n_results: Maximum number of results to return
            drug_filter: Optional drug name to filter by
            
        Returns:
            List of search results with content and metadata
        """
        where_filter = None
        if drug_filter:
            where_filter = {"drug_name": {"$eq": drug_filter}}
        
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })
        
        return formatted
    
    def delete_drug(self, drug_id: str) -> bool:
        """
        Delete all chunks for a specific drug.
        
        Args:
            drug_id: Drug ID to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            self._collection.delete(
                where={"drug_id": {"$eq": drug_id}}
            )
            logger.info(f"Deleted all chunks for drug {drug_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting drug {drug_id}: {e}")
            return False
    
    def get_all_drug_ids(self) -> list[str]:
        """Get list of all unique drug IDs in the store."""
        results = self._collection.get(include=["metadatas"])
        
        drug_ids = set()
        for metadata in results.get("metadatas", []):
            if metadata and "drug_id" in metadata:
                drug_ids.add(metadata["drug_id"])
        
        return list(drug_ids)
    
    def count(self) -> int:
        """Get total number of documents in the store."""
        return self._collection.count()
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Vector store cleared")


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
