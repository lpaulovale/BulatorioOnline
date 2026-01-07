"""
Tests for Vector Store

Tests ChromaDB integration and semantic search functionality.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestVectorStore:
    """Tests for VectorStore class."""
    
    @pytest.fixture
    def mock_chromadb(self):
        """Mock ChromaDB client and collection."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {
            "documents": [["Sample drug content"]],
            "metadatas": [[{"drug_id": "123", "drug_name": "Paracetamol"}]],
            "distances": [[0.1]]
        }
        
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        
        return mock_client, mock_collection
    
    def test_add_document(self, mock_chromadb):
        """Test adding a document to the vector store."""
        mock_client, mock_collection = mock_chromadb
        
        with patch('src.database.vector_store.chromadb.PersistentClient', return_value=mock_client):
            with patch('src.database.vector_store.get_settings') as mock_settings:
                mock_settings.return_value.chroma_persist_path = "./test_data"
                with patch('pathlib.Path.mkdir'):
                    from src.database.vector_store import VectorStore
                    
                    store = VectorStore(persist_path="./test_data")
                    
                    text = "Este medicamento é indicado para dor de cabeça."
                    count = store.add_document(
                        drug_id="123",
                        drug_name="Paracetamol",
                        text_content=text
                    )
                    
                    # Should have called upsert
                    mock_collection.upsert.assert_called_once()
                    assert count > 0
    
    def test_add_empty_document(self, mock_chromadb):
        """Test that empty documents are skipped."""
        mock_client, mock_collection = mock_chromadb
        
        with patch('src.database.vector_store.chromadb.PersistentClient', return_value=mock_client):
            with patch('src.database.vector_store.get_settings') as mock_settings:
                mock_settings.return_value.chroma_persist_path = "./test_data"
                with patch('pathlib.Path.mkdir'):
                    from src.database.vector_store import VectorStore
                    
                    store = VectorStore(persist_path="./test_data")
                    
                    count = store.add_document(
                        drug_id="123",
                        drug_name="Paracetamol",
                        text_content=""
                    )
                    
                    assert count == 0
                    mock_collection.upsert.assert_not_called()
    
    def test_search(self, mock_chromadb):
        """Test semantic search."""
        mock_client, mock_collection = mock_chromadb
        
        with patch('src.database.vector_store.chromadb.PersistentClient', return_value=mock_client):
            with patch('src.database.vector_store.get_settings') as mock_settings:
                mock_settings.return_value.chroma_persist_path = "./test_data"
                with patch('pathlib.Path.mkdir'):
                    from src.database.vector_store import VectorStore
                    
                    store = VectorStore(persist_path="./test_data")
                    
                    results = store.search("dor de cabeça", n_results=5)
                    
                    mock_collection.query.assert_called_once()
                    assert len(results) == 1
                    assert results[0]["metadata"]["drug_name"] == "Paracetamol"
    
    def test_search_with_filter(self, mock_chromadb):
        """Test search with drug filter."""
        mock_client, mock_collection = mock_chromadb
        
        with patch('src.database.vector_store.chromadb.PersistentClient', return_value=mock_client):
            with patch('src.database.vector_store.get_settings') as mock_settings:
                mock_settings.return_value.chroma_persist_path = "./test_data"
                with patch('pathlib.Path.mkdir'):
                    from src.database.vector_store import VectorStore
                    
                    store = VectorStore(persist_path="./test_data")
                    
                    results = store.search(
                        "efeitos colaterais",
                        drug_filter="Paracetamol"
                    )
                    
                    # Verify filter was passed
                    call_kwargs = mock_collection.query.call_args[1]
                    assert call_kwargs["where"]["drug_name"]["$eq"] == "Paracetamol"


class TestMetadataCache:
    """Tests for MetadataCache class."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        """Create a cache with temporary database."""
        with patch('src.database.metadata_cache.get_settings') as mock_settings:
            mock_settings.return_value.sqlite_database_path = str(tmp_path / "test.db")
            
            from src.database.metadata_cache import MetadataCache
            return MetadataCache(db_path=str(tmp_path / "test.db"))
    
    def test_save_and_get_drug(self, cache):
        """Test saving and retrieving a drug."""
        cache.save_drug(
            drug_id="123",
            drug_name="Paracetamol",
            company="Lab XYZ",
            active_ingredient="paracetamol",
            content_hash="abc123"
        )
        
        drug = cache.get_drug("123")
        
        assert drug is not None
        assert drug["drug_name"] == "Paracetamol"
        assert drug["company"] == "Lab XYZ"
    
    def test_needs_update_new_drug(self, cache):
        """Test that new drugs need update."""
        assert cache.needs_update("999", "somehash") is True
    
    def test_needs_update_same_hash(self, cache):
        """Test that same hash doesn't need update."""
        cache.save_drug(
            drug_id="123",
            drug_name="Paracetamol",
            content_hash="abc123"
        )
        
        assert cache.needs_update("123", "abc123") is False
    
    def test_needs_update_different_hash(self, cache):
        """Test that different hash needs update."""
        cache.save_drug(
            drug_id="123",
            drug_name="Paracetamol",
            content_hash="abc123"
        )
        
        assert cache.needs_update("123", "different_hash") is True
    
    def test_mark_indexed(self, cache):
        """Test marking a drug as indexed."""
        cache.save_drug(
            drug_id="123",
            drug_name="Paracetamol",
            is_indexed=False
        )
        
        cache.mark_indexed("123")
        drug = cache.get_drug("123")
        
        assert drug["is_indexed"] == 1  # SQLite stores bool as int
    
    def test_get_stats(self, cache):
        """Test getting cache statistics."""
        cache.save_drug(drug_id="1", drug_name="Drug1", is_indexed=True)
        cache.save_drug(drug_id="2", drug_name="Drug2", is_indexed=False)
        
        stats = cache.get_stats()
        
        assert stats["total_drugs"] == 2
        assert stats["indexed_drugs"] == 1
        assert stats["pending_indexing"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
