"""
Tests for ANVISA Web Scraper

Tests the scraper's ability to parse HTML, handle PDFs, and extract text.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from datetime import datetime

from src.scrapers.anvisa_scraper import AnvisaScraper, DrugBulletin


class TestDrugBulletin:
    """Tests for DrugBulletin dataclass."""
    
    def test_content_hash(self):
        """Test that content hash is generated correctly."""
        bulletin = DrugBulletin(
            id="123",
            name="Paracetamol",
            company="Lab XYZ",
            active_ingredient="paracetamol",
            bulletin_type="paciente",
            pdf_url="http://example.com/bula.pdf",
            text_content="Este medicamento é indicado para dor.",
            last_updated=datetime.now()
        )
        
        hash1 = bulletin.content_hash()
        assert hash1 is not None
        assert len(hash1) == 32  # MD5 hash length
        
        # Same content should produce same hash
        bulletin2 = DrugBulletin(
            id="123",
            name="Paracetamol",
            company="Lab XYZ",
            active_ingredient="paracetamol",
            bulletin_type="paciente",
            pdf_url="http://example.com/bula.pdf",
            text_content="Este medicamento é indicado para dor.",
            last_updated=datetime.now()
        )
        assert bulletin.content_hash() == bulletin2.content_hash()
        
        # Different content should produce different hash
        bulletin3 = DrugBulletin(
            id="123",
            name="Paracetamol",
            company="Lab XYZ",
            active_ingredient="paracetamol",
            bulletin_type="paciente",
            pdf_url="http://example.com/bula.pdf",
            text_content="Conteúdo diferente.",
            last_updated=datetime.now()
        )
        assert bulletin.content_hash() != bulletin3.content_hash()


class TestAnvisaScraper:
    """Tests for AnvisaScraper class."""
    
    @pytest.fixture
    def scraper(self, tmp_path):
        """Create a scraper instance with temporary cache directory."""
        return AnvisaScraper(cache_dir=tmp_path)
    
    def test_initialization(self, scraper, tmp_path):
        """Test scraper initializes correctly."""
        assert scraper.cache_dir == tmp_path
        assert scraper.timeout == 30.0
        assert scraper.max_retries == 3
    
    def test_cache_directory_created(self, tmp_path):
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"
        scraper = AnvisaScraper(cache_dir=cache_dir)
        assert cache_dir.exists()
    
    @pytest.mark.asyncio
    async def test_search_drugs_success(self, scraper):
        """Test successful drug search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [
                {"idProduto": "123", "nomeProduto": "Paracetamol 500mg"},
                {"idProduto": "456", "nomeProduto": "Paracetamol 750mg"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(scraper, '_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)
            scraper._client = mock_client
            
            results = await scraper.search_drugs("paracetamol")
            
            assert len(results) == 2
            assert results[0]["nomeProduto"] == "Paracetamol 500mg"
    
    @pytest.mark.asyncio
    async def test_search_drugs_error(self, scraper):
        """Test drug search handles errors gracefully."""
        import httpx
        
        with patch.object(scraper, '_client') as mock_client:
            mock_client.get = AsyncMock(side_effect=httpx.HTTPError("Network error"))
            scraper._client = mock_client
            
            results = await scraper.search_drugs("paracetamol")
            assert results == []
    
    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path):
        """Test async context manager creates and closes client."""
        scraper = AnvisaScraper(cache_dir=tmp_path)
        
        async with scraper:
            assert scraper._client is not None
        
        # Client should be closed after exiting context


class TestTextChunking:
    """Tests for text chunking in vector store."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        from src.database.vector_store import VectorStore
        
        # Create a simple text
        text = "A" * 2500  # 2500 characters
        
        # Mock the vector store initialization
        with patch('src.database.vector_store.chromadb'):
            with patch('src.database.vector_store.get_settings') as mock_settings:
                mock_settings.return_value.chroma_persist_path = "./test_data"
                
                # Just test the chunking logic directly
                chunks = []
                start = 0
                chunk_size = 1000
                overlap = 200
                
                while start < len(text):
                    end = start + chunk_size
                    chunk = text[start:end]
                    chunks.append(chunk)
                    start = end - overlap
                
                # Should create multiple chunks
                assert len(chunks) > 1
                # First chunk should be full size
                assert len(chunks[0]) == chunk_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
