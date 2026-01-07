"""
Tests for FastAPI API

Tests API endpoints using TestClient.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self):
        """Test health check returns correct status."""
        with patch('src.api.main.get_settings') as mock_settings:
            mock_settings.return_value.cors_origins = []
            mock_settings.return_value.enable_scheduler = False
            
            with patch('src.api.main.setup_scheduler'):
                with patch('src.api.main.get_scheduler') as mock_scheduler:
                    mock_scheduler.return_value.running = False
                    
                    from fastapi.testclient import TestClient
                    from src.api.main import app
                    
                    # Patch dependencies
                    with patch('src.api.main.get_metadata_cache') as mock_cache:
                        with patch('src.api.main.get_vector_store') as mock_store:
                            mock_cache.return_value.get_stats.return_value = {"total_drugs": 0}
                            mock_store.return_value.count.return_value = 0
                            
                            client = TestClient(app)
                            response = client.get("/health")
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert "status" in data


class TestChatEndpoint:
    """Tests for chat API endpoint."""
    
    @pytest.fixture
    def mock_gemini(self):
        """Mock Gemini client."""
        mock = MagicMock()
        mock.query = AsyncMock(return_value="O paracetamol é indicado para dor e febre.")
        return mock
    
    def test_chat_endpoint_success(self, mock_gemini):
        """Test successful chat response."""
        with patch('src.api.routes.chat.get_gemini_client', return_value=mock_gemini):
            with patch('src.api.main.get_settings') as mock_settings:
                mock_settings.return_value.cors_origins = []
                mock_settings.return_value.enable_scheduler = False
                
                with patch('src.api.main.setup_scheduler'):
                    from fastapi.testclient import TestClient
                    from src.api.main import app
                    
                    client = TestClient(app)
                    
                    response = client.post("/api/chat/", json={
                        "message": "Para que serve paracetamol?",
                        "mode": "patient"
                    })
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "response" in data
                    assert data["mode"] == "patient"
    
    def test_chat_endpoint_validation_error(self):
        """Test chat endpoint validates input."""
        with patch('src.api.main.get_settings') as mock_settings:
            mock_settings.return_value.cors_origins = []
            mock_settings.return_value.enable_scheduler = False
            
            with patch('src.api.main.setup_scheduler'):
                from fastapi.testclient import TestClient
                from src.api.main import app
                
                client = TestClient(app)
                
                # Empty message should fail
                response = client.post("/api/chat/", json={
                    "message": "",
                    "mode": "patient"
                })
                
                assert response.status_code == 422  # Validation error


class TestDrugsEndpoint:
    """Tests for drugs API endpoints."""
    
    def test_search_drugs(self):
        """Test drug search endpoint."""
        with patch('src.api.routes.drugs.get_vector_store') as mock_store:
            mock_store.return_value.search.return_value = [
                {
                    "content": "Paracetamol content",
                    "metadata": {
                        "drug_id": "123",
                        "drug_name": "Paracetamol",
                        "company": "Lab XYZ"
                    },
                    "distance": 0.1
                }
            ]
            
            with patch('src.api.main.get_settings') as mock_settings:
                mock_settings.return_value.cors_origins = []
                mock_settings.return_value.enable_scheduler = False
                
                with patch('src.api.main.setup_scheduler'):
                    from fastapi.testclient import TestClient
                    from src.api.main import app
                    
                    client = TestClient(app)
                    
                    response = client.get("/api/drugs/search?q=paracetamol")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["query"] == "paracetamol"
                    assert len(data["results"]) == 1
    
    def test_search_drugs_min_length(self):
        """Test search requires minimum query length."""
        with patch('src.api.main.get_settings') as mock_settings:
            mock_settings.return_value.cors_origins = []
            mock_settings.return_value.enable_scheduler = False
            
            with patch('src.api.main.setup_scheduler'):
                from fastapi.testclient import TestClient
                from src.api.main import app
                
                client = TestClient(app)
                
                response = client.get("/api/drugs/search?q=a")
                
                assert response.status_code == 422  # Query too short
    
    def test_get_drug_not_found(self):
        """Test getting non-existent drug."""
        with patch('src.api.routes.drugs.get_metadata_cache') as mock_cache:
            mock_cache.return_value.get_drug.return_value = None
            
            with patch('src.api.main.get_settings') as mock_settings:
                mock_settings.return_value.cors_origins = []
                mock_settings.return_value.enable_scheduler = False
                
                with patch('src.api.main.setup_scheduler'):
                    from fastapi.testclient import TestClient
                    from src.api.main import app
                    
                    client = TestClient(app)
                    
                    response = client.get("/api/drugs/999")
                    
                    assert response.status_code == 404
    
    def test_get_stats(self):
        """Test stats endpoint."""
        with patch('src.api.routes.drugs.get_metadata_cache') as mock_cache:
            with patch('src.api.routes.drugs.get_vector_store') as mock_store:
                mock_cache.return_value.get_stats.return_value = {
                    "total_drugs": 10,
                    "indexed_drugs": 8,
                    "pending_indexing": 2,
                    "last_scrape": None
                }
                mock_store.return_value.count.return_value = 50
                
                with patch('src.api.main.get_settings') as mock_settings:
                    mock_settings.return_value.cors_origins = []
                    mock_settings.return_value.enable_scheduler = False
                    
                    with patch('src.api.main.setup_scheduler'):
                        from fastapi.testclient import TestClient
                        from src.api.main import app
                        
                        client = TestClient(app)
                        
                        response = client.get("/api/drugs/stats/overview")
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert data["total_drugs"] == 10
                        assert data["vector_documents"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
