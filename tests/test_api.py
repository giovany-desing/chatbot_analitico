"""
Tests para la API FastAPI.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestAPI:

    def test_root(self):
        """Test endpoint raíz"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health(self):
        """Test health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] in ['healthy', 'degraded']
        assert 'databases' in data

    def test_chat_sql_query(self):
        """Test chat con SQL query"""
        response = client.post(
            "/chat",
            json={"message": "¿Cuántas ventas hay?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['intent'] == 'sql'
        assert data['sql_query'] is not None
        assert data['results'] is not None

    def test_chat_general_query(self):
        """Test chat con query general"""
        response = client.post(
            "/chat",
            json={"message": "Hola"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['intent'] == 'general'
        assert data['response'] is not None

    def test_chat_invalid_request(self):
        """Test request inválida"""
        response = client.post(
            "/chat",
            json={"message": ""}
        )
        assert response.status_code == 422  # Validation error