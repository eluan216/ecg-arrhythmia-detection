"""Test suite for FastAPI application."""

import pytest
from fastapi.testclient import TestClient

try:
    from api import app

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


@pytest.mark.skipif(not API_AVAILABLE, reason="API module not available")
class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Fixture for FastAPI test client."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "disclaimer" in data

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_predict_valid_input(self, client):
        """Test prediction endpoint with valid input."""
        payload = {
            "beat_values": [0.1] * 256,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "predicted_class" in data
        assert data["predicted_class"] in ["N", "V", "S", "F", "Q"]
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert "class_probabilities" in data

        # Check all classes are in probabilities
        probs = data["class_probabilities"]
        for cls in ["N", "V", "S", "F", "Q"]:
            assert cls in probs
            assert 0.0 <= probs[cls] <= 1.0

    def test_predict_too_short(self, client):
        """Test prediction with too short input."""
        payload = {
            "beat_values": [0.1] * 50,  # Too short
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 400

    def test_predict_too_long(self, client):
        """Test prediction with too long input."""
        payload = {
            "beat_values": [0.1] * 2000,  # Too long
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 400

    def test_predict_different_window_sizes(self, client):
        """Test prediction with various valid window sizes."""
        for size in [100, 200, 256, 300, 360, 500, 1000]:
            payload = {"beat_values": [0.1] * size}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["predicted_class"] in ["N", "V", "S", "F", "Q"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
