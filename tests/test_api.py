from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "ByteTrack multi-object tracking" in payload["capabilities"]


def test_frontend_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "TrafficVision AI" in response.text
