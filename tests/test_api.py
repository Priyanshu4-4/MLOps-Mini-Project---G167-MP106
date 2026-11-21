
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

valid_data = {
    "cows": 100,
    "buffaloes": 20,
    "fodder_price_rs": 25.5,
    "temp_c": 32.0,
    "vet_visits": 5,
    "feed_kg": 150,
    "breed_mix": "crossbred",
    "village_cluster": "Cluster A"
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_prediction():
    response = client.post("/predict", json=valid_data)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_cows_limit():
    data = valid_data.copy()
    data["cows"] = 351

    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_buffaloes_limit():
    data = valid_data.copy()
    data["buffaloes"] = 2

    response = client.post("/predict", json=data)
    assert response.status_code == 422
