"""Integration tests for src/api.py"""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


SAMPLE_PATIENT = {
    "gender": "Female",
    "family_history_with_overweight": "yes",
    "favc": "yes",
    "caec": "Sometimes",
    "smoke": "no",
    "scc": "no",
    "calc": "Sometimes",
    "mtrans": "Public_Transportation",
    "age": 25,
    "height": 1.65,
    "weight": 95,
    "fcvc": 2.0,
    "ncp": 3.0,
    "ch2o": 2.0,
    "faf": 0.5,
    "tue": 1.0,
}


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_200(client):
    response = client.post("/predict", json=SAMPLE_PATIENT)
    assert response.status_code == 200


def test_predict_response_schema(client):
    response = client.post("/predict", json=SAMPLE_PATIENT)
    data = response.json()
    assert "is_obese" in data
    assert "probability" in data
    assert isinstance(data["is_obese"], bool)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_invalid_input_returns_422(client):
    response = client.post("/predict", json={"gender": "Female"})
    assert response.status_code == 422
