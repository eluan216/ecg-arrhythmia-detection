"""API contract tests using fitted models, without downloaded artifacts."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

import api
from src.data.segment import AAMI_CLASSES, FEATURE_DIM, FEATURE_VERSION


@pytest.fixture
def client(monkeypatch):
    rng = np.random.default_rng(7)
    model = RandomForestClassifier(n_estimators=3, random_state=7).fit(
        rng.normal(size=(30, FEATURE_DIM)), np.arange(30) % 5
    )
    monkeypatch.setattr(api, "_models_loaded", True)
    monkeypatch.setattr(api, "_baseline_model", model)
    monkeypatch.setattr(api, "_cnn_model", None)
    with TestClient(api.app) as test_client:
        yield test_client


def test_root_and_health(client):
    assert "disclaimer" in client.get("/").json()
    assert client.get("/health").json()["baseline_loaded"]
    assert client.get("/ready").status_code == 200


@pytest.mark.parametrize("size", [32, 100, 250, 256, 1000])
def test_predict_valid_input(client, size):
    response = client.post("/predict", json={"signal": [0.1] * size})
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in AAMI_CLASSES
    assert set(data["probabilities"]) == set(AAMI_CLASSES)
    assert sum(data["probabilities"].values()) == pytest.approx(1.0)
    assert data["model_used"] == "baseline"
    assert data["feature_version"] == FEATURE_VERSION


@pytest.mark.parametrize("size", [0, 31, 1001])
def test_invalid_length(client, size):
    response = client.post("/predict", json={"signal": [0.1] * size})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "payload",
    [{"beat_values": [0.1] * 256}, {"signal": [[0.1]]}, {"signal": [0.1], "model": "other"}],
)
def test_invalid_schema(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_unsupported_sampling_rate(client):
    response = client.post("/predict", json={"signal": [0.1] * 256, "sampling_rate": 250})
    assert response.status_code == 400
    assert "360" in response.json()["detail"]


def test_missing_models(client, monkeypatch):
    monkeypatch.setattr(api, "_baseline_model", None)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.post("/predict", json={"signal": [0.1] * 256}).status_code == 503


def test_explicit_missing_cnn(client):
    response = client.post("/predict", json={"signal": [0.1] * 256, "model": "cnn"})
    assert response.status_code == 503


def test_string_class_labels(client, monkeypatch):
    model = RandomForestClassifier(n_estimators=2, random_state=0).fit(
        np.zeros((4, FEATURE_DIM)), ["S", "V", "S", "V"]
    )
    monkeypatch.setattr(api, "_baseline_model", model)
    data = client.post("/predict", json={"signal": [0.1] * 256}).json()
    assert set(data["probabilities"]) == {"S", "V"}


def test_cnn_checkpoint_load_and_predict(tmp_path, monkeypatch):
    import torch

    from src.models.cnn_pytorch import ECG1DCNN

    torch.save(ECG1DCNN().state_dict(), tmp_path / "cnn_model.pth")
    monkeypatch.setenv("ECG_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_models_loaded", False)
    monkeypatch.setattr(api, "_baseline_model", None)
    monkeypatch.setattr(api, "_cnn_model", None)
    api._load_models()
    assert api._cnn_model is not None
    with TestClient(api.app) as test_client:
        response = test_client.post("/predict", json={"signal": [0.1] * 250, "model": "cnn"})
    assert response.status_code == 200
    assert response.json()["model_used"] == "cnn"
