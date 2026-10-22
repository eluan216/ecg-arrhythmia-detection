"""FastAPI application for ECG arrhythmia inference."""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.data.segment import (
    FEATURE_DIM,
    FEATURE_VERSION,
    extract_morphology_features,
    validate_beat_window,
)

warnings.filterwarnings("ignore")

app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description=(
        "Research / educational inference API for AAMI-style ECG beat classification. "
        "Not a medical device. Do not use for clinical decisions."
    ),
    version="0.3.0",
)

# Configurable CORS (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
CNN_INPUT_LENGTH = 256

_device = "cpu"
_cnn_model = None
_baseline_model = None
_baseline_available = False
_cnn_available = False


class BeatRequest(BaseModel):
    """Single-beat inference request."""

    signal: List[float] = Field(..., description="Raw ECG samples for one beat window")
    sampling_rate: float = Field(360.0, description="Sampling rate in Hz")
    model: str = Field("auto", description="auto | baseline | cnn")

    @field_validator("signal")
    @classmethod
    def check_signal(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("signal must be non-empty")
        arr = np.asarray(v, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError("signal contains NaN or Inf")
        if arr.ndim != 1:
            raise ValueError("signal must be 1-D")
        if len(arr) < 32:
            raise ValueError("signal too short (min 32 samples)")
        if len(arr) > 4096:
            raise ValueError("signal too long (max 4096 samples)")
        return v


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]
    model_used: str
    feature_version: str


def _load_models() -> None:
    global _cnn_model, _baseline_model, _baseline_available, _cnn_available, _device
    model_dir = Path(__file__).resolve().parent.parent / "models"

    # Baseline (sklearn)
    baseline_path = model_dir / "baseline_model.pkl"
    if baseline_path.exists():
        try:
            import joblib

            _baseline_model = joblib.load(baseline_path)
            _baseline_available = True
        except Exception as exc:  # pragma: no cover
            print(f"Baseline load failed: {exc}")

    # CNN (torch)
    cnn_path = model_dir / "cnn_model.pt"
    if cnn_path.exists():
        try:
            import torch
            from src.models.cnn1d import ECGCNN1D

            _device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ECGCNN1D(num_classes=len(AAMI_CLASSES), input_length=CNN_INPUT_LENGTH)
            state = torch.load(cnn_path, map_location=_device)
            model.load_state_dict(state)
            model.to(_device)
            model.eval()
            _cnn_model = model
            _cnn_available = True
        except Exception as exc:  # pragma: no cover
            print(f"CNN load failed: {exc}")


@app.on_event("startup")
def startup() -> None:
    _load_models()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "baseline_available": _baseline_available,
        "cnn_available": _cnn_available,
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
    }


@app.get("/ready")
def ready() -> Dict[str, Any]:
    """Readiness: at least one model loaded."""
    if not (_baseline_available or _cnn_available):
        raise HTTPException(status_code=503, detail="No models loaded")
    return {
        "status": "ready",
        "baseline_available": _baseline_available,
        "cnn_available": _cnn_available,
    }


def _predict_baseline(beat_array: np.ndarray) -> PredictionResponse:
    feats = extract_morphology_features(beat_array)
    if feats is None or len(feats) != FEATURE_DIM:
        raise HTTPException(status_code=400, detail="Feature extraction failed")
    proba = _baseline_model.predict_proba(feats.reshape(1, -1))[0]
    pred_idx = int(np.argmax(proba))
    return PredictionResponse(
        predicted_class=AAMI_CLASSES[pred_idx],
        probabilities={c: float(p) for c, p in zip(AAMI_CLASSES, proba)},
        model_used="baseline",
        feature_version=FEATURE_VERSION,
    )


def _predict_cnn(beat_array: np.ndarray) -> PredictionResponse:
    import torch

    # Explicit pad/trim policy for fixed FC layer sized for 256 samples
    if len(beat_array) < CNN_INPUT_LENGTH:
        beat_array = np.pad(beat_array, (0, CNN_INPUT_LENGTH - len(beat_array)), mode="constant")
    elif len(beat_array) > CNN_INPUT_LENGTH:
        beat_array = beat_array[:CNN_INPUT_LENGTH]

    beat_tensor = (
        torch.from_numpy(beat_array.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(_device)
    )
    with torch.no_grad():
        logits = _cnn_model(beat_tensor)
        if not torch.isfinite(logits).all():
            raise HTTPException(status_code=500, detail="CNN produced non-finite logits")
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(proba))
    return PredictionResponse(
        predicted_class=AAMI_CLASSES[pred_idx],
        probabilities={c: float(p) for c, p in zip(AAMI_CLASSES, proba)},
        model_used="cnn",
        feature_version=FEATURE_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: BeatRequest) -> PredictionResponse:
    """Classify a single ECG beat window."""
    arr = np.asarray(req.signal, dtype=np.float64)
    ok, msg = validate_beat_window(arr)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    model_choice = (req.model or "auto").lower()
    if model_choice == "auto":
        if _baseline_available:
            model_choice = "baseline"
        elif _cnn_available:
            model_choice = "cnn"
        else:
            raise HTTPException(status_code=503, detail="No models available")

    if model_choice == "baseline":
        if not _baseline_available:
            raise HTTPException(status_code=503, detail="Baseline model not loaded")
        return _predict_baseline(arr)
    if model_choice == "cnn":
        if not _cnn_available:
            raise HTTPException(status_code=503, detail="CNN model not loaded")
        return _predict_cnn(arr)

    raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
