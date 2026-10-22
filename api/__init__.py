"""FastAPI application for ECG arrhythmia inference."""

import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.data.segment import (
    AAMI_CLASSES,
    DEFAULT_BEAT_LENGTH,
    FEATURE_DIM,
    FEATURE_VERSION,
    extract_morphology_features,
    validate_beat_window,
)

app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description=(
        "Classify ECG heartbeats into AAMI EC57 categories. "
        "Research prototype — not a medical device."
    ),
    version="0.3.0",
)

_models_loaded = False
_baseline_model = None
_cnn_model = None
_device = None
_model_error: Optional[str] = None

CNN_INPUT_LENGTH = 256


def load_models(model_dir: str = "models") -> dict:
    global _models_loaded, _baseline_model, _cnn_model, _device, _model_error

    model_path = Path(model_dir)
    status = {"baseline_loaded": False, "cnn_loaded": False, "error": None}

    baseline_file = model_path / "baseline_model.pkl"
    if baseline_file.exists():
        try:
            import joblib

            _baseline_model = joblib.load(baseline_file)
            status["baseline_loaded"] = True
            print(f"✓ Baseline model loaded from {baseline_file}")
        except Exception as e:
            status["error"] = f"Failed to load baseline model: {e}"
            print(f"✗ {status['error']}")

    cnn_file = model_path / "cnn_model.pth"
    if cnn_file.exists():
        try:
            import torch

            from src.models.cnn_pytorch import ECG1DCNN

            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _cnn_model = ECG1DCNN(num_classes=5, dropout_rate=0.5).to(_device)
            _cnn_model.load_state_dict(torch.load(cnn_file, map_location=_device))
            _cnn_model.eval()
            status["cnn_loaded"] = True
            print(f"✓ CNN model loaded from {cnn_file} (device: {_device})")
        except Exception as e:
            status["error"] = f"Failed to load CNN model: {e}"
            print(f"✗ {status['error']}")

    _models_loaded = True
    if not (status["baseline_loaded"] or status["cnn_loaded"]):
        _model_error = (
            "No trained models found. Run inter-patient training first: "
            "python -m src.eval.inter_patient"
        )
        print(f"⚠ {_model_error}")

    return status


class BeatWindowRequest(BaseModel):
    beat_values: List[float] = Field(
        ...,
        description=(
            f"1-D ECG samples for one R-peak-centered beat. "
            f"Recommended length ~{DEFAULT_BEAT_LENGTH} at 360 Hz. "
            f"CNN path requires exactly {CNN_INPUT_LENGTH} after pad/trim policy."
        ),
    )
    model_preference: Optional[str] = Field(
        default=None,
        description="Optional: 'baseline' or 'cnn'. Default prefers CNN if loaded.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"beat_values": [0.0] * DEFAULT_BEAT_LENGTH},
        }
    )


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict
    model_used: str = ""
    feature_version: str = FEATURE_VERSION
    message: str = ""


@app.on_event("startup")
async def startup_event():
    print("Starting ECG Arrhythmia Detection API...")
    load_models()


@app.get("/")
def read_root():
    return {
        "service": "ECG Arrhythmia Detection",
        "version": "0.3.0",
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "disclaimer": "Research project, NOT a medical diagnostic tool.",
        "models_loaded": {
            "baseline": _baseline_model is not None,
            "cnn": _cnn_model is not None,
        },
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "predict": "/predict",
    }


@app.get("/health")
def health_check():
    """Liveness: process is up."""
    return {"status": "ok", "service": "ECG Arrhythmia Detection API"}


@app.get("/ready")
def readiness_check():
    """Readiness: at least one model is loaded."""
    ready = _baseline_model is not None or _cnn_model is not None
    return {
        "status": "ready" if ready else "not_ready",
        "models_available": {
            "baseline": _baseline_model is not None,
            "cnn": _cnn_model is not None,
        },
        "feature_version": FEATURE_VERSION,
    }


def _predict_cnn(beat_array: np.ndarray) -> PredictionResponse:
    import torch

    # Explicit pad/trim policy for fixed FC layer sized for 256 samples
    if len(beat_array) < CNN_INPUT_LENGTH:
        beat_array = np.pad(
            beat_array, (0, CNN_INPUT_LENGTH - len(beat_array)), mode="constant"
        )
    elif len(beat_array) > CNN_INPUT_LENGTH:
        beat_array = beat_array[:CNN_INPUT_LENGTH]

    beat_tensor = (
        torch.from_numpy(beat_array.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(_device)
    )
    with torch.no_grad():
        logits = _cnn_model(beat_tensor)
        if not torch.isfinite(logits).all():
            raise RuntimeError("CNN produced non-finite logits")
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_idx = int(np.argmax(probs))
    return PredictionResponse(
        predicted_class=AAMI_CLASSES[predicted_idx],
        confidence=float(probs[predicted_idx]),
        class_probabilities={lab: float(p) for lab, p in zip(AAMI_CLASSES, probs)},
        model_used="cnn",
        feature_version=FEATURE_VERSION,
        message=f"1D-CNN (input padded/trimmed to {CNN_INPUT_LENGTH})",
    )


def _predict_baseline(beat_array: np.ndarray) -> PredictionResponse:
    # Shared feature path with training
    features = extract_morphology_features(beat_array).reshape(1, -1)
    predicted_idx = int(_baseline_model.predict(features)[0])
    predicted_class = AAMI_CLASSES[predicted_idx]

    if hasattr(_baseline_model, "predict_proba"):
        probs = _baseline_model.predict_proba(features)[0]
    else:
        probs = np.zeros(len(AAMI_CLASSES))
        probs[predicted_idx] = 1.0

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=float(probs[predicted_idx]),
        class_probabilities={lab: float(p) for lab, p in zip(AAMI_CLASSES, probs)},
        model_used="baseline",
        feature_version=FEATURE_VERSION,
        message=f"Random Forest on {FEATURE_VERSION} morphology features",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: BeatWindowRequest) -> PredictionResponse:
    try:
        beat_array = validate_beat_window(np.asarray(request.beat_values, dtype=np.float64))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    pref = (request.model_preference or "").lower().strip()

    if pref == "baseline":
        if _baseline_model is None:
            raise HTTPException(status_code=503, detail="Baseline model not loaded")
        try:
            return _predict_baseline(beat_array)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Baseline inference failed: {e}") from e

    if pref == "cnn":
        if _cnn_model is None:
            raise HTTPException(status_code=503, detail="CNN model not loaded")
        try:
            return _predict_cnn(beat_array)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"CNN inference failed: {e}") from e

    # Default: CNN then baseline
    if _cnn_model is not None:
        try:
            return _predict_cnn(beat_array)
        except Exception as e:
            warnings.warn(f"CNN inference failed: {e}. Falling back to baseline.")

    if _baseline_model is not None:
        try:
            return _predict_baseline(beat_array)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Baseline inference failed: {e}") from e

    raise HTTPException(
        status_code=503,
        detail=_model_error or "No trained models available.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
