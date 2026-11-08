"""FastAPI application for ECG arrhythmia inference."""

import os
import warnings
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.data.segment import (
    AAMI_CLASSES,
    EXPECTED_FS_HZ,
    FEATURE_DIM,
    FEATURE_VERSION,
    extract_morphology_features,
    normalize_cnn_beats,
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
_device = "cpu"


class BeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: List[float] = Field(..., description="1-D ECG beat samples")
    sampling_rate: float = Field(360.0, gt=0, allow_inf_nan=False)
    model: str = Field("auto", pattern="^(auto|baseline|cnn)$")


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: dict
    model_used: str
    feature_version: str


def _load_models() -> None:
    global _models_loaded, _baseline_model, _cnn_model, _device
    if _models_loaded:
        return
    root_dir = Path(__file__).resolve().parent.parent
    configured_dir = os.getenv("ECG_MODEL_DIR")
    candidates = [
        root_dir / "artifacts" / "models",
        root_dir / "models",
        root_dir / "artifacts" / "eval",
        Path("artifacts/models"),
        Path("models"),
        Path("artifacts/eval"),
    ]
    model_dir = (
        Path(configured_dir)
        if configured_dir
        else next(
            (
                path
                for path in candidates
                if (path / "baseline_model.pkl").exists() or (path / "cnn_model.pth").exists()
            ),
            candidates[0],
        )
    )

    baseline_path = model_dir / "baseline_model.pkl"
    if baseline_path.exists():
        try:
            import joblib

            _baseline_model = joblib.load(baseline_path)
        except Exception as exc:
            warnings.warn(f"Baseline load failed: {exc}")

    cnn_path = model_dir / "cnn_model.pth"
    if cnn_path.exists():
        try:
            import torch

            from src.models.cnn_pytorch import ECG1DCNN

            _device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ECG1DCNN(num_classes=len(AAMI_CLASSES))
            state = torch.load(str(cnn_path), map_location=_device, weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.to(_device)
            model.eval()
            _cnn_model = model
        except Exception as exc:
            warnings.warn(f"CNN load failed: {exc}")

    _models_loaded = True


@app.on_event("startup")
def startup() -> None:
    _load_models()


@app.get("/")
def root():
    return {
        "service": "ECG Arrhythmia Detection API",
        "disclaimer": "Research / education only. Not for clinical decisions.",
    }


@app.get("/health")
def health():
    _load_models()
    return {
        "status": "ok",
        "baseline_loaded": _baseline_model is not None,
        "cnn_loaded": _cnn_model is not None,
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
    }


@app.get("/ready")
def ready():
    _load_models()
    if _baseline_model is None and _cnn_model is None:
        raise HTTPException(status_code=503, detail="No models loaded")
    return {
        "status": "ready",
        "baseline_loaded": _baseline_model is not None,
        "cnn_loaded": _cnn_model is not None,
    }


def _predict_baseline(beat_array: np.ndarray) -> PredictionResponse:
    feats = extract_morphology_features(beat_array)
    if feats is None or len(feats) != FEATURE_DIM:
        raise HTTPException(status_code=400, detail="Feature extraction failed")
    proba = _baseline_model.predict_proba(np.asarray(feats).reshape(1, -1))[0]
    classes = [
        AAMI_CLASSES[int(c)] if isinstance(c, (int, np.integer)) else str(c)
        for c in _baseline_model.classes_
    ]
    pred_idx = int(np.argmax(proba))
    return PredictionResponse(
        predicted_class=str(classes[pred_idx]),
        probabilities={str(c): float(p) for c, p in zip(classes, proba)},
        model_used="baseline",
        feature_version=FEATURE_VERSION,
    )


def _predict_cnn(beat_array: np.ndarray) -> PredictionResponse:
    import torch

    beat_array = normalize_cnn_beats(beat_array)

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
    _load_models()
    arr = np.asarray(req.signal, dtype=np.float64)
    try:
        arr = validate_beat_window(arr)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if req.sampling_rate != EXPECTED_FS_HZ:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FS_HZ} Hz; resample the signal before inference",
        )

    model_choice = (req.model or "auto").lower()
    if model_choice == "auto":
        if _baseline_model is not None:
            model_choice = "baseline"
        elif _cnn_model is not None:
            model_choice = "cnn"
        else:
            raise HTTPException(status_code=503, detail="No models available")

    if model_choice == "baseline":
        if _baseline_model is None:
            raise HTTPException(status_code=503, detail="Baseline model not loaded")
        return _predict_baseline(arr)
    if model_choice == "cnn":
        if _cnn_model is None:
            raise HTTPException(status_code=503, detail="CNN model not loaded")
        return _predict_cnn(arr)

    raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
