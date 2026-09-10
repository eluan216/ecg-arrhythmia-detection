"""FastAPI application for ECG arrhythmia inference."""

import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Initialize FastAPI app
app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description="Classify ECG heartbeats into AAMI EC57 arrhythmia categories",
    version="0.1.0",
)

# Global model state
_models_loaded = False
_baseline_model = None
_cnn_model = None
_device = None
_model_error = None

# AAMI class labels
AAMI_CLASSES = ["N", "V", "S", "F", "Q"]


def load_models(model_dir: str = "models") -> dict:
    """
    Load trained models at startup.

    Args:
        model_dir: Directory containing trained model files.

    Returns:
        Dictionary with load status.
    """
    global _models_loaded, _baseline_model, _cnn_model, _device, _model_error

    model_path = Path(model_dir)
    status = {
        "baseline_loaded": False,
        "cnn_loaded": False,
        "error": None,
    }

    # Try to load baseline model
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

    # Try to load CNN model
    cnn_file = model_path / "cnn_model.pth"
    if cnn_file.exists():
        try:
            import torch

            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            from src.models.cnn_pytorch import ECG1DCNN

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
            "No trained models found. Run: python -m src.train.train_baseline "
            "and python -m src.train.train_cnn"
        )
        print(f"⚠ {_model_error}")

    return status


def extract_morphology_features(beat: np.ndarray) -> np.ndarray:
    """
    Extract 15-dimensional morphology features from a beat window.

    Args:
        beat: Beat window array.

    Returns:
        Feature vector of length 15.
    """
    features = []

    # 1. Peak magnitude
    features.append(np.max(beat))

    # 2. Peak location (normalized)
    features.append(np.argmax(beat) / len(beat))

    # 3. Min value
    features.append(np.min(beat))

    # 4. Min location (normalized)
    features.append(np.argmin(beat) / len(beat))

    # 5. RMS
    features.append(np.sqrt(np.mean(beat**2)))

    # 6. Standard deviation
    features.append(np.std(beat))

    # 7. Mean absolute value
    features.append(np.mean(np.abs(beat)))

    # 8. Mean
    features.append(np.mean(beat))

    # 9. Variance
    features.append(np.var(beat))

    # 10. Slope (mean absolute derivative)
    slope = np.abs(np.diff(beat))
    features.append(np.mean(slope))

    # 11. Kurtosis
    features.append(
        np.mean(((beat - np.mean(beat)) / np.std(beat)) ** 4)
        if np.std(beat) > 0
        else 0
    )

    # 12. Skewness
    features.append(
        np.mean(((beat - np.mean(beat)) / np.std(beat)) ** 3)
        if np.std(beat) > 0
        else 0
    )

    # 13. Zero crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(beat)))) / 2
    features.append(zero_crossings / len(beat))

    # 14-15. Energy and spectral centroid (simplified)
    features.append(np.sum(beat**2))
    try:
        # Simple frequency-domain feature
        fft_magnitude = np.abs(np.fft.fft(beat))
        spectral_centroid = (
            np.sum(np.arange(len(fft_magnitude)) * fft_magnitude)
            / np.sum(fft_magnitude)
        )
        features.append(spectral_centroid / len(fft_magnitude))
    except Exception:
        features.append(0)

    return np.array(features, dtype=np.float32)


class BeatWindowRequest(BaseModel):
    """ECG beat window for inference."""

    beat_values: List[float] = Field(
        ...,
        description="Array of ECG signal samples (250-360 samples recommended)",
        json_schema_extra={"example": [0.1, -0.05, 0.3, -0.2]},
    )

    model_config = ConfigDict(
        json_schema_extra={
            "beat_values": [0.1, -0.05, 0.3, -0.2] * 64,  # 256 samples
        }
    )


class PredictionResponse(BaseModel):
    """Model prediction response."""

    predicted_class: str = Field(
        ...,
        description="Predicted AAMI class: N (Normal), V (VEB), S (SVEB), F (Fusion), Q (Unknown)",
    )
    confidence: float = Field(
        ...,
        description="Predicted probability of the predicted class [0, 1]",
    )
    class_probabilities: dict = Field(
        ...,
        description="Probabilities for all classes",
    )
    model_used: str = Field(
        default="",
        description="Which model was used (baseline or cnn)",
    )
    message: str = Field(
        default="",
        description="Additional information or warnings",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "predicted_class": "N",
            "confidence": 0.95,
            "class_probabilities": {
                "N": 0.95,
                "V": 0.03,
                "S": 0.01,
                "F": 0.005,
                "Q": 0.005,
            },
            "model_used": "cnn",
            "message": "Prediction successful",
        }
    )


@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    print("Starting ECG Arrhythmia Detection API...")
    load_models()


@app.get("/")
def read_root():
    """Root endpoint with API information."""
    status = {
        "baseline": _baseline_model is not None,
        "cnn": _cnn_model is not None,
    }
    return {
        "service": "ECG Arrhythmia Detection",
        "version": "0.1.0",
        "disclaimer": "This is a research project, NOT a medical diagnostic tool.",
        "models_loaded": status,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "ECG Arrhythmia Detection API",
        "models_available": {
            "baseline": _baseline_model is not None,
            "cnn": _cnn_model is not None,
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: BeatWindowRequest) -> PredictionResponse:
    """
    Predict arrhythmia class for an ECG beat window.

    Takes a beat window (array of ECG samples) and returns:
    - Predicted AAMI class (N, V, S, F, Q)
    - Confidence score
    - Per-class probabilities

    **Important:** This is a research prototype, not a clinical tool.
    """

    # Validate input
    if len(request.beat_values) < 100:
        raise HTTPException(
            status_code=400,
            detail="Beat window too short. Provide at least 100 samples.",
        )

    if len(request.beat_values) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Beat window too long. Provide at most 1000 samples.",
        )

    beat_array = np.array(request.beat_values, dtype=np.float32)

    # Try CNN first if available, fallback to baseline
    if _cnn_model is not None:
        try:
            import torch

            # Pad/trim to 256 samples
            if len(beat_array) < 256:
                beat_array = np.pad(
                    beat_array, (0, 256 - len(beat_array)), mode="constant"
                )
            else:
                beat_array = beat_array[:256]

            # Prepare input (1, 1, 256)
            beat_tensor = (
                torch.from_numpy(beat_array).unsqueeze(0).unsqueeze(0).to(_device)
            )

            with torch.no_grad():
                logits = _cnn_model(beat_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            predicted_idx = np.argmax(probs)
            predicted_class = AAMI_CLASSES[predicted_idx]
            confidence = float(probs[predicted_idx])

            class_probs = {
                label: float(prob) for label, prob in zip(AAMI_CLASSES, probs)
            }

            return PredictionResponse(
                predicted_class=predicted_class,
                confidence=confidence,
                class_probabilities=class_probs,
                model_used="cnn",
                message="Prediction from 1D-CNN model",
            )
        except Exception as e:
            warnings.warn(f"CNN inference failed: {e}. Falling back to baseline.")

    if _baseline_model is not None:
        try:
            # Extract morphology features for baseline
            features = extract_morphology_features(beat_array)
            features = features.reshape(1, -1)

            # Get prediction
            predicted_idx = _baseline_model.predict(features)[0]
            predicted_class = AAMI_CLASSES[predicted_idx]

            # Get probabilities if available
            if hasattr(_baseline_model, "predict_proba"):
                probs = _baseline_model.predict_proba(features)[0]
            else:
                probs = np.zeros(5)
                probs[predicted_idx] = 1.0

            confidence = float(probs[predicted_idx])
            class_probs = {
                label: float(prob) for label, prob in zip(AAMI_CLASSES, probs)
            }

            return PredictionResponse(
                predicted_class=predicted_class,
                confidence=confidence,
                class_probabilities=class_probs,
                model_used="baseline",
                message="Prediction from Random Forest baseline model",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Baseline inference failed: {e}",
            )

    # No models available
    raise HTTPException(
        status_code=503,
        detail=_model_error or "No trained models available. Please train models first.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
