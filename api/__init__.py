"""FastAPI application for ECG arrhythmia inference."""

from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description="Classify ECG heartbeats into AAMI EC57 arrhythmia categories",
    version="0.1.0",
)


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
            "message": "Prediction successful",
        }
    )


@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "service": "ECG Arrhythmia Detection",
        "version": "0.1.0",
        "disclaimer": "This is a research project, NOT a medical diagnostic tool.",
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

    # Placeholder prediction logic
    # In production, this would load a trained model and perform inference
    class_labels = ["N", "V", "S", "F", "Q"]

    # Simulate prediction (replace with actual model)
    probs = np.random.dirichlet(np.ones(5))
    predicted_idx = np.argmax(probs)
    predicted_class = class_labels[predicted_idx]
    confidence = float(probs[predicted_idx])

    class_probs = {label: float(prob) for label, prob in zip(class_labels, probs)}

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        class_probabilities=class_probs,
        message="Placeholder prediction. Train a model to get real results.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
