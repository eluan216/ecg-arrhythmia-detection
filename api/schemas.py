"""API request and response models."""

from pydantic import BaseModel, Field
from typing import Dict, List


class BeatWindowRequest(BaseModel):
    """ECG beat window for inference."""
    
    beat_values: List[float] = Field(
        ...,
        description="Array of ECG signal samples",
    )


class PredictionResponse(BaseModel):
    """Model prediction response."""
    
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    message: str = ""
