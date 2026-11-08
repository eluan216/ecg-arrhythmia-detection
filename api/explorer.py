"""Local waveform explorer with explicit synthetic versus measured provenance."""

import json
import os
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.data.segment import AAMI_CLASSES
from src.data.synthetic import generate_synthetic_dataset

router = APIRouter()


@router.get("/explorer", include_in_schema=False)
def explorer_page():
    return FileResponse(Path(__file__).with_name("explorer.html"))


@router.get("/explorer/data")
def explorer_data():
    directory = os.getenv("ECG_MODEL_DIR")
    path = Path(directory) / "explorer_samples.json" if directory else None
    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("source") != "measured-mitbih":
            raise HTTPException(500, "Unexpected explorer artifact provenance")
        return data
    train, labels, _, _ = generate_synthetic_dataset(
        n_samples_per_class=5, beat_length=250, seed=42
    )
    samples = []
    for label, name in enumerate(AAMI_CLASSES):
        for index in np.flatnonzero(labels == label)[:2]:
            samples.append(
                {
                    "id": f"synthetic-{name}-{index}",
                    "reference": name,
                    "signal": np.asarray(train[index], dtype=float).tolist(),
                    "predictions": {},
                }
            )
    return {
        "source": "synthetic",
        "sampling_rate": 360,
        "selection": "Illustrative generated waveforms. Labels describe generator templates, not diagnosis.",
        "samples": samples,
    }
