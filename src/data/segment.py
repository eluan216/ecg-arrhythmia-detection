"""Segmentation, alignment, and feature extraction for ECG beats."""

from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import wfdb

    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False


def segment_beat_around_rpeak(
    signal: np.ndarray,
    rpeak_index: int,
    beat_length: int = 250,
) -> np.ndarray:
    """Extract a beat window centered on the R-peak."""
    half_beat = beat_length // 2
    start = rpeak_index - half_beat
    end = start + beat_length

    pad_left = 0
    pad_right = 0
    if start < 0:
        pad_left = -start
        start = 0
    if end > len(signal):
        pad_right = end - len(signal)
        end = len(signal)

    beat = signal[start:end]
    if pad_left > 0 or pad_right > 0:
        beat = np.pad(beat, (pad_left, pad_right), mode="constant")
    return beat


def extract_morphology_features(beat_window: np.ndarray) -> np.ndarray:
    """Handcrafted morphology features from a beat window (15-D)."""
    features = []
    peak_val = np.max(np.abs(beat_window))
    peak_idx = np.argmax(np.abs(beat_window))
    features.extend([peak_val, peak_idx / len(beat_window)])
    rms = np.sqrt(np.mean(beat_window**2))
    features.append(rms)
    features.append(np.min(beat_window))
    features.append(np.max(beat_window))
    features.append(np.std(beat_window))
    features.append(np.median(beat_window))
    features.append(np.percentile(beat_window, 75) - np.percentile(beat_window, 25))
    features.append(np.mean(beat_window))
    slope = np.diff(beat_window)
    features.append(np.mean(np.abs(slope)))
    features.append(np.max(np.abs(slope)))
    features.append(np.std(slope))
    zero_crossings = np.where(np.diff(np.sign(beat_window)))[0]
    features.append(len(zero_crossings) / len(beat_window))
    mean = np.mean(beat_window)
    std = np.std(beat_window)
    if std > 0:
        kurtosis = np.mean(((beat_window - mean) / std) ** 4) - 3
        skewness = np.mean(((beat_window - mean) / std) ** 3)
    else:
        kurtosis = 0.0
        skewness = 0.0
    features.extend([kurtosis, skewness])
    return np.array(features)


AAMI_CLASS_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "V": "V",
    "[": "V",
    "]": "V",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "F": "F",
    "/": "Q",
    "Q": "Q",
    "e": "Q",
    "|": "Q",
}

AAMI_CLASSES = ["N", "V", "S", "F", "Q"]
AAMI_CLASS_INDEX = {cls: idx for idx, cls in enumerate(AAMI_CLASSES)}


def map_to_aami_class(mitbih_symbol: str) -> str:
    return AAMI_CLASS_MAP.get(mitbih_symbol, "Q")


def _resolve_record_path(record_id: int, record_dir: str) -> str:
    """
    Prefer local files written by download_mitbih ({id}.hea under record_dir).
    Fall back to PhysioNet-style paths only if local files are missing.
    """
    record_path = Path(record_dir)
    local = record_path / str(record_id)
    if local.with_suffix(".hea").exists():
        return str(local)
    # Nested mitdb/ folder (some older layouts)
    nested = record_path / "mitdb" / str(record_id)
    if nested.with_suffix(".hea").exists():
        return str(nested)
    return str(local)


def load_mitbih_record(
    record_id: int,
    record_dir: str = "data/raw",
    beat_length: int = 250,
    channel: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a MIT-BIH record and extract segmented beats with AAMI labels.

    Expects files from `python -m src.data.download` under record_dir.
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb library required for loading MIT-BIH records")

    rec_base = _resolve_record_path(record_id, record_dir)
    if not Path(rec_base + ".hea").exists():
        raise FileNotFoundError(
            f"MIT-BIH record {record_id} not found under {record_dir}. "
            f"Run: python -m src.data.download"
        )

    record = wfdb.rdrecord(rec_base)
    ann = wfdb.rdann(rec_base, "atr")

    signal = record.p_signal[:, channel]
    rpeak_indices = ann.sample
    beat_symbols = ann.symbol

    beat_windows = []
    labels = []
    class_counts = {cls: 0 for cls in AAMI_CLASSES}

    for rpeak_idx, symbol in zip(rpeak_indices, beat_symbols):
        if symbol in ["(", ")", "p", "t", "u", "`"]:
            continue
        beat = segment_beat_around_rpeak(signal, rpeak_idx, beat_length)
        aami_class = map_to_aami_class(symbol)
        class_idx = AAMI_CLASS_INDEX[aami_class]
        beat_windows.append(beat)
        labels.append(class_idx)
        class_counts[aami_class] += 1

    beat_windows = np.array(beat_windows)
    labels = np.array(labels)
    total = max(len(labels), 1)
    dist = " | ".join(
        [f"{cls}={count} ({100 * count / total:.1f}%)" for cls, count in class_counts.items()]
    )
    return beat_windows, labels, [dist]


def create_inter_patient_split(
    record_ids: List[int],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(record_ids)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx].tolist(), shuffled[split_idx:].tolist()


if __name__ == "__main__":
    print("Segmentation module. Run via src.train modules.")
