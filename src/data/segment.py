"""Segmentation, alignment, annotation mapping, and shared morphology features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import wfdb

    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

# Bump when feature definitions change — models trained on older versions must not be mixed.
FEATURE_VERSION = "morphology_v1"
FEATURE_DIM = 15
DEFAULT_BEAT_LENGTH = 250
EXPECTED_FS_HZ = 360

# MIT-BIH: some record IDs are the same physical subject.
# 201 and 202 must never be split across train/test.
SUBJECT_GROUPS: Dict[int, int] = {
    201: 201,
    202: 201,  # same subject as 201
}


def subject_id_for_record(record_id: int) -> int:
    """Map a MIT-BIH record ID to a subject group ID."""
    return SUBJECT_GROUPS.get(int(record_id), int(record_id))


def segment_beat_around_rpeak(
    signal: np.ndarray,
    rpeak_index: int,
    beat_length: int = DEFAULT_BEAT_LENGTH,
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


def validate_beat_window(
    beat_window: np.ndarray,
    *,
    min_length: int = 32,
    max_length: int = 1000,
) -> np.ndarray:
    """
    ECG input contract for morphology features and API inference.

    Raises ValueError on empty, wrong rank, non-finite, or out-of-range length.
    """
    arr = np.asarray(beat_window, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Beat window must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("Beat window is empty")
    if arr.size < min_length:
        raise ValueError(f"Beat window too short: {arr.size} < {min_length}")
    if arr.size > max_length:
        raise ValueError(f"Beat window too long: {arr.size} > {max_length}")
    if not np.isfinite(arr).all():
        raise ValueError("Beat window contains NaN or Inf")
    return arr


def extract_morphology_features(beat_window: np.ndarray) -> np.ndarray:
    """
    Canonical morphology features (FEATURE_VERSION / FEATURE_DIM).

    Training and API **must** call this function — do not reimplement.
    """
    x = validate_beat_window(beat_window)
    n = len(x)
    peak_val = float(np.max(np.abs(x)))
    peak_idx = float(np.argmax(np.abs(x)) / n)
    rms = float(np.sqrt(np.mean(x**2)))
    mn = float(np.min(x))
    mx = float(np.max(x))
    std = float(np.std(x))
    med = float(np.median(x))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    mean = float(np.mean(x))
    slope = np.diff(x)
    mean_abs_slope = float(np.mean(np.abs(slope))) if len(slope) else 0.0
    max_abs_slope = float(np.max(np.abs(slope))) if len(slope) else 0.0
    std_slope = float(np.std(slope)) if len(slope) else 0.0
    zero_crossings = np.where(np.diff(np.sign(x)))[0]
    zcr = float(len(zero_crossings) / n)
    if std > 0:
        z = (x - mean) / std
        kurtosis = float(np.mean(z**4) - 3.0)
        skewness = float(np.mean(z**3))
    else:
        kurtosis = 0.0
        skewness = 0.0
    feats = np.array(
        [
            peak_val,
            peak_idx,
            rms,
            mn,
            mx,
            std,
            med,
            iqr,
            mean,
            mean_abs_slope,
            max_abs_slope,
            std_slope,
            zcr,
            kurtosis,
            skewness,
        ],
        dtype=np.float64,
    )
    assert feats.shape == (FEATURE_DIM,)
    return feats


# --- Annotation mapping (whitelist only; never map unknown → Q) ---

# Explicit beat symbols accepted into AAMI classes
BEAT_SYMBOL_TO_AAMI: Dict[str, str] = {
    # N family
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",  # atrial escape — often folded into N in AAMI practice; kept explicit
    "j": "N",  # nodal escape
    # V family
    "V": "V",
    "E": "V",  # ventricular escape
    # S family
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # Fusion
    "F": "F",
    # Unknown / paced grouped as Q in AAMI
    "/": "Q",  # paced
    "f": "Q",  # fusion of paced and normal
    "Q": "Q",
}

# Non-beat annotation symbols — skip, never label as arrhythmia
NON_BEAT_SYMBOLS = frozenset(
    {
        "+",  # rhythm change
        "~",  # signal quality
        "|",  # isolated QRS-like
        "x",  # non-conducted P
        "(",
        ")",
        "p",
        "t",
        "u",
        "`",
        "^",
        "!",
        "[",
        "]",
        '"',
        "@",
    }
)

AAMI_CLASSES = ["N", "V", "S", "F", "Q"]
AAMI_CLASS_INDEX = {cls: idx for idx, cls in enumerate(AAMI_CLASSES)}


def map_to_aami_class(mitbih_symbol: str) -> Optional[str]:
    """
    Map a MIT-BIH annotation symbol to an AAMI class, or None if not a usable beat.

    Unknown symbols return None (caller must skip) — they are **not** forced to Q.
    """
    if mitbih_symbol in NON_BEAT_SYMBOLS:
        return None
    return BEAT_SYMBOL_TO_AAMI.get(mitbih_symbol)


def _resolve_record_path(record_id: int, record_dir: str) -> str:
    record_path = Path(record_dir)
    local = record_path / str(record_id)
    if local.with_suffix(".hea").exists():
        return str(local)
    nested = record_path / "mitdb" / str(record_id)
    if nested.with_suffix(".hea").exists():
        return str(nested)
    return str(local)


def load_mitbih_record(
    record_id: int,
    record_dir: str = "data/raw",
    beat_length: int = DEFAULT_BEAT_LENGTH,
    channel: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a MIT-BIH record and extract segmented beats with AAMI labels.

    Returns beat windows, integer labels, and a distribution summary string list.
    Annotation stats (accepted / rejected / non-beat) are included in the summary.
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb library required for loading MIT-BIH records")

    rec_base = _resolve_record_path(record_id, record_dir)
    hea = Path(rec_base + ".hea")
    dat = Path(rec_base + ".dat")
    atr = Path(rec_base + ".atr")
    if not hea.exists():
        raise FileNotFoundError(
            f"MIT-BIH record {record_id} missing .hea under {record_dir}. "
            f"Run: python -m src.data.download"
        )
    if not dat.exists():
        raise FileNotFoundError(f"MIT-BIH record {record_id} missing .dat under {record_dir}")
    if not atr.exists():
        raise FileNotFoundError(f"MIT-BIH record {record_id} missing .atr under {record_dir}")

    record = wfdb.rdrecord(rec_base)
    ann = wfdb.rdann(rec_base, "atr")

    signal = record.p_signal[:, channel]
    rpeak_indices = ann.sample
    beat_symbols = ann.symbol

    beat_windows = []
    labels = []
    class_counts = {cls: 0 for cls in AAMI_CLASSES}
    n_non_beat = 0
    n_unmapped = 0
    n_accepted = 0

    for rpeak_idx, symbol in zip(rpeak_indices, beat_symbols):
        if symbol in NON_BEAT_SYMBOLS:
            n_non_beat += 1
            continue
        aami_class = map_to_aami_class(symbol)
        if aami_class is None:
            n_unmapped += 1
            continue
        beat = segment_beat_around_rpeak(signal, rpeak_idx, beat_length)
        class_idx = AAMI_CLASS_INDEX[aami_class]
        beat_windows.append(beat)
        labels.append(class_idx)
        class_counts[aami_class] += 1
        n_accepted += 1

    if not beat_windows:
        raise RuntimeError(
            f"Record {record_id}: no accepted beats "
            f"(non_beat={n_non_beat}, unmapped={n_unmapped})"
        )

    beat_windows = np.array(beat_windows)
    labels = np.array(labels)
    total = max(len(labels), 1)
    dist = " | ".join(
        [f"{cls}={count} ({100 * count / total:.1f}%)" for cls, count in class_counts.items()]
    )
    stats = (
        f"accepted={n_accepted} non_beat_skipped={n_non_beat} "
        f"unmapped_skipped={n_unmapped} subject={subject_id_for_record(record_id)}"
    )
    return beat_windows, labels, [dist, stats]


def create_inter_patient_split(
    record_ids: List[int],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Subject-disjoint split of record IDs.

    Records that share a subject (e.g. 201/202) are kept in the same partition.
    """
    # Build subject → records
    subject_to_records: Dict[int, List[int]] = {}
    for rid in record_ids:
        sid = subject_id_for_record(rid)
        subject_to_records.setdefault(sid, []).append(int(rid))

    subjects = sorted(subject_to_records.keys())
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(subjects)
    split_idx = int(len(shuffled) * train_ratio)
    train_subjects = set(shuffled[:split_idx].tolist())
    test_subjects = set(shuffled[split_idx:].tolist())

    train_records: List[int] = []
    test_records: List[int] = []
    for sid, recs in subject_to_records.items():
        if sid in train_subjects:
            train_records.extend(sorted(recs))
        else:
            test_records.extend(sorted(recs))

    # Integrity: no shared subjects
    train_subj = {subject_id_for_record(r) for r in train_records}
    test_subj = {subject_id_for_record(r) for r in test_records}
    assert train_subj.isdisjoint(test_subj), "Subject leakage in split"
    return train_records, test_records


if __name__ == "__main__":
    print(f"FEATURE_VERSION={FEATURE_VERSION} dim={FEATURE_DIM}")
    print("Segmentation module. Run via src.train / src.eval modules.")
