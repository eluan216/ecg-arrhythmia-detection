"""Annotated beat datasets and causal rhythm context for research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

from src.data.segment import (
    AAMI_CLASS_INDEX,
    EXPECTED_FS_HZ,
    extract_morphology_features,
    map_to_aami_class,
    segment_beat_around_rpeak,
    subject_id_for_record,
)

RR_FEATURE_NAMES = [
    "previous_rr_seconds",
    "previous_rr_over_history",
    "history_rr_seconds",
]


@dataclass
class BeatDataset:
    windows: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    records: np.ndarray
    samples: np.ndarray
    rr: np.ndarray

    def subset(self, indices):
        return BeatDataset(**{key: value[indices] for key, value in vars(self).items()})

    def features(self, rhythm=False):
        morphology = np.vstack([extract_morphology_features(b) for b in self.windows])
        return np.column_stack([morphology, self.rr]) if rhythm else morphology


def causal_rr_features(samples, fs):
    """Use earlier accepted beat locations only; never use the next beat or label."""
    samples = np.asarray(samples, dtype=np.int64)
    if fs <= 0 or not np.isfinite(fs) or np.any(np.diff(samples) <= 0):
        raise ValueError("Beat locations must increase strictly and sampling rate must be positive")
    features = np.zeros((len(samples), 3), dtype=np.float64)
    intervals = np.diff(samples) / fs
    for i in range(1, len(samples)):
        previous = intervals[i - 1]
        # At most ten completed intervals, including the interval just observed.
        history = float(np.mean(intervals[max(0, i - 10) : i]))
        features[i] = [previous, previous / history, history]
    return features


def read_annotated_record(
    base,
    record_id,
    group,
    *,
    lead=None,
    noisy_only=False,
    drop_outside=False,
    audit=None,
    expected_fs=None,
    beat_length=250,
):
    """Read physical mV, resample before segmentation, and map accepted beat labels."""
    record = wfdb.rdrecord(str(base))
    if expected_fs is not None and record.fs != expected_fs:
        raise ValueError(f"{record_id}: expected {expected_fs} Hz, got {record.fs}")
    ann = wfdb.rdann(str(base), "atr")
    channel = 0 if lead is None else record.sig_name.index(lead)
    if record.units[channel].lower() != "mv":
        raise ValueError(f"{record_id}: expected mV, got {record.units[channel]}")
    signal = np.asarray(record.p_signal[:, channel], dtype=np.float64)
    if not np.isfinite(signal).all():
        raise ValueError(f"{record_id}: signal contains missing or non-finite samples")
    ratio = Fraction(EXPECTED_FS_HZ / record.fs).limit_denominator(10000)
    if record.fs != EXPECTED_FS_HZ:
        signal = resample_poly(signal, ratio.numerator, ratio.denominator)
    accepted = [(int(s), map_to_aami_class(c)) for s, c in zip(ann.sample, ann.symbol)]
    accepted = [(s, c) for s, c in accepted if c is not None]
    if not accepted:
        raise ValueError(f"{record_id}: no mapped beats")
    original = np.asarray([s for s, _ in accepted])
    samples = np.rint(original * EXPECTED_FS_HZ / record.fs).astype(np.int64)
    valid = (samples >= 0) & (samples < len(signal))
    if not valid.all() and not drop_outside:
        raise ValueError(f"{record_id}: annotations outside the recording")
    if audit is not None:
        audit[str(record_id)] = {
            "mapped_annotations": len(accepted),
            "outside_recording_excluded": int((~valid).sum()),
            "excluded_original_samples": original[~valid].tolist(),
            "source_sampling_rate_hz": float(record.fs),
            "unmapped_or_nonbeat_annotations": len(ann.sample) - len(accepted),
        }
    accepted = [item for item, keep in zip(accepted, valid) if keep]
    original, samples = original[valid], samples[valid]
    if not accepted:
        raise ValueError(f"{record_id}: no mapped beats inside the recording")
    windows = np.stack([segment_beat_around_rpeak(signal, int(s), beat_length) for s in samples])
    labels = np.asarray([AAMI_CLASS_INDEX[c] for _, c in accepted], dtype=np.int64)
    data = BeatDataset(
        windows.astype(np.float32),
        labels,
        np.full(len(labels), str(group)),
        np.full(len(labels), str(record_id)),
        samples,
        causal_rr_features(original, record.fs),
    )
    if noisy_only:
        seconds = original / record.fs
        # NSTDB: noise begins at 5 minutes, alternating 2 minutes noisy/2 clean.
        mask = (seconds >= 300) & (((seconds - 300) % 240) < 120)
        data = data.subset(mask)
    return data


def concatenate(datasets):
    if not datasets:
        raise ValueError("No datasets supplied")
    return BeatDataset(
        **{key: np.concatenate([getattr(d, key) for d in datasets]) for key in vars(datasets[0])}
    )


def load_mitbih(data_dir, records, *, beat_length=250):
    datasets = []
    for rec in records:
        base = Path(data_dir) / str(rec)
        if not base.with_suffix(".hea").exists():
            base = Path(data_dir) / "mitdb" / str(rec)
        datasets.append(
            read_annotated_record(
                base,
                rec,
                subject_id_for_record(int(rec)),
                expected_fs=360,
                beat_length=beat_length,
            )
        )
    return concatenate(datasets)
