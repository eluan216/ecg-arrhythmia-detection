"""Validation-only normalization/window ablation. Never reads outer test records."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from src.data.research import load_mitbih
from src.data.segment import normalize_cnn_beats
from src.eval.protocols import class_support
from src.eval.research_suite import train_cnn_epochs, validate_partition


def physical_mv(beats):
    return np.asarray(beats, dtype=np.float32)


def run(args):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Diagnostics are immutable; choose a new output")
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_records, _ = validate_partition(split)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    # Read outer TRAIN records only. All arms use identical locations and subjects.
    data = load_mitbih(args.data_dir, train_records, beat_length=500)
    fit_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            data.windows, data.labels, data.groups
        )
    )
    rows = []
    for length in (250, 360, 500):
        for normalization, normalizer in (
            ("per-beat-zscore", normalize_cnn_beats),
            ("physical-mV", physical_mv),
        ):
            # Center on the same reference location, including for odd/even windows.
            start = 250 - length // 2
            fit, val = data.subset(fit_idx), data.subset(val_idx)
            fit.windows = fit.windows[:, start : start + length]
            val.windows = val.windows[:, start : start + length]
            _, selected, history = train_cnn_epochs(
                fit.windows,
                fit.labels,
                args.epochs,
                validation=val,
                normalizer=normalizer,
            )
            rows.append(
                {
                    "beat_length": length,
                    "normalization": normalization,
                    "selected_epoch": selected,
                    "validation_history": history,
                    "best_validation_macro_f1": max(
                        r["validation_macro_f1"] for r in history
                    ),
                }
            )
    report = {
        "scope": "Training-subject validation only; no outer test evaluation or exported model",
        "seed": 42,
        "max_epochs": args.epochs,
        "train_records": train_records,
        "fit_subjects": sorted(np.unique(data.groups[fit_idx]).tolist()),
        "validation_subjects": sorted(np.unique(data.groups[val_idx]).tolist()),
        "class_support": {
            "fit": class_support(data.subset(fit_idx)),
            "validation": class_support(data.subset(val_idx)),
        },
        "arms": rows,
        "interpretation": "Controlled hypotheses, not proof of the cause of the CNN deficit. "
        "RF uses trees and does not require feature standardization. "
        "Existing CNN already uses per-beat z-score normalization. "
        "Any selected change requires a fresh, explicitly documented evaluation.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--split", default="reports/mitbih/2026-11-05/patient_split.json"
    )
    parser.add_argument("--output", default="artifacts/cnn-diagnostics.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
