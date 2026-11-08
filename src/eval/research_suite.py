"""Reproduce morphology/RR ablation and CNN comparison with validation-only selection."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from src.data.download import MITBIH_RECORDS_FILTERED, download_mitbih
from src.data.research import load_mitbih
from src.data.segment import (
    AAMI_CLASSES,
    FEATURE_DIM,
    FEATURE_VERSION,
    normalize_cnn_beats,
)
from src.eval.protocols import (
    class_support,
    fixed_partition,
    nsv_focus,
    shared_subjects,
)
from src.eval.statistics import metrics, subject_bootstrap
from src.models.artifacts import sha256, write_bundle
from src.models.cnn_pytorch import ECG1DCNN


def validate_partition(split, protocol="custom"):

    train = [int(r) for r in split["train_records"]]
    test = [int(r) for r in split["test_records"]]
    if not train or not test or len(set(train + test)) != len(train + test):
        raise ValueError("Split must contain non-empty, unique train/test records")
    expected = (
        set(MITBIH_RECORDS_FILTERED)
        if protocol == "custom"
        else {r for records in fixed_partition(protocol) for r in records}
    )
    if protocol != "custom" and (train, test) != fixed_partition(protocol):
        raise ValueError("Fixed partition must match the declared protocol exactly")
    if set(train + test) != expected:
        raise ValueError("Split must cover the complete declared cohort")
    if shared_subjects(train, test) and protocol != "ds1-ds2":
        raise ValueError("Subject overlap in frozen split")
    return train, test


def train_cnn_epochs(
    windows, labels, epochs, *, validation=None, seed=42, normalizer=normalize_cnn_beats
):
    """Select an epoch using validation subjects only, or train a fixed epoch count."""
    torch.manual_seed(seed)
    model = ECG1DCNN().cpu()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalizer(windows)),
            torch.from_numpy(np.asarray(labels, dtype=np.int64)),
        ),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = np.maximum(np.bincount(labels, minlength=5), 1)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(len(labels) / (5 * counts), dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    selected_epoch, best = epochs, -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for beats, target in loader:
            optimizer.zero_grad()
            loss = criterion(model(beats.unsqueeze(1)), target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
        row = {"epoch": epoch, "training_loss": total_loss / len(labels)}
        if validation is not None:
            proba = cnn_probabilities(model, validation.windows, normalizer=normalizer)
            score = f1_score(
                validation.labels,
                proba.argmax(axis=1),
                labels=range(5),
                average="macro",
                zero_division=0,
            )
            row["validation_macro_f1"] = float(score)
            if score > best:
                best, selected_epoch = score, epoch
        history.append(row)
        print(json.dumps(row), flush=True)
    return model, selected_epoch, history


def cnn_probabilities(model, windows, *, normalizer=normalize_cnn_beats):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(windows), 512):
            beats = torch.from_numpy(normalizer(windows[start : start + 512]))
            result.append(torch.softmax(model(beats.unsqueeze(1)), dim=1).cpu().numpy())
    return np.concatenate(result)


def report_model(y, proba, groups, elapsed, inference_seconds, bootstrap):
    predictions = proba.argmax(axis=1)
    return {
        **metrics(y, predictions),
        "nsv_focus": nsv_focus(y, predictions),
        "macro_f1_interval": subject_bootstrap(y, predictions, groups, bootstrap),
        "training_seconds": elapsed,
        "batch_inference_ms_per_beat": 1000 * inference_seconds / len(y),
        "latency_scope": "Batch model inference including preprocessing; excludes HTTP and disk I/O.",
    }


def aligned_rf_probabilities(model, features):
    result = np.zeros((len(features), 5))
    result[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(features)
    return result


def run(args):
    if args.epochs < 1 or args.bootstrap < 2:
        raise ValueError("epochs >= 1 and bootstrap >= 2 are required")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    output = Path(args.output_dir)
    if (output / "manifest.json").exists():
        raise FileExistsError("Choose a new output directory; completed bundles are immutable")
    output.mkdir(parents=True, exist_ok=True)
    if args.download:
        download_mitbih(args.data_dir)
    protocol_name = getattr(args, "protocol", "custom")
    if protocol_name == "custom":
        split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    else:
        tr, te = fixed_partition(protocol_name)
        split = {"train_records": tr, "test_records": te}
    train_records, test_records = validate_partition(split, protocol_name)
    train = load_mitbih(args.data_dir, train_records)
    test = load_mitbih(args.data_dir, test_records)
    inner_train, validation = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            train.windows, train.labels, train.groups
        )
    )
    fit, val = train.subset(inner_train), train.subset(validation)
    protocol = {
        "dataset": "mitbih",
        "cohort": "custom-45-record" if protocol_name == "custom" else protocol_name,
        "record_split": protocol_name,
        "shared_train_test_subjects": shared_subjects(train_records, test_records),
        "subject_disjoint": not bool(shared_subjects(train_records, test_records)),
        "class_support": {
            name: class_support(data)
            for name, data in [
                ("train", train),
                ("inner_train", fit),
                ("validation", val),
                ("test", test),
            ]
        },
        "seed": 42,
        "train_records": train_records,
        "test_records": test_records,
        "inner_train_subjects": sorted(np.unique(fit.groups).tolist()),
        "validation_subjects": sorted(np.unique(val.groups).tolist()),
        "test_subjects": sorted(np.unique(test.groups).tolist()),
        "n_train": len(train.labels),
        "n_validation": len(val.labels),
        "n_test": len(test.labels),
        "selection": "CNN epoch selected on validation subjects, then refit on all training subjects",
        "max_cnn_epochs": args.epochs,
        "beat_length": 250,
        "sampling_rate_hz": 360,
        "lead_policy": "channel 0, physical mV; not consistently MLII",
        "rr_policy": "three causal features from earlier accepted reference beat locations",
        "source_commit": os.getenv("GITHUB_SHA"),
        "threads": args.threads,
        "data_sha256": {},
    }
    for record in train_records + test_records:
        for suffix in (".dat", ".hea", ".atr"):
            path = Path(args.data_dir) / f"{record}{suffix}"
            if not path.exists():
                path = Path(args.data_dir) / "mitdb" / path.name
            protocol["data_sha256"][path.name] = sha256(path)
    results, probabilities = {}, {}
    for rhythm, name, filename in [
        (False, "RF morphology", "baseline_model.pkl"),
        (True, "RF morphology + RR", "rhythm_model.pkl"),
    ]:
        started = time.perf_counter()
        classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=2,
        )
        classifier.fit(fit.features(rhythm), fit.labels)
        val_pred = classifier.predict(val.features(rhythm))
        validation_score = float(
            f1_score(val.labels, val_pred, labels=range(5), average="macro", zero_division=0)
        )
        classifier.fit(train.features(rhythm), train.labels)
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        proba = aligned_rf_probabilities(classifier, test.features(rhythm))
        inference_seconds = time.perf_counter() - started
        results[name] = report_model(
            test.labels,
            proba,
            test.groups,
            training_seconds,
            inference_seconds,
            args.bootstrap,
        )
        results[name]["validation_macro_f1"] = validation_score
        probabilities[name] = proba
        joblib.dump(classifier, output / filename)
    (output / "baseline_meta.json").write_text(
        json.dumps(
            {
                "feature_version": FEATURE_VERSION,
                "feature_dim": FEATURE_DIM,
                "classes": AAMI_CLASSES,
                "dataset_type": "mitbih",
                "protocol": protocol_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    started = time.perf_counter()
    _, selected_epoch, history = train_cnn_epochs(
        fit.windows, fit.labels, args.epochs, validation=val
    )
    cnn, _, _ = train_cnn_epochs(train.windows, train.labels, selected_epoch)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    proba = cnn_probabilities(cnn, test.windows)
    results["CNN"] = report_model(
        test.labels,
        proba,
        test.groups,
        training_seconds,
        time.perf_counter() - started,
        args.bootstrap,
    )
    results["CNN"]["selected_epoch"] = selected_epoch
    results["CNN"]["validation_history"] = history
    probabilities["CNN"] = proba
    torch.save(cnn.state_dict(), output / "cnn_model.pth")
    samples = []
    # Purposeful per-class correct/error examples; this is not a representative sample.
    for label in range(5):
        indices = np.flatnonzero(test.labels == label)
        for wrong in (True, False):
            chosen = [
                i
                for i in indices
                if any(p[i].argmax() != label for p in probabilities.values()) == wrong
            ][:5]
            for i in chosen:
                samples.append(
                    {
                        "id": f"{test.records[i]}:{test.samples[i]}",
                        "record": str(test.records[i]),
                        "sample": int(test.samples[i]),
                        "reference": AAMI_CLASSES[label],
                        "signal": np.round(test.windows[i], 6).tolist(),
                        "predictions": {
                            name: {
                                "predicted_class": AAMI_CLASSES[int(p[i].argmax())],
                                "probabilities": dict(zip(AAMI_CLASSES, map(float, p[i]))),
                            }
                            for name, p in probabilities.items()
                        },
                    }
                )
    (output / "explorer_samples.json").write_text(
        json.dumps(
            {
                "source": "measured-mitbih",
                "sampling_rate": 360,
                "selection": "Up to five error and five correct examples per reference class.",
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    report = {"protocol": protocol, "models": results, "class_labels": AAMI_CLASSES}
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Measured research comparison",
        "",
        "Five-class AAMI mapping; N/V/S focus is conditional evaluation of the same model.",
        f"Record protocol: {protocol_name}; shared train/test subjects: {shared_subjects(train_records, test_records)}.",
        "",
        "| Model | Five-class macro-F1 | N/V/S focus macro-F1 | 95% patient bootstrap interval (five-class) | Accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, result in results.items():
        ci = result["macro_f1_interval"]
        lines.append(
            f"| {name} | {result['macro_f1']:.4f} | {result['nsv_focus']['macro_f1']:.4f} | "
            f"{ci['lower']:.4f}–{ci['upper']:.4f} | {result['accuracy']:.4f} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_bundle(output, protocol)
    print("RESEARCH_REPORT=" + json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=["custom", "ds1-ds2", "ds1-ds2-disjoint"],
        default="custom",
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--split", default="reports/mitbih/2026-11-05/patient_split.json")
    parser.add_argument("--output-dir", default="artifacts/research")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
