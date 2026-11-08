"""Evaluate frozen research bundles on INCART or NSTDB without fitting or tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import torch
import wfdb

from src.data.research import concatenate, read_annotated_record
from src.data.segment import subject_id_for_record
from src.eval.research_suite import aligned_rf_probabilities, cnn_probabilities
from src.eval.statistics import metrics
from src.models.artifacts import sha256, verify_bundle
from src.models.cnn_pytorch import ECG1DCNN


def nstdb_records(training_records):
    """Never score a noisy derivative whose clean source appeared in training."""
    trained = {subject_id_for_record(int(r)) for r in training_records}
    bases = [r for r in (118, 119) if subject_id_for_record(r) not in trained]
    if not bases:
        raise ValueError("Both NSTDB clean sources overlap training subjects")
    return [
        (f"{base}e{suffix}", base, snr)
        for base in bases
        for suffix, snr in [("24", 24), ("18", 18), ("12", 12), ("06", 6), ("00", 0), ("_6", -6)]
    ]


def ensure_files(dataset, directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    files = [
        f"{record}{suffix}"
        for record in records
        for suffix in (".dat", ".hea", ".atr")
        if not (directory / f"{record}{suffix}").exists()
    ]
    if files:
        wfdb.dl_files(dataset, dl_dir=str(directory), files=files)


def run(args):
    bundle = Path(args.bundle)
    manifest = verify_bundle(bundle)
    torch.set_num_threads(2)
    baseline = joblib.load(bundle / "baseline_model.pkl")
    cnn = ECG1DCNN()
    cnn.load_state_dict(torch.load(bundle / "cnn_model.pth", map_location="cpu", weights_only=True))
    root = Path(args.data_dir)
    datasets, report = {}, {}
    if args.dataset == "nstdb":
        records = nstdb_records(manifest["protocol"]["train_records"])
        if args.download:
            ensure_files("nstdb", root, [r for r, _, _ in records])
        for record, source, snr in records:
            datasets.setdefault(snr, []).append(
                read_annotated_record(root / record, record, source, noisy_only=True)
            )
        report["scope"] = (
            "Noise-on beat centers only, 5 minutes onward, alternating 2-minute intervals. "
            "Clean sources overlapping training are excluded. Very few source subjects; "
            "no population-level confidence interval is claimed."
        )
        report["records"] = [r for r, _, _ in records]
    else:
        records = [f"I{i:02d}" for i in range(1, 76)]
        if args.download:
            ensure_files("incartdb", root, records)
        datasets["all"] = [read_annotated_record(root / r, r, r, lead="II") for r in records]
        report["scope"] = (
            "All 75 INCART records, lead II in mV, polyphase-resampled 257 to 360 Hz. "
            "Reference locations are rounded after resampling; labels use the shared whitelist. "
            "Frozen MIT-BIH models; no fitting. Repeated records share patients: "
            "pooled beat metrics only, no record-as-patient bootstrap."
        )
        report["records"] = records
    report.update(
        dataset=args.dataset,
        bundle_manifest_sha256=sha256(bundle / "manifest.json"),
        evaluations={},
        source_hashes={},
    )
    for key, pieces in datasets.items():
        data = concatenate(pieces)
        predictions = {
            "RF morphology": aligned_rf_probabilities(baseline, data.features()).argmax(axis=1),
            "CNN": cnn_probabilities(cnn, data.windows).argmax(axis=1),
        }
        report["evaluations"][str(key)] = {
            name: metrics(data.labels, pred) for name, pred in predictions.items()
        }
    for record in report["records"]:
        for suffix in (".dat", ".hea", ".atr"):
            path = root / f"{record}{suffix}"
            report["source_hashes"][path.name] = sha256(path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("EXTERNAL_REPORT=" + json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["nstdb", "incartdb"], required=True)
    parser.add_argument("--bundle", default="artifacts/research")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--download", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
