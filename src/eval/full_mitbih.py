"""Run the complete, predeclared MIT-BIH protocol and record provenance.

No beat cap or synthetic fallback is permitted. Missing records fail the run.
"""

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from src.data.download import EXCLUDED_RECORDS, MITBIH_RECORDS_FILTERED, download_mitbih
from src.eval.inter_patient import run_inter_patient_rf


def main() -> None:
    started = time.monotonic()
    data_dir = Path("data/raw")
    output_dir = Path("artifacts/full-mitbih")
    print("Downloading complete MIT-BIH protocol records", flush=True)
    download_mitbih(str(data_dir))
    manifest = {}
    for record in MITBIH_RECORDS_FILTERED:
        for suffix in (".dat", ".hea", ".atr"):
            path = data_dir / f"{record}{suffix}"
            if not path.is_file():
                raise RuntimeError(f"Full-data run requires {path}")
            manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    print("Evaluating all beats: seed=42, five-fold training-patient CV", flush=True)
    results = run_inter_patient_rf(
        data_dir=data_dir,
        output_dir=output_dir,
        n_splits=5,
        max_beats_per_record=None,
        allow_synthetic=False,
        seed=42,
    )
    split = json.loads((output_dir / "patient_split.json").read_text())
    included = {int(r) for r in split["train_records"] + split["test_records"]}
    if included != set(MITBIH_RECORDS_FILTERED):
        raise RuntimeError("Evaluation skipped records; refusing a full-data result")
    if set(split["train_subjects"]) & set(split["test_subjects"]):
        raise RuntimeError("Subject overlap in saved split")
    if results["dataset_type"] != "mitbih":
        raise RuntimeError("Expected real MIT-BIH data")
    provenance = {
        "dataset_url": "https://physionet.org/content/mitdb/1.0.0/",
        "included_records": sorted(included),
        "excluded_records": EXCLUDED_RECORDS,
        "max_beats_per_record": None,
        "seed": 42,
        "train_ratio": 0.8,
        "cv_folds": 5,
        "beat_length": 250,
        "channel": 0,
        "sampling_rate_hz": 360,
        "model": {
            "n_estimators": 200,
            "max_depth": None,
            "class_weight": "balanced_subsample",
            "random_state": 42,
        },
        "source_commit": os.getenv("GITHUB_SHA"),
        "workflow_run_id": os.getenv("GITHUB_RUN_ID"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "python": platform.python_version(),
        "packages": {
            package: version(package)
            for package in ("numpy", "scikit-learn", "wfdb", "joblib", "scipy", "pandas")
        },
        "data_sha256": manifest,
    }
    report = {"results": results, "split": split, "provenance": provenance}
    (output_dir / "full_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MITBIH_FULL_REPORT=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
