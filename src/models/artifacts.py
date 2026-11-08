"""Versioned, checksummed research bundles. Checksums do not establish trust."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from src.data.segment import AAMI_CLASSES, FEATURE_VERSION


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_bundle(directory, protocol):
    directory = Path(directory)
    packages = {
        name: version(name)
        for name in ("numpy", "scipy", "scikit-learn", "torch", "wfdb", "joblib")
    }
    manifest = {
        "schema_version": 1,
        "classes": AAMI_CLASSES,
        "feature_version": FEATURE_VERSION,
        "cnn_preprocessing": "per-beat-zscore-v1",
        "sampling_rate_hz": 360,
        "beat_length": 250,
        "python": platform.python_version(),
        "packages": packages,
        "protocol": protocol,
        "files": {
            p.name: sha256(p)
            for p in sorted(directory.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (directory / "requirements-lock.txt").write_text(
        "\n".join(f"{name}=={value}" for name, value in packages.items()) + "\n", encoding="utf-8"
    )
    # Include the environment lock itself in integrity checks.
    manifest["files"]["requirements-lock.txt"] = sha256(directory / "requirements-lock.txt")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_bundle(directory, check_environment=True):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["classes"] != AAMI_CLASSES:
        raise ValueError("Incompatible model bundle schema or class order")
    if manifest["feature_version"] != FEATURE_VERSION:
        raise ValueError("Incompatible morphology feature version")
    if manifest["cnn_preprocessing"] != "per-beat-zscore-v1":
        raise ValueError("Incompatible CNN preprocessing")
    if manifest["sampling_rate_hz"] != 360 or manifest["beat_length"] != 250:
        raise ValueError("Incompatible sampling or window contract")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha256(directory / name) != digest:
            raise ValueError(f"Artifact integrity check failed: {name}")
    if check_environment:
        for package in ("numpy", "scikit-learn", "torch"):
            if version(package) != manifest["packages"][package]:
                raise ValueError(f"Environment mismatch: {package}; use requirements-lock.txt")
    return manifest
