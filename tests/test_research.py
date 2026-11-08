"""Research regressions: causal features, subject isolation, integrity and explorer."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
from src.data import research
from src.data.research import causal_rr_features
from src.eval.external import nstdb_records
from src.eval.research_suite import validate_partition
from src.eval.statistics import subject_bootstrap
from src.models.artifacts import verify_bundle, write_bundle


def test_rhythm_features_never_look_at_future_beats():
    a = causal_rr_features([0, 360, 720, 1080], 360)
    b = causal_rr_features([0, 360, 720, 1800], 360)
    np.testing.assert_array_equal(a[:3], b[:3])
    np.testing.assert_array_equal(a[0], [0, 0, 0])
    np.testing.assert_array_equal(a[1], [1, 1, 1])


def test_rhythm_features_reject_duplicate_locations():
    with pytest.raises(ValueError):
        causal_rr_features([100, 100], 360)


def test_nstdb_excludes_training_source():
    records = nstdb_records([119])
    assert {source for _, source, _ in records} == {118}
    assert len(records) == 6
    with pytest.raises(ValueError):
        nstdb_records([118, 119])


def test_patient_bootstrap_is_reproducible_and_includes_all_classes():
    labels = np.array([0, 0, 1, 1])
    pred = labels.copy()
    groups = np.array([1, 1, 2, 2])
    a = subject_bootstrap(labels, pred, groups, repetitions=20)
    b = subject_bootstrap(labels, pred, groups, repetitions=20)
    assert a == b
    assert a["upper"] <= 0.4
    with pytest.raises(ValueError):
        subject_bootstrap(labels, pred, [1] * 4)


def test_frozen_partition_rejects_shared_subjects_and_partial_cohort():
    with pytest.raises(ValueError):
        validate_partition({"train_records": [201], "test_records": [202]})


def test_bundle_detects_tampering_before_deserialization(tmp_path):
    model = tmp_path / "baseline_model.pkl"
    model.write_bytes(b"trusted fixture")
    write_bundle(tmp_path, {"train_records": [100]})
    verify_bundle(tmp_path)
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        verify_bundle(tmp_path)


def test_api_rejects_incompatible_bundle(tmp_path, monkeypatch):
    (tmp_path / "baseline_model.pkl").write_bytes(b"must not deserialize")
    write_bundle(tmp_path, {})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["feature_version"] = "old-features"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("ECG_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_models_loaded", False)
    monkeypatch.setattr(api, "_baseline_model", None)
    monkeypatch.setattr(api, "_cnn_model", None)
    with pytest.warns(UserWarning, match="rejected"):
        api._load_models()
    assert api._baseline_model is None


def test_explorer_synthetic_and_measured_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("ECG_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_models_loaded", True)
    with TestClient(api.app) as client:
        assert client.get("/explorer").status_code == 200
        data = client.get("/explorer/data").json()
        assert data["source"] == "synthetic"
        assert len(data["samples"]) == 10
        assert all(not s["predictions"] for s in data["samples"])
        measured = {"source": "measured-mitbih", "samples": [], "sampling_rate": 360}
        (tmp_path / "explorer_samples.json").write_text(json.dumps(measured))
        assert client.get("/explorer/data").json() == measured


def test_external_resampling_aligns_annotations(monkeypatch):
    record = SimpleNamespace(
        sig_name=["II"],
        units=["mV"],
        fs=257,
        p_signal=np.sin(np.arange(1000)[:, None] / 10),
    )
    ann = SimpleNamespace(sample=np.array([257, 514]), symbol=["N", "V"])
    monkeypatch.setattr(research.wfdb, "rdrecord", lambda *a: record)
    monkeypatch.setattr(research.wfdb, "rdann", lambda *a: ann)
    data = research.read_annotated_record("unused", "I01", "patient-1", lead="II")
    assert data.samples.tolist() == [360, 720]
    assert data.windows.shape == (2, 250)
    assert data.labels.tolist() == [0, 1]


def test_outside_annotations_require_explicit_audited_exclusion(monkeypatch):
    record = SimpleNamespace(
        sig_name=["II"],
        units=["mV"],
        fs=257,
        p_signal=np.zeros((1000, 1)),
    )
    ann = SimpleNamespace(sample=np.array([-10, 257, 1001]), symbol=["N", "V", "S"])
    monkeypatch.setattr(research.wfdb, "rdrecord", lambda *a: record)
    monkeypatch.setattr(research.wfdb, "rdann", lambda *a: ann)
    with pytest.raises(ValueError, match="outside"):
        research.read_annotated_record("unused", "I04", "patient-1")
    audit = {}
    data = research.read_annotated_record(
        "unused",
        "I04",
        "patient-1",
        drop_outside=True,
        audit=audit,
    )
    assert data.labels.tolist() == [1]
    assert audit["I04"]["outside_recording_excluded"] == 2
    assert audit["I04"]["excluded_original_samples"] == [-10, 1001]
