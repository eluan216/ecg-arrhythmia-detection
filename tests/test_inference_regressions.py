"""Regression coverage for model metadata and the evaluation data path."""

import json

import numpy as np
import pytest
from sklearn.base import clone

from src.data.segment import AAMI_CLASSES, FEATURE_DIM
from src.eval import inter_patient
from src.models.baseline_sklearn import ECGBaselineRF


def test_baseline_sklearn_contract():
    model = ECGBaselineRF(n_estimators=3, random_state=7)
    cloned = clone(model)
    cloned.set_params(n_estimators=2)
    cloned.fit(np.zeros((4, FEATURE_DIM)), [1, 2, 1, 2])
    assert cloned.model.n_estimators == 2
    np.testing.assert_array_equal(cloned.classes_, [1, 2])
    assert cloned.n_features_in_ == FEATURE_DIM


def test_patient_loader_extracts_features_and_maps_labels(tmp_path, monkeypatch):
    calls = []

    def load_record(record_id, record_dir):
        calls.append((record_id, record_dir))
        return np.ones((3, 250)), np.array([0, 2, 1]), []

    monkeypatch.setattr(inter_patient, "load_mitbih_record", load_record)
    X, y, groups, records = inter_patient.load_patient_beats(
        tmp_path, records=["201", "202"], max_beats_per_record=2
    )
    assert X.shape == (4, FEATURE_DIM)
    assert y.tolist() == ["N", "S", "N", "S"]
    assert len(set(groups)) == 1
    assert records == ["201", "201", "202", "202"]
    assert calls == [(201, str(tmp_path)), (202, str(tmp_path))]


def test_macro_f1_includes_all_declared_classes():
    metrics = inter_patient._metrics_block(np.array(["N"]), np.array(["N"]))
    assert metrics["macro_f1"] == pytest.approx(1 / len(AAMI_CLASSES))
    assert len(metrics["confusion_matrix"]) == len(AAMI_CLASSES)


def test_synthetic_evaluation_is_subject_disjoint(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("test fixture: no dataset")

    monkeypatch.setattr(inter_patient, "load_patient_beats", unavailable)
    result = inter_patient.run_inter_patient_rf(
        tmp_path / "missing", tmp_path / "output", n_splits=2, allow_synthetic=True, seed=7
    )
    assert result["dataset_type"] == "synthetic"
    assert result["feature_dim"] == FEATURE_DIM
    split = json.loads((tmp_path / "output" / "patient_split.json").read_text())
    assert set(split["train_subjects"]).isdisjoint(split["test_subjects"])
    assert result["n_train"] + result["n_test"] == 500
    assert (tmp_path / "output" / "baseline_model.pkl").exists()


def test_real_data_required_by_default(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("test fixture: no dataset")

    monkeypatch.setattr(inter_patient, "load_patient_beats", unavailable)
    with pytest.raises(RuntimeError, match="allow-synthetic"):
        inter_patient.run_inter_patient_rf(tmp_path, tmp_path / "out")


@pytest.mark.parametrize("n_splits,max_beats", [(1, None), (2, 0)])
def test_invalid_evaluation_options(tmp_path, n_splits, max_beats):
    with pytest.raises(ValueError):
        inter_patient.run_inter_patient_rf(
            tmp_path, tmp_path / "out", n_splits=n_splits, max_beats_per_record=max_beats
        )
