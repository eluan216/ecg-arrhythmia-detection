"""Protect published protocol identity and honest minority-class accounting."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.eval.protocols import DS1, DS2, class_support, fixed_partition, nsv_focus
from src.eval.research_suite import validate_partition


def test_published_split_is_exact_and_overlap_is_explicit():
    assert len(DS1) == len(DS2) == 22
    assert not set(DS1) & set(DS2)
    assert set(DS1 + DS2).isdisjoint({102, 104, 107, 217})
    split = dict(zip(("train_records", "test_records"), fixed_partition("ds1-ds2")))
    assert validate_partition(split, "ds1-ds2") == (DS1, DS2)
    with pytest.raises(ValueError):
        validate_partition(split)  # Cannot pass as our strict custom protocol.


def test_disjoint_variant_preserves_ds2():
    train, test = fixed_partition("ds1-ds2-disjoint")
    assert train == [r for r in DS1 if r != 201]
    assert test == DS2
    assert validate_partition(
        {"train_records": train, "test_records": test}, "ds1-ds2-disjoint"
    ) == (train, test)


def test_cannot_smuggle_changed_records_into_published_protocol():
    train, test = fixed_partition("ds1-ds2")
    train[0], test[0] = test[0], train[0]
    with pytest.raises(ValueError, match="exactly"):
        validate_partition({"train_records": train, "test_records": test}, "ds1-ds2")


def test_focus_keeps_fq_predictions_as_errors():
    result = nsv_focus([0, 1, 2, 3, 4], [0, 1, 4, 3, 4])
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["excluded_reference_F_Q"] == 2
    assert result["included_beats"] == 3
    assert result["confusion_matrix"][2][4] == 1


def test_support_counts_independent_subjects_and_absent_classes():
    data = SimpleNamespace(labels=np.array([0, 0, 2]), groups=np.array(["a", "a", "b"]))
    support = class_support(data)
    assert support["N"] == {"beats": 2, "subjects": 1}
    assert support["S"] == {"beats": 1, "subjects": 1}
    assert support["Q"] == {"beats": 0, "subjects": 0}


def test_window_loader_preserves_annotation_center(monkeypatch):
    from src.data import research

    record = SimpleNamespace(
        sig_name=["MLII"],
        units=["mV"],
        fs=360,
        p_signal=np.arange(1000, dtype=float)[:, None],
    )
    annotation = SimpleNamespace(sample=np.array([300, 600]), symbol=["N", "V"])
    monkeypatch.setattr(research.wfdb, "rdrecord", lambda *a: record)
    monkeypatch.setattr(research.wfdb, "rdann", lambda *a: annotation)
    wide = research.read_annotated_record("unused", 100, "patient", beat_length=500)
    narrow = research.read_annotated_record("unused", 100, "patient")
    np.testing.assert_array_equal(wide.windows[:, 125:375], narrow.windows)
    np.testing.assert_array_equal(wide.samples, narrow.samples)
    np.testing.assert_array_equal(wide.rr, narrow.rr)


def test_diagnostic_reads_only_outer_training_records(tmp_path, monkeypatch):
    import json

    from src.data.research import BeatDataset
    from src.eval import cnn_diagnostics as diag

    # Simulate the loader and trainer; inspect data access and all six arms.
    reads, arms = [], []
    dataset = BeatDataset(
        np.ones((20, 500), dtype=np.float32),
        np.arange(20) % 5,
        np.repeat(np.arange(10).astype(str), 2),
        np.repeat("100", 20),
        np.arange(20),
        np.zeros((20, 3)),
    )
    monkeypatch.setattr(diag, "validate_partition", lambda split: ([100, 101], [200]))

    def load(data_dir, records, *, beat_length):
        reads.append(records)
        assert beat_length == 500
        return dataset

    def train(windows, labels, epochs, *, validation, normalizer):
        arms.append((windows.shape[1], normalizer.__name__))
        assert windows.shape[1] == validation.windows.shape[1]
        return None, 1, [{"epoch": 1, "validation_macro_f1": 0.1}]

    monkeypatch.setattr(diag, "load_mitbih", load)
    monkeypatch.setattr(diag, "train_cnn_epochs", train)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({}))
    report = diag.run(
        SimpleNamespace(
            epochs=1,
            threads=1,
            output=str(tmp_path / "diagnostic.json"),
            split=str(split),
            data_dir="unused",
        )
    )
    assert reads == [[100, 101]]
    assert len(set(arms)) == 6
    assert not set(report["fit_subjects"]) & set(report["validation_subjects"])
    assert len(report["arms"]) == 6
