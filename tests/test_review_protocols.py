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
