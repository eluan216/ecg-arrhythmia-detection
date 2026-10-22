"""Subject-disjoint splits (201/202 stay together)."""

from src.data.download import MITBIH_RECORDS_FILTERED
from src.data.segment import (
    create_inter_patient_split,
    subject_id_for_record,
)


def test_201_202_same_subject():
    assert subject_id_for_record(201) == subject_id_for_record(202)


def test_split_keeps_201_202_together():
    ids = sorted(set(MITBIH_RECORDS_FILTERED))
    train, test = create_inter_patient_split(ids, seed=42)
    train_set, test_set = set(train), set(test)
    if 201 in train_set or 202 in train_set:
        assert 201 in train_set and 202 in train_set
        assert 201 not in test_set and 202 not in test_set
    if 201 in test_set or 202 in test_set:
        assert 201 in test_set and 202 in test_set
        assert 201 not in train_set and 202 not in train_set


def test_no_subject_overlap():
    train, test = create_inter_patient_split(MITBIH_RECORDS_FILTERED, seed=0)
    train_subj = {subject_id_for_record(r) for r in train}
    test_subj = {subject_id_for_record(r) for r in test}
    assert train_subj.isdisjoint(test_subj)
