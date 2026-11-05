"""Unit tests for inter-patient split integrity (no network)."""

from src.data.download import MITBIH_RECORDS_FILTERED
from src.data.segment import create_inter_patient_split, subject_id_for_record
from src.eval.inter_patient import save_patient_split


def test_split_no_overlap():
    train, test = create_inter_patient_split(MITBIH_RECORDS_FILTERED, seed=42)
    assert set(train).isdisjoint(set(test))
    assert len(train) + len(test) == len(MITBIH_RECORDS_FILTERED)


def test_split_reproducible():
    a = create_inter_patient_split(MITBIH_RECORDS_FILTERED, seed=42)
    b = create_inter_patient_split(MITBIH_RECORDS_FILTERED, seed=42)
    assert a[0] == b[0] and a[1] == b[1]


def test_save_patient_split(tmp_path):
    train, test = create_inter_patient_split(MITBIH_RECORDS_FILTERED, seed=7)
    path = save_patient_split(
        tmp_path,
        [subject_id_for_record(r) for r in train],
        [subject_id_for_record(r) for r in test],
        train,
        test,
    )
    assert path.exists()
    text = path.read_text()
    assert "train_subjects" in text
    assert "inter-patient" in text
