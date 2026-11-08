"""Regression tests for evaluation isolation and CNN preprocessing parity."""

import numpy as np
import pytest

import api
from src.data.segment import create_inter_patient_split, normalize_cnn_beats
from src.eval import inter_patient
from src.train import train_cnn


def test_missing_record_fails_instead_of_changing_cohort(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("missing record")

    monkeypatch.setattr(inter_patient, "load_mitbih_record", missing)
    with pytest.raises(RuntimeError, match="required record 100"):
        inter_patient.load_patient_beats(tmp_path, records=["100"])


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.1, float("nan"), 0.01])
def test_invalid_split_ratio(ratio):
    with pytest.raises(ValueError):
        create_inter_patient_split([100, 101], train_ratio=ratio)


def test_same_subject_cannot_form_two_partitions():
    with pytest.raises(ValueError, match="two distinct subjects"):
        create_inter_patient_split([201, 202])


def test_cnn_normalization_preserves_length_and_handles_constant_beats():
    beats = np.stack([np.arange(250), np.ones(250)]).astype(np.float32)
    normalized = normalize_cnn_beats(beats)
    assert normalized.shape == beats.shape
    np.testing.assert_allclose(normalized[0].mean(), 0, atol=1e-6)
    np.testing.assert_allclose(normalized[0].std(), 1, atol=1e-6)
    np.testing.assert_array_equal(normalized[1], np.zeros(250))
    np.testing.assert_array_equal(normalize_cnn_beats(beats[0]), normalized[0])


def test_cnn_api_receives_same_samples_as_training(monkeypatch):
    torch = pytest.importorskip("torch")
    captured = []

    class CaptureModel:
        def __call__(self, tensor):
            captured.append(tensor.cpu().numpy())
            return torch.zeros((1, 5))

    monkeypatch.setattr(api, "_cnn_model", CaptureModel())
    monkeypatch.setattr(api, "_device", "cpu")
    beat = np.linspace(-2, 4, 250)
    result = api._predict_cnn(beat)
    np.testing.assert_array_equal(captured[0][0, 0], normalize_cnn_beats(beat[None, :])[0])
    assert len(result.probabilities) == 5


def test_cnn_uses_fixed_epochs_without_test_checkpoint_selection(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    evaluations = []

    class TinyModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.head = torch.nn.Linear(1, 5)

        def forward(self, x):
            if not self.training:
                evaluations.append(len(x))
            return self.head(x.mean(dim=-1))

    rng = np.random.default_rng(42)
    train = rng.normal(size=(10, 250)).astype(np.float32)
    test = rng.normal(size=(4, 250)).astype(np.float32)
    monkeypatch.setattr(train_cnn, "ECG1DCNN", TinyModel)
    monkeypatch.setattr(
        train_cnn,
        "load_or_generate_data",
        lambda **kwargs: (train, np.arange(10) % 5, test, np.zeros(4, dtype=int)),
    )
    metrics = train_cnn.train_cnn(output_dir=str(tmp_path), num_epochs=6, batch_size=10)
    assert evaluations == [4]
    assert metrics["checkpoint_selection"] == "fixed-final-epoch"
    assert metrics["macro_f1"] <= 0.2
