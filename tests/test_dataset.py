"""Test suite for PyTorch ECG dataset and data utilities."""

import importlib.util

import numpy as np
import pytest

try:
    TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from torch.utils.data import DataLoader

    from src.data.dataset import ECGBeatDataset

from src.data.synthetic import AAMI_CLASSES, generate_synthetic_beat, generate_synthetic_dataset


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_beat_normal(self):
        """Test generating a normal beat."""
        beat = generate_synthetic_beat("N", beat_length=256)
        assert len(beat) == 256
        assert isinstance(beat, np.ndarray)

    def test_generate_beat_all_classes(self):
        """Test generating beats for all AAMI classes."""
        for cls in AAMI_CLASSES:
            beat = generate_synthetic_beat(cls, beat_length=256)
            assert len(beat) == 256
            assert not np.any(np.isnan(beat))

    def test_beat_reproducibility(self):
        """Test that same seed produces same beat."""
        beat1 = generate_synthetic_beat("N", seed=42)
        beat2 = generate_synthetic_beat("N", seed=42)
        np.testing.assert_array_almost_equal(beat1, beat2)

    def test_beat_class_differences(self):
        """Test that different classes produce different beats."""
        beats = {cls: generate_synthetic_beat(cls, seed=42) for cls in AAMI_CLASSES}

        # Check that beats are reasonably different
        for cls1 in AAMI_CLASSES:
            for cls2 in AAMI_CLASSES:
                if cls1 != cls2:
                    # Calculate correlation
                    corr = np.corrcoef(beats[cls1], beats[cls2])[0, 1]
                    # Different classes should have lower correlation
                    assert corr < 0.9, f"Classes {cls1} and {cls2} too similar"

    def test_generate_dataset(self):
        """Test generating synthetic dataset."""
        X_train, y_train, X_test, y_test = generate_synthetic_dataset(
            n_samples_per_class=50,
            beat_length=256,
            test_ratio=0.2,
        )

        # Check shapes
        assert X_train.shape[0] + X_test.shape[0] == 250  # 5 classes * 50 samples
        assert X_train.shape[1] == 256
        assert X_test.shape[1] == 256
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

        # Check class distribution is balanced
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        assert len(unique_train) == 5  # All 5 classes present

    def test_dataset_no_nans(self):
        """Test that synthetic dataset contains no NaNs."""
        X_train, y_train, X_test, y_test = generate_synthetic_dataset(n_samples_per_class=50)

        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isnan(X_test))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestECGBeatDataset:
    """Tests for ECG beat PyTorch dataset."""

    def test_dataset_creation(self):
        """Test basic dataset creation."""
        beat_windows = np.random.randn(100, 256).astype(np.float32)
        labels = np.random.randint(0, 5, 100)

        dataset = ECGBeatDataset(beat_windows, labels)

        assert len(dataset) == 100

    def test_dataset_getitem(self):
        """Test getting a single item from dataset."""
        beat_windows = np.random.randn(10, 256).astype(np.float32)
        labels = np.arange(10)

        dataset = ECGBeatDataset(beat_windows, labels)
        beat, label = dataset[0]

        assert beat.shape == (256,)
        assert isinstance(beat, np.ndarray)
        assert label == 0

    def test_dataset_length_mismatch(self):
        """Test that mismatched lengths raise an error."""
        beat_windows = np.random.randn(100, 256)
        labels = np.arange(50)  # Wrong size

        with pytest.raises(AssertionError):
            ECGBeatDataset(beat_windows, labels)

    def test_dataloader_batch(self):
        """Test DataLoader with batching."""
        beat_windows = np.random.randn(50, 256).astype(np.float32)
        labels = np.random.randint(0, 5, 50)

        dataset = ECGBeatDataset(beat_windows, labels)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Get one batch
        beats_batch, labels_batch = next(iter(loader))

        assert beats_batch.shape[0] == 16
        assert beats_batch.shape[1] == 256
        assert len(labels_batch) == 16

    def test_dataset_with_synthetic_data(self):
        """Test dataset using synthetic data."""
        X_train, y_train, _, _ = generate_synthetic_dataset(
            n_samples_per_class=20,
            beat_length=256,
        )

        dataset = ECGBeatDataset(X_train, y_train)

        assert len(dataset) == len(y_train)

        # Check we can iterate
        for i in range(min(5, len(dataset))):
            beat, label = dataset[i]
            assert beat.shape == (256,)
            assert label in range(5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
