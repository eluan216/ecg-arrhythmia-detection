"""PyTorch Dataset class for ECG beats."""

from typing import Tuple

import numpy as np
from torch.utils.data import Dataset


class ECGBeatDataset(Dataset):
    """
    PyTorch Dataset for segmented ECG beats.

    Loads preprocessed beat windows and class labels.
    """

    def __init__(
        self,
        beat_windows: np.ndarray,
        labels: np.ndarray,
        transform=None,
    ):
        """
        Args:
            beat_windows: Array of shape (n_samples, beat_length).
            labels: Array of shape (n_samples,) with integer class indices.
            transform: Optional transform to apply to each sample.
        """
        self.beat_windows = beat_windows
        self.labels = labels
        self.transform = transform

        assert len(beat_windows) == len(labels), "Mismatch between number of beats and labels"

    def __len__(self) -> int:
        return len(self.beat_windows)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        beat = self.beat_windows[idx].astype(np.float32)
        label = self.labels[idx]

        if self.transform:
            beat = self.transform(beat)

        return beat, label


if __name__ == "__main__":
    print("Dataset module. Use via training scripts.")
