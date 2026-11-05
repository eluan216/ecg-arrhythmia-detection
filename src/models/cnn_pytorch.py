"""PyTorch 1D-CNN model for ECG beat classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class ECG1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for ECG beat classification.

    Architecture:
    - Input: Raw beat windows (1 channel, ~250-360 samples)
    - 3 convolutional blocks with BatchNorm, ReLU, MaxPool
    - AdaptiveAvgPool1d so any reasonable beat length works
    - Fully connected head for 5 AAMI classes (N, V, S, F, Q)

    Designed to run on CPU; shallow and compact.
    """

    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),  # (B, 128, 1) regardless of input length
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 1, beat_length)
        Returns:
            logits (batch_size, num_classes)
        """
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    print("CNN model module. Use via training scripts.")
    model = ECG1DCNN()
    for length in (250, 256, 360):
        x = torch.randn(2, 1, length)
        out = model(x)
        print(f"Input shape: {tuple(x.shape)}, Output shape: {tuple(out.shape)}")
