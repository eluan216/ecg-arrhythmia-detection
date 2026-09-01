"""PyTorch 1D-CNN model for ECG beat classification."""

import torch
import torch.nn as nn


class ECG1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for ECG beat classification.

    Architecture:
    - Input: Raw beat windows (1 channel, ~250-360 samples)
    - 2-4 convolutional layers with ReLU
    - Max pooling for down-sampling
    - Fully connected layers for classification
    - Output: 5 classes (N, V, S, F, Q) per AAMI EC57

    Designed to run on CPU, shallow and compact.
    """

    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.5):
        """
        Args:
            num_classes: Number of output classes (default 5 for AAMI).
            dropout_rate: Dropout probability for regularization.
        """
        super(ECG1DCNN, self).__init__()

        # Convolutional blocks
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        # Assuming input length ~256 samples: 256 -> 128 -> 64 -> 32
        self.fc1 = nn.Linear(128 * 32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, beat_length).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)

        # Flatten for dense layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x


if __name__ == "__main__":
    print("CNN model module. Use via training scripts.")
    # Quick test
    model = ECG1DCNN()
    x = torch.randn(2, 1, 256)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
