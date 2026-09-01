"""Generate synthetic ECG beat data for testing and development."""

from pathlib import Path
from typing import Tuple

import numpy as np

from src.data.segment import AAMI_CLASSES


def generate_synthetic_beat(
    class_label: str,
    beat_length: int = 250,
    noise_level: float = 0.1,
    seed: int = None,
) -> np.ndarray:
    """
    Generate a synthetic ECG beat with realistic morphology.

    Different classes have different morphological features:
    - N (Normal): standard QRS-T wave pattern
    - V (VEB): wider QRS, larger amplitude
    - S (SVEB): smaller amplitude, earlier peak
    - F (Fusion): mixture of N and V characteristics
    - Q (Unknown): random pattern

    Args:
        class_label: AAMI class ('N', 'V', 'S', 'F', 'Q').
        beat_length: Length of beat window.
        noise_level: Standard deviation of Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        Synthetic beat window of shape (beat_length,).
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(-1, 1, beat_length)

    if class_label == "N":
        # Normal beat: standard QRS complex
        qrs = 2.0 * np.exp(-30 * t**2)  # QRS
        t_wave = 0.5 * np.exp(-15 * (t - 0.5) ** 2)  # T wave
        beat = qrs + t_wave

    elif class_label == "V":
        # Ventricular beat: wider, larger QRS
        qrs = 3.0 * np.exp(-15 * t**2)  # Wider QRS
        t_wave = -0.3 * np.exp(-10 * (t - 0.6) ** 2)  # Inverted T
        beat = qrs + t_wave

    elif class_label == "S":
        # Supraventricular beat: earlier, smaller
        qrs = 1.5 * np.exp(-40 * (t + 0.3) ** 2)
        t_wave = 0.3 * np.exp(-20 * (t - 0.3) ** 2)
        beat = qrs + t_wave

    elif class_label == "F":
        # Fusion beat: mixture of N and V
        qrs = 2.1 * np.exp(-28 * (t + 0.12) ** 2)
        t_wave = -0.6 * np.exp(-18 * (t - 0.45) ** 2)
        beat = qrs + t_wave

    else:  # 'Q'
        # Unknown: random pattern
        beat = np.random.randn(beat_length) * 0.5

    # Add baseline wander
    baseline = 0.5 * np.sin(2 * np.pi * t / 2)
    beat = beat + baseline

    # Add realistic noise
    beat = beat + np.random.randn(beat_length) * noise_level

    return beat


def generate_synthetic_dataset(
    n_samples_per_class: int = 200,
    beat_length: int = 250,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic ECG dataset with balanced classes.

    Args:
        n_samples_per_class: Number of samples per class.
        beat_length: Length of beat windows.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
    """
    rng = np.random.RandomState(seed)

    all_beats = []
    all_labels = []

    for class_idx, cls_label in enumerate(AAMI_CLASSES):
        class_beats = np.array(
            [
                generate_synthetic_beat(
                    cls_label,
                    beat_length=beat_length,
                    noise_level=0.1,
                    seed=seed + class_idx * 1000 + i,
                )
                for i in range(n_samples_per_class)
            ]
        )

        all_beats.append(class_beats)
        all_labels.extend([class_idx] * n_samples_per_class)

    X = np.vstack(all_beats)
    y = np.array(all_labels)

    # Shuffle
    indices = rng.permutation(len(y))
    X = X[indices]
    y = y[indices]

    # Train/test split
    split_idx = int(len(y) * (1 - test_ratio))

    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def save_synthetic_dataset(
    output_file: str = "data/processed/synthetic_beats.npz",
    n_samples_per_class: int = 200,
    beat_length: int = 250,
) -> None:
    """
    Generate and save a synthetic dataset.

    Args:
        output_file: Path to save the dataset.
        n_samples_per_class: Samples per class.
        beat_length: Beat window length.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic dataset ({n_samples_per_class} samples/class)...")

    X_train, y_train, X_test, y_test = generate_synthetic_dataset(
        n_samples_per_class=n_samples_per_class,
        beat_length=beat_length,
    )

    np.savez(
        output_file,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_labels=AAMI_CLASSES,
    )

    print(f"✓ Synthetic dataset saved to {output_file}")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Test samples: {len(y_test)}")

    # Print class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print("\n  Train class distribution:")
    for cls_idx, count in zip(unique, counts):
        cls_name = AAMI_CLASSES[cls_idx]
        print(f"    {cls_name}: {count}")


if __name__ == "__main__":
    save_synthetic_dataset()
