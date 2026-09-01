"""Segmentation, alignment, and feature extraction for ECG beats."""

import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False


def segment_beat_around_rpeak(
    signal: np.ndarray,
    rpeak_index: int,
    beat_length: int = 250,
) -> np.ndarray:
    """
    Extract a beat window centered on the detected R-peak.
    
    Args:
        signal: Raw ECG signal.
        rpeak_index: Index of the R-peak in the signal.
        beat_length: Length of the output beat window (samples).
    
    Returns:
        Beat window of shape (beat_length,).
        Padded with zeros if near signal boundaries.
    """
    half_beat = beat_length // 2
    start = rpeak_index - half_beat
    end = start + beat_length
    
    # Pad if necessary
    if start < 0:
        pad_left = -start
        start = 0
    else:
        pad_left = 0
    
    if end > len(signal):
        pad_right = end - len(signal)
        end = len(signal)
    else:
        pad_right = 0
    
    beat = signal[start:end]
    if pad_left > 0 or pad_right > 0:
        beat = np.pad(beat, (pad_left, pad_right), mode='constant')
    
    return beat


def extract_morphology_features(beat_window: np.ndarray) -> np.ndarray:
    """
    Extract handcrafted morphology features from a beat window.
    
    Features (15 total):
    - Peak value (max absolute)
    - Peak location (normalized)
    - RMS (root mean square)
    - Min and max values
    - Standard deviation
    - Mean value
    - Mean absolute slope (first derivative)
    - Max absolute slope
    - Std of slope
    - Zero crossing rate
    - Kurtosis
    - Skewness
    
    Args:
        beat_window: 1D array of the beat signal.
    
    Returns:
        Feature vector of shape (15,).
    """
    features = []
    
    # 1. Peak features
    peak_val = np.max(np.abs(beat_window))
    peak_idx = np.argmax(np.abs(beat_window))
    features.extend([peak_val, peak_idx / len(beat_window)])
    
    # 2. RMS and amplitude statistics
    rms = np.sqrt(np.mean(beat_window ** 2))
    features.append(rms)
    features.append(np.min(beat_window))
    features.append(np.max(beat_window))
    features.append(np.std(beat_window))
    
    # 3. Mean value
    features.append(np.mean(beat_window))
    
    # 4. Slope features (first derivative)
    slope = np.diff(beat_window)
    features.append(np.mean(np.abs(slope)))
    features.append(np.max(np.abs(slope)))
    features.append(np.std(slope))
    
    # 5. Zero crossing rate
    zero_crossings = np.where(np.diff(np.sign(beat_window)))[0]
    features.append(len(zero_crossings) / len(beat_window))
    
    # 6. Kurtosis and skewness
    mean = np.mean(beat_window)
    std = np.std(beat_window)
    if std > 0:
        kurtosis = np.mean(((beat_window - mean) / std) ** 4) - 3
        skewness = np.mean(((beat_window - mean) / std) ** 3)
    else:
        kurtosis = 0
        skewness = 0
    features.extend([kurtosis, skewness])
    
    return np.array(features)


# AAMI EC57 class mapping from MIT-BIH annotation symbols
AAMI_CLASS_MAP = {
    # Normal
    'N': 'N',  # Normal beat
    'L': 'N',  # Left main bundle branch block beat
    'R': 'N',  # Right main bundle branch block beat
    
    # Ventricular Ectopic Beat (VEB)
    'V': 'V',  # Premature ventricular contraction
    '[': 'V',  # Start of ventricular flutter wave
    ']': 'V',  # End of ventricular flutter wave
    
    # Supraventricular Ectopic Beat (SVEB)
    'A': 'S',  # Atrial premature beat
    'a': 'S',  # Aberrated atrial premature beat
    'J': 'S',  # Nodal (junctional) premature beat
    'S': 'S',  # Supraventricular premature beat
    
    # Fusion
    'F': 'F',  # Fusion of ventricular and normal beat
    
    # Unknown/Paced
    '/': 'Q',  # Paced beat
    'Q': 'Q',  # Unclassifiable beat
    'e': 'Q',  # Ventricular escape beat
    '|': 'Q',  # Isolated QRS-like artifact
}

AAMI_CLASSES = ['N', 'V', 'S', 'F', 'Q']
AAMI_CLASS_INDEX = {cls: idx for idx, cls in enumerate(AAMI_CLASSES)}


def map_to_aami_class(mitbih_symbol: str) -> str:
    """
    Map MIT-BIH annotation symbol to AAMI EC57 class.
    
    Args:
        mitbih_symbol: Single-character MIT-BIH beat annotation.
    
    Returns:
        AAMI class code ('N', 'V', 'S', 'F', or 'Q').
    """
    return AAMI_CLASS_MAP.get(mitbih_symbol, 'Q')


def load_mitbih_record(
    record_id: int,
    record_dir: str = "data/raw",
    beat_length: int = 250,
    channel: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a MIT-BIH record and extract segmented beats with labels.
    
    Args:
        record_id: MIT-BIH record number (100-234).
        record_dir: Directory containing downloaded records.
        beat_length: Length of beat window in samples.
        channel: Which lead to use (0 or 1).
    
    Returns:
        Tuple of (beat_windows, labels, class_distribution)
        - beat_windows: Array of shape (n_beats, beat_length)
        - labels: Array of shape (n_beats,) with integer class indices
        - class_distribution: String describing class balance
    
    Raises:
        ImportError: If wfdb not available.
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb library required for loading MIT-BIH records")
    
    record_path = Path(record_dir)
    
    # Load record and annotations
    record = wfdb.rdrecord(
        f"mitdb/{record_id}",
        cache_dir=str(record_path)
    )
    ann = wfdb.rdann(
        f"mitdb/{record_id}",
        'atr',
        cache_dir=str(record_path)
    )
    
    signal = record.p_signal[:, channel]
    rpeak_indices = ann.sample
    beat_symbols = ann.symbol
    
    beat_windows = []
    labels = []
    class_counts = {cls: 0 for cls in AAMI_CLASSES}
    
    for rpeak_idx, symbol in zip(rpeak_indices, beat_symbols):
        # Skip non-beat annotations
        if symbol in ['(', ')', 'p', 't', 'u', '`']:
            continue
        
        # Segment beat
        beat = segment_beat_around_rpeak(signal, rpeak_idx, beat_length)
        
        # Map to AAMI class
        aami_class = map_to_aami_class(symbol)
        class_idx = AAMI_CLASS_INDEX[aami_class]
        
        beat_windows.append(beat)
        labels.append(class_idx)
        class_counts[aami_class] += 1
    
    beat_windows = np.array(beat_windows)
    labels = np.array(labels)
    
    # Format class distribution string
    total = len(labels)
    dist = " | ".join([
        f"{cls}={count} ({100*count/total:.1f}%)"
        for cls, count in class_counts.items()
    ])
    
    return beat_windows, labels, [dist]


def create_inter_patient_split(
    record_ids: List[int],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Create an inter-patient train/test split.
    
    Args:
        record_ids: List of all record IDs.
        train_ratio: Fraction of patients for training (0-1).
        seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (train_record_ids, test_record_ids).
    """
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(record_ids)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx].tolist(), shuffled[split_idx:].tolist()


if __name__ == "__main__":
    print("Segmentation module. Run via src.train modules.")
