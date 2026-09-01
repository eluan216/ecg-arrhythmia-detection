"""Test suite for ECG segmentation and data processing."""

import pytest
import numpy as np
from src.data.segment import (
    segment_beat_around_rpeak,
    extract_morphology_features,
    map_to_aami_class,
    create_inter_patient_split,
    AAMI_CLASS_MAP,
    AAMI_CLASSES,
)


class TestSegmentation:
    """Tests for beat segmentation functions."""
    
    def test_segment_beat_around_rpeak_center(self):
        """Test beat segmentation when R-peak is at center."""
        signal = np.arange(100, dtype=float)
        rpeak_index = 50
        beat = segment_beat_around_rpeak(signal, rpeak_index, beat_length=20)
        
        assert len(beat) == 20
        assert beat[10] == 50  # Center should be rpeak_index
    
    def test_segment_beat_padding_left(self):
        """Test beat segmentation with padding at the left edge."""
        signal = np.arange(100, dtype=float)
        rpeak_index = 5
        beat = segment_beat_around_rpeak(signal, rpeak_index, beat_length=20)
        
        assert len(beat) == 20
        # Should be padded on the left
        assert beat[0] == 0  # Padded with zeros
    
    def test_segment_beat_padding_right(self):
        """Test beat segmentation with padding at the right edge."""
        signal = np.arange(100, dtype=float)
        rpeak_index = 95
        beat = segment_beat_around_rpeak(signal, rpeak_index, beat_length=20)
        
        assert len(beat) == 20
        # Should be padded on the right
        assert beat[-1] == 0  # Padded with zeros
    
    def test_segment_beat_exact_window(self):
        """Test beat segmentation with exact window without padding."""
        signal = np.arange(100, dtype=float)
        rpeak_index = 50
        beat = segment_beat_around_rpeak(signal, rpeak_index, beat_length=10)
        
        assert len(beat) == 10
        # Center element should be approximately rpeak_index
        center = len(beat) // 2
        assert beat[center] == rpeak_index
    
    def test_segment_beat_various_lengths(self):
        """Test beat segmentation with different beat lengths."""
        signal = np.sin(np.linspace(0, 10, 500))
        rpeak_index = 250
        
        for beat_length in [100, 200, 256, 360, 512]:
            beat = segment_beat_around_rpeak(signal, rpeak_index, beat_length)
            assert len(beat) == beat_length


class TestMorphologyFeatures:
    """Tests for feature extraction."""
    
    def test_extract_features_normal_beat(self):
        """Test feature extraction from a normal beat."""
        beat = np.random.randn(256)
        features = extract_morphology_features(beat)
        
        assert features.shape == (15,)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_extract_features_constant_signal(self):
        """Test feature extraction from constant signal."""
        beat = np.ones(256)
        features = extract_morphology_features(beat)
        
        assert features.shape == (15,)
        # Peak value should be 1
        assert features[0] == 1.0
        # Standard deviation should be 0
        assert features[5] == 0.0
    
    def test_extract_features_sine_wave(self):
        """Test feature extraction from sine wave."""
        beat = np.sin(np.linspace(0, 2*np.pi, 256))
        features = extract_morphology_features(beat)
        
        assert features.shape == (15,)
        assert not np.any(np.isnan(features))
    
    def test_features_reproducible(self):
        """Test that feature extraction is reproducible."""
        beat = np.random.randn(256)
        features1 = extract_morphology_features(beat)
        features2 = extract_morphology_features(beat)
        
        np.testing.assert_array_equal(features1, features2)


class TestAAMIMapping:
    """Tests for AAMI class mapping."""
    
    def test_aami_mapping_normal(self):
        """Test mapping of normal beat symbols."""
        assert map_to_aami_class('N') == 'N'
        assert map_to_aami_class('L') == 'N'
        assert map_to_aami_class('R') == 'N'
    
    def test_aami_mapping_veb(self):
        """Test mapping of ventricular ectopic beat symbols."""
        assert map_to_aami_class('V') == 'V'
        assert map_to_aami_class('[') == 'V'
        assert map_to_aami_class(']') == 'V'
    
    def test_aami_mapping_sveb(self):
        """Test mapping of supraventricular ectopic beat symbols."""
        assert map_to_aami_class('A') == 'S'
        assert map_to_aami_class('J') == 'S'
        assert map_to_aami_class('S') == 'S'
    
    def test_aami_mapping_fusion(self):
        """Test mapping of fusion beats."""
        assert map_to_aami_class('F') == 'F'
    
    def test_aami_mapping_unknown(self):
        """Test mapping of unknown/paced beats."""
        assert map_to_aami_class('/') == 'Q'
        assert map_to_aami_class('Q') == 'Q'
        assert map_to_aami_class('e') == 'Q'
    
    def test_aami_mapping_unmapped_symbol(self):
        """Test that unmapped symbols default to Q."""
        assert map_to_aami_class('X') == 'Q'
        assert map_to_aami_class('?') == 'Q'
    
    def test_all_mapped_symbols_valid(self):
        """Test that all symbols in map produce valid AAMI classes."""
        for symbol, aami_class in AAMI_CLASS_MAP.items():
            assert aami_class in AAMI_CLASSES
            mapped = map_to_aami_class(symbol)
            assert mapped == aami_class


class TestDataSplitting:
    """Tests for train/test splitting utilities."""
    
    def test_inter_patient_split_basic(self):
        """Test basic inter-patient splitting."""
        records = list(range(100, 130))  # 30 records
        train, test = create_inter_patient_split(records, train_ratio=0.8)
        
        assert len(train) + len(test) == 30
        assert len(train) == 24  # 80%
        assert len(test) == 6    # 20%
        
        # No overlap
        assert len(set(train) & set(test)) == 0
    
    def test_inter_patient_split_ratio(self):
        """Test inter-patient split with different ratios."""
        records = list(range(100, 150))  # 50 records
        
        for ratio in [0.5, 0.6, 0.7, 0.8, 0.9]:
            train, test = create_inter_patient_split(records, train_ratio=ratio)
            
            expected_train = int(50 * ratio)
            assert len(train) == expected_train
            assert len(test) == 50 - expected_train
    
    def test_inter_patient_split_reproducibility(self):
        """Test that same seed produces same split."""
        records = list(range(100, 150))
        
        train1, test1 = create_inter_patient_split(records, seed=42)
        train2, test2 = create_inter_patient_split(records, seed=42)
        
        assert train1 == train2
        assert test1 == test2
    
    def test_inter_patient_split_different_seeds(self):
        """Test that different seeds produce different splits."""
        records = list(range(100, 150))
        
        train1, test1 = create_inter_patient_split(records, seed=42)
        train2, test2 = create_inter_patient_split(records, seed=123)
        
        # Different seeds should produce different splits (with high probability)
        assert train1 != train2 or test1 != test2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
