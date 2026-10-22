"""Training and API must share identical morphology features."""

import numpy as np
import pytest

from src.data.segment import (
    FEATURE_DIM,
    FEATURE_VERSION,
    extract_morphology_features,
    validate_beat_window,
)


def test_feature_dim_and_version():
    x = np.random.randn(250)
    f = extract_morphology_features(x)
    assert f.shape == (FEATURE_DIM,)
    assert FEATURE_VERSION.startswith("morphology_")


def test_api_imports_same_function():
    from api import extract_morphology_features as api_fn
    from src.data.segment import extract_morphology_features as train_fn

    assert api_fn is train_fn


def test_identical_features_for_same_beat():
    from api import extract_morphology_features as api_fn
    from src.data.segment import extract_morphology_features as train_fn

    rng = np.random.default_rng(0)
    beat = rng.normal(0, 1, size=250)
    a = train_fn(beat)
    b = api_fn(beat)
    np.testing.assert_allclose(a, b)


def test_constant_signal():
    beat = np.ones(250)
    f = extract_morphology_features(beat)
    assert f.shape == (FEATURE_DIM,)
    assert np.isfinite(f).all()


def test_rejects_nan():
    beat = np.ones(250)
    beat[10] = np.nan
    with pytest.raises(ValueError, match="NaN|Inf"):
        extract_morphology_features(beat)


def test_rejects_empty():
    with pytest.raises(ValueError):
        validate_beat_window(np.array([]))


def test_rejects_2d():
    with pytest.raises(ValueError, match="1-D"):
        validate_beat_window(np.zeros((10, 10)))
