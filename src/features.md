# Morphology features (`morphology_v1`, 15-D)

**Single source of truth:** `src.data.segment.extract_morphology_features`

Training, evaluation, and the API all call this function. Feature vectors from older code or duplicated API logic are invalid for models trained on `morphology_v1`.

| # | Feature | Definition |
|---|---------|------------|
| 1 | Peak amplitude | max \|signal\| |
| 2 | Peak location | argmax \|signal\| / length |
| 3 | RMS | root-mean-square |
| 4 | Min | minimum sample |
| 5 | Max | maximum sample |
| 6 | Std | standard deviation |
| 7 | Median | median sample |
| 8 | IQR | 75th − 25th percentile |
| 9 | Mean | mean sample |
| 10 | Mean \|slope\| | mean absolute first difference |
| 11 | Max \|slope\| | max absolute first difference |
| 12 | Std slope | std of first difference |
| 13 | Zero-crossing rate | fraction of sign changes |
| 14 | Excess kurtosis | 4th standardized moment − 3 |
| 15 | Skewness | 3rd standardized moment |

`FEATURE_VERSION` is stored with metrics when available. Bump the version string if any definition changes.
