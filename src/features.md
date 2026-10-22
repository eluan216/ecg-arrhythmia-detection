# Morphology features (15-D)

Each beat window (~250 samples @ 360 Hz, R-peak centered) is mapped to a fixed feature vector used by the tabular models.

| # | Feature | Definition |
|---|---------|------------|
| 1 | Peak amplitude | max \|signal\| in the window |
| 2 | Peak location | argmax \|signal\| / length (normalized) |
| 3 | RMS | root-mean-square amplitude |
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

These are deliberately simple, clinically motivated morphology descriptors (amplitude, width proxies, slope, shape). They are **not** a full fiducial P-QRS-T delineation pipeline.
