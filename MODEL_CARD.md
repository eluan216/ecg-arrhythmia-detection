# Model card: ECG beat research prototype

## Purpose and users

I built this project for ECG researchers, ML engineers and students who want
to reproduce beat-classification experiments and inspect errors. It provides
an experiment framework, a local waveform explorer and a model-serving prototype.

It is not intended for patient screening, diagnosis, triage, treatment decisions,
emergency alerts or autonomous interpretation of continuous ECG recordings.
It has not undergone clinical validation or medical-device assessment.

## Inputs and outputs

- One reference-centered 250-sample ECG window at 360 Hz, in physical mV.
- Class order: N, V, S, F, Q. These group beat annotations; they are not five diagnoses.
- RF: fifteen morphology features, unchanged from the historical baseline.
- CNN: per-beat z-score normalization and an adaptive-pooling convolutional model.
- RR ablation: three additional features from earlier accepted reference beat locations.
  No future beat is used. This remains an offline experiment, not a live API model.
- Outputs are class scores and a predicted label. Scores are not calibrated clinical risk.

The API accepts 32–1000 samples for compatibility, but lengths other than 250
are outside the measured research contract. Reference locations are supplied;
the project does not measure R-peak detection or episode detection.

## Data and protocol

MIT-BIH v1.0.0, custom 45-record cohort excluding 102/104/107 and retaining 217.
Records 201/202 share a subject and stay together. Channel 0 is used, including
its lead inconsistency in record 114. This is not a standard 44-record DS1/DS2 comparison.

The research suite uses the historical outer split. It creates an inner validation
split from training subjects, selects the CNN epoch by validation macro-F1,
refits on all outer training subjects, and evaluates held-out patients once.
RF parameters are fixed; morphology and morphology-plus-RR use identical partitions.
The test cohort has already informed development discussions, so it is now a
development benchmark, not a new independent clinical test.

## Evidence

The historical RF report records 35.83% macro-F1 and 94.79% accuracy on 20,934
test beats. S/F/Q performance is poor; F/Q have only 7/4 test beats.
See [the original report](reports/mitbih/2026-11-05/full_report.json).

New comparisons are measured only when the research workflow succeeds.
Each bundle includes a comparison, class supports, confusion matrices,
patient-cluster bootstrap intervals, timings and environment/data provenance.
The interval is conditional on the cohort and model; it does not quantify
clinical certainty or all sources of training variation.

NSTDB evaluation excludes clean sources present in training and scores only
noise-on intervals. Its tiny source population cannot establish broad robustness.
INCART evaluation uses frozen models, lead II and polyphase resampling from
257 to 360 Hz. Its cohort, lead configuration and annotation alignment differ
from MIT-BIH. Pooled results do not prove suitability for other devices.

## Limitations and responsible interpretation

- Historical, small, geographically limited cohorts; no validated fairness claim.
- Rare-class support and subject variability make aggregate scores unstable.
- Reference annotations give better localization than a practical detector may supply.
- Noise, acquisition changes and artifacts can change morphology and predictions.
- The explorer deliberately selects errors and correct examples, not a representative sample.
- Synthetic templates are software fixtures, not real patient physiology.
- A useful research result may be a well-documented failure; no accuracy target is promised.

## Artifact handling

Research bundles record file SHA-256 hashes, class order, feature/preprocessing
versions, window contract and exact core package versions. The API checks these
when a manifest exists before loading weights. Checksums detect changes but do
not make an untrusted pickle safe. Legacy standalone weights remain supported;
use trusted files and matching training environments.

## Further validation needed

Independent prospective cohorts, end-to-end beat localization, device and subgroup
evaluation, calibrated uncertainty, workflow/user studies and appropriate clinical
and regulatory assessment would be needed before considering a healthcare use.
