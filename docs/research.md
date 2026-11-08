# ECG methodology review

Reviewed 8 November 2026 against the repository at `a6ae9dc`.
These notes distinguish implemented corrections from experiments still needed.

## Sources and decisions

| Primary source | Evidence and application |
| --- | --- |
| [MIT-BIH dataset](https://physionet.org/content/mitdb/1.0.0/) | 48 two-channel recordings from 47 subjects, sampled at 360 Hz. Records 201/202 share a subject; keep them together. |
| [MIT-BIH directory introduction](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm) | Acquisition and lead details matter. Channel order is reversed in record 114. Preserve the existing channel-0 protocol and disclose it instead of silently changing historical results. |
| [WFDB annotation codes](https://physionet.org/physiotools/wpg/wpg_36.htm) | Beat, rhythm and noise annotations have different meanings. Keep an explicit beat whitelist. |
| [WFDB evaluation guide](https://physionet.org/physiotools/wag/evnode8.htm) | Evaluation distinguishes beat labels and localization. This repository uses reference locations and does not evaluate a beat detector. Five grouped labels alone do not establish full standards compliance. |
| [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) | Test observations must not drive fitting or selection; apply consistent preprocessing to inference. Remove CNN test-based checkpoint selection and share normalization. |
| [Grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) | Related observations require group-aware partitions. Keep training-subject GroupKFold and reserve test subjects. |
| [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) | Report per-class support and precision/recall/F1; make the macro averaging label universe explicit. CNN now averages all five classes like the RF evaluator. |
| [imbalanced-learn pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html) | Any future resampling must occur inside training folds. Never balance the test cohort or resample before splitting. No resampling was added in this review. |
| [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) | Seeds help but do not ensure identical results across releases, devices and platforms. Preserve environment provenance; do not promise bitwise reproduction. |
| [Model persistence](https://scikit-learn.org/stable/model_persistence.html) | Pickle/joblib requires trusted artifacts and compatible package versions. Document that serving constraint. |

## Corrections implemented

1. **CNN test contamination.** The old trainer evaluated test beats every five epochs and retained the best test-scoring state. Training now uses the predeclared final epoch, followed by one test pass. Future early stopping requires a distinct training-subject validation split.
2. **CNN training/serving mismatch.** Training normalized each beat; serving did not, and serving padded/truncated to 256 despite adaptive pooling. Both paths now call the same per-beat z-score function and preserve window length.
3. **Inconsistent macro-F1.** CNN averaging previously inferred classes from observations. It now explicitly includes all five declared labels.
4. **Silent cohort changes.** The lower-level RF evaluator previously skipped unreadable records. It now fails with the affected record ID.
5. **Split validation.** Reject invalid ratios and single-subject inputs; deduplicate record IDs before partitioning.
6. **Sampling contract.** Reject records whose sampling rate differs from the declared 360 Hz.
7. **Container setup.** Include packaging metadata files in the build, use a standard-library HTTP health check, exclude generated data from build context and document model mounts.
8. **Documentation.** Replace stale development instructions, write the README in the author's first person and archive earlier completion summaries.

## Existing evidence and limitations

The checked-in RF report remains historical evidence: 35.83% macro-F1,
94.79% accuracy, 20,934 test beats. This initial review preserved that report; subsequent experiments are documented in the update below. The report uses the custom 45-record cohort,
including paced record 217; it is not a standard 44-record DS1/DS2 benchmark.

The old CNN checkpoint-selection path cannot support an unbiased test estimate.
Corrected validation-selected CNN measurements are now linked below.

Other review findings remain follow-up work:

- Baseline training helpers also skip failed records; use the strict evaluation runner for published results.
- Cached NPZ inputs do not encode sufficient provenance to prove subject independence or synthetic origin.
- New manifest-backed bundles enforce integrity and feature/environment checks; legacy weights still require trusted, compatible files.
- Older orchestration/setup helpers and stage reports contain historical assumptions.
- A consistent lead protocol and calibrated probabilities remain follow-up work. RR, uncertainty and external evaluation are covered by the new suite.
- Once this holdout has influenced future design choices, it is a development benchmark; a new external test cohort is needed for independent final claims.

## Initial suggested experiment (now implemented below)

Freeze an outer test cohort, split only training subjects into training and
validation, compare morphology with morphology plus RR features, and select
parameters using validation macro-F1. Keep natural test prevalence, report all
class supports and evaluate uncertainty by resampling subjects rather than
individual beats. Publish a new report with code, split, data hashes and versions.



## Implemented research update

The [measured 8 November reports](../reports/research/2026-11-08/) now include
RF morphology versus morphology-plus-RR, validation-selected CNN, patient-cluster
bootstrap intervals, timings, NSTDB stress testing and full INCART evaluation.
See [the experiment guide](experiments.md) for sources and exact contracts.
INCART annotations outside signal boundaries are explicitly excluded and audited.
The explorer, model card, CPU container smoke test and model-bundle verification
make those results inspectable and reproducible. They do not establish clinical utility.
