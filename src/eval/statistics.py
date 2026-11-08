"""Patient-cluster uncertainty and metrics with an explicit label universe."""

import numpy as np
from sklearn.metrics import f1_score

from src.data.segment import AAMI_CLASSES
from src.eval.inter_patient import _metrics_block


def metrics(labels, predictions):
    names = np.asarray(AAMI_CLASSES)
    return _metrics_block(
        names[np.asarray(labels, dtype=int)], names[np.asarray(predictions, dtype=int)]
    )


def subject_bootstrap(labels, predictions, groups, repetitions=200, seed=42):
    """Percentile interval from resampling subjects with replacement, preserving multiplicity."""
    labels, predictions, groups = map(np.asarray, (labels, predictions, groups))
    if len(labels) == 0 or not (len(labels) == len(predictions) == len(groups)):
        raise ValueError("Non-empty labels, predictions and groups must align")
    subjects = np.unique(groups)
    if len(subjects) < 2 or repetitions < 2:
        raise ValueError("At least two subjects and bootstrap repetitions are required")
    clusters = {g: np.flatnonzero(groups == g) for g in subjects}
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repetitions):
        indices = np.concatenate([clusters[g] for g in rng.choice(subjects, len(subjects))])
        scores.append(
            f1_score(
                labels[indices],
                predictions[indices],
                labels=range(5),
                average="macro",
                zero_division=0,
            )
        )
    return {
        "method": "subject-cluster-percentile-bootstrap",
        "confidence": 0.95,
        "lower": float(np.percentile(scores, 2.5)),
        "upper": float(np.percentile(scores, 97.5)),
        "repetitions": repetitions,
        "subjects": len(subjects),
        "seed": seed,
        "limitation": "Conditional on this cohort and trained model; not clinical uncertainty.",
    }
