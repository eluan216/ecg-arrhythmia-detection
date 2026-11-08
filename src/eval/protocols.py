"""Explicit record protocols; the published DS1/DS2 split has a known shared subject."""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.data.segment import AAMI_CLASSES, subject_id_for_record

DS1 = [
    101,
    106,
    108,
    109,
    112,
    114,
    115,
    116,
    118,
    119,
    122,
    124,
    201,
    203,
    205,
    207,
    208,
    209,
    215,
    220,
    223,
    230,
]
DS2 = [
    100,
    103,
    105,
    111,
    113,
    117,
    121,
    123,
    200,
    202,
    210,
    212,
    213,
    214,
    219,
    221,
    222,
    228,
    231,
    232,
    233,
    234,
]


def fixed_partition(protocol):
    if protocol == "ds1-ds2":
        return DS1.copy(), DS2.copy()
    if protocol == "ds1-ds2-disjoint":
        return [r for r in DS1 if r != 201], DS2.copy()
    raise ValueError(f"Unknown fixed protocol: {protocol}")


def shared_subjects(train, test):
    return sorted(
        {subject_id_for_record(r) for r in train} & {subject_id_for_record(r) for r in test}
    )


def class_support(data):
    return {
        name: {
            "beats": int(np.sum(data.labels == i)),
            "subjects": len(np.unique(data.groups[data.labels == i])),
        }
        for i, name in enumerate(AAMI_CLASSES)
    }


def nsv_focus(labels, predictions):
    """Secondary conditional evaluation, NOT a retrained three-class classifier.

    Retain predictions of F/Q as errors on reference N/V/S beats.
    Keep the full confusion matrix so those errors remain visible.
    """
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    mask = np.isin(labels, [0, 1, 2])
    if not mask.any():
        raise ValueError("No reference N/V/S beats")
    truth, pred = labels[mask], predictions[mask]
    report = classification_report(truth, pred, labels=range(5), output_dict=True, zero_division=0)
    return {
        "scope": "Reference N/V/S beats only; unchanged five-class model; F/Q predictions count as errors",
        "class_labels": AAMI_CLASSES,
        "macro_f1": float(
            f1_score(truth, pred, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "included_beats": int(mask.sum()),
        "excluded_reference_F_Q": int((~mask).sum()),
        "per_class": {name: report[str(i)] for i, name in enumerate(AAMI_CLASSES[:3])},
        "confusion_matrix": confusion_matrix(truth, pred, labels=range(5)).tolist(),
    }
