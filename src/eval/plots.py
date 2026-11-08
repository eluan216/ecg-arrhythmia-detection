"""Render standalone figures from a completed comparison and external reports."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.segment import AAMI_CLASSES


def render(bundle, output):
    bundle, output = Path(bundle), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((bundle / "comparison.json").read_text(encoding="utf-8"))
    names = list(report["models"])
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    for ax, name in zip(axes[0], names):
        cm = np.asarray(report["models"][name]["confusion_matrix"])
        normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.set(
            xticks=range(5),
            yticks=range(5),
            xticklabels=AAMI_CLASSES,
            yticklabels=AAMI_CLASSES,
            xlabel="Prediction",
            ylabel="Reference",
            title=name,
        )
        for i in range(5):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if normalized[i, j] > 0.5 else "black",
                )
    fig.tight_layout()
    fig.savefig(output / "confusion-matrices.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, name in enumerate(names):
        scores = [report["models"][name]["per_class"][c]["f1"] for c in AAMI_CLASSES]
        ax.bar(np.arange(5) + (i - 1) * 0.25, scores, width=0.25, label=name)
    ax.set(
        xticks=range(5),
        xticklabels=AAMI_CLASSES,
        ylim=(0, 1),
        ylabel="F1",
        title="Performance on held-out patients",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "per-class-f1.png", dpi=160)
    plt.close(fig)
    noise = bundle / "nstdb.json"
    if noise.exists():
        data = json.loads(noise.read_text(encoding="utf-8"))
        levels = sorted(map(int, data["evaluations"]))
        fig, ax = plt.subplots(figsize=(8, 4))
        for name in data["evaluations"][str(levels[0])]:
            ax.plot(
                levels,
                [data["evaluations"][str(x)][name]["macro_f1"] for x in levels],
                marker="o",
                label=name,
            )
        ax.set(
            xlabel="SNR (dB)",
            ylabel="Macro-F1",
            ylim=(0, 1),
            title="NSTDB noise-on segments; unseen clean sources only",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "noise-robustness.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="artifacts/research")
    parser.add_argument("--output", default="artifacts/figures")
    args = parser.parse_args()
    render(args.bundle, args.output)
