"""Render measured protocol columns only after all reports are available."""

import json
from pathlib import Path


def main():
    paths = ["research", "ds1-ds2", "ds1-ds2-disjoint"]
    reports = [
        json.loads((Path("artifacts") / p / "comparison.json").read_text())
        for p in paths
    ]
    lines = [
        "# Measured protocol comparison",
        "",
        "All models retain five AAMI groups. N/V/S focus excludes reference F/Q beats;",
        "F/Q predictions on included beats remain errors. It is a different estimand.",
        "Published DS1/DS2 shares subject 201/202; the disjoint variant excludes training record 201.",
        "Lead/window/mapping choices still constrain literature comparability.",
        "",
        "| Model / metric | Custom 45 | Published DS1/DS2 | Disjoint variant |",
        "| --- | --- | --- | --- |",
    ]
    for model in reports[0]["models"]:
        for metric in ("five-class", "N/V/S focus"):
            values = [
                (
                    report["models"][model]["macro_f1"]
                    if metric == "five-class"
                    else report["models"][model]["nsv_focus"]["macro_f1"]
                )
                for report in reports
            ]
            lines.append(
                f"| {model} / {metric} | "
                + " | ".join(f"{100 * v:.2f}%" for v in values)
                + " |"
            )
    Path("artifacts/protocol-comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
