"""Build the self-contained public demo. Run from an installed repository checkout."""

import html
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build():
    from src.data.segment import AAMI_CLASSES
    from src.data.synthetic import generate_synthetic_dataset

    train, labels, _, _ = generate_synthetic_dataset(
        n_samples_per_class=5, beat_length=250, seed=42
    )
    samples = [
        {
            "id": f"synthetic-{name}-{int(index)}",
            "reference": name,
            "signal": np.round(train[index], 6).astype(float).tolist(),
            "predictions": {},
        }
        for label, name in enumerate(AAMI_CLASSES)
        for index in np.flatnonzero(labels == label)[:2]
    ]
    payload = {
        "source": "synthetic",
        "sampling_rate": 360,
        "selection": "Illustrative generated templates, not patient recordings or model predictions.",
        "samples": samples,
    }
    report = json.loads(
        (ROOT / "reports/research/2026-11-08/comparison.json").read_text(encoding="utf-8")
    )
    page = (ROOT / "api/explorer.html").read_text(encoding="utf-8")
    page = page.replace(
        "<title>ECG Research Explorer</title>", "<title>ECG Research Explorer — Public Demo</title>"
    )
    page = page.replace(
        'href="/docs"', 'href="https://github.com/eluan216/ecg-arrhythmia-detection#readme"'
    )
    page = page.replace("API documentation ↗", "Repository &amp; full API ↗")
    page = page.replace(
        "Research and education only. This prototype cannot diagnose a condition or guide treatment.",
        "Public browser demo · waveforms are synthetic. Measured benchmark results are shown separately below. Research only; not for diagnosis or treatment.",
    )
    page = page.replace(
        '<input type="checkbox" id="errors">', '<input type="checkbox" id="errors" disabled>'
    )
    page = page.replace("Show model errors only", "Model error filtering: local API only")
    page = page.replace("Run loaded models", "View measured results")
    page = page.replace(
        "File stays in your browser until you run a prediction on this server.",
        "File stays entirely in your browser. This static demo plots it but does not classify it.",
    )
    start = page.index("$('predict').onclick=async()=>")
    end = page.index("$('upload').onchange=", start)
    page = (
        page[:start]
        + "$('predict').onclick=()=>document.getElementById('benchmarks').scrollIntoView({behavior:'smooth'});\n"
        + page[end:]
    )
    page = page.replace(
        "Local waveform loaded; no prediction has been requested.",
        "Local waveform loaded privately. No upload or model prediction is performed.",
    )
    page = page.replace(
        "fetch('/explorer/data').then(r=>{if(!r.ok)throw Error('Could not load examples');return r.json()}).then(data=>{",
        "Promise.resolve(" + json.dumps(payload).replace("</", "<\\/") + ").then(data=>{",
    )
    page = page.replace(
        "Live inference supports the morphology RF and CNN.",
        "Trained-model inference is available in the repository's local FastAPI application. This public static demo does not run those models.",
    )
    rows = []
    options = []
    for name, result in report["models"].items():
        ci = result["macro_f1_interval"]
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{result['macro_f1']:.2%}</td>"
            f"<td>{ci['lower']:.2%}–{ci['upper']:.2%}</td><td>{result['accuracy']:.2%}</td></tr>"
        )
        options.append(f"<option>{html.escape(name)}</option>")
    section = (
        """
<section class="card" id="benchmarks" style="margin-top:24px">
<div class="eyebrow">MEASURED RESULTS · REAL MIT-BIH DATA</div>
<h2>What the models actually achieved</h2>
<p>20,934 beats from nine held-out patients. Epoch selection uses training-subject validation.
These aggregate results are separate from the synthetic waveform examples above.</p>
<div style="overflow-x:auto"><table style="width:100%;text-align:left;border-collapse:collapse;line-height:2.5">
<thead><tr><th>Model</th><th>Macro-F1</th><th>95% patient interval</th><th>Accuracy</th></tr></thead>
<tbody>"""
        + "".join(rows)
        + """</tbody></table></div>
<p>The RR improvement is modest and uncertainty intervals overlap. Minority classes remain weak.
High accuracy does not establish clinical usefulness.</p>
<label for="benchmarkModel">Inspect a measured confusion matrix</label>
<select id="benchmarkModel">"""
        + "".join(options)
        + """</select>
<div id="matrix" style="overflow-x:auto"></div>
<p class="sub">Rows: reference labels. Columns: model predictions. Counts come from the measured holdout.</p>
<p>Frozen-model external testing covered all 75 INCART records (175,867 accepted beats);
seven out-of-range annotations were explicitly excluded and audited.</p>
<a href="https://github.com/eluan216/ecg-arrhythmia-detection/tree/main/reports/research/2026-11-08">Read the measured reports, figures and limitations ↗</a>
</section>
"""
    )
    page = page.replace("</main>", section + "</main>")
    benchmark_script = (
        """
const measuredReport = """
        + json.dumps(report["models"]).replace("</", "<\\/")
        + """;
function renderMatrix(){
 const selected=document.getElementById('benchmarkModel').value, data=measuredReport[selected];
 const table=document.createElement('table');table.style.cssText='width:100%;text-align:center;line-height:2.5;border-collapse:collapse';
 const names=['N','V','S','F','Q'], heading=table.insertRow();
 for(const name of ['Reference / prediction',...names]){const th=document.createElement('th');th.textContent=name;th.scope='col';heading.append(th)}
 data.confusion_matrix.forEach((row,i)=>{
  const tr=table.insertRow(),label=document.createElement('th');label.textContent=names[i];label.scope='row';tr.append(label);
  row.forEach((value,j)=>{const cell=tr.insertCell();cell.textContent=value;cell.style.background=i===j?'#21423f':'#15283d';cell.style.border='1px solid #304257'})
 });
 document.getElementById('matrix').replaceChildren(table);
}
document.getElementById('benchmarkModel').onchange=renderMatrix;renderMatrix();
"""
    )
    page = page.replace("</script>", benchmark_script + "</script>")
    (ROOT / "demo/index.html").write_text(page, encoding="utf-8")
    print("Built demo/index.html: synthetic waveform explorer and measured benchmark inspection.")


if __name__ == "__main__":
    build()
