# Public ECG demo

Open the public demo linked at the top of the repository README. No installation
or sign-in is required. This folder contains the same self-contained HTML source.

Public URL: https://ecg-arrhythmia-detection.vercel.app

Vercel serves this directory as the project's root, using the included
`vercel.json`. No Python installation or server is needed for the public demo.

The demo offers synthetic waveform selection, private local CSV plotting,
measured benchmark tables and interactive confusion matrices. It makes no
predictions for those synthetic or uploaded waveforms. Trained-model inference
and measured per-beat error exploration are available through the local API
with a completed model bundle.

Rebuild after changing the explorer or measured report:

```bash
python demo/build.py
```

To preview without installing scientific dependencies, serve the generated HTML:

```bash
python -m http.server 8000 --directory demo
```

Then open `http://localhost:8000`. The page has no backend, external JavaScript
dependencies or analytics. Uploaded CSV contents stay in the browser.
