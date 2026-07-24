# Crop Disease Prediction and Treatment — Streamlit Dashboard

## Folder structure

```
crop_disease_app/
│
├── app.py                  # Streamlit dashboard (entry point)
├── models.py                # Model architecture + cached loaders
├── inference.py              # OOD check + CNN prediction pipeline
├── explainability.py         # Grad-CAM
├── knowledge_base.py         # Disease information retrieval
├── templates.py              # 10 report templates
├── utils.py                  # Image reading & transforms
├── requirements.txt
│
├── models/
│     ├── cnn_model.pth              # full pickled EfficientNet-B0 model
│     └── autoencoder.pth            # full pickled AutoEncoder model
│
├── data/                            # ⚠️ NOT in your original folder list — add this
│     ├── label_mapping.json         # {"0": "class_key", "1": "...", ...}
│     └── crop_disease_knowledge_base.json   # {"class_key": {...}, ...}
│
└── assets/
      └── logo.png
```

> **Note:** your original folder list didn't include the `label_mapping.json`
> and knowledge-base JSON files used by `information_retrieval()` in the
> notebook. Add a `data/` folder (as above) with those two files, or edit
> the paths at the top of `knowledge_base.py` to point wherever you keep them.

## Install

```bash
cd crop_disease_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`timm` is required even though it isn't imported directly — the CNN model
was saved as a full pickled object built on a `timm` EfficientNet-B0
backbone, and unpickling it requires the same library to be installed.

## Run

```bash
streamlit run app.py
```

## App flow

1. **Upload** — choose "Upload from device" or "Use camera", then provide a leaf image.
2. **Automatic OOD check** — the autoencoder computes a reconstruction loss.
   - If the image looks in-distribution → the app proceeds straight to prediction.
   - If it looks out-of-distribution → a warning is shown with **Yes/No** buttons.
     - **No** → the app resets to a fresh upload state.
     - **Yes** → the app proceeds anyway (result flagged as potentially unreliable).
3. **Prediction** — the CNN model predicts crop + disease + confidence.
4. **Explainability** — two images are shown: original vs. Grad-CAM heatmap overlay.
5. **Report** — one of 10 templated reports is generated from the knowledge base,
   viewable on-screen and downloadable as `.txt`.

## Tuning

- `inference.OOD_THRESHOLD` (default `0.015`) controls the OOD cutoff — same
  value used in the original notebook. Recalibrate if you retrain the autoencoder.
- Model loading is cached with `st.cache_resource`, so the ~2 models load
  once per server process, not on every image.
- Knowledge base / label mapping JSON files are cached with `st.cache_data`.
