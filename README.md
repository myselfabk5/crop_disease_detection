# 🌱 Lightweight Crop Disease Detection System

A lightweight AI-powered crop disease diagnosis system that combines **disease classification**, **out-of-distribution (OOD) detection**, **Grad-CAM explainability**, and **verified treatment recommendations** for reliable agricultural decision support.

## Features

- 🌿 Disease classification using fine-tuned **EfficientNet-B0**
- 🚫 Out-of-Distribution (OOD) detection using a convolutional autoencoder
- 🔍 Grad-CAM visualizations for model interpretability
- 📖 Verified treatment recommendations from a curated knowledge base
- 💻 Interactive web interface built with **Streamlit**

## Model Performance

| Metric | Value |
|--------|------:|
| Validation Accuracy | **89.64%** |
| Test Accuracy | **89.45%** |
| OOD Detection AUROC | **0.8183** |

## Dataset

- 94 crop disease classes
- 13 crop species
- Dataset: https://huggingface.co/datasets/Saon110/bd-crop-vegetable-plant-disease-dataset

## Tech Stack

- Python
- PyTorch
- EfficientNet-B0
- OpenCV
- Streamlit
- Grad-CAM

## Repository Structure

```
├── app.py                 # Streamlit application
├── models/                # Trained model weights
├── utils/                 # Utility functions
├── data/                  # Dataset (not included)
├── images/                # Screenshots
└── requirements.txt
```

## Installation

```bash
git clone <repository-url>
cd <repository-name>

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Citation

If you use this work, please cite the accompanying project report.

## License

This project is intended for educational and research purposes.