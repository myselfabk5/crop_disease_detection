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


## Installation

```bash
git clone git clone https://github.com/myselfabk5/crop_disease_detection.git
cd crop_disease_detection

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Citation

If you use this repository in your research, please cite it as:

```bibtex
@misc{kumar2026cropdisease,
  author       = {Abhishek Kumar},
  title        = {Design of a Lightweight Crop Disease Detection and Decision Support System for South Asia with Out-of-Distribution Detection and Verified Treatment Recommendations},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/myselfabk5/crop_disease_detection}},
}
```

## 📞 Contact

Questions, feedback, and contributions are always welcome. Feel free to open an issue in this repository or reach out via email at **myselfabk5@gmail.com**.


## License

This project is intended for educational and research purposes.
