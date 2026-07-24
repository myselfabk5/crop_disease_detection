"""
models.py
---------
Model architecture definitions and cached model loaders.

The Autoencoder was saved as a *full* pickled model (torch.save(model, ...)),
so the Encoder / Decoder / AutoEncoder classes MUST be importable with the
exact same names/structure that were used when the model was originally
saved, otherwise torch.load(..., weights_only=False) will fail to unpickle.

The CNN (EfficientNet-B0 based) model was also saved as a full pickled
model, built with the `timm` library, so `timm` must be installed even
though we never call it directly here.
"""

from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
CNN_MODEL_PATH = BASE_DIR / "models" / "cnn_model.pth"
AUTOENCODER_MODEL_PATH = BASE_DIR / "models" / "autoencoder.pth"



# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Cached loaders -> loaded exactly once per app session, then reused
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading autoencoder (OOD detector) model...")
def load_autoencoder_model():
    device = get_device()
    model = torch.load(AUTOENCODER_MODEL_PATH, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model, device


@st.cache_resource(show_spinner="Loading crop disease prediction model...")
def load_cnn_model():
    device = get_device()
    model = torch.load(CNN_MODEL_PATH, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model, device