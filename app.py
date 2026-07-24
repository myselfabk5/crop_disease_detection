"""
app.py
------
Streamlit dashboard: Crop Disease Prediction and Treatment.

Flow:
  1. User uploads an image (camera or device).
  2. Autoencoder computes a reconstruction-loss based OOD check.
     - "proceed"      -> silently continue to prediction.
     - "not_proceed"  -> warn the user the image looks out-of-distribution
                          and ask for explicit confirmation before continuing.
  3. CNN model predicts the disease + confidence.
  4. Grad-CAM produces two explainability images.
  5. The fixed knowledge base is queried for treatment info.
  6. A formatted report is rendered from templates.py.
"""

from pathlib import Path

import streamlit as st
import torch.nn as nn
from explainability import run_explainability
from inference import run_prediction, run_ood_check
from knowledge_base import information_retrieval
from models import load_autoencoder_model, load_cnn_model
from templates import language_output
from utils import cnn_transform, image_transform, read_crop_image

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "logo.png"
print(BASE_DIR)
st.set_page_config(
    page_title="Crop Disease Prediction and Treatment",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)




 #--------------------------------------------------------------------------
# Autoencoder architecture (must match training-time definition exactly)
# --------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 14 * 14, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 14 * 14)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 14, 14)
        x = self.decoder(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { background-color: #f7faf7; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }

        .app-title { font-size: 2rem; font-weight: 700; color: #1b5e20; margin-bottom: 0; }
        .app-subtitle { font-size: 1rem; color: #4b5563; margin-top: 0.2rem; margin-bottom: 1.5rem; }

        .section-card {
            background-color: #ffffff;
            border: 1px solid #e3e8e3;
            border-radius: 14px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }

        .metric-pill {
            display: inline-block;
            background-color: #e8f5e9;
            color: #1b5e20;
            font-weight: 600;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-size: 0.95rem;
        }

        .ood-warning {
            background-color: #fff3e0;
            border: 1px solid #ffb74d;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            color: #7a4a00;
        }

        .report-box {
            background-color: #fbfffb;
            border: 1px solid #d7e6d7;
            border-radius: 12px;
            padding: 1rem 1.3rem;
            white-space: pre-wrap;
            font-family: "Source Code Pro", monospace;
            font-size: 0.92rem;
            line-height: 1.5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
DEFAULTS = {
    "stage": "await_upload",       # await_upload -> ood_warning -> results
    "image_bytes": None,
    "ood_loss": None,
    "ood_verdict": None,
    "results": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_app():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


# --------------------------------------------------------------------------
# Cached model loading (loaded once, reused across reruns/sessions)
# --------------------------------------------------------------------------
autoencoder_model, ae_device = load_autoencoder_model()
cnn_model, cnn_device = load_cnn_model()


# --------------------------------------------------------------------------
# Pipeline helpers
# --------------------------------------------------------------------------
def run_full_pipeline(pil_image):
    """Runs prediction + explainability + knowledge base retrieval + report."""
    with st.spinner("Analyzing the leaf and predicting the disease..."):
        predicted_class, confidence, cnn_img_tensor = run_prediction(
            pil_image, cnn_model, cnn_device, cnn_transform
        )

    with st.spinner("Generating explainability visualization (Grad-CAM)..."):
        original_pil, gradcam_pil = run_explainability(
            cnn_model, cnn_img_tensor, pil_image, cnn_device
        )

    with st.spinner("Retrieving treatment information..."):
        retrieved_info = information_retrieval(predicted_class, confidence)
        report = language_output(retrieved_info)

    st.session_state.results = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "original_pil": original_pil,
        "gradcam_pil": gradcam_pil,
        "retrieved_info": retrieved_info,
        "report": report,
    }
    st.session_state.stage = "results"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    st.markdown("### 🌿 About")
    st.write(
        "This tool uses a deep learning model to identify crop diseases from "
        "a leaf image, explains its decision visually, and suggests organic "
        "and chemical treatment options from a curated knowledge base."
    )
    st.markdown("---")
    st.write(
        "**How it works:**\n"
        "1. Upload a leaf image\n"
        "2. Automatic quality/relevance check\n"
        "3. Disease prediction & confidence\n"
        "4. Visual explanation (Grad-CAM)\n"
        "5. Treatment recommendations"
    )
    st.markdown("---")
    if st.session_state.stage != "await_upload":
        if st.button("🔄 Analyze another image", use_container_width=True):
            reset_app()
            st.rerun()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<p class="app-title">Crop Disease Prediction and Treatment</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Upload a photo of a crop leaf to get an instant disease diagnosis, '
    "a visual explanation of the model's decision, and treatment guidance.</p>",
    unsafe_allow_html=True,
)

tab = st.tabs(["🌱 Crop Disease Prediction and Treatment"])[0]

with tab:

    # ----------------------------------------------------------------
    # STAGE 1: Upload
    # ----------------------------------------------------------------
    if st.session_state.stage == "await_upload":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Step 1 · Upload a leaf image")

        input_method = st.radio(
            "Choose how you'd like to provide the image:",
            ["📁 Upload from device", "📷 Use camera"],
            horizontal=True,
        )

        uploaded_file = None
        if input_method == "📁 Upload from device":
            uploaded_file = st.file_uploader(
                "Upload a clear image of the affected crop leaf",
                type=["jpg", "jpeg", "png"],
            )
        else:
            uploaded_file = st.camera_input("Take a photo of the crop leaf")

        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            st.session_state.image_bytes = uploaded_file.getvalue()

            pil_image = read_crop_image(uploaded_file)
            col1, _ = st.columns([1, 2])
            with col1:
                st.image(pil_image, caption="Uploaded image", use_container_width=True)

            with st.spinner("Checking image quality / relevance..."):
                ood_loss, ood_verdict = run_ood_check(
                    pil_image, autoencoder_model, ae_device, image_transform
                )
            st.session_state.ood_loss = ood_loss
            st.session_state.ood_verdict = ood_verdict

            if ood_verdict == "proceed":
                run_full_pipeline(pil_image)
                st.rerun()
            else:
                st.session_state.stage = "ood_warning"
                st.rerun()

    # ----------------------------------------------------------------
    # STAGE 2: OOD warning + confirmation
    # ----------------------------------------------------------------
    elif st.session_state.stage == "ood_warning":
        pil_image = read_crop_image(st.session_state.image_bytes)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(pil_image, caption="Uploaded image", use_container_width=True)

        with col2:
            st.markdown(
                f"""
                <div class="ood-warning">
                <b>⚠️ This image looks out-of-distribution.</b><br><br>
                The uploaded image doesn't closely resemble the crop leaf images the
                model was trained on (reconstruction error: <b>{st.session_state.ood_loss:.4f}</b>).
                If you proceed, the disease prediction and confidence score
                <b>may not be reliable</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.write("**Do you still want to proceed with the prediction?**")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes, proceed anyway", use_container_width=True, type="primary"):
                    run_full_pipeline(pil_image)
                    st.rerun()
            with c2:
                if st.button("❌ No, start over", use_container_width=True):
                    reset_app()
                    st.rerun()

    # ----------------------------------------------------------------
    # STAGE 3: Results
    # ----------------------------------------------------------------
    elif st.session_state.stage == "results":
        results = st.session_state.results
        retrieved_info = results["retrieved_info"]

        if st.session_state.ood_verdict == "not_proceed":
            st.warning(
                "⚠️ This result was generated from an image flagged as out-of-distribution. "
                "The prediction confidence may not be reliable.",
                icon="⚠️",
            )

        disease_name = retrieved_info.get(
            "display_name", retrieved_info.get("original_class_label", "Unknown")
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Step 2 · Prediction Result")

        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<span class="metric-pill">🌾 Crop: {retrieved_info.get("crop", "N/A")}</span>', unsafe_allow_html=True)
        m2.markdown(f'<span class="metric-pill">🦠 Disease: {disease_name}</span>', unsafe_allow_html=True)
        m3.markdown(f'<span class="metric-pill">🎯 Confidence: {results["confidence"]:.2%}</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Step 3 · Where the model is looking (Explainability)")
        g1, g2 = st.columns(2)
        with g1:
            st.image(results["original_pil"], caption="Original Image", use_container_width=True)
        with g2:
            st.image(results["gradcam_pil"], caption="Disease Highlighted Image (Grad-CAM)", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Step 4 · Diagnosis & Treatment Report")
        st.markdown(f'<div class="report-box">{results["report"]}</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download report as .txt",
            data=results["report"],
            file_name="crop_disease_report.txt",
            mime="text/plain",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.button("🔄 Analyze another image", on_click=reset_app)
