"""
knowledge_base.py
------------------
Loads the fixed disease knowledge base and label mapping, and retrieves
the record for a predicted class.

Expected files (place them under the `data/` folder, or update the paths
below):
  - data/label_mapping.json                        -> {"0": "class_key", ...}
  - data/crop_disease_knowledge_base.json           -> {"class_key": {...}, ...}

Each knowledge-base entry is expected to contain at least the keys used by
templates.py: crop, symptoms, cause, organic_treatment, chemical_treatment,
prevention, severity, source, source_url, and optionally display_name /
original_class_label.
"""

import json
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
LABEL_MAPPING_PATH = BASE_DIR / "data" / "label_mapping.json"
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "crop_disease_knowledge_base.json"


@st.cache_data(show_spinner=False)
def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def information_retrieval(predicted_disease, disease_confidence):
    """
    Maps the predicted class index -> class key -> knowledge base record,
    and attaches the model's confidence score to the record.
    """
    label_mapping = _load_json(LABEL_MAPPING_PATH)
    disease_database = _load_json(KNOWLEDGE_BASE_PATH)

    predicted_disease_class = label_mapping[str(predicted_disease)]
    retrieved_info = dict(disease_database[predicted_disease_class])  # copy, don't mutate cache
    retrieved_info["predicted_probability"] = disease_confidence

    return retrieved_info
