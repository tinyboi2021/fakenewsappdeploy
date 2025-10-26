import streamlit as st
import time
import os
from pathlib import Path
import gdown

# Imports from transformers *after* the model files are available
def try_import_transformers():
    global AutoTokenizer, AutoModelForSequenceClassification
    import importlib
    transformers = importlib.import_module("transformers")
    AutoTokenizer = getattr(transformers, "AutoTokenizer")
    AutoModelForSequenceClassification = getattr(transformers, "AutoModelForSequenceClassification")

# --- Google Drive Settings ---
GOOGLE_DRIVE_FOLDER_ID = "1Xgg_vwh2pVrsi4rzHFymnrXvr0RfU3-M"  # Replace with your actual folder ID
MODEL_FOLDER = "./BertFinalModel"

@st.cache_resource
def download_and_load_model():
    # Make sure the folder exists
    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
    config_path = os.path.join(MODEL_FOLDER, "config.json")

    if not os.path.exists(config_path):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}"
            try:
                # gdown requires the folder to be public and accessible
                gdown.download_folder(
                    url=url,
                    output=MODEL_FOLDER,
                    quiet=False,
                    use_cookies=False
                )
                st.success("Model files downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                st.stop()

    # Imports from transformers only after files are available
    try_import_transformers()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER)
        labels = model.config.id2label
        return tokenizer, model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def classify_text(text, tokenizer, model, labels):
    import torch
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred_idx = int(torch.argmax(logits, dim=1).item())
    confidence = float(torch.softmax(logits, dim=1)[0][pred_idx].item())
    label = labels[pred_idx] if isinstance(labels, dict) else str(pred_idx)
    return label, confidence

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("🔍 Fake News Detector")
st.markdown("---")

# Model loading
tokenizer, model, labels = download_and_load_model()

# Input section
st.markdown("### Enter Text to Classify")
text_input = st.text_area(
    "Paste your news article or text here:",
    height=200,
    placeholder="Enter the text you want to analyze..."
)

if st.button("🚀 Classify Text", use_container_width=True):
    if text_input.strip():
        with st.spinner("⏳ Analyzing..."):
            time.sleep(0.5)
            prediction, confidence = classify_text(
                text_input, tokenizer, model, labels
            )
        st.markdown("### 📊 Results")
        if prediction.upper() == "FAKE":
            st.error(f"🚨 Prediction: {prediction}")
        else:
            st.success(f"✅ Prediction: {prediction}")
        st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
    else:
        st.warning("Please enter text to classify.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Built with Streamlit & BERT | Fake News Detection System"
    "</div>",
    unsafe_allow_html=True
)
