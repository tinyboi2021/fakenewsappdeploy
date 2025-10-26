import streamlit as st
import time
import os
from pathlib import Path
import gdown

# Lazy import to avoid issues before files present
def import_transformers():
    global AutoTokenizer, AutoModelForSequenceClassification
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Google Drive Model Folder ---
GOOGLE_DRIVE_FOLDER_ID = "1Xgg_vwh2pVrsi4rzHFymnrXvr0RfU3-M"  # Your actual folder ID
MODEL_FOLDER = "./BertFinalModel"

@st.cache_resource
def download_and_load_model():
    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
    config_path = os.path.join(MODEL_FOLDER, "config.json")

    # Download model if necessary
    if not os.path.exists(config_path):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}"
            try:
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
    # Import after files are present
    import_transformers()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER)
        labels = {0: "TRUE NEWS", 1: "FAKE NEWS"}  # Explicit label mapping
        return tokenizer, model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def classify_text(text, tokenizer, model):
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
    return pred_idx, confidence

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("🔍 Fake News Detector")
st.markdown("---")

tokenizer, model, label_map = download_and_load_model()

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
            pred_idx, confidence = classify_text(
                text_input, tokenizer, model
            )
        pred_label = label_map.get(pred_idx, f"Label {pred_idx}")
        st.markdown("### 📊 Results")
        if pred_idx == 1:
            st.error(f"🚨 Prediction: {pred_label}")
        elif pred_idx == 0:
            st.success(f"✅ Prediction: {pred_label}")
        else:
            st.warning(f"Prediction: {pred_label}")
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
