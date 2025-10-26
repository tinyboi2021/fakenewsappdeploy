import streamlit as st
import time
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gdown

# --- Google Drive model folder ID ---
GOOGLE_DRIVE_FOLDER_ID = "1Xgg_vwh2pVrsi4rzHFymnrXvr0RfU3-M"  # Replace with your folder ID
MODEL_FOLDER = "https://drive.google.com/drive/folders/1Xgg_vwh2pVrsi4rzHFymnrXvr0RfU3-M?usp=drive_link"

@st.cache_resource
def load_model():
    """
    Downloads the model from Google Drive (if not already cached) and loads it.
    """
    model_path = Path(MODEL_FOLDER)
    model_path.mkdir(exist_ok=True)
    
    # Check if model files exist locally
    config_path = model_path / "config.json"
    
    if not config_path.exists():
        with st.spinner("📥 Downloading model from Google Drive (first time only)..."):
            try:
                # Download folder from Google Drive
                gdown.download_folder(
                    id=GOOGLE_DRIVE_FOLDER_ID,
                    output=MODEL_FOLDER,
                    quiet=False,
                    use_cookies=False
                )
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Error downloading model: {e}")
                st.stop()
    
    try:
        # Load the tokenizer and model from the specified folder
        tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER)
        
        # Get label mappings from the model's config
        labels = model.config.id2label
        
        return tokenizer, model, labels
        
    except OSError as e:
        st.error(f"❌ Error: Model files not found at '{MODEL_FOLDER}'.")
        st.error(f"Please ensure the Google Drive folder ID is correct. Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

def classify_text(text, tokenizer, model, labels):
    """
    Classifies text as True or Fake news using the BERT model.
    """
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
    prediction = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1)[0][prediction].item()
    
    return labels[prediction], confidence

# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("🔍 Fake News Detector")
st.markdown("---")

# Load model
tokenizer, model, labels = load_model()

# Input section
st.markdown("### Enter Text to Classify")
text_input = st.text_area(
    "Paste your news article or text here:",
    height=200,
    placeholder="Enter the text you want to analyze..."
)

# Classify button
if st.button("🚀 Classify Text", use_container_width=True):
    if text_input.strip():
        with st.spinner("⏳ Analyzing..."):
            time.sleep(0.5)  # Brief delay for UX
            
            prediction, confidence = classify_text(
                text_input, 
                tokenizer, 
                model, 
                labels
            )
        
        # Display results
        st.markdown("### 📊 Results")
        
        # Color-coded prediction
        if prediction == "FAKE":
            st.error(f"🚨 **Prediction: {prediction}**")
        else:
            st.success(f"✅ **Prediction: {prediction}**")
        
        # Confidence bar
        st.metric(
            label="Confidence",
            value=f"{confidence*100:.2f}%"
        )
        
        # Detailed analysis
        st.markdown("#### Detailed Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Classification:** {prediction}")
        with col2:
            st.write(f"**Confidence Score:** {confidence:.4f}")
    
    else:
        st.warning("⚠️ Please enter some text to classify!")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Built with Streamlit & BERT | Fake News Detection System"
    "</div>",
    unsafe_allow_html=True
)
