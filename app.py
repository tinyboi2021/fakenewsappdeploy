import streamlit as st
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- This is your final, corrected model path ---
MODEL_FOLDER = r"C:\Users\pauls\Downloads\Mini_Project Demo\BertFinalModel\BertFinalModel" 
# --- ---

@st.cache_resource
def load_model(folder_path):
    """
    Loads the saved Hugging Face model and tokenizer from a local folder.
    """
    try:
        # Load the tokenizer and model from the specified folder
        tokenizer = AutoTokenizer.from_pretrained(folder_path)
        model = AutoModelForSequenceClassification.from_pretrained(folder_path)
        
        # Get label mappings from the model's config (e.g., 0: "TRUE", 1: "FAKE")
        labels = model.config.id2label
        
        return tokenizer, model, labels
    
    except OSError as e:
        # This error happens if the folder path is wrong
        st.error(f"Error: Model folder not found at '{folder_path}'.")
        st.error(f"Please make sure this path is correct. Full error: {e}")
        st.stop()
    except Exception as e:
        # This catches other errors, like the 'model_type' error
        st.error(f"An error occurred while loading the model: {e}")
        st.error("This often means the 'config.json' file is missing 'model_type': 'bert'. Please see the instructions.")
        st.stop()

# --- Load the Model ---
# This will only run once at the start
model_data = load_model(MODEL_FOLDER)

if model_data:
    tokenizer, model, id2label = model_data
    
    # --- STREAMLIT UI ---
    
    st.set_page_config(page_title="Fake News Detector", page_icon="📰")

    st.title("📰 Fake News Detector")
    st.markdown(
        "Enter a news headline or article text below to check if it's likely true or fake."
    )

    # Text area for user input
    user_input = st.text_area(
        "Enter news text here:", 
        "", 
        height=200,
        placeholder="e.g., 'Scientists discover new planet made entirely of diamond...'"
    )

    # Analyze button
    if st.button("Analyze News", type="primary"):
        if user_input:
            # Show a spinner while processing
            with st.spinner("Analyzing..."):
                
                # 1. Tokenize the user input
                inputs = tokenizer(
                    user_input, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                
                # 2. Make a prediction
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # 3. Get the raw prediction scores (logits)
                logits = outputs.logits
                
                # 4. Convert logits to probabilities
                probabilities = torch.softmax(logits, dim=1)
                
                # 5. Get the most likely class index and its confidence
                confidence = probabilities.max().item() * 100
                prediction_index = torch.argmax(probabilities, dim=1).item()

            # 4. Display the result
            
            label_name = ""
            if id2label:
                # Use the label name from the model's config (e.g., "FAKE" or "TRUE")
                label_name = id2label[prediction_index]
            else:
                # Fallback if no labels are in the config
                label_name = f"Class {prediction_index}"
                st.warning(f"Model config does not contain 'id2label' mappings. Displaying raw class index.")
                st.warning("Using index 0 as 'TRUE' and index 1 as 'FAKE' as requested.")

            # --- YOUR FINAL LABEL LOGIC ---
            # You stated that Label 1 is "Fake" and Label 0 is "True".
            
            # Check if the predicted index is 0 (TRUE News)
            if prediction_index == 0:
                st.success(f"**Prediction: TRUE News**")
                st.write(f"Confidence: {confidence:.2f}%")
            # Check if the predicted index is 1 (FAKE News)
            elif prediction_index == 1:
                st.error(f"**Prediction: FAKE News**")
                st.write(f"Confidence: {confidence:.2f}%")
            else:
                # A fallback for any other unexpected labels
                st.warning(f"**Prediction: {label_name or prediction_index}**")
                st.write(f"Confidence: {confidence:.2f}%")
            # --- End of final logic ---
                    
        else:
            # Show a warning if the text area is empty
            st.warning("Please enter some text to analyze.")

    # --- Sidebar with instructions ---
    st.sidebar.header("How to Run")
    st.sidebar.markdown(
        f"""
        1.  Save this file as `app.py`.
        2.  Save the `requirements.txt` file in the same folder.
        3.  Install dependencies: `pip install -r requirements.txt`
        4.  Run the app: `streamlit run app.py`
        """
    )
    st.sidebar.header("Your Model Path")
    st.sidebar.code(MODEL_FOLDER, language="text")
    
    st.sidebar.header("Label Configuration")
    st.sidebar.info(
        "This app is configured for your model:\n"
        "* **Label 0 = TRUE News (Green)**\n"
        "* **Label 1 = FAKE News (Red)**"
    )

