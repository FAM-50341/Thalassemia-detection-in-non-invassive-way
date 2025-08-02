import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
import pytesseract
from PIL import Image
import cv2
import re
import os

# Set Tesseract path for Render
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Streamlit app title
st.title("Thalassemia Classification from CBC Report Image")

# Feature names
feature_columns = [
    'Age', 'Hb', 'Hct', 'MCV', 'MCH', 'MCHC', 'RDW', 'RBC count',
    'Sex_female', 'Sex_male', 'RDW_Hb_ratio'
]

# Load the trained model
@st.cache_resource
def load_thalassemia_model():
    try:
        model = tf_load_model("thalassemia_1dcnn_model.h5")
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
model = load_thalassemia_model()
if model is None:
    st.stop()

# Preprocess image for OCR
def preprocess_image_for_ocr(image):
    try:
        img = np.array(image.convert('L'))  # Convert to grayscale
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.bitwise_not(img_bin)
        return img_inv
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None

# Extract text with OCR
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"OCR extraction failed: {e}")
        return ""

# Parse features from OCR text
def parse_cbc_features(text):
    features = {
        "Age": None, "Sex": None, "Hb": None, "Hct": None,
        "MCV": None, "MCH": None, "MCHC": None,
        "RDW": None, "RBC count": None
    }

    # Age and Sex
    age_sex = re.search(r"Age[:\s]+(\d+)\s+Sex[:\s]+(\w+)", text, re.IGNORECASE)
    if age_sex:
        features["Age"] = int(age_sex.group(1))
        features["Sex"] = age_sex.group(2)

    # Extract values for remaining features
    for key in ["Hb", "Hct", "MCV", "MCH", "MCHC", "RDW", "RBC count"]:
        pattern = re.compile(rf"{key}[:\s]+([\d.]+)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            features[key] = float(match.group(1))

    return features

# File uploader for CBC report image
st.header("Upload CBC Report Image")
uploaded_file = st.file_uploader("Choose a CBC report image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded CBC Report", use_column_width=True)

    # Preprocess and extract text
    preprocessed_img = preprocess_image_for_ocr(image)
    if preprocessed_img is not None:
        ocr_text = extract_text_from_image(preprocessed_img)
        st.subheader("Extracted Text")
        st.text(ocr_text)

        # Parse features
        parsed_features = parse_cbc_features(ocr_text)
        st.subheader("Parsed Features")
        st.write(parsed_features)

        # Check for missing features
        if None in parsed_features.values():
            st.error("Some features could not be extracted from the image. Please check the image quality or format.")
        else:
            # Compute derived features
            parsed_features["RDW_Hb_ratio"] = parsed_features["RDW"] / parsed_features["Hb"] if parsed_features["Hb"] != 0 else 0.0
            parsed_features["Sex_female"] = 1.0 if parsed_features["Sex"].lower() == "female" else 0.0
            parsed_features["Sex_male"] = 1.0 if parsed_features["Sex"].lower() == "male" else 0.0

            # Create input array
            input_data = np.array([[
                parsed_features["Age"], parsed_features["Hb"], parsed_features["Hct"],
                parsed_features["MCV"], parsed_features["MCH"], parsed_features["MCHC"],
                parsed_features["RDW"], parsed_features["RBC count"],
                parsed_features["Sex_female"], parsed_features["Sex_male"],
                parsed_features["RDW_Hb_ratio"]
            ]])
            input_data_cnn = input_data.reshape(1, len(feature_columns), 1)

            # Save to CSV
            try:
                csv_df = pd.DataFrame([parsed_features], columns=feature_columns)
                csv_filename = "extracted_features.csv"
                csv_df.to_csv(csv_filename, index=False)
                st.success(f"Features saved to {csv_filename}")
                with open(csv_filename, "rb") as f:
                    st.download_button("Download Extracted Features CSV", f, file_name=csv_filename)
                if os.path.exists(csv_filename):
                    os.remove(csv_filename)
            except Exception as e:
                st.error(f"Failed to save CSV: {e}")

            # Make prediction
            try:
                prediction = model.predict(input_data_cnn)
                predicted_class = np.argmax(prediction, axis=1)[0]
                st.subheader("Prediction")
                st.write(f"**Predicted Class**: {predicted_class} (Probability: {prediction[0][predicted_class]:.4f})")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


