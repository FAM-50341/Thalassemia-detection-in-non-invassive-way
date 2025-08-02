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
import platform
import subprocess

# Set Tesseract path explicitly for Windows
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Verify Tesseract installation
def verify_tesseract():
    try:
        # Check if tesseract executable exists
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            raise FileNotFoundError(f"Tesseract executable not found at {pytesseract.pytesseract.tesseract_cmd}")
        
        # Run tesseract --version to verify
        result = subprocess.run([pytesseract.pytesseract.tesseract_cmd, "--version"], 
                              capture_output=True, text=True, check=True)
        st.info(f"Tesseract OCR is installed: {result.stdout.splitlines()[0]}")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        st.error(f"Tesseract is not installed or not accessible: {e}")
        st.error(f"Expected Tesseract at: {tesseract_path}")
        st.error("1. Verify 'tesseract.exe' exists in 'C:\\Program Files\\Tesseract-OCR'.")
        st.error("2. Reinstall Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        st.error("3. Add 'C:\\Program Files\\Tesseract-OCR' to your system PATH using PowerShell:")
        st.code('$tesseractPath = "C:\\Program Files\\Tesseract-OCR"\n$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")\nif ($currentPath -notlike "*$tesseractPath*") {\n    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$tesseractPath", "Machine")\n}')
        st.error("4. Restart PowerShell or your computer after updating PATH.")
        return False

if not verify_tesseract():
    st.stop()

# Streamlit app title
st.title("Thalassemia Classification from CBC Report Image")

# Feature names
feature_columns = [
    "Age", "Hb", "Hct", "MCV", "MCH", "MCHC", "RDW", "RBC count",
    "Sex_female", "Sex_male", "RDW_Hb_ratio"
]

# Load the trained model
@st.cache_resource
def load_thalassemia_model():
    try:
        model = tf_load_model("thalassemia_1dcnn_model.h5")
        model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
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
        img = np.array(image.convert("L"))  # Convert to grayscale
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img_inv = cv2.bitwise_not(img_bin)
        return img_inv
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None

# Extract text with OCR
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()
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
    age_sex = re.search(r"Age[:\s]*(\d+)\s*(?:Sex[:\s]*(\w+))?", text, re.IGNORECASE)
    if age_sex:
        features["Age"] = int(age_sex.group(1))
        if age_sex.group(2):
            features["Sex"] = age_sex.group(2).lower()

    # Extract values for remaining features with more flexible regex
    for key in ["Hb", "Hct", "MCV", "MCH", "MCHC", "RDW", "RBC count"]:
        pattern = re.compile(rf"{key}[\s:]*([\d.]+)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            try:
                features[key] = float(match.group(1))
            except ValueError:
                st.warning(f"Could not convert {key} value to float: {match.group(1)}")

    return features

# File uploader for CBC report image
st.header("Upload CBC Report Image")
uploaded_file = st.file_uploader("Choose a CBC report image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display the uploaded image
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded CBC Report", use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        st.stop()

    # Preprocess and extract text
    preprocessed_img = preprocess_image_for_ocr(image)
    if preprocessed_img is not None:
        ocr_text = extract_text_from_image(preprocessed_img)
        st.subheader("Extracted Text")
        st.text(ocr_text if ocr_text else "No text extracted. Check image quality or Tesseract configuration.")

        # Parse features
        parsed_features = parse_cbc_features(ocr_text)
        st.subheader("Parsed Features")
        st.write(parsed_features)

        # Check for missing features
        missing_features = [key for key, value in parsed_features.items() if value is None]
        if missing_features:
            st.error(f"Some features could not be extracted: {', '.join(missing_features)}. Please check the image quality or format.")
        else:
            # Compute derived features
            try:
                parsed_features["RDW_Hb_ratio"] = parsed_features["RDW"] / parsed_features["Hb"] if parsed_features["Hb"] != 0 else 0.0
                parsed_features["Sex_female"] = 1.0 if parsed_features["Sex"] == "female" else 0.0
                parsed_features["Sex_male"] = 1.0 if parsed_features["Sex"] == "male" else 0.0
            except Exception as e:
                st.error(f"Error computing derived features: {e}")
                st.stop()

            # Create input array
            try:
                input_data = np.array([[
                    parsed_features["Age"], parsed_features["Hb"], parsed_features["Hct"],
                    parsed_features["MCV"], parsed_features["MCH"], parsed_features["MCHC"],
                    parsed_features["RDW"], parsed_features["RBC count"],
                    parsed_features["Sex_female"], parsed_features["Sex_male"],
                    parsed_features["RDW_Hb_ratio"]
                ]])
                input_data_cnn = input_data.reshape(1, len(feature_columns), 1)
            except Exception as e:
                st.error(f"Error preparing input data: {e}")
                st.stop()

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
