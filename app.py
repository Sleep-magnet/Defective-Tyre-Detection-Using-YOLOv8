import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARN"] = "1"
os.environ["PYTHONASYNCIODEBUG"] = "1"
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best (2).pt")

model = load_model()

# Streamlit UI
st.title("🛞 Defective Tyre Detection")
st.write("Upload an image to detect defects using a YOLO model.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and show original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temp file because ultralytics expects a path or ndarray
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        results = model(tmp.name)

    # Draw detections
    result_image = Image.fromarray(results[0].plot())
    st.image(result_image, caption="Detection Result", use_column_width=True)
