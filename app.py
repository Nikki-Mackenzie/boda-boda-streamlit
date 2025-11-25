# app.py
"""
Boda Boda Safety Compliance Detector - Streamlit app
Detects: helmet / no_helmet, reflector / no_reflector, overload / no_overload
Place best.pt in the same folder as this file.
"""

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = "best.pt"

st.set_page_config(
    page_title="Boda Boda Safety Detector",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Boda Boda Safety")
    st.markdown(
        """
Use this demo to verify helmet, reflector and overload compliance.
- Upload one or more images.
- Adjust confidence threshold and preview results.
"""
    )
    conf = st.slider("Confidence threshold", 0.01, 0.9, 0.25, step=0.01)
    show_boxes = st.checkbox("Show bounding boxes (annotated image)", value=True)
    show_json = st.checkbox("Show prediction JSON", value=False)
    st.markdown("---")
    st.caption("Model must be named `best.pt` and located in the app folder.")

# --- HELPERS ---
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load YOLO model once and cache it."""
    return YOLO(path)

def run_inference(model, pil_image: Image.Image, conf_thres: float):
    """Run model and return annotated image (PIL) and results dict."""
    start = time.time()
    results = model.predict(source=np.array(pil_image), conf=conf_thres, imgsz=640)
    elapsed = time.time() - start

    # annotated image (numpy)
    annotated = results[0].plot()
    annotated_pil = Image.fromarray(annotated)

    # Build simple JSON-style summary
    preds = []
    for det in results[0].boxes.data.tolist():  # [x1,y1,x2,y2,score,class]
        x1, y1, x2, y2, score, cls = det
        preds.append(
            {
                "class": results[0].names[int(cls)],
                "confidence": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            }
        )

    meta = {"elapsed_seconds": elapsed, "num_predictions": len(preds)}
    return annotated_pil, preds, meta

def pil_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# --- MAIN ---
st.title("🛵 Boda Boda Safety Compliance Detector")
st.write(
    "A Streamlit demo using a YOLOv8 model to detect helmet usage, reflectors, and overloading."
)

# Check model availability
if not Path(MODEL_PATH).exists():
    st.error(f"Model file not found at `{MODEL_PATH}`. Upload `best.pt` to the app folder.")
    st.stop()

# Load model (cached)
with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH)

# File uploader (allow multiple)
uploaded_files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(2)
    left_col = cols[0]
    right_col = cols[1]

    for idx, ufile in enumerate(uploaded_files):
        # Open image
        image = Image.open(ufile).convert("RGB")

        # Run inference
        annotated_img, preds, meta = run_inference(model, image, conf)

        # Layout headers for first image
        if idx == 0:
            left_col.subheader("Annotated output")
            right_col.subheader("Predictions & info")

        # Show annotated image or original
        if show_boxes:
            left_col.image(annotated_img, caption=f"Annotated: {ufile.name}", use_column_width=True)
        else:
            left_col.image(image, caption=f"Input: {ufile.name}", use_column_width=True)

        # Prediction panel
        with right_col.expander(f"Details — {ufile.name}", expanded=True):
            st.write(f"**Inference time:** {meta['elapsed_seconds']:.2f} s")
            st.write(f"**Detections:** {meta['num_predictions']}")
            if show_json:
                st.json(preds)
            # Download annotated image
            img_bytes = pil_to_bytes(annotated_img)
            st.download_button(
                label="Download annotated image",
                data=img_bytes,
                file_name=f"annotated_{Path(ufile.name).stem}.png",
                mime="image/png",
            )

else:
    st.info("Upload images to run the detector. Try photos taken on Thika Road, Mombasa Road, or Kangundo Road.")

# Footer / credits
st.markdown("---")
st.markdown("**Notes:** Model trained for six classes: helmet, no_helmet, reflector, no_reflector, overload, no_overload.")
st.markdown("If inference fails on Streamlit Cloud, ensure `best.pt` is present and `requirements.txt` matches the training environment.")
