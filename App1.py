import os
import random
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SAR Oil Spill Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "unet_oilspill.h5"

# Local validation images folder
IMAGE_DIRS = ["val_images"]
MASK_DIRS = ["val_images"]

# ---------------- STYLE ----------------
st.markdown(
    """
<style>
    .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(120,120,120,0.16);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(18,18,18,0.04), rgba(18,18,18,0.01));
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2.05rem; line-height: 1.15; }
    .hero p { margin: 0.35rem 0 0 0; color: rgba(85,85,85,0.95); font-size: 0.98rem; }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.35rem 0 0.75rem 0;
    }
    .card {
        border: 1px solid rgba(120,120,120,0.16);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        background: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.035);
        height: 100%;
    }
    .card .label {
        font-size: 0.82rem;
        color: rgba(100,100,100,0.92);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }
    .card .value {
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0;
    }
    .card .note {
        margin-top: 0.2rem;
        color: rgba(90,90,90,0.92);
        font-size: 0.9rem;
    }
    .small-note {
        color: rgba(100,100,100,0.88);
        font-size: 0.88rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- HELPERS ----------------
@st.cache_resource
def load_unet():
    return load_model(MODEL_PATH, compile=False)

def resolve_dir(candidates):
    for d in candidates:
        if os.path.exists(d):
            return d
    return None

IMAGE_DIR = resolve_dir(IMAGE_DIRS)
MASK_DIR = resolve_dir(MASK_DIRS)

def list_files(image_dir):
    if not image_dir or not os.path.exists(image_dir):
        return []
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    return sorted([f for f in os.listdir(image_dir) if f.lower().endswith(exts)])

def read_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def preprocess(img):
    img256 = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
    inp = img256[np.newaxis, :, :, np.newaxis]
    return img256, inp

def predict(model, img):
    img256, inp = preprocess(img)
    pred = model.predict(inp, verbose=0)[0, :, :, 0]
    mask = (pred > threshold).astype(np.uint8)
    return img256, pred, mask

def overlay_image(img256, mask):
    base = (img256 * 255).astype(np.uint8)
    overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    color = overlay.copy()
    color[mask == 1] = [255, 0, 0]
    return cv2.addWeighted(overlay, 0.74, color, 0.26, 0)

def compute_iou_dice(gt_mask, pred_mask):
    gt = (gt_mask > 127).astype(np.uint8)
    pr = (pred_mask > 0).astype(np.uint8)

    inter = np.sum(gt * pr)
    union = np.sum(gt) + np.sum(pr) - inter
    iou = inter / (union + 1e-6)

    dice = (2 * inter) / (np.sum(gt) + np.sum(pr) + 1e-6)
    return iou, dice

def metric_box(label, value, note):
    st.markdown(
        f"""
        <div class="card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- HEADER ----------------
st.markdown(
    """
<div class="hero">
    <h1>🌊 SAR Oil Spill Detection Dashboard</h1>
    <p>U-Net based segmentation dashboard for satellite oil spill detection.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Controls")
threshold = st.sidebar.slider("Threshold", 0.10, 0.90, 0.50, 0.01)

files = list_files(IMAGE_DIR)
if files:
    min_val = min(3, len(files))
    max_val = min(9, len(files))
    default_val = min(6, len(files))
    if min_val < max_val:
        sample_count = st.sidebar.slider("Validation samples", min_val, max_val, default_val, 1)
    else:
        sample_count = max_val
else:
    sample_count = 6

st.sidebar.markdown("---")
st.sidebar.caption("Model: U-Net | Input: 256×256 grayscale | Output: binary mask")

# ---------------- LOAD MODEL ----------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

with st.spinner("Loading model..."):
    model = load_unet()

# ---------------- TOP METRICS ----------------
st.markdown('<div class="section-title">Model Snapshot</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_box("Architecture", "U-Net", "Semantic segmentation model")
with m2:
    metric_box("Input Size", "256×256", "Normalized grayscale image")
with m3:
    metric_box("Output", "Mask", "Pixel-wise oil spill region")
with m4:
    metric_box("Threshold", f"{threshold:.2f}", "Mask binarization")

st.markdown("---")

# ---------------- LIVE PREDICTION ----------------
st.markdown('<div class="section-title">Live Prediction</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload SAR image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("Could not read the uploaded image.")
    else:
        img256, pred, mask = predict(model, img)
        overlay = overlay_image(img256, mask)

        coverage = float(mask.mean() * 100)
        max_prob = float(pred.max())
        mean_prob = float(pred.mean())

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_box("Coverage", f"{coverage:.2f}%", "Predicted spill area")
        with c2:
            metric_box("Max Confidence", f"{max_prob:.3f}", "Peak probability")
        with c3:
            metric_box("Mean Confidence", f"{mean_prob:.3f}", "Average probability")

        g1, g2, g3 = st.columns(3)
        with g1:
            st.image(img256, caption="Input", use_container_width=True, clamp=True)
        with g2:
            st.image(mask * 255, caption="Prediction", use_container_width=True, clamp=True)
        with g3:
            st.image(overlay, caption="Overlay", use_container_width=True, clamp=True)
else:
    st.info("Upload an image to run the model.")

st.markdown("---")

# ---------------- VALIDATION SAMPLE ----------------
st.markdown('<div class="section-title">Validation Sample</div>', unsafe_allow_html=True)

if not files or not IMAGE_DIR or not MASK_DIR:
    st.warning("Validation dataset not found at the configured path.")
else:
    selected = st.selectbox("Select a validation image", files, index=0)

    img_path = os.path.join(IMAGE_DIR, selected)
    mask_path = os.path.join(MASK_DIR, selected)

    val_img = read_gray(img_path)
    val_mask = read_gray(mask_path) if os.path.exists(mask_path) else None

    if val_img is None or val_mask is None:
        st.warning("Could not load the selected validation image or mask.")
    else:
        val_img = cv2.resize(val_img, (256, 256))
        val_mask = cv2.resize(val_mask, (256, 256))

        val_img_norm, val_pred, val_pred_mask = predict(model, val_img)
        val_overlay = overlay_image(val_img_norm, val_pred_mask)
        iou, dice = compute_iou_dice(val_mask, val_pred_mask)

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            metric_box("IoU", f"{iou:.4f}", "Prediction vs mask")
        with v2:
            metric_box("Dice", f"{dice:.4f}", "Shape overlap score")
        with v3:
            metric_box("Coverage", f"{val_pred_mask.mean() * 100:.2f}%", "Predicted area")
        with v4:
            metric_box("Peak Prob.", f"{val_pred.max():.3f}", "Strongest response")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(val_img, caption="Validation Input", use_container_width=True, clamp=True)
        with c2:
            st.image(val_mask, caption="Ground Truth", use_container_width=True, clamp=True)
        with c3:
            st.image(val_overlay, caption="Prediction Overlay", use_container_width=True, clamp=True)

        st.markdown("### More Validation Samples")
        sample_files = random.sample(files, min(sample_count, len(files)))
        cols = st.columns(3)

        for i, fname in enumerate(sample_files):
            ip = os.path.join(IMAGE_DIR, fname)
            mp = os.path.join(MASK_DIR, fname)

            img = read_gray(ip)
            msk = read_gray(mp) if os.path.exists(mp) else None

            if img is None or msk is None:
                continue

            img = cv2.resize(img, (256, 256))
            msk = cv2.resize(msk, (256, 256))

            _, _, pred_mask = predict(model, img)
            over = overlay_image(img.astype(np.float32) / 255.0, pred_mask)

            with cols[i % 3]:
                st.image(img, caption=fname, use_container_width=True, clamp=True)
                st.image(msk, caption="Mask", use_container_width=True, clamp=True)
                st.image(over, caption="Overlay", use_container_width=True, clamp=True)

st.markdown("---")

# ---------------- TRAINING RESULT SNAPSHOT ----------------
st.markdown('<div class="section-title">Training Snapshot</div>', unsafe_allow_html=True)
t1, t2, t3 = st.columns(3)
with t1:
    metric_box("IoU", "0.5778", "Reported test score")
with t2:
    metric_box("Dice", "0.6944", "Reported test score")
with t3:
    metric_box("ROC-AUC", "> 0.50", "Above random baseline")

st.caption("Research demo only. Validate outputs before operational use.")