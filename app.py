
import io
import time

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

from config import CLASS_NAMES, IMAGE_SIZE, RISK_COLORS
from gradcam import GradCAM, overlay_heatmap_on_image
from model import build_model, load_checkpoint, load_lightweight_checkpoint, predict
from preprocessing import bgr_to_tensor, remove_hair_dullrazor, resize_image

st.set_page_config(
    page_title="Skin Lesion Classification & Melanoma Detection",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model weights...")
def get_model(backbone_name: str, checkpoint_path: str = ""):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path and checkpoint_path.endswith(".pkl"):
        try:
            model = load_lightweight_checkpoint(checkpoint_path, device=device)
            st.sidebar.success(f"Loaded lightweight checkpoint: {checkpoint_path} "
                                f"(backbone: {model.backbone_name})")
            return model, device
        except FileNotFoundError:
            st.sidebar.warning(
                "No lightweight checkpoint found at that path -- using an ImageNet-pretrained "
                "backbone only. Run extract_features.py then train_head.py to produce one "
                "(see README.md's fast CPU workflow)."
            )
        except Exception as exc:
            st.sidebar.error(f"Failed to load lightweight checkpoint: {exc}")
        model = build_model(backbone_name=backbone_name, pretrained=True, device=device)
        return model, device

    model = build_model(backbone_name=backbone_name, pretrained=True, device=device)
    if checkpoint_path:
        try:
            model = load_checkpoint(model, checkpoint_path, device=device)
            st.sidebar.success(f"Loaded full checkpoint: {checkpoint_path}")
        except FileNotFoundError:
            st.sidebar.warning(
                "No checkpoint found at that path -- using an ImageNet-pretrained backbone only."
            )
        except Exception as exc:
            st.sidebar.error(f"Failed to load checkpoint: {exc}")

    return model, device


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Converts a PIL RGB image (from st.file_uploader) into an OpenCV BGR array."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_display(image_bgr: np.ndarray) -> np.ndarray:
    """Converts BGR (OpenCV) back to RGB for st.image display."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def render_risk_banner(risk_level: str, predicted_class: str, confidence: float):
    color = RISK_COLORS.get(risk_level, "#616161")
    st.markdown(
        f"""
        <div style="background-color:{color};padding:18px 24px;border-radius:10px;
                    color:white;font-size:20px;font-weight:600;margin-bottom:10px;">
            {risk_level} Risk &nbsp;|&nbsp; {predicted_class} &nbsp;|&nbsp;
            Confidence: {confidence * 100:.2f}%
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_chart(class_names, probabilities):
    sorted_pairs = sorted(zip(class_names, probabilities), key=lambda p: p[1], reverse=True)
    names = [p[0] for p in sorted_pairs]
    values = [p[1] * 100 for p in sorted_pairs]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(color=values, colorscale="RdYlGn_r"),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Class Probability Distribution",
        xaxis_title="Probability (%)",
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("🩺 Automated Skin Lesion Classification & Melanoma Detection")
    st.caption(
        "Research / educational demo only -- **NOT** a medical device and **NOT** a "
        "substitute for professional dermatological diagnosis."
    )

    
    st.sidebar.header("Model Settings")
    backbone_name = st.sidebar.selectbox(
        "Model architecture", options=["efficientnet_b4", "resnet50"], index=0
    )
    checkpoint_path = st.sidebar.text_input(
        "Trained checkpoint path (optional)", value="checkpoints/skin_lesion_model.pkl",
        help="A .pkl path (from train_head.py) auto-detects its own backbone. "
             "A .pth path (from train.py) is loaded onto the backbone selected above."
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence flag threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Predictions below this confidence are flagged as 'uncertain' in the UI."
    )
    apply_hair_removal = st.sidebar.checkbox("Apply DullRazor hair removal", value=True)
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM heatmap", value=True)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This demo ships with an ImageNet-pretrained backbone. For clinically meaningful "
        "predictions, fine-tune on HAM10000 using `train.py` first (see README.md)."
    )

    model, device = get_model(backbone_name, checkpoint_path)

    
    uploaded_file = st.file_uploader(
        "Upload a dermoscopic lesion image (PNG/JPG)", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is None:
        st.info("Upload an image to begin analysis.")
        return

    try:
        pil_image = Image.open(io.BytesIO(uploaded_file.read()))
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")
        return

    raw_bgr = pil_to_bgr(pil_image)

    with st.spinner("Preprocessing image..."):
        try:
            clean_bgr = remove_hair_dullrazor(raw_bgr) if apply_hair_removal else raw_bgr.copy()
            resized_raw = resize_image(raw_bgr, IMAGE_SIZE)
            resized_clean = resize_image(clean_bgr, IMAGE_SIZE)
            input_tensor = bgr_to_tensor(resized_clean)
        except Exception as exc:
            st.error(f"Preprocessing failed: {exc}")
            return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Uploaded Image")
        st.image(bgr_to_display(resized_raw), use_container_width=True)
    with col2:
        label = "Hair-Removed Clean Image" if apply_hair_removal else "Clean Image (hair removal off)"
        st.subheader(label)
        st.image(bgr_to_display(resized_clean), use_container_width=True)

    with st.spinner("Running inference..."):
        try:
            start = time.time()
            result = predict(model, input_tensor, device=device)
            elapsed = time.time() - start
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return

    st.markdown("### Prediction")
    render_risk_banner(result.risk_level, result.predicted_class, result.confidence)
    st.caption(f"Inference time: {elapsed * 1000:.1f} ms on {device.upper()}")

    if result.confidence < confidence_threshold:
        st.warning(
            f"⚠️ Confidence ({result.confidence * 100:.1f}%) is below your configured threshold "
            f"({confidence_threshold * 100:.0f}%). Treat this prediction as uncertain."
        )

    col3, col4 = st.columns(2)

    with col3:
        if show_gradcam:
            st.subheader("Grad-CAM Lesion Focus")
            try:
                target_layer = model.get_target_layer()
                cam_engine = GradCAM(model, target_layer)
                heatmap, used_class_idx = cam_engine.generate(input_tensor, device=device)
                overlay = overlay_heatmap_on_image(heatmap, resized_clean)
                st.image(
                    bgr_to_display(overlay), use_container_width=True,
                    caption=f"Heatmap explains prediction: {CLASS_NAMES[used_class_idx]}",
                )
            except Exception as exc:
                st.error(f"Grad-CAM generation failed: {exc}")

    with col4:
        render_probability_chart(result.class_names, result.probabilities)

    with st.expander("Full probability table"):
        for name, prob in sorted(zip(result.class_names, result.probabilities),
                                  key=lambda p: p[1], reverse=True):
            st.write(f"**{name}**: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()