import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

from models import get_model
from utils import load_model, get_device, load_config
from data import IMAGENET_MEAN, IMAGENET_STD
from gradcam import gradcam_for_pil_image

CONFIG = load_config("configs/config.yaml")
CHECKPOINT_PATH = "checkpoints/best_model_densenet121.pt"
THRESHOLD = CONFIG.get("threshold", 0.3)
IMAGE_SIZE = CONFIG.get("image_size", 224)
MODEL_NAME = CONFIG.get("model", "densenet121")

EXAMPLE_IMAGES = {
    "NORMAL — IM-0001-0001.jpeg": "data/chest_xray/test/NORMAL/IM-0001-0001.jpeg",
    "NORMAL — IM-0003-0001.jpeg": "data/chest_xray/test/NORMAL/IM-0003-0001.jpeg",
    "PNEUMONIA — person100_bacteria_475.jpeg": "data/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg",
    "PNEUMONIA — person100_bacteria_477.jpeg": "data/chest_xray/test/PNEUMONIA/person100_bacteria_477.jpeg",
}


@st.cache_resource
def load_trained_model():
    device = get_device()
    model = get_model(MODEL_NAME, pretrained=False)
    model = load_model(model, CHECKPOINT_PATH, device)
    model = model.to(device)
    model.eval()
    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0)


def run_inference(image: Image.Image, model, device):
    tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        output = model(tensor).squeeze(1)
        prob = torch.sigmoid(output).item()
    label = "PNEUMONIA" if prob >= THRESHOLD else "NORMAL"
    return label, prob


def show_prediction(label, prob):
    st.divider()
    st.subheader("Prediction")

    if label == "PNEUMONIA":
        st.error(f"⚠️ **PNEUMONIA DETECTED**")
    else:
        st.success(f"✅ **NORMAL** — No pneumonia detected")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classification", label)
    with col2:
        st.metric("Probability", f"{prob:.4f}")

    st.divider()
    st.subheader("Confidence")
    st.progress(min(prob, 1.0))
    st.caption(
        "Higher probability → more confident pneumonia detection. "
        f"Threshold: {THRESHOLD}"
    )


def show_gradcam(image: Image.Image, model, device):
    st.divider()
    with st.expander("🔍 Grad-CAM Explainability — What did the model look at?", expanded=True):
        st.caption(
            "Grad-CAM highlights the regions most influential for the model's decision. "
            "In a well-trained model, activations should concentrate on the lung fields, "
            "not on scanner borders or text annotations."
        )
        with st.spinner("Computing Grad-CAM…"):
            heatmap, overlay = gradcam_for_pil_image(
                image=image,
                model=model,
                model_name=MODEL_NAME,
                image_size=IMAGE_SIZE,
                device=device,
            )

        col1, col2 = st.columns(2)
        with col1:
            st.image(heatmap, caption="Activation heatmap", clamp=True, use_container_width=True)
        with col2:
            st.image(overlay, caption="Overlay", use_container_width=True)


def main():
    st.set_page_config(
        page_title="Pneumonia Detection",
        page_icon="🫁",
        layout="centered",
    )

    st.title("🫁 Pneumonia Detection from Chest X-Rays")
    st.caption(
        "Upload a chest X-ray image to classify it as **Normal** or **Pneumonia**. "
        "Powered by a fine-tuned DenseNet-121 model."
    )

    model, device = load_trained_model()

    st.divider()

    tab_upload, tab_examples = st.tabs(["📤 Upload Image", "🖼️ Example Images"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-ray", use_container_width=True)
            label, prob = run_inference(image, model, device)
            show_prediction(label, prob)
            show_gradcam(image, model, device)

    with tab_examples:
        selected = st.selectbox("Select an example image", list(EXAMPLE_IMAGES.keys()))
        if selected:
            path = EXAMPLE_IMAGES[selected]
            if os.path.exists(path):
                image = Image.open(path).convert("RGB")
                st.image(image, caption=selected, use_container_width=True)
                if st.button("Run Prediction"):
                    label, prob = run_inference(image, model, device)
                    show_prediction(label, prob)
                    show_gradcam(image, model, device)
            else:
                st.warning(f"Example image not found: {path}")


if __name__ == "__main__":
    main()
