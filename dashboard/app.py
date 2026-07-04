import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import get_eval_transforms, get_denormalize_transform, CIFAR100_MEAN, CIFAR100_STD
from src.models.custom_cnn import CustomCNN
from src.models.vgg16_model import VGG16Transfer
from src.models.resnet50_model import ResNet50Transfer
from src.models.mobilenet_model import MobileNetV2Transfer
from src.models.inception_model import InceptionV3Transfer
from src.evaluation.grad_cam import GradCAM, get_target_layer_for_model

st.set_page_config(
    page_title="CNN Benchmark Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_data
def load_results(results_dir="results"):
    results_path = Path(results_dir)
    all_results = {}
    for json_file in results_path.glob("*_results.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            model_name = json_file.stem.replace("_results", "")
            all_results[model_name] = data
    return all_results


@st.cache_resource
def load_model(model_name, model_path, num_classes=100, device="cpu"):
    models_map = {
        "custom_cnn_relu": lambda: CustomCNN(num_classes=num_classes, activation="relu"),
        "custom_cnn_leaky_relu": lambda: CustomCNN(num_classes=num_classes, activation="leaky_relu"),
        "custom_cnn_gelu": lambda: CustomCNN(num_classes=num_classes, activation="gelu"),
        "vgg16_feature_extraction": lambda: VGG16Transfer(num_classes=num_classes, mode="feature_extraction"),
        "vgg16_fine_tuning": lambda: VGG16Transfer(num_classes=num_classes, mode="fine_tuning"),
        "resnet50_feature_extraction": lambda: ResNet50Transfer(num_classes=num_classes, mode="feature_extraction"),
        "resnet50_fine_tuning": lambda: ResNet50Transfer(num_classes=num_classes, mode="fine_tuning"),
        "mobilenet_v2_feature_extraction": lambda: MobileNetV2Transfer(num_classes=num_classes, mode="feature_extraction"),
        "mobilenet_v2_fine_tuning": lambda: MobileNetV2Transfer(num_classes=num_classes, mode="fine_tuning"),
        "inception_v3_feature_extraction": lambda: InceptionV3Transfer(num_classes=num_classes, mode="feature_extraction"),
        "inception_v3_fine_tuning": lambda: InceptionV3Transfer(num_classes=num_classes, mode="fine_tuning"),
    }

    for key, factory in models_map.items():
        if key in model_name.lower():
            model = factory()
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            return model

    return None


def predict_image(model, image, transform, device, class_names=None):
    model.eval()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.time()
        output = model(input_tensor)
        inference_time = (time.time() - start) * 1000

    probs = torch.softmax(output, dim=1)
    top5_probs, top5_indices = probs.topk(5)

    predictions = []
    for i in range(5):
        idx = top5_indices[0][i].item()
        prob = top5_probs[0][i].item()
        class_name = class_names[idx] if class_names else f"Class {idx}"
        predictions.append({
            "class": class_name,
            "confidence": prob * 100,
        })

    return predictions, inference_time, input_tensor


def render_overview(results):
    st.markdown('<div class="main-header">CNN Benchmark Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Comparing Custom CNN vs Transfer Learning on CIFAR-100</div>',
        unsafe_allow_html=True,
    )

    if not results:
        st.warning("No results found. Run `python scripts/train_all.py` first to generate results.")
        return

    col1, col2, col3, col4 = st.columns(4)

    best_model = max(results.items(), key=lambda x: x[1].get("test_acc", 0))
    with col1:
        st.metric("Best Model", best_model[0].replace("_", " ").title())
    with col2:
        st.metric("Best Accuracy", f"{best_model[1].get('test_acc', 0):.1f}%")
    with col3:
        st.metric("Total Models", len(results))
    with col4:
        avg_acc = np.mean([r.get("test_acc", 0) for r in results.values()])
        st.metric("Average Accuracy", f"{avg_acc:.1f}%")

    st.markdown("---")

    st.subheader("Model Comparison Table")
    rows = []
    for name, data in results.items():
        rows.append({
            "Model": name.replace("_", " ").title(),
            "Accuracy (%)": f"{data.get('test_acc', 0):.2f}",
            "F1 Score (%)": f"{data.get('f1_macro', 0):.2f}",
            "Top-5 Acc (%)": f"{data.get('top5_acc', 0):.2f}",
            "Size (MB)": f"{data.get('model_size_mb', 0):.1f}",
            "Inference (ms/img)": f"{data.get('time_per_image', 0):.2f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Accuracy Comparison")
    chart_data = pd.DataFrame({
        "Model": [r["Model"] for r in rows],
        "Accuracy": [float(r["Accuracy (%)"]) for r in rows],
    })
    st.bar_chart(chart_data.set_index("Model"))


def render_training_curves(results):
    st.subheader("Training Curves")

    if not results:
        st.warning("No training history found.")
        return

    model_names = list(results.keys())
    selected = st.selectbox("Select Model", model_names)

    if selected and "history" in results[selected]:
        history = results[selected]["history"]

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
        axes[0].plot(epochs, history["val_loss"], "r-", label="Validation", linewidth=2)
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
        axes[1].plot(epochs, history["val_acc"], "r-", label="Validation", linewidth=2)
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()

        axes[2].plot(epochs, history["lr"], "g-", linewidth=2)
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].set_yscale("log")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No training history available for this model.")


def render_gradcam_explorer(results):
    st.subheader("Grad-CAM Explorer")
    st.write("Upload an image to see where each model focuses its attention.")

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        saved_models_dir = Path("saved_models")
        model_files = list(saved_models_dir.glob("*_best.pth"))

        if model_files:
            device = load_device()
            selected_model = st.selectbox(
                "Select Model for Grad-CAM",
                [f.stem.replace("_best", "") for f in model_files],
            )

            if st.button("Generate Grad-CAM"):
                model_path = saved_models_dir / f"{selected_model}_best.pth"
                model = load_model(selected_model, str(model_path), device=device)

                if model:
                    transform = get_eval_transforms()
                    predictions, inference_time, input_tensor = predict_image(
                        model, image, transform, device
                    )

                    try:
                        target_layer = get_target_layer_for_model(model)
                        grad_cam = GradCAM(model, target_layer)
                        cam, pred_class, output = grad_cam.generate(input_tensor)

                        denorm = get_denormalize_transform()
                        original = denorm(input_tensor[0].cpu()).numpy()
                        cam_overlay = grad_cam.overlay_cam(original, cam)

                        with col2:
                            st.image(cam_overlay, caption="Grad-CAM Heatmap", use_container_width=True)

                        st.write(f"**Predicted:** {predictions[0]['class']} "
                                 f"({predictions[0]['confidence']:.1f}%) | "
                                 f"**Inference:** {inference_time:.1f}ms")
                    except Exception as e:
                        st.error(f"Grad-CAM failed: {e}")

                    st.subheader("Top-5 Predictions")
                    pred_df = pd.DataFrame(predictions)
                    st.dataframe(pred_df, hide_index=True)


def render_live_demo(results):
    st.subheader("Live Prediction Demo")
    st.write("Upload an image and get predictions from all trained models.")

    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="demo_upload")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Image", width=200)

        saved_models_dir = Path("saved_models")
        model_files = list(saved_models_dir.glob("*_best.pth"))

        if not model_files:
            st.warning("No trained models found. Run training first.")
            return

        device = load_device()
        transform = get_eval_transforms()

        all_predictions = {}
        for model_file in model_files:
            model_name = model_file.stem.replace("_best", "")
            model = load_model(model_name, str(model_file), device=device)
            if model:
                preds, inference_time, _ = predict_image(model, image, transform, device)
                all_predictions[model_name] = {
                    "predictions": preds,
                    "inference_time": inference_time,
                }

        if all_predictions:
            st.markdown("---")
            for model_name, data in all_predictions.items():
                with st.expander(f"{model_name.replace('_', ' ').title()}", expanded=True):
                    cols = st.columns([2, 1])
                    with cols[0]:
                        pred_df = pd.DataFrame(data["predictions"])
                        st.dataframe(pred_df, hide_index=True)
                    with cols[1]:
                        st.metric("Top-1", f"{data['predictions'][0]['confidence']:.1f}%")
                        st.metric("Inference", f"{data['inference_time']:.1f}ms")


def main():
    results = load_results()

    with st.sidebar:
        st.image("https://img.icons8.com/doodle/96/neural-network.png", width=80)
        st.markdown("## Navigation")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Training Curves",
        "Grad-CAM Explorer",
        "Live Demo",
    ])

    with tab1:
        render_overview(results)
    with tab2:
        render_training_curves(results)
    with tab3:
        render_gradcam_explorer(results)
    with tab4:
        render_live_demo(results)

    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #999; font-size: 0.8rem;">'
        "CNN Benchmark Lab | Custom CNN vs Transfer Learning on CIFAR-100"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
