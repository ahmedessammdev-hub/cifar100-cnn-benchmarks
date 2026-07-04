# 🔬 CNN Benchmark Lab

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Weights & Biases](https://img.shields.io/badge/W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

A clean, production-grade deep learning benchmark laboratory comparing **custom-built convolutional neural networks (CNNs)** against state-of-the-art **transfer learning** architectures (VGG16, ResNet50, MobileNetV2, InceptionV3) on the CIFAR-100 image classification task. 

This repository showcases clean PyTorch engineering, modular codebase design, real-time training telemetry, model explainability via Grad-CAM, interactive visual dashboard, and microservice API deployment.

---

## 🌟 Key Features

* **Refactored Custom CNN** — Engineered with clean, high-performance `nn.Sequential` block structures, featuring configurable conv channel depths, batch normalization, multiple activation modules (ReLU, LeakyReLU, GELU), and dropout regularization.
* **Transfer Learning Benchmarks** — Implements pre-trained architectures with toggleable modes for either fast **Feature Extraction** (frozen base weights) or joint **Fine-Tuning** (selective layered optimizer updates).
* **Systematic Experimentation** — Built-in automated scripts to train and evaluate 11 different model configurations systematically under reproducible seeds and unified hyperparameters.
* **Explainable AI (XAI)** — Custom Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for deep interpretability, allowing developers to visualize activation overlays and see exactly where models focus their attention.
* **Production REST API** — High-performance FastAPI server providing asynchronous endpoints for individual model predictions, multi-model side-by-side inference comparisons, and metadata querying.
* **Interactive Web Dashboard** — A beautiful Streamlit application displaying aggregated model metrics, live Matplotlib training curve plotting, model size comparisons, and interactive Grad-CAM visualizations.
* **Experiment Tracking** — Seamless Weights & Biases (W&B) integration for robust training run metric logging, system resource utilization tracking, and hyperparameter search comparison.

---

## 🛠️ Tech Stack & Ecosystem

| Component | Tooling | Description |
|---|---|---|
| **Core Framework** | `PyTorch 2.0+` | Backbone tensor library for GPU/CPU graph computing and automatic differentiation. |
| **Vision Tools** | `torchvision` | Standard utility datasets, pre-trained ImageNet architectures, and transformation steps. |
| **API Backend** | `FastAPI` & `Uvicorn` | Modern, asynchronous ASGI framework and server for blazing fast inference REST requests. |
| **Interactive UI**| `Streamlit` | Frontend web framework for dashboard visualization and interactive evaluation demo. |
| **Explainability**| `OpenCV` | Real-time image processing, matrix transformation, and custom Jet heatmap overlay mapping. |
| **Tracking** | `Weights & Biases` | Professional ML experiment telemetry and performance dashboard logging. |
| **Testing** | `pytest` | Standardized unit tests verifying model execution graphs and data transformations. |
| **Deployment** | `Docker` | Containerized environment specifications for cross-system, scalable cloud deployments. |

---

## 📂 Codebase Structure

```directory
cnn-benchmark-lab/
├── config/
│   └── config.yaml          # Hyperparameters, paths, and training options
├── src/
│   ├── data/
│   │   ├── dataset.py       # CIFAR-100 dataset wrapper and dataloader factory
│   │   └── transforms.py    # Custom augmentations (CutOut, Color Jitter, crops)
│   ├── models/
│   │   ├── custom_cnn.py    # Sequential custom CNN architecture
│   │   ├── vgg16_model.py   # VGG16 transfer learning wrapper
│   │   ├── resnet50_model.py# ResNet50 transfer learning wrapper
│   │   ├── mobilenet_model.py# MobileNetV2 transfer learning wrapper
│   │   └── inception_model.py# InceptionV3 transfer learning wrapper
│   ├── training/
│   │   ├── trainer.py       # Standardized PyTorch trainer with early stopping
│   │   └── callbacks.py     # Optimizers, learning rate scheduling (Cosine Annealing)
│   ├── evaluation/
│   │   ├── metrics.py       # Multi-class Accuracy, F1-score, Confusion Matrix
│   │   └── grad_cam.py      # Explainable Grad-CAM activation generator
│   ├── visualization/
│   │   └── plots.py         # Visual utility for plotting training telemetry
│   └── api/
│       └── main.py          # FastAPI application serving production predictions
├── dashboard/
│   └── app.py               # Streamlit dashboard application
├── scripts/
│   ├── train_all.py         # Sequential automation script training all 11 configurations
│   ├── evaluate_all.py      # Automated benchmarking and performance compilation
│   └── download_data.py     # Clean utility to pre-fetch CIFAR-100 dataset
├── tests/
│   ├── test_models.py       # Graph shapes, parameter counts, and activations test suite
│   ├── test_data.py         # Data transformations and augmentation test suite
│   └── test_api.py          # Grad-CAM and REST API integration validation
├── Dockerfile               # Production Docker container image definition
├── Dockerfile.streamlit     # Streamlit container image definition
├── requirements.txt         # Package dependency manifest
└── setup.py                 # Package setup and dependencies installer
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10+** installed. We highly recommend a CUDA-compatible GPU environment.

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/cnn-benchmark-lab.git
cd cnn-benchmark-lab
pip install -r requirements.txt
```

### 3. Model Training
Run the training automation script to download the CIFAR-100 dataset and sequentially train all 11 model configurations:
```bash
python scripts/train_all.py
```
*To enable Weights & Biases experiment logging, set the environment flag:*
```bash
$env:USE_WANDB="true"  # On Windows PowerShell
# Or export USE_WANDB=true on Linux/macOS
python scripts/train_all.py
```

### 4. Run Interactive Dashboard
Launch the interactive Streamlit dashboard to compare accuracy charts, view metrics, and generate live heatmaps:
```bash
streamlit run dashboard/app.py
```

### 5. Run FastAPI Prediction Service
Start the local REST API server:
```bash
python -m uvicorn src.api.main:app --reload --port 8000
```
Access the interactive OpenAPI Swagger docs at `http://localhost:8000/docs`.

### 6. Containerized Deployment
Build and run the FastAPI server via Docker:
```bash
docker build -t cnn-benchmark-api .
docker run -p 8000:8000 cnn-benchmark-api
```

---

## 🧬 Custom CNN Model Architecture

The custom CNN architecture is designed around reusable `ConvBlock` modules inheriting directly from `nn.Sequential` to maximize performance and code legibility:

```
Input (3x32x32)
   ↓
ConvBlock (3 → 64 Channels)   → [Conv2d → BatchNorm → Activation → MaxPool]
   ↓
ConvBlock (64 → 128 Channels) → [Conv2d → BatchNorm → Activation → MaxPool]
   ↓
ConvBlock (128 → 256 Channels)→ [Conv2d → BatchNorm → Activation → MaxPool]
   ↓
ConvBlock (256 → 512 Channels)→ [Conv2d → BatchNorm → Activation → MaxPool]
   ↓
Global Average Pooling (1x1)
   ↓
Linear Classifier (512 → 256) → [Linear → BatchNorm → ReLU → Dropout]
   ↓
Output Head (256 → 100 Classes)
```

---

## 📈 Quality Assurance & Testing

The codebase includes standard unit tests to ensure stability when performing architecture refactoring or package upgrades. Run tests directly via Python:

```bash
# Model graph validation, activation compatibility, and parameter counts
python tests/test_models.py

# Normalization, image loading transforms, and CutOut validation
python tests/test_data.py

# Grad-CAM hooks and model gradient backward evaluation
python tests/test_api.py
```

---

## 📄 License

This repository is licensed under the **MIT License**. Check out the `LICENSE` file for more details.
