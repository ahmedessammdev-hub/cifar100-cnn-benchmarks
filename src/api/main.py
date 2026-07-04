import io
import time
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.data.transforms import get_eval_transforms
from src.models.custom_cnn import CustomCNN
from src.models.vgg16_model import VGG16Transfer
from src.models.resnet50_model import ResNet50Transfer
from src.models.mobilenet_model import MobileNetV2Transfer
from src.models.inception_model import InceptionV3Transfer

app = FastAPI(
    title="CNN Benchmark Lab API",
    description="Image classification API comparing custom CNN vs Transfer Learning models on CIFAR-100",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path("saved_models")
RESULTS_DIR = Path("results")
NUM_CLASSES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_eval_transforms()

loaded_models = {}
model_metadata = {}


def get_model_factory(model_name):
    factories = {
        "custom_cnn_relu": lambda: CustomCNN(num_classes=NUM_CLASSES, activation="relu"),
        "custom_cnn_leaky_relu": lambda: CustomCNN(num_classes=NUM_CLASSES, activation="leaky_relu"),
        "custom_cnn_gelu": lambda: CustomCNN(num_classes=NUM_CLASSES, activation="gelu"),
        "vgg16_feature_extraction": lambda: VGG16Transfer(num_classes=NUM_CLASSES, mode="feature_extraction"),
        "vgg16_fine_tuning": lambda: VGG16Transfer(num_classes=NUM_CLASSES, mode="fine_tuning"),
        "resnet50_feature_extraction": lambda: ResNet50Transfer(num_classes=NUM_CLASSES, mode="feature_extraction"),
        "resnet50_fine_tuning": lambda: ResNet50Transfer(num_classes=NUM_CLASSES, mode="fine_tuning"),
        "mobilenet_v2_feature_extraction": lambda: MobileNetV2Transfer(num_classes=NUM_CLASSES, mode="feature_extraction"),
        "mobilenet_v2_fine_tuning": lambda: MobileNetV2Transfer(num_classes=NUM_CLASSES, mode="fine_tuning"),
        "inception_v3_feature_extraction": lambda: InceptionV3Transfer(num_classes=NUM_CLASSES, mode="feature_extraction"),
        "inception_v3_fine_tuning": lambda: InceptionV3Transfer(num_classes=NUM_CLASSES, mode="fine_tuning"),
    }
    return factories.get(model_name)


def load_model(model_name):
    if model_name in loaded_models:
        return loaded_models[model_name]

    model_path = MODELS_DIR / f"{model_name}_best.pth"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    factory = get_model_factory(model_name)
    if not factory:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_name}")

    model = factory()
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.to(DEVICE)

    loaded_models[model_name] = model
    return model


def load_all_models():
    model_files = list(MODELS_DIR.glob("*_best.pth"))
    for model_file in model_files:
        model_name = model_file.stem.replace("_best", "")
        try:
            load_model(model_name)
        except Exception:
            pass


@app.on_event("startup")
async def startup_event():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    load_all_models()

    for json_file in RESULTS_DIR.glob("*_results.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            model_name = json_file.stem.replace("_results", "")
            model_metadata[model_name] = {
                "accuracy": data.get("test_acc", 0),
                "f1_score": data.get("f1_macro", 0),
                "model_size_mb": data.get("model_size_mb", 0),
                "time_per_image_ms": data.get("time_per_image", 0),
            }


@app.get("/")
async def root():
    return {
        "message": "CNN Benchmark Lab API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "models_loaded": len(loaded_models),
    }


@app.get("/models")
async def list_models():
    available_models = []
    for model_file in MODELS_DIR.glob("*_best.pth"):
        model_name = model_file.stem.replace("_best", "")
        size_mb = model_file.stat().st_size / (1024 * 1024)
        available_models.append({
            "name": model_name,
            "size_mb": round(size_mb, 2),
            "metadata": model_metadata.get(model_name, {}),
        })
    return {"models": available_models}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = "resnet50_fine_tuning",
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        model = load_model(model_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        inference_time = (time.time() - start_time) * 1000

    probs = torch.softmax(output, dim=1)
    top5_probs, top5_indices = probs.topk(5)

    predictions = []
    for i in range(5):
        predictions.append({
            "class_index": top5_indices[0][i].item(),
            "confidence": round(top5_probs[0][i].item() * 100, 2),
        })

    return {
        "model": model_name,
        "predictions": predictions,
        "inference_time_ms": round(inference_time, 2),
    }


@app.post("/predict/all")
async def predict_all(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    results = {}

    for model_name, model in loaded_models.items():
        try:
            with torch.no_grad():
                start_time = time.time()
                output = model(input_tensor)
                inference_time = (time.time() - start_time) * 1000

            probs = torch.softmax(output, dim=1)
            top5_probs, top5_indices = probs.topk(5)

            predictions = []
            for i in range(5):
                predictions.append({
                    "class_index": top5_indices[0][i].item(),
                    "confidence": round(top5_probs[0][i].item() * 100, 2),
                })

            results[model_name] = {
                "predictions": predictions,
                "inference_time_ms": round(inference_time, 2),
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}

    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
