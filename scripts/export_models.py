import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.custom_cnn import CustomCNN
from src.models.vgg16_model import VGG16Transfer
from src.models.resnet50_model import ResNet50Transfer
from src.models.mobilenet_model import MobileNetV2Transfer
from src.models.inception_model import InceptionV3Transfer


def export_to_onnx(model, model_name, input_size, save_dir="saved_models"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)
    onnx_path = save_dir / f"{model_name}.onnx"

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        print(f"Exported {model_name} to {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"Failed to export {model_name}: {e}")
        return None


def main():
    models_dir = Path("saved_models")
    model_files = list(models_dir.glob("*_best.pth"))

    for model_file in model_files:
        model_name = model_file.stem.replace("_best", "")

        if "custom_cnn" in model_name:
            activation = "relu"
            if "leaky" in model_name:
                activation = "leaky_relu"
            elif "gelu" in model_name:
                activation = "gelu"
            model = CustomCNN(num_classes=100, activation=activation)
            input_size = 32
        elif "vgg16" in model_name:
            mode = "fine_tuning" if "fine" in model_name else "feature_extraction"
            model = VGG16Transfer(num_classes=100, mode=mode)
            input_size = 224
        elif "resnet50" in model_name:
            mode = "fine_tuning" if "fine" in model_name else "feature_extraction"
            model = ResNet50Transfer(num_classes=100, mode=mode)
            input_size = 224
        elif "mobilenet" in model_name:
            mode = "fine_tuning" if "fine" in model_name else "feature_extraction"
            model = MobileNetV2Transfer(num_classes=100, mode=mode)
            input_size = 224
        elif "inception" in model_name:
            mode = "fine_tuning" if "fine" in model_name else "feature_extraction"
            model = InceptionV3Transfer(num_classes=100, mode=mode)
            input_size = 299
        else:
            continue

        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        export_to_onnx(model, model_name, input_size)


if __name__ == "__main__":
    main()
