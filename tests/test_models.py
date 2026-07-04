import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.custom_cnn import CustomCNN
from src.models.vgg16_model import VGG16Transfer
from src.models.resnet50_model import ResNet50Transfer
from src.models.mobilenet_model import MobileNetV2Transfer
from src.models.inception_model import InceptionV3Transfer


def test_custom_cnn_forward():
    model = CustomCNN(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100), f"Expected (2, 100), got {out.shape}"
    print("CustomCNN forward pass: OK")


def test_custom_cnn_activations():
    for act in ["relu", "leaky_relu", "gelu"]:
        model = CustomCNN(num_classes=100, activation=act)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 100)
    print("CustomCNN all activations: OK")


def test_transfer_models():
    models = [
        VGG16Transfer(num_classes=100, mode="feature_extraction"),
        ResNet50Transfer(num_classes=100, mode="feature_extraction"),
        MobileNetV2Transfer(num_classes=100, mode="feature_extraction"),
    ]
    for model in models:
        x = torch.randn(2, 3, model.input_size, model.input_size)
        out = model(x)
        assert out.shape == (2, 100), f"{type(model).__name__}: Expected (2, 100), got {out.shape}"
    print("Transfer models forward pass: OK")


def test_model_param_counts():
    model = CustomCNN(num_classes=100)
    total = sum(p.numel() for p in model.parameters())
    assert total > 0
    print(f"CustomCNN params: {total:,}")

    model = ResNet50Transfer(num_classes=100, mode="feature_extraction")
    counts = model.get_param_count()
    assert counts["total"] > 0
    assert counts["trainable"] < counts["total"]
    print(f"ResNet50 trainable: {counts['trainable']:,} / {counts['total']:,}")


if __name__ == "__main__":
    test_custom_cnn_forward()
    test_custom_cnn_activations()
    test_transfer_models()
    test_model_param_counts()
    print("\nAll tests passed!")
