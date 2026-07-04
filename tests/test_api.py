import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from src.evaluation.grad_cam import GradCAM, get_target_layer_for_model
from src.models.custom_cnn import CustomCNN
from src.data.transforms import get_eval_transforms


def test_gradcam():
    model = CustomCNN(num_classes=100, activation="relu")
    target_layer = get_target_layer_for_model(model)
    grad_cam = GradCAM(model, target_layer)

    dummy_input = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_input)
    transform = get_eval_transforms()
    input_tensor = transform(image).unsqueeze(0)

    cam, pred_class, output = grad_cam.generate(input_tensor)
    assert cam is not None
    assert pred_class is not None
    print(f"Grad-CAM output shape: {cam.shape}")
    print(f"Predicted class: {pred_class}")
    print("Grad-CAM test: OK")


if __name__ == "__main__":
    test_gradcam()
    print("\nAll Grad-CAM tests passed!")
