import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import get_train_transforms, get_eval_transforms, get_imagenet_transforms


def test_train_transforms():
    transform = get_train_transforms()
    import numpy as np
    dummy = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    from PIL import Image
    img = Image.fromarray(dummy)
    tensor = transform(img)
    assert tensor.shape == (3, 32, 32), f"Expected (3, 32, 32), got {tensor.shape}"
    print("Train transforms: OK")


def test_eval_transforms():
    transform = get_eval_transforms()
    import numpy as np
    dummy = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    from PIL import Image
    img = Image.fromarray(dummy)
    tensor = transform(img)
    assert tensor.shape == (3, 32, 32)
    print("Eval transforms: OK")


def test_imagenet_transforms():
    for size in [224, 299]:
        transform = get_imagenet_transforms(image_size=size, train=True)
        import numpy as np
        dummy = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        from PIL import Image
        img = Image.fromarray(dummy)
        tensor = transform(img)
        assert tensor.shape == (3, size, size)
    print("ImageNet transforms: OK")


if __name__ == "__main__":
    test_train_transforms()
    test_eval_transforms()
    test_imagenet_transforms()
    print("\nAll data tests passed!")
