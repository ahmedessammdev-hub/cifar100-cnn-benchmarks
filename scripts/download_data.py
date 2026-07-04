from datasets import load_dataset
from pathlib import Path

DATA_DIR = "data"


def main():
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)

    print("Downloading CIFAR-100 from HuggingFace...")
    ds = load_dataset("uoft-cs/cifar100", cache_dir=str(data_dir / "hf_cache"))

    print(f"Train: {len(ds['train'])} samples")
    print(f"Test: {len(ds['test'])} samples")
    print("Done! You can now run: python scripts/train_all.py")


if __name__ == "__main__":
    main()
