import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CIFAR100DataModule
from src.evaluation.metrics import compute_metrics, get_top_k_accuracy
from src.visualization.plots import plot_confusion_matrix_heatmap


def load_config():
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    results_dir = Path(config["paths"]["results"])

    print("Loading dataset...")
    data_module = CIFAR100DataModule(config["data"])
    data_module.prepare_data()
    data_module.setup()
    test_loader = data_module.get_test_loader()

    print("\nEvaluation Results:")
    print("="*80)

    all_results = {}
    for json_file in results_dir.glob("*_results.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            model_name = json_file.stem.replace("_results", "")
            all_results[model_name] = data

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get("test_acc", 0),
        reverse=True,
    )

    print(f"{'Model':<40} {'Accuracy':>10} {'F1':>10} {'Size':>10} {'Speed':>12}")
    print("-"*82)
    for name, data in sorted_results:
        print(f"{name:<40} "
              f"{data.get('test_acc', 0):>9.2f}% "
              f"{data.get('f1_macro', 0):>9.2f}% "
              f"{data.get('model_size_mb', 0):>8.1f}MB "
              f"{data.get('time_per_image', 0):>9.2f}ms")

    best_model = sorted_results[0] if sorted_results else None
    if best_model:
        print(f"\nBest Model: {best_model[0]} ({best_model[1]['test_acc']:.2f}%)")

    print(f"\nResults available in: {results_dir}")


if __name__ == "__main__":
    main()
