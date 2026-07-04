import os
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset as _hf_load_dataset

import sys
import json
import time
import torch
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CIFAR100DataModule
from src.data.transforms import get_eval_transforms
from src.models.custom_cnn import CustomCNN
from src.models.vgg16_model import VGG16Transfer
from src.models.resnet50_model import ResNet50Transfer
from src.models.mobilenet_model import MobileNetV2Transfer
from src.models.inception_model import InceptionV3Transfer
from src.training.trainer import Trainer, EarlyStopping
from src.training.callbacks import get_optimizer, get_scheduler, get_transfer_optimizer
from src.evaluation.metrics import compute_metrics, get_top_k_accuracy
from src.visualization.plots import plot_training_curves, plot_model_comparison

USE_WANDB = os.environ.get("USE_WANDB", "false").lower() == "true"


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_custom_cnn(config, data_module, device, activations=None):
    if activations is None:
        activations = ["relu"]

    results = {}
    base_config = config["custom_cnn"]
    train_config = config["training"]

    for activation in activations:
        print(f"\n{'='*60}", flush=True)
        print(f"Training Custom CNN with {activation}", flush=True)
        print(f"{'='*60}", flush=True)

        model = CustomCNN(
            num_classes=config["data"]["num_classes"],
            conv_channels=base_config["conv_layers"],
            fc_dims=base_config["fc_layers"],
            dropout=base_config["dropout"],
            use_batch_norm=base_config["use_batch_norm"],
            activation=activation,
        )

        optimizer = get_optimizer(model, train_config)
        scheduler = get_scheduler(optimizer, train_config)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_dir=config["paths"]["saved_models"],
            use_wandb=USE_WANDB,
        )

        early_stopping = EarlyStopping(
            patience=train_config.get("early_stopping_patience", 10),
            min_delta=train_config.get("min_delta", 0.001),
        )

        history = trainer.train(
            train_loader=data_module.get_train_loader(),
            val_loader=data_module.get_val_loader(),
            epochs=train_config["epochs"],
            early_stopping=early_stopping,
            model_name=f"custom_cnn_{activation}",
        )

        trainer.load_checkpoint(f"custom_cnn_{activation}")
        eval_results = trainer.evaluate(data_module.get_test_loader())

        metrics = compute_metrics(
            eval_results["targets"],
            eval_results["predictions"],
        )

        results[f"custom_cnn_{activation}"] = {
            "test_acc": eval_results["test_acc"],
            "test_loss": eval_results["test_loss"],
            "f1_macro": metrics["f1_macro"],
            "top5_acc": get_top_k_accuracy(
                eval_results["targets"],
                eval_results["probabilities"],
                k=5,
            ),
            "time_per_image": eval_results["time_per_image"],
            "model_size_mb": trainer.get_model_size(),
            "history": {
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
                "train_acc": history["train_acc"],
                "val_acc": history["val_acc"],
                "lr": history["lr"],
            },
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }

        plot_training_curves(
            history,
            model_name=f"Custom CNN ({activation})",
            save_dir=config["paths"]["plots"],
        )

    return results


def train_transfer_model(model_class, model_name, config, data_module, device, mode="feature_extraction"):
    print(f"\n{'='*60}", flush=True)
    print(f"Training {model_name} - {mode}", flush=True)
    print(f"{'='*60}", flush=True)

    num_classes = config["data"]["num_classes"]
    model = model_class(num_classes=num_classes, mode=mode)

    tl_config = config["transfer_learning"]
    if mode == "feature_extraction":
        train_epochs = tl_config["feature_extraction"]["epochs"]
        lr = tl_config["feature_extraction"]["lr"]
    else:
        train_epochs = tl_config["fine_tuning"]["epochs"]
        lr = tl_config["fine_tuning"]["lr"]

    optimizer = get_transfer_optimizer(model, mode, lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)

    full_name = f"{model_name}_{mode}"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config["paths"]["saved_models"],
        use_wandb=USE_WANDB,
    )

    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    history = trainer.train(
        train_loader=data_module.get_train_loader(),
        val_loader=data_module.get_val_loader(),
        epochs=train_epochs,
        early_stopping=early_stopping,
        model_name=full_name,
    )

    trainer.load_checkpoint(full_name)
    eval_results = trainer.evaluate(data_module.get_test_loader())

    metrics = compute_metrics(
        eval_results["targets"],
        eval_results["predictions"],
    )

    result = {
        "test_acc": eval_results["test_acc"],
        "test_loss": eval_results["test_loss"],
        "f1_macro": metrics["f1_macro"],
        "top5_acc": get_top_k_accuracy(
            eval_results["targets"],
            eval_results["probabilities"],
            k=5,
        ),
        "time_per_image": eval_results["time_per_image"],
        "model_size_mb": trainer.get_model_size(),
        "history": {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_acc": history["train_acc"],
            "val_acc": history["val_acc"],
            "lr": history["lr"],
        },
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }

    plot_training_curves(
        history,
        model_name=full_name.replace("_", " ").title(),
        save_dir=config["paths"]["plots"],
    )

    return full_name, result


def train_all_transfer(config, data_module, device):
    models_to_train = [
        (VGG16Transfer, "vgg16"),
        (ResNet50Transfer, "resnet50"),
        (MobileNetV2Transfer, "mobilenet_v2"),
        (InceptionV3Transfer, "inception_v3"),
    ]

    results = {}
    for model_class, model_name in models_to_train:
        for mode in ["feature_extraction", "fine_tuning"]:
            try:
                name, result = train_transfer_model(
                    model_class, model_name, config, data_module, device, mode
                )
                results[name] = result
            except Exception as e:
                print(f"Error training {model_name} ({mode}): {e}", flush=True)

    return results


def save_results(results, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for model_name, data in results.items():
        output = {k: v for k, v in data.items() if k != "history"}
        output["history"] = data.get("history", {})

        with open(save_dir / f"{model_name}_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)

    comparison = {}
    for name, data in results.items():
        comparison[name] = {
            "test_acc": data["test_acc"],
            "f1_macro": data["f1_macro"],
            "model_size_mb": data.get("model_size_mb", 0),
            "time_per_image": data.get("time_per_image", 0),
        }

    with open(save_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)


def main():
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if USE_WANDB:
        import wandb
        wandb.login()

    print("Loading dataset...", flush=True)
    data_module = CIFAR100DataModule(config["data"])
    data_module.setup()

    print(f"\nDataset: CIFAR-100", flush=True)
    print(f"Train: {len(data_module.train_dataset)}", flush=True)
    print(f"Val: {len(data_module.val_dataset)}", flush=True)
    print(f"Test: {len(data_module.test_dataset)}", flush=True)

    all_results = {}

    print("\n" + "="*60, flush=True)
    print("PHASE 1: Custom CNN Experiments", flush=True)
    print("="*60, flush=True)
    custom_results = train_custom_cnn(config, data_module, device)
    all_results.update(custom_results)

    print("\n" + "="*60, flush=True)
    print("PHASE 2: Transfer Learning", flush=True)
    print("="*60, flush=True)
    transfer_results = train_all_transfer(config, data_module, device)
    all_results.update(transfer_results)

    print("\n" + "="*60, flush=True)
    print("Saving Results", flush=True)
    print("="*60, flush=True)
    save_results(all_results, config["paths"]["results"])

    plot_model_comparison(
        {k: v for k, v in all_results.items()},
        metrics=["test_acc", "f1_macro"],
        save_dir=config["paths"]["plots"],
    )

    print("\n" + "="*60, flush=True)
    print("FINAL RESULTS", flush=True)
    print("="*60, flush=True)
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["test_acc"],
        reverse=True,
    )
    for name, data in sorted_results:
        print(f"{name:40s} | Acc: {data['test_acc']:6.2f}% | "
              f"F1: {data['f1_macro']:6.2f}% | "
              f"Size: {data.get('model_size_mb', 0):6.1f}MB | "
              f"Time: {data.get('time_per_image', 0):6.2f}ms/img", flush=True)

    print(f"\nResults saved to: {config['paths']['results']}", flush=True)
    print("Training complete!", flush=True)


if __name__ == "__main__":
    main()
