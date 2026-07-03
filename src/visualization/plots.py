import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_training_curves(history, model_name="Model", save_dir=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0].set_title(f"{model_name} - Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    axes[1].set_title(f"{model_name} - Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    axes[2].plot(epochs, history["lr"], "g-", linewidth=2)
    axes[2].set_title(f"{model_name} - Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")

    plt.tight_layout()
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{model_name}_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_model_comparison(results, metrics=None, save_dir=None):
    if metrics is None:
        metrics = ["test_acc", "f1_macro"]

    model_names = list(results.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))

    if len(metrics) == 1:
        axes = [axes]

    colors = sns.color_palette("husl", len(model_names))

    for ax, metric in zip(axes, metrics):
        values = [results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(model_names, values, color=colors)
        ax.set_title(f"Model Comparison - {metric}")
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(values) * 1.2)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_confusion_matrix_heatmap(cm, model_name="Model", figsize=(20, 16), save_dir=None):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=False, cmap="Blues", ax=ax, square=True)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"{model_name} - Confusion Matrix", fontsize=14)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{model_name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_per_class_accuracy(targets, predictions, class_names=None, model_name="Model", save_dir=None):
    from sklearn.metrics import accuracy_score

    unique_targets = np.unique(targets)
    per_class_acc = []

    for cls in unique_targets:
        mask = targets == cls
        if mask.sum() > 0:
            acc = accuracy_score(targets[mask] == cls, predictions[mask] == cls) * 100
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0)

    per_class_acc = np.array(per_class_acc)

    fig, ax = plt.subplots(figsize=(20, 6))
    x = range(len(per_class_acc))
    ax.bar(x, per_class_acc, color=sns.color_palette("viridis", len(per_class_acc)))
    ax.set_xlabel("Class Index")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{model_name} - Per-Class Accuracy")
    ax.set_ylim(0, 105)

    if class_names and len(class_names) == len(per_class_acc):
        ax.set_xticks(range(0, len(class_names), 5))
        ax.set_xticklabels([class_names[i] for i in range(0, len(class_names), 5)],
                           rotation=45, ha="right")

    plt.tight_layout()
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{model_name}_per_class_acc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_activation_comparison(results, save_dir=None):
    activations = ["ReLU", "LeakyReLU", "GELU"]
    available = [a for a in activations if a in results]

    if len(available) < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = sns.color_palette("husl", len(available))

    for i, act in enumerate(available):
        history = results[act]
        epochs = range(1, len(history["val_acc"]) + 1)
        axes[0].plot(epochs, history["val_acc"], label=act, linewidth=2, color=colors[i])
        axes[1].plot(epochs, history["val_loss"], label=act, linewidth=2, color=colors[i])

    axes[0].set_title("Activation Functions - Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()

    axes[1].set_title("Activation Functions - Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "activation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_gradcam_visualization(original, cam_overlay, predicted_class, true_class,
                               confidence, model_name="Model", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if isinstance(original, np.ndarray):
        if original.ndim == 3 and original.shape[0] in [1, 3]:
            original = np.transpose(original, (1, 2, 0))
        if original.max() <= 1.0:
            original = (original * 255).astype(np.uint8)

    axes[0].imshow(original)
    axes[0].set_title(f"Original\nTrue: {true_class}")
    axes[0].axis("off")

    axes[1].imshow(cam_overlay)
    axes[1].set_title(f"Grad-CAM\nPred: {predicted_class} ({confidence:.1f}%)")
    axes[1].axis("off")

    plt.suptitle(f"{model_name} - Grad-CAM Visualization", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
