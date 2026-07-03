import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def compute_metrics(targets, predictions, class_names=None):
    acc = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average="macro", zero_division=0)
    recall = recall_score(targets, predictions, average="macro", zero_division=0)
    f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
    cm = confusion_matrix(targets, predictions)

    metrics = {
        "accuracy": acc * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_macro": f1 * 100,
        "f1_per_class": f1_per_class * 100,
        "confusion_matrix": cm,
    }

    if class_names:
        metrics["classification_report"] = classification_report(
            targets, predictions, target_names=class_names, zero_division=0
        )

    return metrics


def plot_confusion_matrix(cm, class_names=None, figsize=(20, 16), save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=False, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def get_top_k_accuracy(targets, probabilities, k=5):
    top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
    correct = sum(1 for i, t in enumerate(targets) if t in top_k_preds[i])
    return correct / len(targets) * 100
