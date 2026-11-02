#!/usr/bin/env python3
"""Evaluation metrics and plotting helpers for three-class sentiment experiments."""

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
LABELS_ORDER = [-1, 0, 1]


def evaluate_predictions(y_true: Sequence[int], y_pred: Sequence[int], model_name: str = "") -> Dict:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if y_true_arr.shape != y_pred_arr.shape or y_true_arr.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays with equal lengths")

    acc = accuracy_score(y_true_arr, y_pred_arr)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=LABELS_ORDER, average="macro", zero_division=0
    )
    _, _, f1_weight, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=LABELS_ORDER, average="weighted", zero_division=0
    )
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=LABELS_ORDER, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=LABELS_ORDER)

    return {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weight), 4),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(float(p_cls[i]), 4),
                "recall": round(float(r_cls[i]), 4),
                "f1": round(float(f1_cls[i]), 4),
                "support": int(sup_cls[i]),
            }
            for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
    }


def bootstrap_macro_f1_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap interval for macro-F1 on a fixed held-out sample."""
    from sklearn.metrics import f1_score

    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    if yt.size == 0 or yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must be non-empty arrays with equal lengths")
    rng = np.random.default_rng(seed)
    scores = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, yt.size, size=yt.size)
        scores[i] = f1_score(yt[idx], yp[idx], labels=LABELS_ORDER, average="macro", zero_division=0)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1.0 - alpha))


def plot_dual_confusion_matrix(y_true: List[int], y_pred: List[int], title: str, save_path: str) -> None:
    """Plot raw confusion counts and row-normalized recall percentages."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDER)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(title, fontsize=14, y=1.02)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0], annot_kws={"size": 12})
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Reference label")
    annot_pct = np.array([[f"{v * 100:.0f}%" for v in row] for row in cm_norm])
    sns.heatmap(cm_norm, annot=annot_pct, fmt="", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1], vmin=0.0, vmax=1.0)
    axes[1].set_title("Row-normalized (recall)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Reference label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_macro_f1_comparison(
    model_names: List[str], val_scores: List[float], test_scores: List[float],
    save_path: str = "figures/macro_f1_val_test.png",
) -> None:
    """Compare validation and held-out-test macro-F1 (both use the same label protocol)."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    y = np.arange(len(model_names))
    height = 0.36
    fig, ax = plt.subplots(figsize=(10, max(5.8, 0.55 * len(model_names))))
    ax.barh(y - height / 2, val_scores, height, label="Validation", color="#2b7bba")
    rects = ax.barh(y + height / 2, test_scores, height, label="Held-out test", color="#eb6b34")
    ax.set_xlabel("Macro-F1")
    ax.set_title("Validation vs held-out test Macro-F1")
    ax.set_yticks(y)
    ax.set_yticklabels(model_names)
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.3)
    for rect in rects:
        width = rect.get_width()
        ax.annotate(f"{width:.3f}", (width + 0.01, rect.get_y() + rect.get_height() / 2), va="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
