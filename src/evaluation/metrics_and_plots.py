#!/usr/bin/env python3
"""
Module đánh giá mô hình và trực quan hóa kết quả (Bước 5):
- Tính toán: Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1 và chỉ số theo từng lớp
- Vẽ Confusion Matrix 2 bảng (Số đếm + Chuẩn hóa theo hàng Recall %)
- Vẽ biểu đồ thanh ngang so sánh Macro-F1 giữa Val (nhãn sao) và Test (nhãn gold)
"""

import os
from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
LABELS_ORDER = [-1, 0, 1]


def evaluate_predictions(y_true: List[int], y_pred: List[int], model_name: str = "") -> Dict:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=LABELS_ORDER, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
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


def plot_dual_confusion_matrix(y_true: List[int], y_pred: List[int], title: str, save_path: str) -> None:
    """Vẽ Confusion Matrix gồm 2 panel (Số đếm & Chuẩn hóa theo hàng Recall %) và lưu vào save_path."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDER)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(title, fontsize=14, y=1.02)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
        annot_kws={"size": 12},
    )
    axes[0].set_title("Số đếm", fontsize=11)
    axes[0].set_xlabel("Dự đoán")
    axes[0].set_ylabel("Nhãn gold")

    annot_pct = np.array([[f"{val * 100:.0f}%" for val in row] for row in cm_norm])
    sns.heatmap(
        cm_norm,
        annot=annot_pct,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
        vmin=0.0,
        vmax=1.0,
        annot_kws={"size": 12},
    )
    axes[1].set_title("Chuẩn hóa theo hàng (recall)", fontsize=11)
    axes[1].set_xlabel("Dự đoán")
    axes[1].set_ylabel("Nhãn gold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_macro_f1_comparison(
    model_names: List[str],
    val_scores: List[float],
    test_scores: List[float],
    save_path: str = "figures/macro_f1_val_test.png",
) -> None:
    """Vẽ biểu đồ thanh ngang kép so sánh Macro-F1 giữa Val (nhãn sao) và Test (nhãn gold)."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    y = np.arange(len(model_names))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(y - height / 2, val_scores, height, label="Val (nhãn sao)", color="#2b7bba")
    rects2 = ax.barh(y + height / 2, test_scores, height, label="Test (nhãn gold)", color="#eb6b34")

    ax.set_xlabel("Macro-F1")
    ax.set_title("Macro-F1 theo mô hình: val (nhãn sao) và test (nhãn gold)", loc="left", fontsize=13)
    ax.set_yticks(y)
    ax.set_yticklabels(model_names)
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    ax.grid(axis="x", linestyle="-", alpha=0.3)

    for rect in rects2:
        w = rect.get_width()
        ax.annotate(
            f"{w:.3f}",
            xy=(w + 0.01, rect.get_y() + rect.get_height() / 2),
            va="center",
            ha="left",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
