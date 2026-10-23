#!/usr/bin/env python3
"""
Script thực thi Bước 3 (Trích chọn đặc trưng BoW, TF-IDF, Word2Vec),
Bước 4A (Huấn luyện Baseline, Naïve Bayes, SVM) và Bước 5 (Đánh giá & xuất Confusion Matrix).

Cách chạy:
    python scripts/train_classical.py
"""

import csv
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.metrics_and_plots import (
    evaluate_predictions,
    plot_dual_confusion_matrix,
    plot_macro_f1_comparison,
)
from src.features.build_features import (
    extract_bow_features,
    extract_tfidf_features,
    extract_word2vec_features,
)
from src.models.classical import train_baseline, train_naive_bayes, train_svm


def load_split(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [r["text_clean"] for r in rows]
    y_star = [int(r["label_star"]) for r in rows]
    y_target = [int(r["label_target"]) for r in rows]
    return rows, texts, y_star, y_target


def main():
    for path in ["data/processed/train.csv", "data/processed/val.csv", "data/processed/test_gold.csv"]:
        if not os.path.exists(path):
            print(f"[INFO] Chưa có {path}. Đang tự động chạy auto_label_review.py...")
            from scripts.auto_label_review import main as run_auto_label

            run_auto_label()
            break

    _, X_tr_txt, _, y_train = load_split("data/processed/train.csv")
    _, X_val_txt, y_val_star, _ = load_split("data/processed/val.csv")
    _, X_te_txt, _, y_test_gold = load_split("data/processed/test_gold.csv")

    print("[1/3] Trích chọn đặc trưng (BoW, TF-IDF, Word2Vec)...")
    _, X_tr_bow, X_val_bow, X_te_bow = extract_bow_features(X_tr_txt, X_val_txt, X_te_txt)
    _, X_tr_tfidf, X_val_tfidf, X_te_tfidf = extract_tfidf_features(X_tr_txt, X_val_txt, X_te_txt)
    _, X_tr_w2v, X_val_w2v, X_te_w2v = extract_word2vec_features(X_tr_txt, X_val_txt, X_te_txt)

    print("[2/3] Huấn luyện các mô hình (Baseline, Naïve Bayes, SVM)...")
    models_cfg = [
        ("Baseline (lớp đông nhất)", train_baseline(X_tr_bow, y_train), X_val_bow, X_te_bow, "cm_baseline_lop_dong_nhat.png"),
        ("BoW + NB", train_naive_bayes(X_tr_bow, y_train, "bow", alpha=1.0), X_val_bow, X_te_bow, "cm_bow_nb.png"),
        ("BoW + SVM", train_svm(X_tr_bow, y_train, C=0.5), X_val_bow, X_te_bow, "cm_bow_svm.png"),
        ("TF-IDF + NB", train_naive_bayes(X_tr_tfidf, y_train, "tfidf", alpha=0.5), X_val_tfidf, X_te_tfidf, "cm_tf_idf_nb.png"),
        ("TF-IDF + SVM", train_svm(X_tr_tfidf, y_train, C=1.0), X_val_tfidf, X_te_tfidf, "cm_tf_idf_svm.png"),
        ("W2V + NB", train_naive_bayes(X_tr_w2v, y_train, "w2v"), X_val_w2v, X_te_w2v, "cm_w2v_nb.png"),
        ("W2V + SVM", train_svm(X_tr_w2v, y_train, C=1.0), X_val_w2v, X_te_w2v, "cm_w2v_svm.png"),
    ]

    all_metrics = []
    model_names = []
    val_f1s = []
    test_f1s = []

    print("\n" + "=" * 88)
    print(f"{'Mô hình':<28} | {'Acc (Test)':<10} | {'Macro-P':<8} | {'Macro-R':<8} | {'Macro-F1':<8} | {'Weighted-F1':<11}")
    print("-" * 88)

    for name, clf, X_v, X_t, fig_name in models_cfg:
        val_pred = clf.predict(X_v)
        test_pred = clf.predict(X_t)

        val_res = evaluate_predictions(y_val_star, val_pred, model_name=f"{name} (val)")
        test_res = evaluate_predictions(y_test_gold, test_pred, model_name=name)
        test_res["val_macro_f1"] = val_res["macro_f1"]
        all_metrics.append(test_res)

        model_names.append(name)
        val_f1s.append(val_res["macro_f1"])
        test_f1s.append(test_res["macro_f1"])

        print(
            f"{name:<28} | {test_res['accuracy']:<10.4f} | {test_res['macro_precision']:<8.4f} | "
            f"{test_res['macro_recall']:<8.4f} | {test_res['macro_f1']:<8.4f} | {test_res['weighted_f1']:<11.4f}"
        )

        plot_dual_confusion_matrix(
            y_test_gold,
            test_pred,
            title=f"{name} (test gold, n={len(y_test_gold)})",
            save_path=os.path.join("figures", fig_name),
        )

    print("=" * 88)

    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/classical_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    plot_macro_f1_comparison(model_names, val_f1s, test_f1s, save_path="figures/macro_f1_val_test.png")
    print("[3/3] Đã lưu bảng chỉ số tại outputs/metrics/classical_metrics.json và biểu đồ tại figures/")


if __name__ == "__main__":
    main()
