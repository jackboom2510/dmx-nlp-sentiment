#!/usr/bin/env python3
"""Train and compare classical sentiment baselines across BoW, TF-IDF and Word2Vec."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.metrics_and_plots import (
    bootstrap_macro_f1_ci,
    evaluate_predictions,
    plot_dual_confusion_matrix,
    plot_macro_f1_comparison,
)
from src.features.build_features import extract_bow_features, extract_tfidf_features, extract_word2vec_features
from src.models.classical import train_baseline, train_naive_bayes, train_svm


def load_split(filepath: Path):
    with filepath.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    texts = [row["text_clean"] for row in rows]
    labels = [int(row["label_target"]) for row in rows]
    return rows, texts, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT_DIR / "data/processed/train.csv")
    parser.add_argument("--validation", type=Path, default=ROOT_DIR / "data/processed/val.csv")
    parser.add_argument("--test", type=Path, default=ROOT_DIR / "data/processed/test.csv")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for test Macro-F1 95% CI")
    args = parser.parse_args()

    required = [args.train, args.validation, args.test]
    if not all(p.exists() for p in required):
        print("[INFO] Processed split missing; generating it from raw reviews.")
        from scripts.auto_label_review import main as run_auto_label
        run_auto_label()

    train_rows, train_text, y_train = load_split(args.train)
    val_rows, val_text, y_val = load_split(args.validation)
    test_rows, test_text, y_test = load_split(args.test)
    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Train, validation and test splits must all contain rows")

    print(f"Rows: train={len(y_train):,}; validation={len(y_val):,}; held-out test={len(y_test):,}")
    print("All evaluation targets are label_target (hybrid auto-labels); test is not human-gold annotated.")
    print("[1/3] Extracting features; vectorizers and Word2Vec fit on train only...")
    _, Xtr_bow, Xv_bow, Xte_bow = extract_bow_features(train_text, val_text, test_text)
    _, Xtr_tfidf, Xv_tfidf, Xte_tfidf = extract_tfidf_features(train_text, val_text, test_text)
    _, Xtr_w2v, Xv_w2v, Xte_w2v = extract_word2vec_features(train_text, val_text, test_text)

    specs = [
        ("Majority baseline", lambda: train_baseline(Xtr_bow, y_train), Xv_bow, Xte_bow),
        ("BoW + MultinomialNB", lambda: train_naive_bayes(Xtr_bow, y_train, "bow", alpha=1.0), Xv_bow, Xte_bow),
        ("BoW + LinearSVC", lambda: train_svm(Xtr_bow, y_train, C=0.5), Xv_bow, Xte_bow),
        ("TF-IDF + MultinomialNB", lambda: train_naive_bayes(Xtr_tfidf, y_train, "tfidf", alpha=0.5), Xv_tfidf, Xte_tfidf),
        ("TF-IDF + LinearSVC", lambda: train_svm(Xtr_tfidf, y_train, C=1.0), Xv_tfidf, Xte_tfidf),
        ("Word2Vec + GaussianNB", lambda: train_naive_bayes(Xtr_w2v, y_train, "w2v"), Xv_w2v, Xte_w2v),
        ("Word2Vec + LinearSVC", lambda: train_svm(Xtr_w2v, y_train, C=1.0), Xv_w2v, Xte_w2v),
    ]

    metrics, names, val_f1, test_f1 = [], [], [], []
    predictions_dir = ROOT_DIR / "outputs/predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = ROOT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name, fit_model, X_val, X_test in specs:
        train_start = time.perf_counter()
        model = fit_model()
        training_seconds = time.perf_counter() - train_start
        pred_val = model.predict(X_val).astype(int).tolist()
        pred_test = model.predict(X_test).astype(int).tolist()
        val_result = evaluate_predictions(y_val, pred_val, model_name=name)
        test_result = evaluate_predictions(y_test, pred_test, model_name=name)
        lo, hi = bootstrap_macro_f1_ci(y_test, pred_test, n_bootstrap=args.bootstrap)
        test_result.update({
            "validation_macro_f1": val_result["macro_f1"],
            "test_macro_f1_ci95": [round(lo, 4), round(hi, 4)],
            "test_label_source": "label_target (hybrid auto-label; not human gold)",
            "training_seconds": round(training_seconds, 4),
        })
        metrics.append(test_result)
        names.append(name)
        val_f1.append(val_result["macro_f1"])
        test_f1.append(test_result["macro_f1"])

        slug = name.lower().replace("+", "").replace(" ", "_").replace("-", "_")
        test_result["prediction_file"] = f"outputs/predictions/{slug}.csv"
        test_result["confusion_matrix_file"] = f"figures/cm_{slug}.png"
        plot_dual_confusion_matrix(y_test, pred_test, f"{name} — held-out test (n={len(y_test)})",
                                   str(figures_dir / f"cm_{slug}.png"))
        with (predictions_dir / f"{slug}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["review_id", "category", "content", "y_true", "y_pred", "correct"])
            writer.writeheader()
            for row, gold, pred in zip(test_rows, y_test, pred_test):
                writer.writerow({"review_id": row.get("review_id", ""), "category": row.get("category", ""),
                                 "content": row.get("content", ""), "y_true": gold, "y_pred": pred,
                                 "correct": int(gold == pred)})

    metric_path = ROOT_DIR / "outputs/metrics/classical_metrics.json"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_macro_f1_comparison(names, val_f1, test_f1, str(figures_dir / "macro_f1_val_test.png"))
    print(f"[3/3] Saved metrics: {metric_path}")
    for item in sorted(metrics, key=lambda x: x["macro_f1"], reverse=True):
        print(f"{item['model']:<30} accuracy={item['accuracy']:.3f} macro-F1={item['macro_f1']:.3f} "
              f"95% CI={item['test_macro_f1_ci95']}")


if __name__ == "__main__":
    main()
