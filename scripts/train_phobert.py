#!/usr/bin/env python3
"""
Script thực thi Bước 4B (Bổ sung Ý 4): Huấn luyện Mô hình Hiện đại PhoBERT (`vinai/phobert-base-v2`)
kết hợp Weighted Cross-Entropy Loss trên tập dữ liệu Điện Máy Xanh.

Cách chạy:
    python scripts/train_phobert.py --epochs 4 --batch-size 16
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.metrics_and_plots import evaluate_predictions, plot_dual_confusion_matrix
from src.models.phobert import (
    DMXSentimentDataset,
    ID2LABEL,
    ID2NAME,
    WeightedPhoBERTTrainer,
    compute_balanced_weights,
    compute_hf_metrics,
)


def load_segmented_split(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [r["text_segmented"] for r in rows]
    y_target = [int(r["label_target"]) for r in rows]
    return texts, y_target


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT cho phân tích cảm xúc Điện Máy Xanh")
    parser.add_argument("--model-name", default="vinai/phobert-base-v2", help="Tên model HuggingFace")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default="outputs/phobert_checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_texts, y_train = load_segmented_split("data/processed/train.csv")
    val_texts, y_val = load_segmented_split("data/processed/val.csv")
    test_texts, y_test = load_segmented_split("data/processed/test_gold.csv")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = DMXSentimentDataset(train_texts, y_train, tokenizer, max_length=args.max_length)
    val_ds = DMXSentimentDataset(val_texts, y_val, tokenizer, max_length=args.max_length)
    test_ds = DMXSentimentDataset(test_texts, y_test, tokenizer, max_length=args.max_length)

    class_weights = compute_balanced_weights(y_train)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2NAME,
        label2id={v: k for k, v in ID2NAME.items()},
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
    )

    trainer = WeightedPhoBERTTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_hf_metrics,
    )

    trainer.train()

    test_out = trainer.predict(test_ds)
    pred_ids = np.argmax(test_out.predictions, axis=-1)
    y_test_pred = [ID2LABEL[int(i)] for i in pred_ids]

    metrics = evaluate_predictions(y_test, y_test_pred, model_name="PhoBERT")
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/phobert_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_dual_confusion_matrix(
        y_test,
        y_test_pred,
        title=f"PhoBERT (test gold, n={len(y_test)})",
        save_path="figures/cm_phobert.png",
    )
    print("[DONE] Kết quả PhoBERT trên tập test_gold:", json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
