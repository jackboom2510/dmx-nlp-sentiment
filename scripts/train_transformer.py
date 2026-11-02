#!/usr/bin/env python3
"""Fine-tune a modern multilingual or Vietnamese transformer for DMX sentiment.

Examples:
  python scripts/train_transformer.py --model-name vinai/phobert-base-v2
  python scripts/train_transformer.py --model-name FacebookAI/xlm-roberta-large --batch-size 4 --max-length 160

PhoBERT is fed segmented Vietnamese text; XLM-R uses original review text by default.
The held-out labels are hybrid-auto-label targets, not independently human-gold annotations.
"""

import argparse
import csv
import inspect
import json
import re
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    EarlyStoppingCallback,
    TrainingArguments,
)

from src.evaluation.metrics_and_plots import (
    bootstrap_macro_f1_ci,
    evaluate_predictions,
    plot_dual_confusion_matrix,
)
from src.models.transformer import (
    ID2LABEL,
    ID2NAME,
    SentimentDataset,
    WeightedSequenceClassificationTrainer,
    balanced_class_weights,
)


def read_split(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows, [int(row["label_target"]) for row in rows]


def compute_metrics(eval_prediction):
    logits, label_ids = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(np.mean(predictions == label_ids)),
        "macro_f1": float(f1_score(label_ids, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="vinai/phobert-base-v2")
    parser.add_argument("--input-field", choices=["auto", "content", "text_segmented", "text_clean"], default="auto")
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs/models")
    parser.add_argument("--train", type=Path, default=ROOT_DIR / "data/processed/train.csv")
    parser.add_argument("--validation", type=Path, default=ROOT_DIR / "data/processed/val.csv")
    parser.add_argument("--test", type=Path, default=ROOT_DIR / "data/processed/test.csv")
    args = parser.parse_args()

    if not all(p.exists() for p in [args.train, args.validation, args.test]):
        raise FileNotFoundError("Processed splits not found; first run `python auto_label_review.py`.")

    is_phobert = "phobert" in args.model_name.lower()
    input_field = args.input_field
    if input_field == "auto":
        input_field = "text_segmented" if is_phobert else "content"

    train_rows, y_train = read_split(args.train)
    val_rows, y_val = read_split(args.validation)
    test_rows, y_test = read_split(args.test)
    for name, rows in [("train", train_rows), ("validation", val_rows), ("test", test_rows)]:
        if not rows:
            raise ValueError(f"{name} split is empty")
        missing = [i for i, row in enumerate(rows) if not row.get(input_field, "").strip()]
        if missing:
            raise ValueError(f"{name} has {len(missing)} empty values in {input_field}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label={i: name for i, name in ID2NAME.items()},
        label2id={name: i for i, name in ID2NAME.items()},
    )
    train_ds = SentimentDataset([r[input_field] for r in train_rows], y_train, tokenizer, args.max_length)
    val_ds = SentimentDataset([r[input_field] for r in val_rows], y_val, tokenizer, args.max_length)
    test_ds = SentimentDataset([r[input_field] for r in test_rows], y_test, tokenizer, args.max_length)

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.model_name).strip("_").lower()
    model_dir = args.output_dir / slug
    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    training_kwargs = {
        "output_dir": str(model_dir),
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size * 2,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": 0.1,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "logging_strategy": "steps",
        "logging_steps": 50,
        "seed": args.seed,
        "report_to": "none",
        "fp16": bool(torch.cuda.is_available()),
        "dataloader_num_workers": 0,
    }
    training_kwargs[eval_key] = "epoch"
    training_args = TrainingArguments(**training_kwargs)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    trainer_kwargs = {
        "class_weights": balanced_class_weights(y_train),
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = WeightedSequenceClassificationTrainer(**trainer_kwargs)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Fine-tuning {args.model_name} | input={input_field} | device={device_name}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - train_start
    inference_start = time.perf_counter()
    val_metrics = trainer.evaluate(val_ds, metric_key_prefix="validation")
    test_output = trainer.predict(test_ds, metric_key_prefix="test")
    inference_seconds = time.perf_counter() - inference_start
    logits = test_output.predictions
    pred_ids = np.argmax(logits, axis=-1)
    y_pred = [ID2LABEL[int(idx)] for idx in pred_ids]
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    result = evaluate_predictions(y_test, y_pred, model_name=args.model_name)
    ci = bootstrap_macro_f1_ci(y_test, y_pred, n_bootstrap=args.bootstrap, seed=args.seed)
    result.update({
        "validation_macro_f1": round(float(val_metrics.get("validation_macro_f1", 0.0)), 4),
        "test_macro_f1_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "model_name": args.model_name,
        "input_field": input_field,
        "max_length": args.max_length,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "training_seconds": round(training_seconds, 2),
        "evaluation_seconds": round(inference_seconds, 2),
        "device": device_name,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if torch.cuda.is_available() else None,
        "test_label_source": "label_target (hybrid auto-label; not human gold)",
        "prediction_file": f"outputs/predictions/{slug}.csv",
        "confusion_matrix_file": f"figures/cm_{slug}.png",
    })

    metrics_dir = ROOT_DIR / "outputs/metrics"
    predictions_dir = ROOT_DIR / "outputs/predictions"
    figures_dir = ROOT_DIR / "figures"
    for folder in (metrics_dir, predictions_dir, figures_dir):
        folder.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{slug}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_dual_confusion_matrix(y_test, y_pred, f"{args.model_name} — held-out test (n={len(y_test)})",
                               str(figures_dir / f"cm_{slug}.png"))
    pred_path = predictions_dir / f"{slug}.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["review_id", "category", "content", "y_true", "y_pred", "confidence", "p_negative", "p_neutral", "p_positive"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, target, pred, probability in zip(test_rows, y_test, y_pred, probs):
            writer.writerow({
                "review_id": row.get("review_id", ""), "category": row.get("category", ""),
                "content": row.get("content", ""), "y_true": target, "y_pred": pred,
                "confidence": float(probability.max()), "p_negative": float(probability[0]),
                "p_neutral": float(probability[1]), "p_positive": float(probability[2]),
            })
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[DONE] Model directory: {model_dir}\nMetrics: {metrics_dir / f'{slug}.json'}\nPredictions: {pred_path}")


if __name__ == "__main__":
    main()
