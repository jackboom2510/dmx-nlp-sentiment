#!/usr/bin/env python3
"""
Module xây dựng và huấn luyện Mô hình Hiện đại PhoBERT (Bước 4B - Bổ sung Ý 4):
Fine-tuning End-to-End mô hình `vinai/phobert-base-v2` với hàm mất mát Weighted Cross-Entropy Loss.
"""

from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABEL2ID: Dict[int, int] = {-1: 0, 0: 1, 1: 2}
ID2LABEL: Dict[int, int] = {0: -1, 1: 0, 2: 1}
ID2NAME: Dict[int, str] = {0: "Negative", 1: "Neutral", 2: "Positive"}


class DMXSentimentDataset(Dataset):
    def __init__(self, segmented_texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            list(segmented_texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = [LABEL2ID[int(y)] for y in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedPhoBERTTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_hf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
    }


def compute_balanced_weights(y_train: List[int]) -> torch.Tensor:
    mapped = np.array([LABEL2ID[int(y)] for y in y_train])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=mapped,
    )
    return torch.tensor(weights, dtype=torch.float)
