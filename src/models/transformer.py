"""Reusable weighted-loss trainer and dataset for Hugging Face sequence classifiers."""

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import Trainer

LABEL2ID = {-1: 0, 0: 1, 1: 2}
ID2LABEL = {0: -1, 1: 0, 2: 1}
ID2NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}


class SentimentDataset(Dataset):
    """Tokenize on construction; labels use stable Negative/Neutral/Positive IDs."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_length: int = 160):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length, padding=False)
        self.labels = [LABEL2ID[int(label)] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def balanced_class_weights(labels: Sequence[int]) -> torch.Tensor:
    encoded = np.asarray([LABEL2ID[int(y)] for y in labels])
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=encoded)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedSequenceClassificationTrainer(Trainer):
    """Trainer applying inverse-frequency weighted cross entropy on training labels."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))(
            outputs.logits.reshape(-1, model.config.num_labels), labels.reshape(-1)
        )
        return (loss, outputs) if return_outputs else loss
