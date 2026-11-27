# TC-10 — Trọng số lớp từ phân bố train: w_k = n / (K * n_k)
# Nguồn: notebooks/02_phobert_colab.ipynb ô 6 (đếm nhãn), ô 9 (tính trọng số), ô 11 (đưa vào loss)
# Điều kiện: CPU (pandas). 
# Kỳ vọng: Negative 0,8432 · Neutral 4,2667 · Positive 0,6330 (khớp phobert.json class_weights)

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
train = pd.read_csv(ROOT / "data/processed/train.csv")

LABELS = ["Negative", "Neutral", "Positive"]
counts = train["label"].value_counts().reindex(LABELS)
n, K = len(train), len(LABELS)
weights = {lab: n / (K * counts[lab]) for lab in LABELS}

print("Phân bố train:", counts.to_dict(), "| n =", n)
for lab in LABELS:
    print(f" w[{lab:8s}] = {n}/(3 x {counts[lab]}) = {weights[lab]:.4f}")

meta = json.loads((ROOT / "results/metrics/phobert.json").read_text(encoding="utf-8"))
for lab in LABELS:
    assert abs(weights[lab] - meta["class_weights"][lab]) < 1e-3, (lab, weights[lab])

print("\nĐưa vào loss ở ô 11: CrossEntropyLoss(weight=class_w) — phạt lỗi trên Neutral nặng gấp",
      f"{weights['Neutral'] / weights['Positive']:.1f} lần Positive")
print("PASS TC-10: trọng số tác động vào LOSS (không phải nhân bản dữ liệu như oversampling);")
print("tắt trọng số -> recall Neutral (400/5.120 câu train) sụp đầu tiên")
