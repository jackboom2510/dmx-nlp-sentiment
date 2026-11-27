# TC-16 — Bootstrap CI95 kiểu PAIRED & chấm lại bằng nhãn người thứ hai
# Nguồn: src/evaluate.py robustness() dòng 119 · results/robustness.csv · Bảng 3.10
# Điều kiện: CPU (numpy/pandas), CHỈ đọc predictions đã lưu — không huấn luyện lại.
# Kỳ vọng: PhoBERT CI95 [0,8639; 0,9353], human2 0,8497 · TF-IDF+SVM [0,7511; 0,8527], 0,7054.
# "Paired": cùng một ma trận chỉ số resample (2000 x 500, seed 42) cho MỌI mô hình.

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
LABELS = ["Negative", "Neutral", "Positive"]
CODE = {lab: i for i, lab in enumerate(LABELS)}
N_BOOT, SEED = 2000, 42

def fast_macro_f1(y: np.ndarray, p: np.ndarray, k: int = 3) -> np.ndarray:
    """Macro-F1 cho từng hàng của mảng (B, n) — chép đúng evaluate.py:109."""
    f1 = []
    for c in range(k):
        tp = ((y == c) & (p == c)).sum(1)
        denom = (y == c).sum(1) + (p == c).sum(1)
        f1.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return np.mean(f1, axis=0)

test = pd.read_csv(ROOT / "data/processed/test.csv")
classic = pd.read_csv(ROOT / "results/predictions/classic_test.csv").set_index("review_id").loc[test["review_id"]]
pho = pd.read_csv(ROOT / "results/predictions/phobert_test.csv").set_index("review_id").loc[test["review_id"]]
preds = {"PhoBERT": pho["pred"], "TF-IDF + SVM": classic["pred_tf_idf_svm"]}

y = test["label"].map(CODE).to_numpy()
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))  # CÙNG idx cho mọi mô hình -> paired

gold = pd.read_csv(ROOT / "data/gold/gold_test.csv")[["review_id", "label_human_2"]]
h2 = test[["review_id"]].merge(gold, on="review_id", how="left")["label_human_2"]
has = h2.notna().to_numpy()

for name, p in preds.items():
    pc = pd.Series(p).map(CODE).to_numpy()
    boot = fast_macro_f1(y[idx], pc[idx])
    print(f"{name:13s} macro-F1 {f1_score(test['label'], p, average='macro'):.4f} "
          f"CI95 [{np.percentile(boot, 2.5):.4f}; {np.percentile(boot, 97.5):.4f}] "
          f"| human2 (n={int(has.sum())}) macro-F1 "
          f"{f1_score(h2[has], p[has], average='macro', zero_division=0):.4f} acc {accuracy_score(h2[has], p[has]):.4f}")

rob = pd.read_csv(ROOT / "results/robustness.csv").set_index("model")
assert abs(rob.loc["PhoBERT", "ci95_low"] - 0.8639) < 1e-3 and abs(rob.loc["PhoBERT", "human2_macro_f1"] - 0.8497) < 1e-3
print("\nPASS TC-16: hai CI gần như không chồng lấn -> chênh 0,902 vs 0,804 có căn cứ thống kê.")
print("Lưu ý: CI95 này có trong robustness.csv nhưng báo cáo v1.6 CHƯA đưa vào — hãy chủ động nêu")
