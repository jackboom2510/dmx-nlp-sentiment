# TC-15 — Tính lại toàn bộ Bảng 3.7 từ file dự đoán (không train lại)
# Nguồn: src/evaluate.py metrics() dòng 49 · results/comparison.csv, test_metrics.json
# Điều kiện: CPU (pandas + sklearn).
# Kỳ vọng: PhoBERT 0,948/0,9023 · TF-IDF+SVM 0,890/0,8038 · baseline 0,328/0,1647;
#   hàng Neutral: TF-IDF+SVM [15,23,9], PhoBERT [3,42,2].

import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
LABELS = ["Negative", "Neutral", "Positive"]
test = pd.read_csv(ROOT / "data/processed/test.csv")
classic = pd.read_csv(ROOT / "results/predictions/classic_test.csv").set_index("review_id").loc[test["review_id"]]
pho = pd.read_csv(ROOT / "results/predictions/phobert_test.csv").set_index("review_id").loc[test["review_id"]]
y = test["label"]

preds = {"TF-IDF + SVM": classic["pred_tf_idf_svm"].to_numpy(), "PhoBERT": pho["pred"].to_numpy()}
for name, p in preds.items():
    cm = confusion_matrix(y, p, labels=LABELS)
    print(f"\n{name}: accuracy={accuracy_score(y, p):.4f} macro-F1={f1_score(y, p, average='macro'):.4f}")
    print(" ma trận nhầm lẫn (hàng = nhãn thật Neg/Neu/Pos):\n", cm)
    print(f" hàng Neutral: {cm[1].tolist()} -> recall Neutral = {cm[1][1]}/{cm[1].sum()} = {cm[1][1] / cm[1].sum():.3f}")

assert confusion_matrix(y, preds["PhoBERT"], labels=LABELS)[1].tolist() == [3, 42, 2]
assert confusion_matrix(y, preds["TF-IDF + SVM"], labels=LABELS)[1].tolist() == [15, 23, 9]
comp = pd.read_csv(ROOT / "results/comparison.csv").set_index("model")
assert abs(comp.loc["PhoBERT", "test_macro_f1"] - 0.9023) < 1e-3
assert abs(comp.loc["Baseline (lớp đông nhất)", "test_accuracy"] - 0.328) < 1e-3
print("\nPASS TC-15: mọi số Bảng 3.7 suy được từ file dự đoán; accuracy cao vẫn có thể vô dụng "
      "(baseline 0,328 acc nhưng macro-F1 0,1647)")
