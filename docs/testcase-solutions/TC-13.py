# TC-13 — Dự đoán lại từ checkpoint & assert nhất quán (ô 13 của notebook)
# Nguồn: notebooks/02_phobert_colab.ipynb ô 13 · results/predictions/phobert_{val,test}.csv
# Điều kiện: CPU (pandas + sklearn), chỉ đọc file dự đoán đã lưu.
# Kỳ vọng: tính lại từ phobert_test.csv -> macro-F1 0,9023, Acc 0,948;
#   assert của notebook: |macro-F1 tính lại trên val - best| < 1e-3 (ngăn nạp nhầm checkpoint).
#   CHỈ seed 42 mới có file dự đoán trên test; seed 43/44 chỉ có file val.

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
LABELS = ["Negative", "Neutral", "Positive"]
meta = json.loads((ROOT / "results/metrics/phobert.json").read_text(encoding="utf-8"))

for name in ("val", "test"):
    p = pd.read_csv(ROOT / f"results/predictions/phobert_{name}.csv")
    probs = p[["prob_Negative", "prob_Neutral", "prob_Positive"]].to_numpy()
    assert (abs(probs.sum(axis=1) - 1.0) < 1e-3).all()                 # softmax -> tổng = 1
    assert (p["pred"] == pd.Series(probs.argmax(axis=1)).map(dict(enumerate(LABELS)))).all()
    mf1 = f1_score(p["label"], p["pred"], average="macro")
    print(f"{name:4s}: n={len(p)} macro-F1={mf1:.4f} accuracy={accuracy_score(p['label'], p['pred']):.4f}")
    if name == "val":  # chính là phép kiểm của assert ở ô 13
        assert abs(mf1 - meta["val_best"]["macro_f1"]) < 1e-3, (mf1, meta["val_best"])
    else:
        assert abs(mf1 - 0.9023) < 1e-3, mf1

print("\nDanh sách file predictions:", sorted(x.name for x in (ROOT / 'results/predictions').glob('phobert*.csv')))
print("PASS TC-13: notebook KHÔNG chấm test — 0,9023 thuộc về Phase 6 (src/evaluate.py) đọc file này")
