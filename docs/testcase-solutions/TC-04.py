# TC-04 — Độ dài bình luận theo nhãn: người chê viết dài hơn hẳn
# Nguồn: notebooks/01_eda.ipynb ô 10 · liên hệ Bảng 3.4
# Điều kiện: CPU (pandas). Pool = train+val của data/processed, nhãn = label_text.
# Kỳ vọng: Negative n=2367 mean 37,4 trung vị 24 max 889 · Neutral 465/17,3/12 · Positive 3192/13,1/9

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
tr = pd.read_csv(ROOT / "data/processed/train.csv")
va = pd.read_csv(ROOT / "data/processed/val.csv")
pool = pd.concat([tr, va], ignore_index=True)

label_col = "label_text" if "label_text" in pool.columns else "label"
pool["n_words"] = pool["text"].astype(str).str.split().str.len()
desc = pool.groupby(label_col)["n_words"].describe(percentiles=[0.25, 0.5, 0.75])
print(desc[["count", "mean", "50%", "max"]].round(1).to_string())

neg, pos = desc.loc["Negative"], desc.loc["Positive"]
assert int(neg["count"]) == 2367 and int(pos["count"]) == 3192
assert abs(neg["50%"] - 24) < 1e-9 and abs(pos["50%"] - 9) < 1e-9
print("\nPASS TC-04: trung vị Negative (24 từ) gần gấp 3 Positive (9 từ) — "
      "người chê kể sự cố/bảo hành, người khen viết ngắn")
print("Suy ra: trần 128 token của PhoBERT ảnh hưởng lớp Negative nhiều nhất (xem TC-20)")
