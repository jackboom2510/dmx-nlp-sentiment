# TC-30 — Vẽ phân bố độ dài bình luận theo nhãn (box plot)
# Nguồn: notebooks/01_eda.ipynb ô 10 · results/figures/eda_length_by_label.png · nối tiếp TC-04
# Điều kiện: CPU. Pool = train+val (6.024), nhãn chữ; cắt hiển thị tại 120 từ để hộp dễ đọc.
# Kỳ vọng trung vị: Negative 24 · Neutral 12 · Positive 9 từ; Negative dài nhất 889 từ.
# Đầu ra: testcase-charts/TC-30.png

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
OUT = Path(os.environ.get("CHART_OUT", Path.cwd() / "testcase-charts")); OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["Negative", "Neutral", "Positive"]
COL = ["#eb6834", "#e8b13f", "#2a78d6"]

pool = pd.concat([pd.read_csv(ROOT / "data/processed/train.csv"),
                  pd.read_csv(ROOT / "data/processed/val.csv")], ignore_index=True)
label_col = "label_text" if "label_text" in pool.columns else "label"
pool["n_words"] = pool["text"].astype(str).str.split().str.len()
data = [pool.loc[pool[label_col] == lab, "n_words"] for lab in LABELS]
medians = {lab: float(s.median()) for lab, s in zip(LABELS, data)}
print("Trung vị số từ:", medians, "| dài nhất:", int(pool["n_words"].max()))
assert medians == {"Negative": 24.0, "Neutral": 12.0, "Positive": 9.0}

fig, ax = plt.subplots(figsize=(8, 4.4))
bp = ax.boxplot([s.clip(upper=120) for s in data], tick_labels=LABELS, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], COL):
    patch.set_facecolor(c); patch.set_alpha(0.75)
for i, (lab, s) in enumerate(zip(LABELS, data), 1):
    ax.text(i, medians[lab] + 3, f"trung vị {medians[lab]:.0f}", ha="center", fontsize=9)
ax.set_ylabel("Số từ mỗi bình luận (clip hiển thị 120)")
ax.set_title("Độ dài bình luận theo nhãn (train+val, n=6.024)", loc="left", fontsize=12)
ax.grid(axis="y", color="#e7e2d9", linewidth=0.8); ax.set_axisbelow(True)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.tight_layout()
path = OUT / "TC-30.png"; fig.savefig(path, dpi=150)
print("saved", path)
print("PASS TC-30: hộp Negative cao/kéo dài nhất — liên hệ trần 128 token (TC-20, TC-25) và nhóm lỗi câu dài")
