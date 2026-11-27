# TC-23 — Vẽ Hình 3.2: biểu đồ chồng 100% nhãn nội dung theo mức sao
# Nguồn: Báo cáo Hình 3.2 · notebooks/01_eda.ipynb ô 16 · results/label_noise_pool.csv
# Điều kiện: CPU (matplotlib) + label_noise_pool.csv (pool train+val 6.024 câu).
# Kỳ vọng: 3 sao: 73,8% Negative / 18,3% Neutral / 7,9% Positive — mức sao nhiễu nhất.
# Đầu ra: testcase-charts/TC-23.png. Lưu ý: hình này là POOL; bản gold test lệch 83,3% ở 3 sao (TC-03).

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
OUT = Path(os.environ.get("CHART_OUT", Path.cwd() / "testcase-charts")); OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["Negative", "Neutral", "Positive"]
COL = {"Negative": "#eb6834", "Neutral": "#e8b13f", "Positive": "#2a78d6"}

noise = pd.read_csv(ROOT / "results/label_noise_pool.csv", index_col="star")
share = noise[LABELS].div(noise["n"], axis=0) * 100
print((share).round(1).to_string())
assert abs(share.loc[3, "Negative"] - 73.8) < 0.2

fig, ax = plt.subplots(figsize=(8.5, 4.2))
xs = [f"{s}★\n(n={n})" for s, n in zip(share.index, noise["n"])]
bottom = np.zeros(len(share))
for lab in LABELS:
    v = share[lab].to_numpy()
    ax.bar(xs, v, 0.6, bottom=bottom, color=COL[lab], label=lab, edgecolor="white", linewidth=1.2)
    for xi, (b, h) in enumerate(zip(bottom, v)):
        if h >= 6:
            ax.text(xi, b + h / 2, f"{h:.0f}%", ha="center", va="center", fontsize=9, color="white")
    bottom += v
ax.set_ylim(0, 100)
ax.set_ylabel("% bình luận của mức sao")
ax.set_title("Hình 3.2 — Nhãn theo nội dung ở từng mức sao (train+val, n=6.024)", loc="left", fontsize=12)
ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
path = OUT / "TC-23.png"; fig.savefig(path, dpi=150)
print("saved", path)
print("PASS TC-23: cột 3★ gần như không có Neutral thật — bằng chứng trực quan cho việc bỏ nhãn sao")
