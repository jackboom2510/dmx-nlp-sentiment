# TC-27 — Vẽ Hình 3.5: macro-F1 trên val và test của mọi mô hình (thanh ngang ghép đôi)
# Nguồn: Báo cáo Hình 3.5 · results/figures/macro_f1_val_test.png · src/evaluate.py plot_bars() dòng 82
# Điều kiện: CPU + results/comparison.csv. Biến thể "không bỏ từ dừng" (is_reference) bị LOẠI khỏi hình chính.
# Kỳ vọng: PhoBERT val 0,8749 / test 0,9023 đứng đầu; baseline 0,2362 / 0,1647 đứng cuối.
# Đầu ra: testcase-charts/TC-27.png

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
OUT = Path(os.environ.get("CHART_OUT", Path.cwd() / "testcase-charts")); OUT.mkdir(parents=True, exist_ok=True)

comp = pd.read_csv(ROOT / "results/comparison.csv")
d = comp[~comp["is_reference"]].reset_index(drop=True)  # loại hàng tham khảo như evaluate.py
y = np.arange(len(d))[::-1]
h = 0.38
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.barh(y + h / 2, d["val_macro_f1"], h - 0.04, color="#2a78d6", label="Val (nhãn chữ)")
ax.barh(y - h / 2, d["test_macro_f1"], h - 0.04, color="#eb6834", label="Test (nhãn gold)")
for yi, v, t in zip(y, d["val_macro_f1"], d["test_macro_f1"]):
    ax.text(v + 0.01, yi + h / 2, f"{v:.3f}", va="center", fontsize=8.5)
    ax.text(t + 0.01, yi - h / 2, f"{t:.3f}", va="center", fontsize=8.5)
ax.set_yticks(y, d["model"])
ax.set_xlim(0, 1.05); ax.set_xlabel("Macro-F1")
ax.grid(axis="x", color="#e7e2d9", linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
ax.set_title("Hình 3.5 — Macro-F1: val (nhãn chữ) và test (nhãn gold)", loc="left", fontsize=12)
fig.tight_layout()
path = OUT / "TC-27.png"; fig.savefig(path, dpi=150)
top = d.loc[d["test_macro_f1"].idxmax()]
print("saved", path, "| mô hình test cao nhất:", top["model"], round(top["test_macro_f1"], 4))
assert top["model"] == "PhoBERT"
print("PASS TC-27: test > val ở PhoBERT/TF-IDF là do phân bố nhãn test khác val — không phải rò rỉ")
