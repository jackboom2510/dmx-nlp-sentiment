# TC-21 — Vẽ biểu đồ PHỄU làm sạch dữ liệu (waterfall)
# Nguồn: notebooks/01_eda.ipynb ô 3 · results/figures/eda_cleaning.png · Bảng 3.1
# Điều kiện: CPU (matplotlib) + chỉ đọc clean_stats.json.
# Kỳ vọng dữ liệu: còn lại sau từng bước = 9.481, 9.464, 7.488, 7.485, 7.485, 6.795, 6.540.
# Đầu ra: testcase-charts/TC-21.png

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
OUT = Path(os.environ.get("CHART_OUT", Path.cwd() / "testcase-charts")); OUT.mkdir(parents=True, exist_ok=True)
stats = json.loads((ROOT / "results/metrics/clean_stats.json").read_text(encoding="utf-8"))

labels = ["Thô", "Rỗng", "Dưới 3 từ", "Không phải\ntiếng Việt", "Phản hồi\nnhân viên", "Trùng\ny hệt", "Gần trùng\n(cos>0,9)", "Sạch"]
drops = [0, stats["drop_empty"], stats["drop_short_lt3_words"], stats["drop_non_vietnamese"],
         stats["drop_staff_reply"], stats["drop_exact_dup"], stats["drop_near_dup_cos_gt_0.9"], 0]
remain, n = [], stats["raw"]
for d in drops:
    n -= d
    remain.append(n)
remain[0] = stats["raw"]
expected = [9481, 9464, 7488, 7485, 7485, 6795, 6540]
assert remain[1:7] == expected[1:] and remain[-1] == 6540, remain

fig, ax = plt.subplots(figsize=(9, 4.2))
colors = ["#2a78d6"] + ["#eb6834"] * 6 + ["#1f7a4d"]
ax.bar(labels, remain, color=colors, width=0.66)
for i, (r, d) in enumerate(zip(remain, drops)):
    ax.text(i, r + 120, f"{r:,}".replace(",", "."), ha="center", fontsize=9)
    if d:
        ax.text(i, max(r - 500, 200), f"-{d:,}".replace(",", "."), ha="center", fontsize=8.5, color="#b3372f")
ax.set_ylim(0, 10800)
ax.set_ylabel("Số bình luận còn lại")
ax.set_title("Phễu làm sạch dữ liệu: 9.481 -> 6.540 (Bảng 3.1)", loc="left", fontsize=12)
ax.grid(axis="y", color="#e7e2d9", linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
path = OUT / "TC-21.png"; fig.savefig(path, dpi=150)
print("saved", path, "| remain =", remain)
print("PASS TC-21: bước 'dưới 3 từ' (-1.976) là bậc thang lớn nhất của phễu")
