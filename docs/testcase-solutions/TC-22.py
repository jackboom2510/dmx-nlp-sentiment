# TC-22 — Vẽ Hình 3.1: số bình luận theo ngành hàng (thanh ngang)
# Nguồn: Báo cáo Hình 3.1 · results/figures/eda_categories.png · reports/make_figures.py categories() dòng 72
# Điều kiện: CPU (matplotlib) + clean_stats.json. 
# Kỳ vọng: 19 ngành, tổng 6.540; lớn nhất may-lanh 882, tu-lanh 833; nhỏ nhất bep-tu 34.
# Đầu ra: testcase-charts/TC-22.png

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
OUT = Path(os.environ.get("CHART_OUT", Path.cwd() / "testcase-charts")); OUT.mkdir(parents=True, exist_ok=True)
stats = json.loads((ROOT / "results/metrics/clean_stats.json").read_text(encoding="utf-8"))
cat = pd.Series(stats["clean_by_category"]).sort_values()
assert len(cat) == 19 and int(cat.sum()) == 6540 and int(cat.max()) == 882

fig, ax = plt.subplots(figsize=(8.5, 5.6))
ax.barh(cat.index, cat.values, height=0.65, color="#2a78d6")
for y, v in enumerate(cat.values):
    ax.text(v + 12, y, f"{v}", va="center", fontsize=9)
ax.set_xlabel("Số bình luận sau khi làm sạch")
ax.set_title("Hình 3.1 — Số bình luận theo ngành hàng (n=6.540, 72 sản phẩm)", loc="left", fontsize=12)
ax.grid(axis="x", color="#e7e2d9", linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
path = OUT / "TC-22.png"; fig.savefig(path, dpi=150)
print(cat.sort_values(ascending=False).head(3).to_string())
print("saved", path)
print("PASS TC-22: biểu đồ cho thấy lệch ngành (máy lạnh + tủ lạnh chiếm 26%) — phải nêu khi nói tính đại diện")
