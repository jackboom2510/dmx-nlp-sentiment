# TC-26 — Vẽ đường học PhoBERT: train loss & val macro-F1 theo epoch (3 seed)
# Nguồn: results/metrics/phobert.json + phobert_seed43/44.json (history) · Bảng 3.6
# Điều kiện: chỉ đọc artifact. Biểu đồ KHÔNG có sẵn trong báo cáo — tự vẽ để trình bày quá khớp.
# Kỳ vọng: seed 42 đỉnh epoch 2 (0,8749); seed 43/44 đỉnh epoch 4 (0,8694/0,8978); loss giảm đơn điệu.
# Đầu ra: testcase-charts/TC-26.png

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
OUT = Path(os.environ.get("CHART_OUT", Path.cwd() / "testcase-charts")); OUT.mkdir(parents=True, exist_ok=True)

runs = {}
for seed, f in ((42, "phobert.json"), (43, "phobert_seed43.json"), (44, "phobert_seed44.json")):
    runs[seed] = json.loads((ROOT / f"results/metrics/{f}").read_text(encoding="utf-8"))

fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax2 = ax.twinx()
for seed, style in ((42, "-o"), (43, "--s"), (44, "--^")):
    h = runs[seed]["history"]
    ep = [x["epoch"] for x in h]
    f1 = [x["val_macro_f1"] for x in h]
    ax.plot(ep, f1, style, label=f"val macro-F1 seed {seed} (đỉnh e{runs[seed]['best_epoch']})", linewidth=1.8)
    b = runs[seed]["best_epoch"]
    ax.annotate(f"{max(f1):.3f}", (b, max(f1)), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
h42 = runs[42]["history"]
ax2.plot([x["epoch"] for x in h42], [x["train_loss"] for x in h42], ":D", color="#b3372f",
         label="train loss seed 42", linewidth=1.6)
ax.set_xlabel("Epoch"); ax.set_ylabel("Val macro-F1"); ax2.set_ylabel("Train loss (seed 42)", color="#b3372f")
ax.set_xticks([1, 2, 3, 4]); ax.set_ylim(0.80, 0.93)
ax.grid(color="#e7e2d9", linewidth=0.8); ax.set_axisbelow(True)
ax.set_title("Đường học PhoBERT: loss giảm đều nhưng val F1 đỉnh ở epoch 2 (seed 42)", loc="left", fontsize=12)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="lower left")
fig.tight_layout()
path = OUT / "TC-26.png"; fig.savefig(path, dpi=150)
print("saved", path)
assert runs[42]["best_epoch"] == 2 and runs[43]["best_epoch"] == runs[44]["best_epoch"] == 4
print("PASS TC-26: một hình kể trọn câu chuyện chọn checkpoint + biến thiên theo seed (0,881 ± 0,015)")
