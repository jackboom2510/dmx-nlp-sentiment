# TC-29 — Vẽ forest plot khoảng tin cậy 95% (CI95) của macro-F1 trên test
# Nguồn: results/robustness.csv (bootstrap 2.000 mẫu paired, evaluate.py robustness() dòng 119)
# Điều kiện: CPU + robustness.csv. Biểu đồ này CHƯA có trong báo cáo v1.6 — vẽ bổ sung khi bảo vệ.
# Kỳ vọng: PhoBERT 0,9023 [0,8639; 0,9353]; TF-IDF+SVM 0,8038 [0,7511; 0,8527] — hai khoảng gần như không chồng.
# Đầu ra: testcase-charts/TC-29.png

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
OUT = Path(os.environ.get("CHART_OUT", Path.cwd() / "testcase-charts")); OUT.mkdir(parents=True, exist_ok=True)

rob = pd.read_csv(ROOT / "results/robustness.csv")
# Loại baseline lớp đông nhất (0,165) khỏi trục chính để các mô hình thật không bị ép dải;
# baseline vẫn là mốc đối chiếu trong robustness.csv (macro-F1 0,1647, CI [0,149; 0,179]).
d = rob[rob["test_macro_f1"].notna() & rob["ci95_low"].notna()
        & (rob["model"] != "Baseline (lớp đông nhất)")].sort_values("test_macro_f1")
y = range(len(d))
fig, ax = plt.subplots(figsize=(8.5, 4.8))
colors = ["#c15f3c" if m == "PhoBERT" else "#2a78d6" for m in d["model"]]
ax.errorbar(d["test_macro_f1"], list(y),
            xerr=[d["test_macro_f1"] - d["ci95_low"], d["ci95_high"] - d["test_macro_f1"]],
            fmt="o", color="#1c1917", ecolor="#a8a29e", elinewidth=1.6, capsize=4, markersize=5)
for yi, (_, r), c in zip(y, d.iterrows(), colors):
    ax.plot(r["test_macro_f1"], yi, "o", color=c, markersize=7)
    ax.text(r["ci95_high"] + 0.008, yi, f"{r['test_macro_f1']:.3f} [{r['ci95_low']:.3f}; {r['ci95_high']:.3f}]",
            va="center", fontsize=8.5)
ax.set_yticks(list(y), d["model"])
ax.set_xlim(0.55, 1.28); ax.set_xlabel("Macro-F1 trên test (bootstrap 2.000 mẫu, paired)")
ax.grid(axis="x", color="#e7e2d9", linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("CI95 của macro-F1 trên test (bootstrap paired)", loc="left", fontsize=12)
fig.tight_layout()
path = OUT / "TC-29.png"; fig.savefig(path, dpi=150)
pho = rob.set_index("model").loc["PhoBERT"]
print("saved", path, f"| PhoBERT CI95 [{pho['ci95_low']}, {pho['ci95_high']}]")
assert abs(pho["ci95_low"] - 0.8639) < 1e-3
print("PASS TC-29: đây là hình nên bổ sung vào báo cáo — trả lời thẳng câu '0,902 hơn 0,804 có ý nghĩa không?'")
