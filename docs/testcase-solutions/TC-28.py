# TC-28 — Vẽ Hình 3.6: ma trận nhầm lẫn TF-IDF+SVM vs PhoBERT (số đếm + % theo hàng)
# Nguồn: Báo cáo Hình 3.6 · src/evaluate.py plot_cm() dòng 58 · results/figures/report_confusion.png
# Điều kiện: CPU + results/predictions/classic_test.csv (cột pred_tf_idf_svm, pred_phobert).
# Kỳ vọng: hàng Neutral: TF-IDF+SVM [15,23,9] · PhoBERT [3,42,2]; màu theo tỷ lệ hàng (recall).
# Đầu ra: testcase-charts/TC-28.png

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
OUT = Path(os.environ.get("CHART_OUT", Path.cwd() / "testcase-charts")); OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["Negative", "Neutral", "Positive"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

pred = pd.read_csv(ROOT / "results/predictions/classic_test.csv")
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
for ax, (col, title) in zip(axes, (("pred_tf_idf_svm", "(a) TF-IDF + SVM"),
                                   ("pred_phobert", "(b) PhoBERT"))):
    cm = confusion_matrix(pred["label"], pred[col], labels=LABELS)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j] * 100:.0f}%)", ha="center", va="center",
                    fontsize=10, color="#ffffff" if norm[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Nhãn gold"); ax.set_title(title, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    print(title, "hàng Neutral:", cm[1].tolist())
    if col == "pred_phobert":
        assert cm[1].tolist() == [3, 42, 2]
    else:
        assert cm[1].tolist() == [15, 23, 9]
fig.suptitle("Hình 3.6 — Ma trận nhầm lẫn trên test (số câu và % theo hàng)", fontsize=12)
fig.tight_layout()
path = OUT / "TC-28.png"; fig.savefig(path, dpi=150)
print("saved", path)
print("PASS TC-28: % theo hàng = recall từng lớp; đọc thêm CỘT Neutral của PhoBERT để thấy precision 0,677")
