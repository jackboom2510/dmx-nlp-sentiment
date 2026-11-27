# TC-25 — Vẽ Hình 3.4: phân bố số token PhoBERT ở ba tập + vạch cắt 128 (BẢNG/HÌNH ĐANG THIẾU ARTIFACT)
# Nguồn: Báo cáo Hình 3.4 + Bảng 3.4 · notebooks/02_phobert_colab.ipynb ô 9 (encode, TRUNC)
# Điều kiện: nhánh chính xác cần transformers (tokenizer vinai/phobert-base-v2) — đo trên text_seg
#   (notebook đo trên cột x sau RDRSegmenter nên số có thể lệch rất nhỏ). Thiếu transformers:
#   script vẽ bằng token tách từ (text_seg) và ghi rõ là bản xấp xỉ.
# Kỳ vọng (tokenizer, theo báo cáo): trung vị 15/15/17; vượt 128: train 78 (1,5%), val 14, test 10.
#   Đo xấp xỉ bằng text_seg trong môi trường kiểm thử: trung vị 12/12/14, vượt 128: 71/12/9 (BPE tách nhỏ hơn nên ít hơn).
# Đầu ra: testcase-charts/TC-25.png

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
OUT = Path(os.environ.get("CHART_OUT", Path.cwd() / "testcase-charts")); OUT.mkdir(parents=True, exist_ok=True)
SETS = ["train", "val", "test"]
frames = {n: pd.read_csv(ROOT / f"data/processed/{n}.csv") for n in SETS}

mode = "token tách từ text_seg (xấp xỉ)"
lens = {n: d["text_seg"].fillna("").str.split().str.len().to_numpy() for n, d in frames.items()}
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    lens = {n: np.array([len(x) for x in tok(list(d["text_seg"].fillna("")))["input_ids"]])
            for n, d in frames.items()}
    mode = "token BPE PhoBERT (trên text_seg; notebook gốc đo trên cột x sau RDRSegmenter)"
except Exception as e:
    print("(không dùng được transformers:", repr(e)[:120], "-> dùng bản xấp xỉ)")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True)
report_med = {"train": 15, "val": 15, "test": 17}
report_over = {"train": 78, "val": 14, "test": 10}
for ax, n in zip(axes, SETS):
    v = lens[n]
    ax.hist(np.clip(v, 0, 180), bins=np.arange(0, 185, 5), color="#2a78d6", alpha=0.85)
    ax.axvline(128, color="#b3372f", linestyle="--", linewidth=1.4, label="trần 128")
    ax.axvline(np.median(v), color="#1c1917", linewidth=1.2, label=f"trung vị đo: {np.median(v):.0f}")
    over = int((v > 128).sum())
    ax.set_title(f"{n}: trung vị {np.median(v):.0f} | >128: {over} ({over / len(v) * 100:.1f}%)", fontsize=10, loc="left")
    ax.grid(axis="y", color="#e7e2d9", linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    print(f"{n:5s} n={len(v)} median={np.median(v):.0f} >128={over} "
          f"| mốc báo cáo: trung vị {report_med[n]}, >128: {report_over[n]}")
axes[0].set_ylabel("Số bình luận")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"Hình 3.4 — Phân bố độ dài token mỗi bình luận ({mode})", fontsize=12)
fig.supxlabel("Số token / bình luận (clip tại 180)", fontsize=9)
fig.tight_layout()
path = OUT / "TC-25.png"; fig.savefig(path, dpi=150)
print("saved", path)
print("PASS TC-25: đây là script mà báo cáo còn thiếu cho Bảng 3.4 — chạy bản tokenizer để chốt số chính thức")
