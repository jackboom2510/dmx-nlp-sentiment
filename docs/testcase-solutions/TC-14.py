# TC-14 — Biến thiên theo seed & bản đối chứng huấn luyện bằng NHÃN SAO
# Nguồn: results/metrics/phobert_seeds.json · results/archive_star/ (notebook + metrics + comparison)
# Điều kiện: chỉ đọc artifact.
# Kỳ vọng: seed 42/43/44 -> 0,8749/0,8694/0,8978 (mean 0,8807 ± 0,0151); bản nhãn sao test = 0,632.

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))

seeds = json.loads((ROOT / "results/metrics/phobert_seeds.json").read_text(encoding="utf-8"))
vals = []
for s, r in seeds["seeds"].items():
    print(f"seed {s}: best_epoch={r['best_epoch']} val_macro_f1_best={r['val_macro_f1_best']}")
    vals.append(r["val_macro_f1_best"])
print(f"mean {np.mean(vals):.4f} ± std(ddof=1) {np.std(vals, ddof=1):.4f} "
      f"(artifact ghi {seeds['val_macro_f1_best_epoch_mean']} ± {seeds['val_macro_f1_best_epoch_std']})")
assert abs(np.mean(vals) - 0.8807) < 1e-3

star_meta = json.loads((ROOT / "results/archive_star/metrics/phobert.json").read_text(encoding="utf-8"))
print("\nBản nhãn sao (archive): best_epoch =", star_meta["best_epoch"],
      "| val_best macro-F1 =", star_meta["val_best"]["macro_f1"])
arch = pd.read_csv(ROOT / "results/archive_star/comparison.csv").set_index("model")
main = pd.read_csv(ROOT / "results/comparison.csv").set_index("model")
print(f"PhoBERT test macro-F1: nhãn sao {arch.loc['PhoBERT', 'test_macro_f1']:.4f} "
      f"-> nhãn nội dung {main.loc['PhoBERT', 'test_macro_f1']:.4f}")
assert abs(arch.loc["PhoBERT", "test_macro_f1"] - 0.632) < 1e-3

print("\nPASS TC-14: (1) kết quả dao động ±0,015 theo seed — phải báo cáo kèm độ lệch; "
      "seed 43/44 đạt đỉnh ở epoch 4")
print("(2) đổi nhãn sao -> nội dung đáng giá ~+0,27 macro-F1, nhưng không gán trọn 100%: "
      "cùng lúc có sửa từ điển teencode và loại 16 câu trùng")
