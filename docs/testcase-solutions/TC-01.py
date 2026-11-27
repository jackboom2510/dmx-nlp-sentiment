# TC-01 — Tái lập phễu làm sạch 9.481 -> 6.540
# Nguồn: notebooks/01_eda.ipynb ô 3 · Báo cáo Bảng 3.1 · src/clean_label.py main() dòng 100
# Điều kiện: chỉ đọc artifact (không cần chạy lại crawl/làm sạch).
# Cách chạy: cd <repo sentiment-classification> && python TC-01.py
#   (hoặc SC_REPO=/duong/dan/repo python TC-01.py)
# Kỳ vọng: phễu 9481 -> 9464 -> 7488 -> 7485 -> 7485 -> 6795 -> 6540

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
stats = json.loads((ROOT / "results/metrics/clean_stats.json").read_text(encoding="utf-8"))

STEPS = [
    ("Thô (sau crawl)", None, "raw"),
    ("Rỗng", "drop_empty", None),
    ("Dưới 3 từ", "drop_short_lt3_words", None),
    ("Không phải tiếng Việt", "drop_non_vietnamese", None),
    ("Phản hồi nhân viên", "drop_staff_reply", None),
    ("Trùng y hệt", "drop_exact_dup", None),
    ("Gần trùng (cosine TF-IDF > 0,9)", "drop_near_dup_cos_gt_0.9", None),
]

n = stats["raw"]
print(f"{'Bước':38s} {'Bị loại':>8s} {'Còn lại':>8s}")
for name, drop_key, _ in STEPS:
    if drop_key is None:
        print(f"{name:38s} {'—':>8s} {n:8d}")
    else:
        d = stats[drop_key]
        n -= d
        print(f"{name:38s} {d:8d} {n:8d}")

print("\nSao sau sạch:", stats["clean_by_star"])
print("Ngành hàng:", stats["n_categories"], "| Sản phẩm:", stats["n_products"],
      "| Trung vị số từ:", stats["words_median"])
print("Cặp gần trùng khác sao:", stats["near_dup_pairs_with_different_star"])
print("Gold:", stats["gold"], "câu |", stats["gold_n_products"], "sản phẩm | pool còn lại:",
      stats["pool_after_gold"])

assert n == stats["clean"] == 6540, n
print("\nPASS TC-01: khớp phễu Bảng 3.1 (9.481 -> 6.540)")
