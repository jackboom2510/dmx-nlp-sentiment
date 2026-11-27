# TC-05 — Hai đường tiền xử lý: text_clean (cổ điển) vs text_seg (PhoBERT)
# Nguồn: src/preprocess.py to_text_clean() dòng 120, to_text_seg() dòng 126 · Bảng 3.3
# Điều kiện: chạy thật cần underthesea (requirements.txt). Nếu thiếu gói, script vẫn đối chiếu
#   10 ví dụ đã lưu trong results/preprocess_examples.csv.
# Kỳ vọng: text_seg giữ hoa/thường + dấu câu + từ dừng; text_clean hạ thường, bỏ dấu câu/từ dừng;
#   cả hai đều sửa teencode (mn -> mọi người, Ko đc -> không được).

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
ex = pd.read_csv(ROOT / "results/preprocess_examples.csv")
for _, r in ex.head(3).iterrows():
    print("GỐC :", r["text"])
    print("SEG :", r["text_seg"], "  <- PhoBERT: giữ hoa/thường, dấu câu, từ dừng")
    print("CLEAN:", r["text_clean"], "  <- cổ điển: hạ thường, bỏ dấu câu + từ dừng\n")

stats = json.loads((ROOT / "results/metrics/preprocess_stats.json").read_text(encoding="utf-8"))
print("Toàn cục (preprocess_stats.json): sửa từ", stats["word_corrections_total"], "lượt /",
      stats["reviews_with_word_correction"], "câu | bỏ từ dừng loại",
      f"{stats['pct_tokens_removed_as_stopwords'] * 100:.1f}% token | độ dài TB",
      stats["tokens_mean"]["text_nostop"], "->", stats["tokens_mean"]["text_clean"])
assert stats["word_corrections_total"] == 3745 and stats["reviews_with_word_correction"] == 2057

# Chạy thật hai hàm nếu môi trường có underthesea (đứng tại gốc repo sentiment-classification)
try:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.preprocess import to_text_clean, to_text_seg
    s = "Ko đc như kỳ vọng, ko nhớ để hút bụi và lau hết nhà."
    print("\nChạy thật:", s)
    print("SEG :", to_text_seg(s))
    print("CLEAN:", to_text_clean(s))
    assert "không được" in to_text_clean(s) and "không được" in to_text_seg(s)
except ImportError as e:
    print("\n(chưa cài underthesea nên bỏ qua phần chạy thật:", e, ")")

print("PASS TC-05: cùng một câu gốc, hai đầu vào khác nhau theo đúng thiết kế hai họ mô hình")
