# TC-20 — Câu dài nhất của test (314 từ) dưới trần 128 token của PhoBERT
# Nguồn: data/processed/test.csv · notebooks/02_phobert_colab.ipynb ô 9 · Bảng 3.4
# Điều kiện: CPU (pandas) + đọc predictions đã lưu.
# Kỳ vọng: review_id 2c87d5468027... 314 từ, text_seg 271 token -> bị cắt còn 128;
#   PhoBERT vẫn đoán Negative (prob 0,9891), TF-IDF+SVM cũng Negative.
# Điểm nâng cao: truncation chỉ "an toàn" vì tín hiệu chê nằm ngay đầu câu; câu khen dài
#   mà ý chê nằm ở cuối mới là trường hợp rủi ro thật của max_len=128.

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
test = pd.read_csv(ROOT / "data/processed/test.csv")
test["n_words"] = test["text"].astype(str).str.split().str.len()
row = test.loc[test["n_words"].idxmax()]

seg_tokens = len(str(row["text_seg"]).split())
print(f"Câu dài nhất test: review_id={str(row['review_id'])[:12]}... | {row['n_words']} từ gốc | "
      f"text_seg {seg_tokens} token | nhãn gold={row['label']} | {row['star']} sao")
print("Nội dung mở đầu:", str(row["text"])[:150], "...")
assert row["n_words"] == 314 and seg_tokens == 271 and seg_tokens > 128

pho = pd.read_csv(ROOT / "results/predictions/phobert_test.csv").set_index("review_id").loc[row["review_id"]]
classic = pd.read_csv(ROOT / "results/predictions/classic_test.csv").set_index("review_id").loc[row["review_id"]]
print(f"\nPhoBERT (chỉ thấy ~128/{seg_tokens} token): {pho['pred']} (prob {pho['prob_' + pho['pred']]:.4f})")
print(f"TF-IDF+SVM (thấy toàn bộ câu)          : {classic['pred_tf_idf_svm']}")
assert pho["pred"] == classic["pred_tf_idf_svm"] == row["label"] == "Negative"

print("\nPASS TC-20: 128 là số TOKEN BPE dưới-mẫu (subword), không phải số từ;")
print("nâng max_len 128->256 làm chi phí attention tăng ~4 lần (O(n^2)) trong khi chỉ 1,5-2% câu hưởng lợi")
