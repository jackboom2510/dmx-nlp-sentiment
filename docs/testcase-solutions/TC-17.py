# TC-17 — Nhóm lỗi PHỦ ĐỊNH: TF-IDF+SVM sai cả 4, PhoBERT đúng cả 4
# Nguồn: results/errors.csv (nhom_loi = "Phủ định") · Bảng 3.12–3.13
# Điều kiện: chỉ đọc artifact. Cơ chế: mô hình túi từ cộng điểm từ "ngon" mà bỏ qua "ko/chưa/không"
#   đứng trước; PhoBERT dùng attention ngữ cảnh nên bắt được quan hệ phủ định.
# Kỳ vọng: 4 câu dưới đây — cổ điển sai, PhoBERT (pred_other_model) đúng.

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
err = pd.read_csv(ROOT / "results/errors.csv")
neg = err[(err["model"] == "TF-IDF + SVM") & (err["nhom_loi"] == "Phủ định")]

print(f"Số câu phủ định trong 15 lỗi mẫu của TF-IDF+SVM: {len(neg)}")
for _, r in neg.iterrows():
    print(f"\n- gold={r['label']} | TF-IDF+SVM đoán {r['pred']} (SAI) | PhoBERT đoán {r['pred_other_model']}")
    print(" ", str(r["text"])[:130])
    print("  ghi chú:", r["ghi_chu"])

assert len(neg) == 4
assert (neg["pred"] != neg["label"]).all()                 # cổ điển sai cả 4
assert (neg["pred_other_model"] == neg["label"]).all()     # PhoBERT đúng cả 4
pho_neg = err[(err["model"] == "PhoBERT") & (err["nhom_loi"] == "Phủ định")]
print(f"\nSố câu phủ định trong 15 lỗi mẫu của PhoBERT: {len(pho_neg)}")
assert len(pho_neg) == 0
print("PASS TC-17: phủ định xuất hiện 4 lần ở lỗi cổ điển, 0 lần ở lỗi PhoBERT (Bảng 3.12)")
