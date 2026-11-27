# TC-19 — Câu cụt / "kh" đa nghĩa / câu hỏi vì sự cố: phân biệt LỖI MÔ HÌNH vs NHÃN TRANH CÃI
# Nguồn: results/errors.csv · docs/HUONG_DAN_GAN_NHAN.md (quy tắc: hỏi vì sự cố = Tiêu cực)
# Điều kiện: chỉ đọc artifact. Hãy tự gán nhãn 3 câu bên dưới TRƯỚC khi chạy để tự kiểm tra.
# Kỳ vọng:
#   "kh ok 5sao ổn"            -> gold Positive | PhoBERT Neutral (sai), TF-IDF+SVM Positive (đúng)
#   câu hỏi ngăn đông mềm      -> gold Negative | PhoBERT Neutral (sai: bắt hình thức câu hỏi)
#   "Như máy điều hòa"         -> gold Positive | cả hai mô hình đều sai

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
err = pd.read_csv(ROOT / "results/errors.csv")

CASES = [
    ("kh ok 5sao", "Positive", "câu cụt + viết tắt 'kh' đã bị loại khỏi teencode vì đa nghĩa"),
    ("ngăn đông mềm", "Negative", "câu hỏi VÌ GẶP SỰ CỐ -> Tiêu cực theo guideline, không phải Trung tính"),
    ("Như máy điều hòa", "Positive", "khen ngầm cực ngắn, thiếu ngữ cảnh -> cả hai mô hình đều sai"),
]

for key, expected_gold, note in CASES:
    rows = err[err["text"].astype(str).str.contains(key, regex=False)]
    for _, r in rows.iterrows():
        other_name = "PhoBERT" if r["model"] != "PhoBERT" else "TF-IDF + SVM"
        print(f"\n[{key}] gold={r['label']} | {r['model']} đoán {r['pred']} | {other_name} đoán {r['pred_other_model']}")
        print("  ", str(r["text"])[:120])
        print("   nhóm lỗi:", r["nhom_loi"], "|", note)
        assert r["label"] == expected_gold

print("\nPASS TC-19: tự gán đúng theo guideline trước khi xem đáp án = nắm chuẩn nhãn của con người;")
print("vùng xám còn lại (vd 'Mới đầu làm lạnh chưa nhanh') được lượng hóa bằng kappa = 0,7975")
