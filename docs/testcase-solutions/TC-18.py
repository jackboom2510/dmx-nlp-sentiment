# TC-18 — Câu khen-chê lẫn lộn & thí nghiệm cắt bỏ một từ (ablation)
# Nguồn: src/predict.py DEFAULT dòng 15, phobert_predictor() dòng 23 · Báo cáo §3.7.3
# Điều kiện: cần models/phobert_best (checkpoint KHÔNG có sẵn trong repo — phải chạy lại
#   notebooks/02_phobert_colab.ipynb trên GPU, hoặc dùng máy đã từng huấn luyện).
#   Thiếu checkpoint -> phần PhoBERT bị predict.py bỏ qua lặng lẽ (đó cũng là một hành vi cần thuộc).
# Kỳ vọng (khớp §3.7.3): nguyên câu -> Tiêu cực p=0,62; bỏ "giặt" -> Trung tính p=0,68.

import sys
from pathlib import Path

ROOT = Path.cwd()  # chạy tại gốc repo sentiment-classification
sys.path.insert(0, str(ROOT))

SENT = "máy giặt chạy êm nhưng vắt không khô"   # câu mặc định đầu tiên của predict.py
ABLATED = SENT.replace("giặt ", "")              # -> "máy chạy êm nhưng vắt không khô"

try:
    from src.predict import phobert_predictor, best_classic
    import joblib
    from src.config import MODELS
    from src.preprocess import to_text_clean

    name = best_classic()  # đọc từ results/metrics/val_classic.json
    clf = joblib.load(MODELS / (name.replace(" + ", "_").replace("-", "") + ".joblib"))
    print("Mô hình cổ điển tốt nhất trên val:", name,
          "(LinearSVC không có predict_proba -> chỉ in nhãn)")
    for s in (SENT, ABLATED):
        print(f"\nReview: {s}\n  {name}: {clf.predict([to_text_clean(s)])[0]}")

    pho = phobert_predictor()
    if pho is None:
        print("\n[!] Chưa có models/phobert_best hoặc thiếu torch/transformers -> PhoBERT bị bỏ qua.")
        print("Kỳ vọng theo báo cáo §3.7.3: nguyên câu -> Negative p=0,62 | bỏ 'giặt' -> Neutral p=0,68")
    else:
        for s, (lab, p) in zip((SENT, ABLATED), pho([SENT, ABLATED])):
            print(f"  PhoBERT: {lab} ({p:.2f})   <- {s}")
        print("\nCả hai xác suất đều < 0,7 -> quyết định mong manh ở vùng ranh giới khen-chê;")
        print("đề xuất cải tiến: ngưỡng từ chối dự đoán (abstention) khi xác suất thấp")
except ImportError as e:
    print("Thiếu thư viện để chạy inference:", e)
    print("Kỳ vọng theo báo cáo §3.7.3: nguyên câu -> Negative p=0,62 | bỏ 'giặt' -> Neutral p=0,68")
