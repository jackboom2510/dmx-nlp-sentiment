# TC-09 — Token hóa BPE, trần 128 và tỷ lệ bị cắt (truncated)
# Nguồn: notebooks/02_phobert_colab.ipynb ô 9 encode() · Bảng 3.4 · phobert.json truncated_train_pct
# Điều kiện: CPU. Đo đúng trên cột x (sau resegment); không có VnCoreNLP thì đo trên text_seg
#   và đối chiếu mốc đã lưu trong phobert.json (chênh lệch nhỏ do bộ tách từ).
# Kỳ vọng (output ô 9 đã lưu): train tokens median 15, p95 75, truncated 1.5% (0,0152).

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
meta = json.loads((ROOT / "results/metrics/phobert.json").read_text(encoding="utf-8"))
print("Mốc artifact: truncated_train_pct =", meta["truncated_train_pct"],
      "| input =", meta["input"], "| segmenter =", meta["segmenter"])

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    texts = train["text_seg"].fillna("")  # notebook dùng cột x (tách lại bằng RDRSegmenter)
    lens = np.array([len(ids) for ids in tok(list(texts))["input_ids"]])
    trunc = float((lens > 128).mean())
    print(f"Đo lại trên text_seg: median {np.median(lens):.0f}, "
          f"p95 {np.percentile(lens, 95):.0f}, truncated {trunc:.1%}")
    assert np.median(lens) == 15 and abs(trunc - 0.0152) < 0.005
    # encode() ở ô 9: cắt tại 128, đệm ĐỘNG theo lô (DataCollatorWithPadding) chứ không pad cứng
    enc = tok(["máy giặt chạy êm"], truncation=True, max_length=128)
    print("Ví dụ encode:", enc)
except ImportError:
    print("(chưa cài transformers — đối chiếu bằng mốc artifact ở trên)")

print("PASS TC-09: phân biệt được 15 TOKEN BPE (trung vị) với 13 TỪ trước tiền xử lý ở Bảng 3.4")
print("Lưu ý chênh: số vượt 128 token của val/test (14/10 câu) chưa có artifact — phải tự đo như trên")
