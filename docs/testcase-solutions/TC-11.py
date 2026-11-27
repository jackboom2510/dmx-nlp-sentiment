# TC-11 — Vòng huấn luyện & chọn checkpoint theo macro-F1 trên val
# Nguồn: notebooks/02_phobert_colab.ipynb ô 11 train_run() · Bảng 3.6 · phobert.json history
# Điều kiện: kiểm chứng bằng artifact (không cần GPU). Muốn chạy lại: mở notebook trên máy có GPU.
# Kỳ vọng: loss 0,6488/0,3558/0,2433/0,1672 · val F1 0,8427/0,8749/0,8609/0,8621 · best epoch 2.
#
# Lõi của train_run() (ô 11) — cơ chế chọn checkpoint:
#   for ep in 1..4: train 1 epoch -> s = val_scores(val) 
#                   if s["macro_f1"] > best["macro_f1"]: model.save_pretrained(MODEL_DIR)
#   -> chỉ lưu khi TỐT HƠN mọi epoch trước; vẫn chạy đủ 4 epoch (không có patience).

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
meta = json.loads((ROOT / "results/metrics/phobert.json").read_text(encoding="utf-8"))

print(f"{'epoch':>5s} {'train_loss':>10s} {'val_macro_f1':>12s} {'val_acc':>8s} {'F1_Neu':>7s}  lưu?")
best = -1.0
for h in meta["history"]:
    saved = h["val_macro_f1"] > best
    best = max(best, h["val_macro_f1"])
    print(f"{h['epoch']:5d} {h['train_loss']:10.4f} {h['val_macro_f1']:12.4f} "
          f"{h['val_accuracy']:8.4f} {h['val_f1_Neutral']:7.4f}  {'LƯU' if saved else '-'}")

assert meta["best_epoch"] == 2 and abs(meta["val_best"]["macro_f1"] - 0.8749) < 1e-9
losses = [h["train_loss"] for h in meta["history"]]
assert losses == sorted(losses, reverse=True)  # loss giảm đơn điệu...
vals = [h["val_macro_f1"] for h in meta["history"]]
assert vals[1] == max(vals)                     # ...nhưng val F1 đỉnh ở epoch 2 -> quá khớp sau đó

print("\nPASS TC-11: train loss giảm đều nhưng val macro-F1 giảm sau epoch 2 = bắt đầu quá khớp;")
print("checkpoint epoch 2 đóng vai trò dừng sớm theo val — test KHÔNG tham gia chọn epoch")
