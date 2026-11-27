# TC-12 — Fallback OOM: batch 16 -> batch 8 + tích lũy gradient 2 (batch hiệu dụng không đổi)
# Nguồn: notebooks/02_phobert_colab.ipynb ô 11 (try/except OutOfMemoryError, train_run)
# Điều kiện: mô phỏng bằng số học, không cần GPU. Lần chạy thật: batch_used=16 (không dính OOM).
#
# Luồng gốc (ô 11):
#   try:    history, best = train_run(batch=16, accum=1)
#   except torch.cuda.OutOfMemoryError:
#           torch.cuda.empty_cache(); history, best = train_run(batch=8, accum=2)  # chạy lại từ đầu
# Trong train_run: loss = loss_fn(...)/accum; chỉ step optimizer khi (i+1) % accum == 0,
#   kèm scaler.unscale_ + clip_grad_norm_(1.0).

import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
N_TRAIN, EPOCHS = 5120, 4

def describe(batch: int, accum: int) -> None:
    eff = batch * accum
    batches = math.ceil(N_TRAIN / batch)
    steps_per_epoch = batches // accum
    print(f"batch={batch:2d} accum={accum} -> batch hiệu dụng {eff:2d} | "
          f"{batches} lô/epoch | {steps_per_epoch} bước optimizer/epoch | "
          f"{steps_per_epoch * EPOCHS} bước cho {EPOCHS} epoch (warmup 10% = {int(0.1 * steps_per_epoch * EPOCHS)})")

describe(16, 1)   # cấu hình chính thức
describe(8, 2)    # cấu hình fallback khi OOM

meta = json.loads((ROOT / "results/metrics/phobert.json").read_text(encoding="utf-8"))
cfg = meta["config"]
print("\nArtifact lần chạy thật: batch_used =", cfg["batch_used"],
      "| grad_accum_used =", cfg["grad_accum_used"],
      "| thời gian/epoch (giây):", [h["sec"] for h in meta["history"]])
assert cfg["batch_used"] == 16 and cfg["grad_accum_used"] == 1
print("PASS TC-12: 16x1 = 8x2 = 16 — đổi cấu hình bộ nhớ mà giữ nguyên quy mô cập nhật;")
print("lưu ý fallback chạy lại TỪ ĐẦU (không resume) và LayerNorm của RoBERTa ít nhạy với batch nhỏ")
