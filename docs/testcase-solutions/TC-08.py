# TC-08 — Định tuyến môi trường & seed trong notebook PhoBERT
# Nguồn: notebooks/02_phobert_colab.ipynb ô 2, 3, 4 · results/metrics/phobert.json
# Điều kiện: chỉ đọc notebook + artifact. Bẫy: tên file "colab" nhưng lần chạy chính thức là LOCAL.
# Kỳ vọng: SEED=42 -> dự đoán val+test; 43/44 -> chỉ val. phobert.json: in_colab=false, RTX 3060.

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))

# --- Logic ô 2: suy SUFFIX và MODEL_DIR từ biến môi trường PHOBERT_SEED ---
def route(seed: int) -> dict:
    suffix = "" if seed == 42 else f"_seed{seed}"
    return {
        "seed": seed,
        "suffix": suffix,
        "model_dir": f"models/phobert_best{suffix}",
        "metrics_file": f"results/metrics/phobert{suffix}.json",
        "predict_on": ["val", "test"] if seed == 42 else ["val"],  # ô 13 của notebook
    }

for s in (42, 43, 44):
    print(route(s))

meta = json.loads((ROOT / "results/metrics/phobert.json").read_text(encoding="utf-8"))
print("\nLần chạy sinh ra báo cáo:")
print(" in_colab :", meta["in_colab"], " <- False = chạy LOCAL dù notebook tên 'colab'")
print(" GPU      :", meta["gpu"])
print(" seed     :", meta["config"]["seed"], "| batch_used:", meta["config"]["batch_used"])
print(" segmenter:", meta["segmenter"])

# Kiểm chứng các ô gốc thật sự chứa logic trên
nb = json.loads((ROOT / "notebooks/02_phobert_colab.ipynb").read_text(encoding="utf-8"))
src2 = "".join(nb["cells"][2]["source"])
assert 'os.environ.get("PHOBERT_SEED", 42)' in src2 and "IN_COLAB" in src2
print("\nPASS TC-08: thuộc cơ chế seed/thư mục -> phân biệt được người từng chạy notebook")
