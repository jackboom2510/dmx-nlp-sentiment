# TC-03 — Nhiễu nhãn sao: 20,9% (pool 6.024) vs 31,2% (gold test 500)
# Nguồn: notebooks/01_eda.ipynb ô 16 · Hình 3.2 · results/label_noise_pool.csv · gold_agreement.json
# Điều kiện: chỉ đọc artifact. Bẫy kinh điển: hai con số khác nhau vì mẫu số & cách bốc mẫu khác nhau.
# Kỳ vọng: pool theo sao 0,8/0,7/81,7/29,4/5,8 (%), chung 20,9%; gold test lệch 31,2%.

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
noise = pd.read_csv(ROOT / "results/label_noise_pool.csv", index_col="star")

print("Pool train+val (6.024 câu) — nhãn chữ theo mức sao:")
print(noise[["Negative", "Neutral", "Positive", "n"]].to_string())
overall_pool = float((noise["n"] * noise["pct_label_text_differs_from_star"]).sum() / noise["n"].sum())
print("\nTỷ lệ nhãn chữ khác nhãn sao theo sao (%):")
print((noise["pct_label_text_differs_from_star"] * 100).round(1).to_string())
print(f"Chung (pool): {overall_pool * 100:.1f}%")

gold = json.loads((ROOT / "results/metrics/gold_agreement.json").read_text(encoding="utf-8"))
print(f"\nGold test (500 câu, bốc theo hạn ngạch sao 100/70/150/80/100): "
      f"sao lệch nội dung = {gold['pct_star_label_differs_from_final'] * 100:.1f}%")

assert abs(overall_pool - 0.209) < 0.002 and abs(gold["pct_star_label_differs_from_final"] - 0.312) < 1e-9
print("PASS TC-03: 20,9% (pool 6.024, LLM gán) != 31,2% (gold 500, có duyệt người) — không mâu thuẫn")
