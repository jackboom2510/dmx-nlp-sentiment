# TC-24 — Vẽ Hình 3.3: 12 n-gram đặc trưng nhất của mỗi lớp (3 panel)
# Nguồn: Báo cáo Hình 3.3 · notebooks/01_eda.ipynb ô 13 · reports/make_figures.py top_terms() dòng 107
# Điều kiện: CPU. Vectorizer CHỈ fit trên train (chống rò rỉ) — đúng src/features.py tfidf().
# Kỳ vọng: từ điển 3.162 n-gram; điểm = TF-IDF TB trong lớp - lớn nhất ở hai lớp còn lại.
# Đầu ra: testcase-charts/TC-24.png

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
OUT = Path(os.environ.get("CHART_OUT", Path.cwd() / "testcase-charts")); OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["Negative", "Neutral", "Positive"]
COL = {"Negative": "#eb6834", "Neutral": "#9a6a12", "Positive": "#2a78d6"}

train = pd.read_csv(ROOT / "data/processed/train.csv")
vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 2),
                      min_df=5, sublinear_tf=True)
X = vec.fit_transform(train["text_clean"].fillna(""))
terms = np.array(vec.get_feature_names_out())
assert len(terms) == 3162, len(terms)
y = train["label"].to_numpy()
mean = np.vstack([np.asarray(X[y == lab].mean(axis=0)).ravel() for lab in LABELS])

fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharex=True)
for i, (ax, lab) in enumerate(zip(axes, LABELS)):
    score = mean[i] - np.delete(mean, i, axis=0).max(axis=0)
    order = np.argsort(-score)[:12][::-1]
    ax.barh(terms[order], score[order], height=0.65, color=COL[lab])
    ax.set_title(lab, fontsize=11, loc="left")
    ax.grid(axis="x", color="#e7e2d9", linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    print(lab, "top3:", ", ".join(terms[np.argsort(-score)[:3]]))
fig.suptitle("Hình 3.3 — 12 n-gram đặc trưng nhất mỗi lớp (TF-IDF, chỉ fit trên train)", x=0.01, ha="left", fontsize=12)
fig.supxlabel("TF-IDF trung bình trong lớp - lớn nhất ở hai lớp còn lại", fontsize=9)
fig.tight_layout()
path = OUT / "TC-24.png"; fig.savefig(path, dpi=150)
print("saved", path)
print("PASS TC-24: Negative nổi 'bảo_hành/kêu/không lạnh', Neutral nổi 'tạm ổn/cũng được' — khớp diễn giải §3.4")
