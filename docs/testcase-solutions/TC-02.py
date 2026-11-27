# TC-02 — Top n-gram đặc trưng từng lớp, CHỈ fit trên train
# Nguồn: notebooks/01_eda.ipynb ô 13 · Hình 3.3 · src/features.py tfidf() dòng 27
# Điều kiện: CPU (pandas + sklearn). Minh họa kỷ luật chống rò rỉ: vectorizer chỉ fit trên train.
# Kỳ vọng: từ điển 3.162 n-gram (1,2) min_df=5;
#   Negative có "không lạnh", Neutral có "tạm ổn/cũng được", Positive có "rất tốt/chạy êm".

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(os.environ.get("SC_REPO", Path.cwd()))
train = pd.read_csv(ROOT / "data/processed/train.csv")

# Giống hệt src/features.py tfidf(): token đã tách sẵn nên pattern là \S+ (giữ "máy_giặt" = 1 token)
vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False,
                      ngram_range=(1, 2), min_df=5, sublinear_tf=True)
X = vec.fit_transform(train["text_clean"].fillna(""))  # FIT TRÊN TRAIN — không phải toàn bộ dữ liệu
terms = np.array(vec.get_feature_names_out())
print(f"Từ vựng TF-IDF (1,2), min_df=5, fit trên train: {len(terms)} n-gram")
assert len(terms) == 3162, len(terms)

label_col = "label_text" if "label_text" in train.columns else "label"
y = train[label_col].to_numpy()
LABELS = ["Negative", "Neutral", "Positive"]
mean = np.vstack([np.asarray(X[y == lab].mean(axis=0)).ravel() for lab in LABELS])

print("\nTop 10 bigram đặc trưng mỗi lớp (điểm = TF-IDF TB trong lớp - lớn nhất ở lớp khác):")
top = {}
for i, lab in enumerate(LABELS):
    score = mean[i] - np.delete(mean, i, axis=0).max(axis=0)
    order = np.argsort(-score)
    bigrams = [terms[j] for j in order if " " in terms[j]][:10]
    top[lab] = bigrams
    print(f"{lab:9s}: {', '.join(bigrams)}")

assert "không lạnh" in top["Negative"] and "tạm ổn" in top["Neutral"] and "rất tốt" in top["Positive"]
print("\nPASS TC-02: bigram phủ định 'không lạnh' lọt top Negative — bigram giữ được một phần thứ tự")
