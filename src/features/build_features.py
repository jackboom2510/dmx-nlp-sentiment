#!/usr/bin/env python3
"""
Module trích chọn đặc trưng (Feature Extraction) cho bài toán Phân loại Cảm xúc:
1. Bag of Words (BoW) - CountVectorizer
2. TF-IDF - TfidfVectorizer
3. Word Embeddings - Word2Vec Skip-gram (Mean Pooling)
"""

from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def extract_bow_features(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: int = 15000,
):
    """Trích chọn đặc trưng Bag of Words (chỉ fit trên tập Train để chống Data Leakage)."""
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_val, X_test


def extract_tfidf_features(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: int = 15000,
    norm: str = "l2",
):
    """Trích chọn đặc trưng TF-IDF (chỉ fit trên tập Train để chống Data Leakage)."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        norm=norm,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_val, X_test


def extract_word2vec_features(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    epochs: int = 15,
    seed: int = 42,
):
    """
    Huấn luyện Word2Vec Skip-gram trên tập Train và biểu diễn mỗi văn bản
    bằng trung bình cộng các vector từ (Mean Pooling).
    """
    from gensim.models import Word2Vec

    train_sentences = [doc.split() for doc in train_texts]
    w2v_model = Word2Vec(
        sentences=train_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,  # deterministic training for reproducible report metrics
        sg=sg,
        epochs=epochs,
        seed=seed,
    )

    def doc_to_vec(doc: str) -> np.ndarray:
        words = [w for w in doc.split() if w in w2v_model.wv]
        if not words:
            return np.zeros(vector_size, dtype=np.float32)
        return np.mean(w2v_model.wv[words], axis=0).astype(np.float32)

    X_train = np.vstack([doc_to_vec(d) for d in train_texts])
    X_val = np.vstack([doc_to_vec(d) for d in val_texts])
    X_test = np.vstack([doc_to_vec(d) for d in test_texts])
    return w2v_model, X_train, X_val, X_test
