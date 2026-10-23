#!/usr/bin/env python3
"""
Module xây dựng và huấn luyện các mô hình Học máy truyền thống (Bước 4A):
- Baseline: DummyClassifier(strategy="most_frequent")
- Naïve Bayes: MultinomialNB (cho BoW/TF-IDF) và GaussianNB (cho Word2Vec) kết hợp Hybrid Resampling
- Support Vector Machine (SVM): LinearSVC(class_weight="balanced")
"""

from typing import Tuple
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC


def hybrid_resample_numpy(
    X,
    y: np.ndarray,
    undersample_pos: int = 2500,
    oversample_neg: int = 1500,
    oversample_neu: int = 1200,
    seed: int = 42,
) -> Tuple[ object, np.ndarray]:
    """
    Thực hiện Hybrid Sampling (Undersampling lớp đa số Positive=1, Oversampling lớp thiểu số Negative=-1, Neutral=0)
    hoạt động trực tiếp với cả scipy sparse matrix (BoW/TF-IDF) và numpy dense array (Word2Vec).
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y)
    target_map = {1: undersample_pos, -1: oversample_neg, 0: oversample_neu}
    sampled_indices = []

    for cls, target_n in target_map.items():
        cls_idx = np.where(y_arr == cls)[0]
        if len(cls_idx) == 0:
            continue
        if len(cls_idx) > target_n:
            chosen = rng.choice(cls_idx, size=target_n, replace=False)
        elif len(cls_idx) < target_n:
            chosen = rng.choice(cls_idx, size=target_n, replace=True)
        else:
            chosen = cls_idx
        sampled_indices.append(chosen)

    all_idx = np.concatenate(sampled_indices)
    rng.shuffle(all_idx)
    return X[all_idx], y_arr[all_idx]


def train_baseline(X_train, y_train):
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    return clf


def train_naive_bayes(
    X_train,
    y_train,
    feature_type: str = "tfidf",
    alpha: float = 0.5,
    use_hybrid_sampling: bool = True,
    seed: int = 42,
):
    if use_hybrid_sampling:
        X_tr, y_tr = hybrid_resample_numpy(X_train, y_train, seed=seed)
    else:
        X_tr, y_tr = X_train, np.asarray(y_train)

    if feature_type in ("bow", "tfidf"):
        clf = MultinomialNB(alpha=alpha)
    else:
        clf = GaussianNB()

    clf.fit(X_tr, y_tr)
    return clf


def train_svm(
    X_train,
    y_train,
    C: float = 1.0,
    class_weight: str = "balanced",
    max_iter: int = 5000,
    seed: int = 42,
):
    clf = LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    return clf
