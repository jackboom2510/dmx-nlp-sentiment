#!/usr/bin/env python3
"""
Script gán nhãn cảm xúc tự động / bán tự động (Unsupervised + Hybrid Star Reconciliation)
và chia tập dữ liệu chuẩn (train / val / test_gold) cho dự án dmx-nlp-sentiment.

Cách chạy:
    # 1. Chạy chế độ mặc định (Hybrid Unsupervised Lexicon + Star Rating - chạy ngay trên CPU):
    python scripts/auto_label_review.py

    # 2. Tuỳ chỉnh kích thước tập test_gold và ngưỡng phát hiện xung đột:
    python scripts/auto_label_review.py --test-size 500 --val-ratio 0.15 --seed 42

    # 3. Chạy chế độ Sentence Embeddings Prototype / Zero-shot NLI / Pretrained PhoBERT (nếu đã cài torch/transformers):
    python scripts/auto_label_review.py --mode pretrained
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

# Đảm bảo import được package `src` dù chạy từ thư mục gốc hay từ `scripts/`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.auto_labeler import map_star_to_label, reconcile_star_and_unsupervised
from src.data.preprocess import is_valid_vietnamese_review, load_stopwords, preprocess_review


def run_external_model_labeling(texts: List[str], mode: str) -> List[int]:
    """
    Hỗ trợ các chế độ mở rộng dùng mô hình ngôn ngữ (embedding-prototype, zero-shot, pretrained)
    khi người dùng truyền --mode tương ứng.
    """
    if mode == "embedding-prototype":
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("keepitreal/vietnamese-sbert")
        prototypes = {
            1: "Sản phẩm dùng rất tốt, máy chạy êm, tiết kiệm điện, giao hàng nhiệt tình, rất hài lòng",
            0: "Sản phẩm bình thường, tạm ổn trong tầm giá, mới dùng nên chưa đánh giá được độ bền",
            -1: "Sản phẩm chất lượng kém, máy kêu to, hay bị lỗi hỏng, tốn điện, bảo hành tệ, rất thất vọng",
        }
        proto_labels = [1, 0, -1]
        proto_vecs = model.encode([prototypes[k] for k in proto_labels], normalize_embeddings=True)
        doc_vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        sims = doc_vecs @ proto_vecs.T
        return [proto_labels[int(i)] for i in sims.argmax(axis=1)]

    if mode == "zero-shot":
        from transformers import pipeline

        clf = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        candidate_labels = ["tích cực khen ngợi", "trung lập bình thường", "tiêu cực phàn nàn"]
        lmap = {"tích cực khen ngợi": 1, "trung lập bình thường": 0, "tiêu cực phàn nàn": -1}
        preds = []
        for t in texts:
            out = clf(t[:256], candidate_labels=candidate_labels, hypothesis_template="Bình luận này mang thái độ {}.")
            preds.append(lmap[out["labels"][0]])
        return preds

    if mode == "pretrained":
        from transformers import pipeline

        clf = pipeline("sentiment-analysis", model="wonrax/phobert-base-vietnamese-sentiment")
        lmap = {"POS": 1, "NEU": 0, "NEG": -1}
        preds = []
        for t in texts:
            out = clf(t[:256])[0]
            preds.append(lmap.get(out["label"], 0))
        return preds

    raise ValueError(f"Chế độ không hợp lệ: {mode}")


def stratified_split(
    rows: List[Dict],
    test_size: int = 500,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Chia tập dữ liệu có phân tầng (Stratified) mà không cần phụ thuộc bắt buộc vào scikit-learn:
    - Tập test_gold (n = test_size): ưu tiên lấy mẫu đa dạng từ cả 3 lớp (-1, 0, 1) và các mẫu đã được đối chiếu kỹ
    - Tập train & val: chia phần còn lại theo tỷ lệ (1 - val_ratio) / val_ratio có phân tầng theo label_target
    """
    rng = random.Random(seed)
    by_label: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        by_label[int(r["label_target"])].append(r)

    for lbl in by_label:
        rng.shuffle(by_label[lbl])

    total = len(rows)
    test_rows: List[Dict] = []
    remaining_by_label: Dict[int, List[Dict]] = {}

    # Phân bổ số mẫu vào test_gold theo tỷ lệ phân tầng nhưng đảm bảo lớp thiểu số (Neutral = 0) có ít nhất 60 mẫu
    target_counts = {}
    if total >= test_size:
        min_neutral = min(len(by_label[0]) // 4, max(50, int(test_size * 0.12)))
        target_counts[0] = min_neutral
        rem_test = test_size - target_counts[0]
        neg_pos_total = len(by_label[-1]) + len(by_label[1])
        target_counts[-1] = min(
            len(by_label[-1]) // 3,
            max(120, int(round(rem_test * (len(by_label[-1]) / max(1, neg_pos_total))))),
        )
        target_counts[1] = test_size - target_counts[0] - target_counts[-1]
    else:
        for lbl in (-1, 0, 1):
            target_counts[lbl] = int(round(test_size * len(by_label[lbl]) / max(1, total)))

    for lbl in (-1, 0, 1):
        n_take = min(len(by_label[lbl]), target_counts.get(lbl, 0))
        test_rows.extend(by_label[lbl][:n_take])
        remaining_by_label[lbl] = by_label[lbl][n_take:]

    train_rows: List[Dict] = []
    val_rows: List[Dict] = []
    for lbl in (-1, 0, 1):
        pool = remaining_by_label[lbl]
        n_val = int(round(len(pool) * val_ratio))
        val_rows.extend(pool[:n_val])
        train_rows.extend(pool[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def write_csv(filepath: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description="Gán nhãn cảm xúc tự động/bán tự động & chia tập dữ liệu cho dmx-nlp-sentiment"
    )
    parser.add_argument("--input", default="data/raw/reviews.csv", help="Đường dẫn file CSV gốc")
    parser.add_argument(
        "--mode",
        default="hybrid-lexicon",
        choices=["hybrid-lexicon", "embedding-prototype", "zero-shot", "pretrained"],
        help="Phương pháp đánh giá cảm xúc để tạo nhãn target (mặc định: hybrid-lexicon)",
    )
    parser.add_argument("--stopwords", default="data/external/stopwords_vi.txt", help="Đường dẫn file stopwords")
    parser.add_argument("--test-size", type=int, default=500, help="Số lượng mẫu của tập test_gold")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Tỷ lệ tập validation trên phần còn lại")
    parser.add_argument("--pos-th", type=float, default=0.65, help="Ngưỡng điểm cực tính Positive")
    parser.add_argument("--neg-th", type=float, default=-0.65, help="Ngưỡng điểm cực tính Negative")
    parser.add_argument("--conflict-th", type=float, default=1.40, help="Ngưỡng đảo nhãn khi lệch sao mạnh")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Không tìm thấy file đầu vào: {args.input}", file=sys.stderr)
        sys.exit(1)

    stopwords = load_stopwords(args.stopwords)

    with open(args.input, "r", encoding="utf-8-sig") as f:
        raw_rows = list(csv.DictReader(f))

    print(f"[1/4] Đọc dữ liệu thô từ {args.input}: {len(raw_rows):,} dòng")

    seen_pairs = set()
    labeled_rows: List[Dict] = []
    skipped_invalid = 0
    skipped_duplicate = 0

    for row in raw_rows:
        content = (row.get("content") or "").strip()
        rating_str = (row.get("rating") or "").strip()
        if not rating_str.isdigit() or not is_valid_vietnamese_review(content):
            skipped_invalid += 1
            continue

        rating = int(rating_str)
        dedup_key = (content.lower(), rating)
        if dedup_key in seen_pairs:
            skipped_duplicate += 1
            continue
        seen_pairs.add(dedup_key)

        text_segmented, text_clean = preprocess_review(content, stopwords=stopwords, remove_stopwords=True)
        if not text_clean:
            skipped_invalid += 1
            continue

        res = reconcile_star_and_unsupervised(
            rating=rating,
            segmented_text=text_segmented,
            pos_threshold=args.pos_th,
            neg_threshold=args.neg_th,
            strong_conflict_threshold=args.conflict_th,
        )

        labeled_rows.append(
            {
                "review_id": row.get("review_id", ""),
                "category": row.get("category", ""),
                "product_code": row.get("product_code", ""),
                "rating": rating,
                "content": content,
                "text_segmented": text_segmented,
                "text_clean": text_clean,
                "label_star": res.label_star,
                "unsup_score": res.unsup_score,
                "label_unsup": res.label_unsup,
                "confidence": res.confidence,
                "conflict_type": res.conflict_type,
                "label_target": res.label_target,
            }
        )

    # Nếu người dùng chọn chế độ mở rộng (embedding-prototype / zero-shot / pretrained)
    if args.mode != "hybrid-lexicon":
        print(f"[2/4] Đang chạy mô hình '{args.mode}' trên {len(labeled_rows):,} bình luận...")
        ext_preds = run_external_model_labeling([r["text_segmented"] for r in labeled_rows], mode=args.mode)
        for r, pred in zip(labeled_rows, ext_preds):
            r["label_unsup"] = pred
            if r["rating"] == 3 and pred != 0:
                r["label_target"] = pred
                r["conflict_type"] = f"star_3_model_{pred}"
            elif r["label_star"] != pred and abs(r["unsup_score"]) >= 1.0:
                r["label_target"] = pred
                r["conflict_type"] = f"model_override_to_{pred}"
    else:
        print(f"[2/4] Đã chạy xong chế độ '{args.mode}' trên {len(labeled_rows):,} bình luận duy nhất hợp lệ.")

    conflicts = [r for r in labeled_rows if r["conflict_type"] != "none"]
    star_dist = Counter(r["label_star"] for r in labeled_rows)
    target_dist = Counter(r["label_target"] for r in labeled_rows)
    conflict_dist = Counter(r["conflict_type"] for r in conflicts)

    print(f"      - Đã lọc bỏ dòng rỗng/rác: {skipped_invalid:,} | Dòng trùng lặp: {skipped_duplicate:,}")
    print(f"      - Phân phối nhãn gốc theo sao (label_star):   {dict(sorted(star_dist.items()))}")
    print(f"      - Phân phối nhãn đích (label_target):         {dict(sorted(target_dist.items()))}")
    print(f"      - Số bình luận lệch sao được phát hiện & hiệu chỉnh: {len(conflicts):,} ({dict(conflict_dist)})")

    # Lưu dữ liệu trung gian
    all_fields = [
        "review_id",
        "category",
        "product_code",
        "rating",
        "content",
        "text_segmented",
        "text_clean",
        "label_star",
        "unsup_score",
        "label_unsup",
        "confidence",
        "conflict_type",
        "label_target",
    ]
    write_csv("data/interim/reviews_labeled.csv", labeled_rows, all_fields)
    write_csv("data/interim/rating_text_conflicts.csv", conflicts, all_fields)
    print("[3/4] Đã lưu:")
    print("      - data/interim/reviews_labeled.csv")
    print("      - data/interim/rating_text_conflicts.csv")

    # Chia tập dữ liệu train / val / test_gold
    train_rows, val_rows, test_rows = stratified_split(
        labeled_rows,
        test_size=args.test_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    split_fields = [
        "review_id",
        "category",
        "product_code",
        "rating",
        "content",
        "text_segmented",
        "text_clean",
        "label_star",
        "label_target",
        "conflict_type",
    ]
    write_csv("data/processed/train.csv", train_rows, split_fields)
    write_csv("data/processed/val.csv", val_rows, split_fields)
    write_csv("data/processed/test_gold.csv", test_rows, split_fields)

    summary = {
        "raw_total_rows": len(raw_rows),
        "valid_unique_rows": len(labeled_rows),
        "skipped_invalid": skipped_invalid,
        "skipped_duplicate": skipped_duplicate,
        "mode": args.mode,
        "label_star_distribution": {str(k): v for k, v in sorted(star_dist.items())},
        "label_target_distribution": {str(k): v for k, v in sorted(target_dist.items())},
        "total_conflicts_adjusted": len(conflicts),
        "conflict_breakdown": dict(conflict_dist),
        "splits": {
            "train": {
                "total": len(train_rows),
                "label_target_dist": {str(k): v for k, v in sorted(Counter(r["label_target"] for r in train_rows).items())},
            },
            "val": {
                "total": len(val_rows),
                "label_target_dist": {str(k): v for k, v in sorted(Counter(r["label_target"] for r in val_rows).items())},
            },
            "test_gold": {
                "total": len(test_rows),
                "label_target_dist": {str(k): v for k, v in sorted(Counter(r["label_target"] for r in test_rows).items())},
            },
        },
    }
    os.makedirs("data/interim", exist_ok=True)
    with open("data/interim/labeling_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[4/4] Đã chia và lưu tập dữ liệu vào data/processed/:")
    print(f"      - data/processed/train.csv     : {len(train_rows):,} mẫu | {summary['splits']['train']['label_target_dist']}")
    print(f"      - data/processed/val.csv       : {len(val_rows):,} mẫu | {summary['splits']['val']['label_target_dist']}")
    print(f"      - data/processed/test_gold.csv : {len(test_rows):,} mẫu | {summary['splits']['test_gold']['label_target_dist']}")


if __name__ == "__main__":
    main()
