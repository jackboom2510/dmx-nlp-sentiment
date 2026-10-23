#!/usr/bin/env python3
"""
Module gán nhãn cảm xúc tự động / bán tự động (Unsupervised & Hybrid Weak Supervision)
cho bình luận sản phẩm Điện Máy Xanh.

Hỗ trợ 4 chế độ (--mode):
1. hybrid-lexicon (Mặc định): Kết hợp Từ điển cảm xúc miền Điện máy + Luật phủ định/Chuyển ý + Số sao.
   Chạy siêu tốc trên CPU không cần tải model ngoài, minh bạch 100% về mặt học thuật.
2. embedding-prototype: Unsupervised Sentence Embeddings (Cosine Similarity với 3 câu Prototype).
3. zero-shot: Multilingual NLI Zero-shot Classification (mDeBERTa-v3-base-mnli-xnli).
4. pretrained: Teacher model tiếng Việt có sẵn (wonrax/phobert-base-vietnamese-sentiment).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# 1. Từ điển cảm xúc tích cực miền Điện Máy Xanh (trọng số > 0)
POSITIVE_LEXICON: Dict[str, float] = {
    "rất_tốt": 1.6,
    "tuyệt_vời": 1.6,
    "xuất_sắc": 1.6,
    "rất_ưng_ý": 1.6,
    "hài_lòng": 1.4,
    "ưng_ý": 1.4,
    "đáng_tiền": 1.3,
    "tiết_kiệm_điện": 1.3,
    "tiết_kiệm": 1.0,
    "lạnh_nhanh": 1.3,
    "lạnh_sâu": 1.3,
    "êm_ái": 1.3,
    "nhiệt_tình": 1.2,
    "chu_đáo": 1.2,
    "tận_tình": 1.2,
    "dễ_thương": 1.1,
    "mềm_dẻo": 1.2,
    "chín_đều": 1.2,
    "rõ_nét": 1.2,
    "sắc_nét": 1.2,
    "mượt_mà": 1.2,
    "sang_trọng": 1.1,
    "hiện_đại": 1.0,
    "nhỏ_gọn": 0.9,
    "dễ_dàng": 0.9,
    "ổn_định": 1.0,
    "hợp_lý": 1.0,
    "tốt": 1.0,
    "êm": 1.1,
    "mát": 1.0,
    "ngon": 1.1,
    "bền": 1.1,
    "đẹp": 0.9,
    "nhanh": 0.8,
    "sạch": 0.9,
    "mượt": 1.0,
    "nét": 0.9,
    "ổn": 0.7,
    "thích": 0.9,
}

# 2. Từ điển cảm xúc tiêu cực miền Điện Máy Xanh (trọng số < 0)
NEGATIVE_LEXICON: Dict[str, float] = {
    "thất_vọng": -1.7,
    "bực_mình": -1.6,
    "phí_tiền": -1.6,
    "tiếc_tiền": -1.6,
    "kêu_to": -1.5,
    "rung_lắc": -1.5,
    "rỉ_nước": -1.5,
    "chảy_nước": -1.5,
    "đọng_nước": -1.4,
    "ra_mồ_hôi": -1.4,
    "đổ_mồ_hôi": -1.4,
    "không_lạnh": -1.6,
    "không_mát": -1.6,
    "lâu_lạnh": -1.3,
    "hao_điện": -1.4,
    "tốn_điện": -1.4,
    "khét_đáy": -1.6,
    "bị_cháy": -1.6,
    "bị_sống": -1.5,
    "bị_nhão": -1.4,
    "bị_khô": -1.3,
    "bong_tróc": -1.5,
    "tróc_sơn": -1.5,
    "tróc_men": -1.5,
    "chập_chờn": -1.5,
    "hư_hỏng": -1.6,
    "bất_thường": -1.3,
    "khó_chịu": -1.3,
    "bất_tiện": -1.1,
    "lỗi": -1.5,
    "hỏng": -1.6,
    "hư": -1.5,
    "tệ": -1.6,
    "kém": -1.4,
    "ồn": -1.3,
    "kêu": -1.1,
    "rung": -1.1,
    "khét": -1.5,
    "nhão": -1.2,
    "sống": -1.2,
    "trào": -1.1,
    "nóng": -1.0,
    "chậm": -0.9,
    "lag": -1.1,
    "đơ": -1.2,
    "chán": -1.3,
    "đắt": -0.8,
}

# 3. Từ khóa trung tính / dè dặt (Neutral / Hedging)
NEUTRAL_MARKERS = {
    "bình_thường",
    "tạm_ổn",
    "tạm",
    "thời_gian",
    "mới_mua",
    "xem_sao",
}

NEGATION_WORDS = {"không", "chưa", "chẳng", "chả", "đừng"}
INTENSIFIERS: Dict[str, float] = {
    "rất": 1.35,
    "quá": 1.35,
    "cực": 1.45,
    "cực_kỳ": 1.5,
    "lắm": 1.25,
    "hơi": 0.65,
    "khá": 0.85,
}
CONTRAST_CONJUNCTIONS = {"nhưng", "tuy_nhiên", "mỗi_tội", "có_điều", "mà"}


@dataclass
class AutoLabelResult:
    label_star: int
    unsup_score: float
    label_unsup: int
    confidence: float
    label_target: int
    conflict_type: str


def map_star_to_label(rating: int) -> int:
    """Quy đổi số sao (1-5) sang nhãn cảm xúc sơ bộ (-1, 0, 1) theo chuẩn báo cáo mẫu."""
    if rating >= 4:
        return 1
    if rating == 3:
        return 0
    return -1


def score_segmented_review_unsupervised(
    segmented_text: str,
    pos_threshold: float = 0.65,
    neg_threshold: float = -0.65,
) -> Tuple[float, int, float, bool, int, int]:
    """
    Tính điểm cực tính cảm xúc (Unsupervised Lexicon + Rule-based) cho một bình luận đã tách từ.
    Trả về:
        (total_score, predicted_label, confidence, has_contrast, pos_hits, neg_hits)
    """
    if not segmented_text or not segmented_text.strip():
        return 0.0, 0, 0.3, False, 0, 0

    raw_tokens = [
        t.strip(".,!?;:()[]\"'")
        for t in segmented_text.lower().split()
        if t.strip(".,!?;:()[]\"'")
    ]
    if not raw_tokens:
        return 0.0, 0, 0.3, False, 0, 0

    # Nếu câu chỉ có từ trung tính rõ rệt ("bình_thường", "tạm_ổn")
    if set(raw_tokens).issubset(NEUTRAL_MARKERS | {"thì", "cũng", "ạ", "nhé", "thôi"}):
        return 0.0, 0, 0.85, False, 0, 0

    # Kiểm tra có liên từ chuyển ý ("nhưng", "tuy_nhiên", "mỗi_tội") ở giữa câu hay không
    contrast_idx = -1
    for idx, tok in enumerate(raw_tokens):
        if tok in CONTRAST_CONJUNCTIONS and 0 < idx < len(raw_tokens) - 1:
            contrast_idx = idx
            break
    has_contrast = contrast_idx != -1

    total_score = 0.0
    pos_hits = 0
    neg_hits = 0

    for i, tok in enumerate(raw_tokens):
        val = POSITIVE_LEXICON.get(tok, NEGATIVE_LEXICON.get(tok, 0.0))
        if val == 0.0:
            continue

        prev_1 = raw_tokens[i - 1] if i >= 1 else ""
        prev_2 = raw_tokens[i - 2] if i >= 2 else ""
        next_1 = raw_tokens[i + 1] if i + 1 < len(raw_tokens) else ""

        # Xử lý đảo cực khi có từ phủ định đứng trước (trong cửa sổ 1-2 từ)
        if prev_1 in NEGATION_WORDS or prev_2 in NEGATION_WORDS:
            if val > 0:
                # Ví dụ: "không tốt", "không hài_lòng", "không chín" -> Tiêu cực mạnh
                val = -abs(val) * 1.15
            else:
                # Ví dụ: "không ồn", "không kêu", "không lỗi" -> Tích cực nhẹ
                val = abs(val) * 0.70

        # Nhân hệ số mức độ (Intensifier / Diminisher)
        if prev_1 in INTENSIFIERS:
            val *= INTENSIFIERS[prev_1]
        elif next_1 in {"lắm", "quá"}:
            val *= 1.25

        # Mệnh đề sau liên từ chuyển ý ("nhưng", "tuy_nhiên", "mỗi_tội") mang trọng số quyết định cao hơn
        if has_contrast and i > contrast_idx:
            val *= 1.45

        if val > 0:
            pos_hits += 1
        elif val < 0:
            neg_hits += 1

        total_score += val

    # Quy đổi điểm sang nhãn (-1, 0, 1) và độ tự tin (confidence)
    abs_score = abs(total_score)
    confidence = min(0.98, 0.50 + 0.20 * abs_score)

    if total_score >= pos_threshold:
        pred_label = 1
    elif total_score <= neg_threshold:
        pred_label = -1
    else:
        pred_label = 0
        confidence = max(0.55, 0.75 - 0.25 * abs_score)

    return round(total_score, 4), pred_label, round(confidence, 4), has_contrast, pos_hits, neg_hits


def reconcile_star_and_unsupervised(
    rating: int,
    segmented_text: str,
    pos_threshold: float = 0.65,
    neg_threshold: float = -0.65,
    strong_conflict_threshold: float = 1.40,
) -> AutoLabelResult:
    """
    Kết hợp nhãn số sao (Silver Label) và bộ đánh giá cảm xúc Unsupervised trên văn bản
    để tạo nhãn đích (label_target) chính xác và giải quyết hiện tượng lệch sao - nội dung.
    """
    label_star = map_star_to_label(rating)
    score, label_unsup, conf, has_contrast, pos_hits, neg_hits = score_segmented_review_unsupervised(
        segmented_text,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold,
    )

    label_target = label_star
    conflict_type = "none"

    # Trường hợp 1: Đánh giá 3 sao (Neutral mặc định), nhưng nội dung khen hoặc chê rất rõ ràng
    if rating == 3:
        if score <= -1.0 and neg_hits >= 1 and pos_hits == 0:
            label_target = -1
            conflict_type = "star_3_clear_negative"
        elif score >= 1.0 and pos_hits >= 1 and neg_hits == 0 and not has_contrast:
            label_target = 1
            conflict_type = "star_3_clear_positive"

    # Trường hợp 2: Đánh giá 4-5 sao (Positive mặc định), nhưng nội dung phàn nàn/báo lỗi gay gắt
    elif rating >= 4:
        if score <= -strong_conflict_threshold and neg_hits >= 1 and pos_hits == 0:
            label_target = -1
            conflict_type = "star_pos_text_strong_negative"
        elif rating == 4 and has_contrast and pos_hits >= 1 and neg_hits >= 1 and -0.9 <= score <= 0.6:
            # Ví dụ 4 sao: "Nhìn chung cũng ok. Nhưng có điều tìm kiếm giọng nói hơi lag..."
            label_target = 0
            conflict_type = "star_4_mixed_contrast_to_neutral"

    # Trường hợp 3: Đánh giá 1-2 sao (Negative mặc định), nhưng nội dung toàn lời khen (bấm nhầm sao)
    elif rating <= 2:
        if score >= strong_conflict_threshold and pos_hits >= 1 and neg_hits == 0 and not has_contrast:
            label_target = 1
            conflict_type = "star_neg_text_strong_positive"

    return AutoLabelResult(
        label_star=label_star,
        unsup_score=score,
        label_unsup=label_unsup,
        confidence=conf,
        label_target=label_target,
        conflict_type=conflict_type,
    )
