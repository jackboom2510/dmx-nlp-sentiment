#!/usr/bin/env python3
"""
Module tiền xử lý văn bản tiếng Việt cho bình luận sản phẩm Điện Máy Xanh.
Hỗ trợ:
- Chuẩn hóa Unicode dựng sẵn (NFC)
- Lọc bình luận rác (chỉ chứa ký tự lặp như "Hhhhh", số hoặc ký tự đặc biệt)
- Chuẩn hóa từ viết tắt, teencode và cụm từ không dấu phổ biến trên TMĐT
- Tách từ tiếng Việt (sử dụng pyvi/underthesea nếu có, tự động fallback sang từ điển ghép từ Longest-Matching)
- Lọc stopword tiếng Việt có bảo toàn từ phủ định & từ chỉ mức độ cảm xúc
"""

import os
import re
import unicodedata
from typing import Optional, Set, Tuple

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

try:
    from underthesea import word_tokenize as uts_word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False

# Từ điển chuẩn hóa từ viết tắt / teencode đơn âm tiết
TEENCODE_MAP = {
    "đmx": "điện_máy_xanh",
    "dmx": "điện_máy_xanh",
    "tgdd": "thế_giới_di_động",
    "sp": "sản_phẩm",
    "nv": "nhân_viên",
    "ko": "không",
    "k": "không",
    "kh": "không",
    "hok": "không",
    "kg": "không",
    "đc": "được",
    "dc": "được",
    "dk": "được",
    "ok": "tốt",
    "oke": "tốt",
    "okay": "tốt",
    "okie": "tốt",
    "good": "tốt",
    "bh": "bảo_hành",
    "sd": "sử_dụng",
    "ae": "anh_em",
    "mn": "mọi_người",
    "mk": "mình",
    "vs": "với",
    "cx": "cũng",
    "wa": "quá",
    "qá": "quá",
    "dt": "điện_thoại",
    "đt": "điện_thoại",
    "tv": "tivi",
    "tũ": "tủ",
    "cữa": "cửa",
    "xún": "xuống",
    "sài": "xài",
}

# Cụm từ không dấu phổ biến trong bình luận Điện Máy Xanh -> có dấu chuẩn
UNACCENTED_PHRASES = [
    (r"\brat tot\b", "rất tốt"),
    (r"\brat ung y\b", "rất ưng ý"),
    (r"\bung y\b", "ưng ý"),
    (r"\bhai long\b", "hài lòng"),
    (r"\bnhiet tinh\b", "nhiệt tình"),
    (r"\bchu dao\b", "chu đáo"),
    (r"\btan tinh\b", "tận tình"),
    (r"\btiet kiem\b", "tiết kiệm"),
    (r"\bde thuong\b", "dễ thương"),
    (r"\bbao hanh\b", "bảo hành"),
    (r"\bsan pham\b", "sản phẩm"),
    (r"\bnhan vien\b", "nhân viên"),
    (r"\bdien may xanh\b", "điện máy xanh"),
    (r"\bsu dung\b", "sử dụng"),
    (r"\bho tro\b", "hỗ trợ"),
]

# Danh sách từ ghép tiếng Việt miền Điện máy & Cảm xúc phục vụ tách từ fallback (khi chưa cài pyvi)
COMPOUND_WORDS_VI = sorted(
    [
        "điện máy xanh", "thế giới di động", "nhân viên", "sản phẩm", "bảo hành",
        "sử dụng", "tiết kiệm", "tiết kiệm điện", "hài lòng", "ưng ý", "nhiệt tình",
        "chu đáo", "tận tình", "dễ thương", "thất vọng", "bực mình", "khó chịu",
        "bình thường", "tạm ổn", "ổn định", "tuyệt vời", "xuất sắc", "đáng tiền",
        "phí tiền", "tiếc tiền", "hợp lý", "máy giặt", "tủ lạnh", "máy lạnh",
        "điều hòa", "nồi cơm", "nồi cơm điện", "máy lọc nước", "lọc nước",
        "kêu to", "rung lắc", "êm ái", "lạnh nhanh", "lạnh sâu", "lâu lạnh",
        "không lạnh", "không mát", "rỉ nước", "chảy nước", "đọng nước", "ra mồ hôi",
        "đổ mồ hôi", "hao điện", "tốn điện", "bong tróc", "tróc sơn", "tróc men",
        "chập chờn", "hư hỏng", "bất thường", "nguyên nhân", "giải quyết",
        "dứt điểm", "chính sách", "đổi trả", "lắp đặt", "giao hàng", "tư vấn",
        "tổng đài", "chăm sóc", "khách hàng", "chất lượng", "thiết kế", "nhỏ gọn",
        "sang trọng", "hiện đại", "dễ dàng", "bất tiện", "cảm ứng", "điều khiển",
        "giọng nói", "hình ảnh", "âm thanh", "rõ nét", "sắc nét", "mượt mà",
        "mềm dẻo", "chín đều", "khét đáy", "bị cháy", "bị sống", "bị nhão",
        "bị khô", "tuy nhiên", "mỗi tội", "có điều", "nhìn chung", "hiện tại",
        "thời gian", "kinh nghiệm", "lắp ráp",
    ],
    key=lambda x: len(x.split()),
    reverse=True,
)

# Các từ phủ định & chỉ mức độ cảm xúc BẮT BUỘC GIỮ LẠI khi lọc stopword
NEGATION_AND_SENTIMENT_KEEP = {
    "không", "chưa", "chẳng", "chả", "đừng", "chớ",
    "rất", "quá", "lắm", "hơi", "khá", "cực", "cực_kỳ",
    "tốt", "tệ", "kém", "ổn", "đẹp", "xấu", "lỗi", "hỏng", "hư",
    "ồn", "êm", "mát", "nóng", "nhanh", "chậm", "ngon", "bền",
    "nhưng", "tuy_nhiên", "mỗi_tội", "có_điều", "bị",
}


def load_stopwords(filepath: str = "data/external/stopwords_vi.txt") -> Set[str]:
    """Đọc danh sách stopword và bảo toàn các từ phủ định/cảm xúc."""
    stopwords: Set[str] = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                if line and not line.startswith("#"):
                    stopwords.add(line.replace(" ", "_"))
    else:
        stopwords = {
            "thì", "là", "mà", "và", "của", "những", "các", "một", "này", "đó",
            "cho", "với", "trong", "khi", "để", "từ", "về", "như", "theo", "do",
            "tại", "vì", "nếu", "hay", "hoặc", "vào", "ra", "lên", "xuống", "ạ",
        }
    return stopwords - NEGATION_AND_SENTIMENT_KEEP


def is_valid_vietnamese_review(text: str) -> bool:
    """
    Kiểm tra bình luận có hợp lệ không:
    - Không rỗng
    - Không phải chuỗi lặp vô nghĩa kiểu 'Hhhhh', 'aaaa', '11111'
    - Có ít nhất 1 từ có nghĩa (>= 2 ký tự chữ cái)
    """
    if not isinstance(text, str):
        return False
    cleaned = re.sub(r"[^\w\s]", " ", text.strip().lower())
    cleaned = re.sub(r"\d+", " ", cleaned)
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    if not tokens:
        return False
    # Nếu toàn bộ câu chỉ gồm 1 ký tự lặp lại (vd: "hhhhh", "aaaaa") -> loại bỏ
    joined = "".join(tokens)
    if len(set(joined)) <= 1 and len(joined) >= 3:
        return False
    return True


def segment_vietnamese(text: str) -> str:
    """Tách từ tiếng Việt bằng pyvi / underthesea hoặc fallback từ điển ghép từ."""
    if HAS_PYVI:
        return ViTokenizer.tokenize(text)
    if HAS_UNDERTHESEA:
        return uts_word_tokenize(text, format="text")
    # Fallback: nối các từ ghép trong COMPOUND_WORDS_VI bằng dấu '_'
    out = f" {text} "
    for phrase in COMPOUND_WORDS_VI:
        underscored = phrase.replace(" ", "_")
        out = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", underscored, out)
    return " ".join(out.split())


def preprocess_review(
    raw_text: str,
    stopwords: Optional[Set[str]] = None,
    remove_stopwords: bool = True,
) -> Tuple[str, str]:
    """
    Trả về bộ 2 chuỗi:
    - text_segmented: Đã chuẩn hóa Unicode, teencode và tách từ ghép (giữ stopword cho PhoBERT)
    - text_clean: Đã lọc thêm dấu câu và stopword (bảo toàn từ phủ định) cho BoW / TF-IDF / Word2Vec
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return "", ""

    # 1. Chuẩn hóa bảng mã Unicode dựng sẵn NFC
    text = unicodedata.normalize("NFC", raw_text.strip())
    text = text.lower()

    # 2. Xóa URL, HTML tags
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    # 3. Kéo giãn ký tự lặp liên tiếp >= 3 lần (vd: "tốtttt" -> "tốt", "ồn quáaaa" -> "ồn quá")
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # 4. Chuẩn hóa cụm từ không dấu phổ biến
    for pattern, replacement in UNACCENTED_PHRASES:
        text = re.sub(pattern, replacement, text)

    # 5. Thay thế dấu câu bằng khoảng trắng nhẹ để chuẩn hóa token viết tắt
    text_alnum = re.sub(
        r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ.,!?]",
        " ",
        text,
    )

    tokens = []
    for tok in text_alnum.split():
        clean_tok = tok.strip(".,!?")
        if clean_tok in TEENCODE_MAP:
            tokens.append(TEENCODE_MAP[clean_tok])
        else:
            tokens.append(tok)
    normalized_text = " ".join(tokens)

    # 6. Tách từ tiếng Việt (Word Segmentation)
    text_segmented = segment_vietnamese(normalized_text)

    # 7. Tạo bản text_clean cho BoW / TF-IDF / Word2Vec
    clean_only_letters = re.sub(
        r"[^\w\s_àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
        " ",
        text_segmented,
    )
    clean_only_letters = re.sub(r"\b\d+\b", " ", clean_only_letters)
    words = [w.strip("_") for w in clean_only_letters.split() if len(w.strip("_")) >= 2]
    if remove_stopwords and stopwords:
        words = [w for w in words if w not in stopwords]
    text_clean = " ".join(words)

    return text_segmented, text_clean
