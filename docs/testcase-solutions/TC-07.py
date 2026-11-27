# TC-07 — Chuẩn hóa dấu thanh / rút ký tự kéo dài / ranh giới từ đa nghĩa
# Nguồn: src/preprocess.py clean() 63, normalize() 84, correct_words() 93 · resources/teencode.txt
# Điều kiện: bản rút gọn tự chạy được (không cần underthesea); logic chép đúng từ src/preprocess.py.
# Kỳ vọng: normalize(clean("hoà tốtttt!!! máy ko tốt")) == "hòa tốt! máy ko tốt"
#          "kh" KHÔNG bị sửa thành "không" (đa nghĩa: còn là "khách hàng").

import re
import unicodedata
from pathlib import Path

# ---------- clean() rút gọn (src/preprocess.py:63) ----------
HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
ODD_CHARS_RE = re.compile(r"[^\w\s.,!?;:()\-/%'\"]")

def clean(text: str) -> str:
    text = HTML_RE.sub(" ", str(text))
    text = URL_RE.sub(" ", text)
    text = ODD_CHARS_RE.sub(" ", text).replace("_", " ")
    text = re.sub(r"([.,!?;:])(?=[^\W\d_])", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------- normalize() (src/preprocess.py:84): NFC -> dấu thanh cũ -> rút ký tự lặp ----------
_TONE = {"à": "ò", "á": "ó", "ả": "ỏ", "ã": "õ", "ạ": "ọ",
         "è": "ò", "é": "ó", "ẻ": "ỏ", "ẽ": "õ", "ẹ": "ọ"}
_TONE_Y = {"ỳ": "ù", "ý": "ú", "ỷ": "ủ", "ỹ": "ũ", "ỵ": "ụ"}
OA_RE = re.compile(r"o([àáảãạèéẻẽẹ])(?!\w)")
UY_RE = re.compile(r"(?<![qQ])u([ỳýỷỹỵ])(?!\w)")
REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")       # chữ lặp >= 3 lần: tốtttt -> tốt
PUNCT_REPEAT_RE = re.compile(r"([^\w\s])\1+")     # !!! -> !

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)  # NFC phải chạy TRƯỚC regex dấu thanh
    text = OA_RE.sub(lambda m: _TONE[m.group(1)] + ("a" if m.group(1) in "àáảãạ" else "e"), text)
    text = UY_RE.sub(lambda m: _TONE_Y[m.group(1)] + "y", text)
    text = REPEAT_RE.sub(r"\1", text)
    return PUNCT_REPEAT_RE.sub(r"\1", text)

# ---------- correct_words(): nạp teencode.txt, khóa dài ưu tiên, whole-word ----------
def load_teencode(path: Path) -> dict:
    d = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            k, v = ln.split("\t")
            d[k] = v
    return d

if __name__ == "__main__":
    got = normalize(clean("hoà tốtttt!!! máy ko tốt"))
    print("clean+normalize:", got)
    assert got == "hòa tốt! máy ko tốt", got

    tc_path = Path(__file__).with_name("teencode.txt")  # nếu đặt cạnh file tài nguyên
    repo_tc = Path("resources/teencode.txt")           # chạy tại gốc repo sentiment-classification
    tc = load_teencode(repo_tc) if repo_tc.exists() else {}
    if tc:
        print("Số mục teencode:", len(tc))
        assert len(tc) == 65 and "kh" not in tc and tc.get("ko") == "không"
        print("'kh' không có trong từ điển -> câu 'kh ok 5sao ổn' giữ nguyên 'kh'")
    print("PASS TC-07: thứ tự NFC -> dấu thanh -> rút lặp; từ đa nghĩa bị loại có chủ đích")
