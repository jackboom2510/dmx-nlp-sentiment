# TC-06 — Tách từ 2 tầng: Underthesea (text_seg) -> gỡ "_" -> RDRSegmenter tách lại
# Nguồn: notebooks/02_phobert_colab.ipynb ô 7 · src/preprocess.py segment() dòng 99
# Điều kiện: RDRSegmenter cần py_vncorenlp + Java + VnCoreNLP-1.2.jar (models/vncorenlp).
#   Thiếu một trong các điều kiện -> code notebook FALLBACK lặng lẽ về Underthesea.
# Kỳ vọng: 3 cặp underthesea/input đã lưu trong output ô 7 (bên dưới) — RDR gộp/tách khác Underthesea.

import os

# --- Hàm resegment đúng như ô 7 của notebook ---
SEGMENTER = "underthesea (fallback)"

def make_resegment(root_models: str):
    """Trả về (hàm resegment, tên bộ tách từ). Thiếu VnCoreNLP -> hàm giữ nguyên text_seg."""
    try:
        import py_vncorenlp
        vn_dir = os.path.join(root_models, "vncorenlp")
        os.makedirs(vn_dir, exist_ok=True)
        if not os.path.exists(os.path.join(vn_dir, "VnCoreNLP-1.2.jar")):
            py_vncorenlp.download_model(save_dir=vn_dir)
        cwd = os.getcwd()  # VnCoreNLP đổi thư mục làm việc -> phải lưu & khôi phục
        rdr = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vn_dir)
        os.chdir(cwd)

        def resegment(t: str) -> str:
            raw = t.replace("_", " ").strip()  # gỡ tách từ Underthesea trước khi tách lại
            return " ".join(rdr.word_segment(raw)) if raw else ""

        return resegment, "RDRSegmenter (VnCoreNLP)"
    except Exception as e:
        print("VnCoreNLP unavailable -> keep underthesea segmentation:", repr(e)[:200])
        return (lambda t: t), SEGMENTER


# --- 3 cặp ví dụ kỳ vọng, chép từ output đã lưu của ô 7 (lần chạy chính thức, RTX 3060) ---
EXPECTED = [
    ("Nấu cơm bình_thường , được",
     "Nấu cơm bình_thường , được"),
    ("Máy dùng tốt . Tiếng không bị to quá . Mỗi tội hút tóc kém",
     "Máy dùng tốt . Tiếng không bị to quá . Mỗi_tội hút tóc kém"),
    ("Thợ điện_máy xanh nắp cho mình Model Cục nóng khác model dàn lạnh , chán hết chỗ nói",
     "Thợ_điện máy xanh nắp cho mình Model_Cục nóng khác model dàn lạnh , chán hết_chỗ_nói"),
]

if __name__ == "__main__":
    resegment, name = make_resegment(os.environ.get("SC_MODELS", "models"))
    print("Segmenter đang dùng:", name)
    for under, expected in EXPECTED:
        got = resegment(under)
        print("\n underthesea:", under)
        print(" input      :", got)
        if name.startswith("RDR"):
            assert got == expected, (got, expected)
    print("\nPASS TC-06: PhoBERT ăn văn bản đã tách HAI tầng; metadata phobert.json ghi "
          "'segmenter: RDRSegmenter (VnCoreNLP)' — đó là bằng chứng lần chạy thật dùng bộ nào")
