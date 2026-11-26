# Research v1.6 — Bản đồ tổng thể mục / bảng / hình

Đối chiếu từng mục, bảng và hình của `BAO_CAO_v1.6.pdf` (38 trang) với nguồn sơ cấp: code và artifacts của repo nguồn `sannguyen125/sentiment-classification` @ `01843665a5e31f64e6dce1adf66a0116ece3fbc0`, cùng tài liệu gốc của các thư viện/mô hình. Trạng thái: **Khớp** = số liệu/khẳng định truy được về nguồn; **Có điều kiện** = đúng nhưng thiếu điều kiện/ngữ cảnh; **Chênh** = chưa khớp hoặc không có artifact truy vết. Quy trình fine-tune PhoBERT được nghiên cứu sâu riêng tại `docs/research-phobert-finetuning-pipeline.md`.

| Mục / Bảng / Hình | Nguồn sơ cấp | Chi tiết | Trạng thái |
|---|---|---|---|
| 1.1–1.4 Giới thiệu, phát biểu, thách thức | `docs/HUONG_DAN_GAN_NHAN.md` (định nghĩa nhãn) | 01 | Khớp (còn câu lặp 1.4, typo "vôi") |
| 2.1 Quy trình phân loại văn bản | `src/features.py`, `src/split.py` | 01 | Khớp |
| 2.2.1 Làm sạch, chuẩn hóa | `src/preprocess.py: clean/normalize` | 01 | Có điều kiện (chữ thường) |
| 2.2.2 Chuẩn hóa từ viết tắt | `resources/teencode.txt` (65 mục) | 01 | Khớp |
| 2.2.3 Tách từ | Underthesea; VnCoreNLP RDRSegmenter | 01 | Khớp |
| 2.2.4 Từ dừng | `resources/stopwords_vi.txt` (111), `keep_words.txt` (39) | 01 | Khớp |
| 2.2.5 Stemming/lemmatization | (không áp dụng trong code) | 01 | Khớp |
| 2.3.1 BoW, n-gram | `src/features.py`, `results/metrics/val_grid.csv` | 02 | Khớp |
| 2.3.2 TF-IDF, cosine | `src/clean_label.py` (khử gần trùng), `src/features.py` | 02 | Có điều kiện (thiếu trường hợp f = 0) |
| 2.3.3 Word2Vec Skip-gram | `src/features.py` (gensim) | 02 | Khớp ở mức mô tả |
| 2.3.4 Véc-tơ câu trọng số TF-IDF | `src/features.py` | 02 | Khớp |
| 2.4.1 Naive Bayes | scikit-learn (MultinomialNB/GaussianNB) | 02 | Khớp |
| 2.4.2 SVM | scikit-learn (LinearSVC/SVC) | 02 | Có điều kiện (thiếu công thức RBF/γ) |
| 2.5 PhoBERT | VinAI phobert-base-v2; paper PhoBERT | 02 | Có điều kiện (thiếu công thức attention) |
| 2.6 Đánh giá mô hình | `src/evaluate.py` | 02 | Khớp |
| 3.1 Môi trường, công cụ | `requirements.txt`, `results/metrics/phobert.json` | 03 | Có điều kiện (LLM chưa nêu tên) |
| 3.2 Thu thập dữ liệu | `src/crawl/`, `results/metrics/crawl_stats.json` | 03 | Khớp |
| 3.3.1 Làm sạch — Bảng 3.1 | `src/clean_label.py`, `results/metrics/clean_stats.json` | 03 | Khớp |
| 3.3.2 Gán nhãn — Hình 3.2, Bảng 3.2 | `src/gold_llm.py`, `src/pool_llm.py`, `results/metrics/gold_agreement.json`, `split_stats.json` | 03 | Khớp |
| 3.4 Tiền xử lý — Bảng 3.3, Hình 3.3 | `src/preprocess.py`, `results/metrics/preprocess_stats.json`, `results/preprocess_examples.csv` | 01 | Khớp |
| Bảng 3.4 + Hình 3.4 (độ dài, >128 token, OOV) | Một phần `results/metrics/phobert.json` (1,52%) | 03 | Chênh (thiếu artifact cho OOV/val/test) |
| 3.5 Huấn luyện — Bảng 3.5 (siêu tham số) | `results/metrics/val_grid.csv`, `src/train_classic.py` | 04 | Khớp |
| Bảng 3.6 (PhoBERT theo epoch) | `results/metrics/phobert.json` | 04 | Khớp |
| 3.6.1 — Bảng 3.7, Hình 3.5 | `results/comparison.csv`, `results/metrics/test_metrics.json`, `results/figures/macro_f1_val_test.png` | 04 | Khớp |
| 3.6.2 — Bảng 3.8, Hình 3.6 | `results/comparison.csv`, `results/figures/cm_*.png` | 04 | Khớp |
| 3.6.3 Có/không bỏ từ dừng | `results/figures/cm_tf_idf_svm_khong_bo_stopword_tham_khao.png`, metrics val | 04 | Khớp |
| 3.6.4 — Bảng 3.9 (sao vs nội dung) | `results/archive_star/comparison.csv` | 04 | Khớp |
| 3.6.5 — Bảng 3.10 (nhãn người) | `results/robustness.csv`, `results/metrics/gold_agreement.json` | 04 | Khớp |
| 3.7.1–3.7.2 — Bảng 3.11–3.13 (lỗi) | `results/errors.csv`, `results/error_groups.csv` | 04 | Khớp |
| 3.7.3 Câu khen chê lẫn lộn | `src/predict.py` (câu mặc định đầu tiên) | 04 | Khớp |
| 3.8 Kết luận, hướng phát triển | Tổng hợp các mục trên | 04 | Khớp |
