# dmx-nlp-sentiment
**Hệ thống Phân loại Cảm xúc Bình luận Sản phẩm trên Điện Máy Xanh (Vietnamese Product Review Sentiment Analysis)**

Dự án thuộc học phần **Phân tích và Khai phá Dữ liệu Văn bản** (Học viện Công nghệ Bưu chính Viễn thông – PTIT), được tổ chức theo kiến trúc chuẩn của một kho mã nguồn Phân loại Văn bản (NLP Text Classification Repository) thực tế.

Hệ thống phân loại bình luận sản phẩm tiếng Việt thành **3 lớp cảm xúc**:
- **Tích cực (`Positive`, quy ước `1`)**: Bình luận khen chất lượng sản phẩm, dịch vụ giao hàng/lắp đặt tốt.
- **Trung tính (`Neutral`, quy ước `0`)**: Bình luận nhận xét trung lập, hỏi đáp hoặc vừa khen vừa góp ý nhẹ.
- **Tiêu cực (`Negative`, quy ước `-1`)**: Bình luận phàn nàn về lỗi kỹ thuật, độ ồn, hao điện, bảo hành hoặc dịch vụ kém.

---

## Kiến trúc Thư mục Chuẩn hóa (Production NLP Project Structure)

```text
dmx-nlp-sentiment/
├── configs/
│   └── config.yaml                       # Cấu hình trung tâm (đường dẫn, ngưỡng gán nhãn, siêu tham số mô hình)
├── data/
│   ├── raw/                              # Dữ liệu thô crawl từ Điện Máy Xanh (12,189 bình luận, bất biến)
│   │   ├── reviews.jsonl
│   │   ├── reviews.csv
│   │   └── reviews_drop_na.csv
│   ├── external/
│   │   └── stopwords_vi.txt              # Danh sách stopword tiếng Việt (đã bảo toàn từ phủ định & cảm xúc)
│   ├── interim/                          # Dữ liệu trung gian sinh bởi auto_label_review.py
│   │   ├── reviews_labeled.csv           # 9,516 bình luận duy nhất đã tách từ & gán nhãn target
│   │   ├── rating_text_conflicts.csv     # 490 bình luận bị lệch giữa số sao và nội dung câu chữ
│   │   └── labeling_summary.json         # Thống kê chi tiết trước/sau khi hiệu chỉnh nhãn tự động
│   ├── processed/                        # Dữ liệu sạch đã chia tập sẵn sàng cho huấn luyện & đánh giá
│   │   ├── train.csv                     # Tập huấn luyện (7,664 mẫu)
│   │   ├── val.csv                       # Tập kiểm định (1,352 mẫu)
│   │   └── test_gold.csv                 # Tập kiểm tra chuẩn (500 mẫu)
│   └── debug/                            # HTML & ảnh chụp màn hình phục vụ kiểm tra selector khi crawl
├── docs/
│   └── Bài Tập Lớn I - Lazada.pdf        # Báo cáo mẫu tham khảo của PTIT
├── figures/                              # Nơi lưu biểu đồ thực nghiệm của dự án Điện Máy Xanh
│   └── reference/                        # ⚠️ Lưu trữ 10 biểu đồ THAM KHẢO từ một repo khác cùng chủ đề
│       ├── cm_*.png
│       └── macro_f1_val_test.png
├── outputs/
│   ├── metrics/                          # Kết quả đánh giá JSON/CSV (Accuracy, Precision, Recall, F1)
│   └── predictions/                      # Kết quả dự đoán của các mô hình trên tập test_gold
├── scripts/
│   ├── crawl/                            # Bước 1A: Các script thu thập bình luận từ Điện Máy Xanh
│   │   ├── crawl_dmx_reviews.py
│   │   ├── multi_thread_crawler.py
│   │   ├── protected_multi_thread_crawler.py
│   │   └── json_to_csv.py
│   ├── auto_label_review.py              # Bước 1B & 2: Gán nhãn tự động/bán tự động & chia tập dữ liệu
│   ├── train_classical.py                # Bước 3, 4A & 5: Trích chọn BoW/TF-IDF/W2V, huấn luyện NB/SVM & vẽ biểu đồ
│   └── train_phobert.py                  # Bước 4B (Bổ sung Ý 4): Fine-tune mô hình hiện đại PhoBERT
├── src/                                  # Package mã nguồn lõi (Core NLP Modules)
│   ├── data/
│   │   ├── preprocess.py                 # Chuẩn hóa Unicode NFC, teencode, tách từ tiếng Việt, lọc stopword
│   │   └── auto_labeler.py               # Bộ gán nhãn Unsupervised Lexicon + Luật phủ định/chuyển ý + Số sao
│   ├── features/
│   │   └── build_features.py             # Trích chọn đặc trưng BoW, TF-IDF, Word2Vec Skip-gram
│   ├── models/
│   │   ├── classical.py                  # Baseline, Naïve Bayes (kèm Hybrid Sampling), LinearSVC (balanced)
│   │   └── phobert.py                    # Dataset & WeightedPhoBERTTrainer cho vinai/phobert-base-v2
│   └── evaluation/
│       └── metrics_and_plots.py          # Tính chỉ số đánh giá & vẽ Confusion Matrix 2-panel + biểu đồ Macro-F1
├── auto_label_review.py                  # Wrapper chạy nhanh tại thư mục gốc -> scripts/auto_label_review.py
├── requirements.txt                      # Danh sách thư viện phụ thuộc
├── PLAN.md                               # Kế hoạch thực hiện dự án
├── INSTRUCTIONS.md                       # Hướng dẫn chi tiết từng bước & khung viết báo cáo PTIT
├── AGENTS.md                             # Quy tắc bất biến dành cho AI Coding Agents
├── SKILLS.md                             # Danh mục 7 kỹ năng tác nghiệp cho AI Coding Agents
└── CLAUDE.md                             # Tài liệu tham chiếu nhanh cho Claude Code
```

> ### ⚠️ Lưu ý quan trọng về thư mục `figures/reference/`
> Các file ảnh trong `figures/reference/` (`cm_*.png`, `macro_f1_val_test.png`) **chỉ là các biểu đồ mẫu tham khảo từ một repository khác có cùng chủ đề** (được tách riêng vào `figures/reference/` để làm chuẩn định dạng trực quan hóa). Khi chạy `python scripts/train_classical.py` và `python scripts/train_phobert.py`, các biểu đồ kết quả thực tế của bộ dữ liệu Điện Máy Xanh sẽ được sinh mới trực tiếp vào `figures/`.

---

## Hướng dẫn Chạy Nhanh Toàn bộ Pipeline (Quickstart)

### 1. Cài đặt môi trường
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Bước 1 & 2: Tự động gán nhãn cảm xúc (`label_target`), tiền xử lý & chia tập dữ liệu
Chạy script `auto_label_review.py` để khử trùng lặp, tiền xử lý tiếng Việt, kết hợp **Số sao (`rating`) + Mô hình Unsupervised (`hybrid-lexicon` / `embedding-prototype` / `zero-shot` / `pretrained`)** nhằm phát hiện & hiệu chỉnh các bình luận bị lệch sao, đồng thời chia tập `train.csv` / `val.csv` / `test_gold.csv`:

```bash
# Chạy chế độ mặc định (Hybrid Unsupervised Lexicon + Star Reconciliation - chạy ngay trên CPU)
python3 auto_label_review.py

# Hoặc chạy với mô hình Pretrained / Zero-shot / Sentence Embeddings:
python3 auto_label_review.py --mode pretrained
```

**Kết quả thực tế sau khi chạy `auto_label_review.py` trên dữ liệu Điện Máy Xanh**:
- Từ **12,189 dòng thô** $\rightarrow$ lọc bỏ `49` dòng rỗng/rác (`"Hhhhh"`, ...) và `2,624` dòng trùng lặp $\rightarrow$ còn **9,516 bình luận duy nhất hợp lệ**.
- Phát hiện và hiệu chỉnh tự động **490 bình luận lệch giữa số sao và nội dung câu chữ** (lưu tại `data/interim/rating_text_conflicts.csv`), bao gồm:
  - `294` bình luận chấm 3 sao nhưng nội dung phàn nàn/báo lỗi rõ rệt (`star_3_clear_negative` $\rightarrow$ `-1`).
  - `79` bình luận chấm 4–5 sao nhưng nội dung chê gay gắt (`star_pos_text_strong_negative` $\rightarrow$ `-1`).
  - `67` bình luận chấm 3 sao nhưng nội dung khen hoàn toàn (`star_3_clear_positive` $\rightarrow$ `1`).
  - `41` bình luận chấm 4 sao có cấu trúc khen–chê đối lập (`star_4_mixed_contrast_to_neutral` $\rightarrow$ `0`).
  - `9` bình luận chấm 1–2 sao (bấm nhầm sao) nhưng nội dung khen ngợi (`star_neg_text_strong_positive` $\rightarrow$ `1`).
- Chia tập dữ liệu vào `data/processed/`:
  - `data/processed/train.csv`: **7,664 mẫu** (`-1: 1,951`, `0: 487`, `1: 5,226`)
  - `data/processed/val.csv`: **1,352 mẫu** (`-1: 344`, `0: 86`, `1: 922`)
  - `data/processed/test_gold.csv`: **500 mẫu** (`-1: 120`, `0: 60`, `1: 320`)

### 3. Bước 3, 4A & 5: Trích chọn đặc trưng (BoW, TF-IDF, Word2Vec) & Huấn luyện Naïve Bayes, SVM
```bash
python3 scripts/train_classical.py
```

### 4. Bước 4B (Bổ sung Ý 4): Huấn luyện mô hình hiện đại PhoBERT (`vinai/phobert-base-v2`)
```bash
python3 scripts/train_phobert.py --epochs 4 --batch-size 16
```

---

## Tài liệu Chi tiết trong Repository
- **[`PLAN.md`](PLAN.md)**: Kế hoạch tổng thể thực hiện dự án, kiến trúc hệ thống và phân công nhóm.
- **[`INSTRUCTIONS.md`](INSTRUCTIONS.md)**: Hướng dẫn chi tiết cách thực hiện từng bước và khung viết báo cáo bám sát mẫu `docs/Bài Tập Lớn I - Lazada.pdf`.
- **[`AGENTS.md`](AGENTS.md)** & **[`SKILLS.md`](SKILLS.md)**: Quy tắc bất biến và danh mục 7 kỹ năng dành cho AI Coding Agents.
