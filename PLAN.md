# Kế hoạch dự án: Phân loại cảm xúc bình luận Điện Máy Xanh

## 1. Mục tiêu

Xây dựng pipeline tiếng Việt có thể tái lập để phân loại bình luận thiết bị gia dụng thành **Negative (-1), Neutral (0), Positive (1)**. Dự án so sánh đặc trưng sparse/dense, mô hình ML truyền thống và Transformer, cung cấp phân tích lỗi có chiều sâu cùng notebook định dạng báo cáo PTIT.

**Ưu tiên phương pháp luận:** nhãn `label_target` hiện là nhãn hybrid tự động (rating + lexicon), do đó `test.csv` là held-out pseudo-label chứ chưa phải gold label. Cần human annotation độc lập trước khi kết luận chất lượng thật. Các chỉ số hiện có không được mô tả là kết quả trên nhãn chân trị.

## 2. Trạng thái triển khai

| Giai đoạn | Trạng thái | Artifact / ghi chú |
|---|---|---|
| 1. Thu thập dữ liệu | Có pipeline | Crawler tại `scripts/crawl/`, dữ liệu gốc `data/raw/` |
| 2. Làm sạch / gán nhãn / chia tập | Đã triển khai | `auto_label_review.py`, `src/data/`; nhãn hybrid, conflict log, stratified splits |
| 3. Tiền xử lý tiếng Việt | Đã triển khai | NFC, teencode, word segmentation, negation-safe stopwords |
| 4. Đặc trưng & mô hình cổ điển | Đã triển khai | BoW, TF-IDF, Word2Vec + NB / LinearSVC + baseline |
| 5. Transformer | Đã triển khai / cần benchmark | PhoBERT fine-tuning và generic runner hỗ trợ XLM-RoBERTa Large |
| 6. Evaluation & reporting | Đã mở rộng | Per-class, confusion matrix, bootstrap CI, prediction artifacts, notebook report |
| 7. Human gold audit | Cần thực hiện | Export mẫu mù bằng `scripts/export_annotation_sample.py`, hai annotator + adjudication |

## 3. Thiết kế khoa học và protocol

### Câu hỏi nghiên cứu
- **RQ1:** TF-IDF có cải thiện Macro-F1 so với BoW không?
- **RQ2:** LinearSVC có tăng chất lượng lớp thiểu số so với NB không?
- **RQ3:** Word2Vec mean pooling mất bao nhiêu tín hiệu so với n-gram?
- **RQ4:** PhoBERT và XLM-R Large có tăng hiệu năng per-class, đặc biệt Neutral/Negative?
- **RQ5:** Các loại lỗi nào chi phối và bao nhiêu phần bắt nguồn từ nhãn tự động?

### Thiết kế đánh giá
- Dùng cùng split và cùng `label_target` cho mọi model; fit vectorizer/embedding chỉ trên train.
- Chọn cấu hình theo validation Macro-F1; chỉ đánh giá test ở bước cuối.
- Báo cáo Accuracy, macro P/R/F1, weighted-F1, F1 theo lớp, confusion matrix, bootstrap CI.
- Ghi rõ benchmark pseudo-label và benchmark human-audited như hai kết quả riêng.
- Bổ sung group split theo `product_code` để đo tổng quát hóa sang sản phẩm chưa thấy.

## 4. Kiến trúc hiện tại

```text
configs/config.yaml
 data/raw/                   # Reviews thô
 data/external/              # Stopwords
 data/interim/               # Nhãn hybrid, conflicts, summary
 data/processed/             # train.csv / val.csv / test.csv (pseudo-label)
 src/data/                   # Tiền xử lý & auto-labeling
 src/features/               # BoW, TF-IDF, Word2Vec
 src/models/                 # Baselines, PhoBERT / Transformer
 src/evaluation/             # Metrics, CI, confusion matrices
 scripts/                    # Auto-label, train, crawl, human annotation sample
 notebooks/01_sentiment_analysis_report.ipynb
 figures/reference/          # Biểu đồ tham khảo repo khác, không phải kết quả DMX
 outputs/                    # Metrics, predictions, model checkpoints, annotation sheets
```

## 5. Kết quả preprocessing hiện có

Lần chạy mặc định: 12,189 dòng thô → 9,516 dòng hợp lệ duy nhất (49 dòng rác và 2,624 trùng lặp được loại); 490 trường hợp rating–text bị đánh dấu/hiệu chỉnh. Split stratified có 7,664 train, 1,352 validation và 500 held-out test. Nhãn test phân bố `-1:120, 0:60, 1:320`. Các con số này mô tả tập pseudo-labeled hiện có, không phải human-gold test.

## 6. Ma trận thực nghiệm

| Biểu diễn | Model | Mục đích |
|---|---|---|
| BoW | Majority, MultinomialNB, LinearSVC | Baseline và so sánh classifier |
| TF-IDF | MultinomialNB, LinearSVC | Trọng số thông tin từ/ngram |
| Word2Vec mean pooling | GaussianNB, LinearSVC | Dense static embedding baseline |
| `text_segmented` | PhoBERT base v2 fine-tune | Encoder tiếng Việt |
| Raw review content | XLM-RoBERTa Large fine-tune | Encoder đa ngôn ngữ dung lượng lớn |

XLM-R Large là đối chứng hiện đại, không tuyên bố SOTA khi chưa benchmark trên gold labels và cùng thiết lập. Transformer checkpoints không commit vào Git.

## 7. Deliverables báo cáo

- `notebooks/01_sentiment_analysis_report.ipynb`: narrative, EDA, integrity checks, runner toggles, model leaderboard, per-class assessment, error analysis, limitations.
- `docs/report_latex/`: báo cáo XeLaTeX nhiều chương, trang bìa/phân công dành cho nhóm 4, bảng kết quả thật và checklist biểu đồ theo yêu cầu bài tập.
- `outputs/metrics/*.json`: metrics và bootstrap interval.
- `outputs/predictions/*.csv`: dự đoán từng review để phân tích lỗi.
- `figures/cm_*.png`, `figures/macro_f1_val_test.png`: biểu đồ sinh từ chính dataset này.
- `outputs/annotation/annotation_sheet.csv`: mẫu nhãn độc lập (sinh bởi script, không có sẵn cho tới khi chạy).

Tài liệu tham khảo: `docs/Bài Tập Lớn I - Lazada.pdf` dùng để bám khung trình bày; các biểu đồ tại `figures/reference/` đến từ repo khác cùng chủ đề và chỉ phục vụ tham khảo phong cách.
