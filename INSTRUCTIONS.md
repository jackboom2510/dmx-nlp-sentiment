# Hướng dẫn triển khai và viết báo cáo

Tài liệu hướng dẫn thực thi pipeline phân tích cảm xúc bình luận Điện Máy Xanh và phát triển nội dung báo cáo theo bố cục học thuật của `docs/Bài Tập Lớn I - Lazada.pdf`. Notebook chi tiết, có mã phân tích và biểu đồ, đặt tại `notebooks/01_sentiment_analysis_report.ipynb`.

> **Cảnh báo về nhãn:** các split hiện tại dùng `label_target` sinh bằng hybrid rating + lexicon, bao gồm cả `data/processed/test.csv`. Đây là tập test giữ lại theo pseudo-label, không phải tập gold thủ công. Không mô tả điểm số từ đó như accuracy trên chân trị. Trước báo cáo kết luận, nên tạo nhãn độc lập bằng quy trình tại Mục 7.

## 1. Cài đặt & cấu hình

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Cấu hình đường dẫn, seed, tham số feature/model trong `configs/config.yaml`. Transformers tải model từ Hugging Face; cần internet lúc tải lần đầu. XLM-R Large nên chạy GPU đủ bộ nhớ.

## 2. Bước 1 — Thu thập và tạo nhãn

Dữ liệu crawl hiện nằm ở `data/raw/reviews.csv` / JSONL. Các crawler ở `scripts/crawl/`; tuân thủ điều khoản trang web, tốc độ hợp lý, và chỉ crawl dữ liệu công khai. Không ghi đè dữ liệu gốc khi tái xử lý.

```bash
python auto_label_review.py
# tùy chọn
python auto_label_review.py --mode hybrid-lexicon --test-size 500 --val-ratio 0.15 --seed 42
```

Quy trình khử rác/trùng lặp, chuẩn hóa tiếng Việt, tạo nhãn `label_star`, chấm lexicon (`label_unsup` / `unsup_score`), reconciliate các xung đột rating-text thành `label_target`, rồi chia train/val/test có phân tầng. Tệp sinh ra:

- `data/interim/reviews_labeled.csv`
- `data/interim/rating_text_conflicts.csv`
- `data/interim/labeling_summary.json`
- `data/processed/train.csv`, `val.csv`, `test.csv`

Các chế độ `embedding-prototype`, `zero-shot`, `pretrained` là phương án gán nhãn/teacher phụ thuộc model bên ngoài. Ghi lại model, phiên bản, seed, và chế độ khi chạy; không trộn các pseudo-label từ các teacher khác nhau mà không lưu nguồn nhãn.

### Kiểm tra nhãn trước huấn luyện

Đọc `labeling_summary.json`, xem phân bố lớp, lấy mẫu rating-text conflicts và kiểm tra các trường hợp đa khía cạnh / mỉa mai / phủ định. Ghi rõ chính sách sửa nhãn. Bộ gán nhãn lexicon là weak supervision; số rating không phải ground truth hoàn hảo và nội dung ngắn có thể không đủ tín hiệu.

## 3. Bước 2 — Tiền xử lý tiếng Việt

Module `src/data/preprocess.py` thực hiện:
1. Unicode NFC, lowercase, xóa URL/HTML và chuẩn hóa lặp ký tự.
2. Sửa một số teencode/viết tắt và cụm từ không dấu phổ biến.
3. Word segmentation (`pyvi` / `underthesea`, fallback từ điển miền).
4. Tạo hai phiên bản: `text_segmented` cho PhoBERT (giữ stopwords), `text_clean` cho đặc trưng cổ điển (lọc stopwords).
5. Bảo toàn phủ định, cường độ và từ cảm xúc trong stopword handling.

Báo cáo nên có bảng ví dụ **raw → segmented → clean**, giải thích vì sao không loại “không/chưa/nhưng”, và lưu ý quy tắc heuristic có thể gây mất nghĩa. Không fit phép biến đổi học từ dữ liệu (vocabulary/IDF/embedding) ngoài train.

## 4. Bước 3 — Trích chọn đặc trưng

`src/features/build_features.py` cung cấp:

- **BoW:** `CountVectorizer`, unigram/bigram.
- **TF-IDF:** unigram/bigram, sublinear TF.
- **Word2Vec:** Skip-gram train trên train split, mean pooling thành vector câu.

Mô tả số chiều vocabulary / min_df / max_df thực tế khi chạy, thời gian tính và ưu/nhược điểm. BoW/TF-IDF giữ tín hiệu từ khóa hiếm tốt nhưng không hiểu ngữ cảnh; Word2Vec giảm thưa nhưng mean pooling làm mất thứ tự và phạm vi phủ định.

## 5. Bước 4 — Huấn luyện và so sánh mô hình

### 5.1 ML cổ điển

```bash
python scripts/train_classical.py --bootstrap 2000
```

Đối chiếu majority baseline, MultinomialNB (BoW/TF-IDF), GaussianNB (Word2Vec), LinearSVC (cả ba feature). NB dùng hybrid resampling; SVM dùng `class_weight=balanced`. Ghi seed và tham số. Mọi mô hình nhận cùng train / validation / test, cùng `label_target`.

### 5.2 Transformer hiện đại

PhoBERT:

```bash
python scripts/train_transformer.py --model-name vinai/phobert-base-v2 --epochs 5 --batch-size 8
```

XLM-RoBERTa Large — đối chứng đa ngôn ngữ dung lượng cao:

```bash
python scripts/train_transformer.py --model-name FacebookAI/xlm-roberta-large \\
  --epochs 5 --batch-size 2 --gradient-accumulation-steps 8 --max-length 160
```

Script tự chọn `text_segmented` cho PhoBERT và `content` cho model khác ở chế độ `--input-field auto`; có thể chỉ định field. Weighted cross-entropy lấy class weights từ train; validation Macro-F1 chọn checkpoint và early stopping; test được chấm sau cùng. Ghi cấu hình GPU, thời gian, checkpoint, max length và peak VRAM nếu có. XLM-R Large là baseline mạnh/hiện đại, không gắn nhãn “SOTA” nếu không có so sánh theo benchmark được công nhận và cùng nhãn đánh giá.

### 5.3 Artifact

- JSON: `outputs/metrics/`
- Dự đoán từng dòng (kèm confidence với Transformer): `outputs/predictions/`
- Confusion matrices / so sánh Macro-F1: `figures/`
- Checkpoints: `outputs/models/` (không bắt buộc đưa checkpoint lớn vào Git)

## 6. Bước 5 — Đánh giá chi tiết

Notebook tạo bảng và biểu đồ khi có artifacts. Chương kết quả nên bao gồm:

1. **Bảng leaderboard**: Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1; validation Macro-F1 để mô tả lựa chọn và test Macro-F1 kèm bootstrap CI 95%.
2. **Bảng từng lớp**: Precision/Recall/F1/Support của Negative, Neutral, Positive. Phân tích rõ lớp Neutral yếu hơn như thế nào và rủi ro bỏ sót Negative.
3. **Confusion matrix**: số đếm và tỷ lệ recall theo hàng; giải thích các cặp nhầm lẫn lớn nhất bằng ví dụ.
4. **So sánh có kiểm soát**: BoW vs TF-IDF, NB vs SVM, Word2Vec vs n-grams, PhoBERT vs XLM-R. Cùng split, cùng target, không lựa mô hình dựa vào test.
5. **Phân tích lỗi**: xem các câu sai dự đoán theo category, độ dài, rating, conflict type; phân loại nguyên nhân thành phủ định, tương phản, đa khía cạnh, từ lóng, câu ngắn/không rõ, lỗi nhãn.
6. **Độ bất định**: bootstrap CI Macro-F1; CI trên pseudo-label không khắc phục annotation bias.

Chỉ số chính nên là Macro-F1 do lệch lớp. Accuracy có thể cao vì lớp Positive phổ biến. Ghi rõ hỗ trợ (support) để người đọc thấy độ tin cậy của lớp nhỏ. Không so sánh kết quả hiện tại với hình `figures/reference/`: hình đó đến từ repository khác, chỉ tham khảo trực quan.

## 7. Tạo tập nhãn người gán (khuyến nghị bắt buộc trước kết luận mạnh)

```bash
python scripts/export_annotation_sample.py --n 300 --seed 42
```

Script mặc định xuất mẫu phân tầng đã ẩn pseudo-label vào `outputs/annotation/annotation_sheet.csv`; `--sampling random` lấy mẫu theo prevalence nếu cần ước lượng chỉ số tổng thể. Hai người gán độc lập, chỉ xem nội dung (ẩn cả rating để tránh anchoring), dùng `-1=Negative, 0=Neutral, 1=Positive`; bất đồng được adjudicate. Mẫu phân tầng hữu ích để tăng support lớp hiếm nhưng không nên diễn giải Accuracy trực tiếp như ước lượng prevalence nếu không hiệu chỉnh trọng số. Lưu protocol và ví dụ ranh giới giữa Neutral/Positive. Sau đó:

- Tính Cohen’s kappa và confusion giữa hai annotator.
- Chấm model trên `adjudicated_label`; tách hẳn kết quả đánh giá human-audited khỏi kết quả pseudo-label.
- Nếu dùng tập này để điều chỉnh model, cần tạo một test audit khác hoặc không tuyên bố test độc lập.

Notebook tự tìm file đã hoàn thiện để tính kappa và classification report. Có thể mở rộng CLI để tính metric trực tiếp với model predictions.

## 8. Chạy notebook và viết báo cáo

```bash
jupyter lab notebooks/01_sentiment_analysis_report.ipynb
```

Notebook có phần narrative theo logic báo cáo: giới thiệu/mục tiêu → dữ liệu → phương pháp → thực nghiệm → đánh giá → error analysis → hạn chế/kết luận. Mặc định lệnh train tắt; bật cờ ở cell “Chạy thực nghiệm”. Export HTML/PDF từ Jupyter sau khi chạy toàn bộ cell. Thay đoạn kết luận bằng số liệu thật, không để lại placeholders.

Bản báo cáo nguồn LaTeX cho **nhóm 4 thành viên** nằm tại `docs/report_latex/` (các chương tách thành file riêng). Điền tên/MSSV/giảng viên/tỷ lệ đóng góp trước khi nộp; biên dịch XeLaTeX theo `docs/report_latex/README.md`. Cấu trúc có tham khảo cách chia chương và trình bày của báo cáo Clustering do người dùng cung cấp, nhưng toàn bộ thuật toán và biểu đồ được thiết kế lại cho phân loại cảm xúc.

### Dàn ý báo cáo đề xuất

1. **Mở đầu:** bài toán, động lực, mục tiêu, câu hỏi nghiên cứu và phạm vi.
2. **Cơ sở lý thuyết:** sentiment tiếng Việt, BoW/TF-IDF/Word2Vec, NB/SVM, Transformer; giải thích lựa chọn PhoBERT và XLM-R.
3. **Dữ liệu & tiền xử lý:** nguồn crawl, tiêu chí lọc, label protocol, thống kê lớp/ngành/rating, ví dụ chuyển đổi văn bản.
4. **Thiết kế hệ thống & thực nghiệm:** sơ đồ pipeline, split, chống leakage, hyperparameters, phần cứng và reproducibility.
5. **Kết quả:** bảng mô hình, per-class, confusion matrices, CI và thời gian/tài nguyên.
6. **Thảo luận:** trả lời từng RQ, lỗi điển hình, sai lệch nhãn, so sánh model và hạn chế.
7. **Kết luận & hướng phát triển:** annotation thật, group split theo sản phẩm, calibrate confidence, aspect-based sentiment / model distillation.
8. **Phụ lục:** cấu hình, ví dụ lỗi, chi tiết gán nhãn, đóng góp thành viên.

## 9. Diễn giải giới hạn

- Test hiện tại không phải gold. Đây là hạn chế quan trọng nhất; mọi kết quả cần được gọi là pseudo-label agreement.
- Split ngẫu nhiên có thể để cùng product_code ở cả train và test. Nếu bài toán triển khai là sản phẩm mới, đánh giá bổ sung group split.
- Lexicon có thể encode thiên lệch và domain assumptions; kiểm tra mẫu thủ công, đừng coi conflict adjustment là ground truth.
- XLM-R Large tốn bộ nhớ và thời gian. Dùng cùng protocol; mô tả compute thay vì chỉ nêu điểm.
- Khác biệt điểm nhỏ cần được diễn giải cẩn trọng, đặc biệt khi khoảng CI chồng lấp.
