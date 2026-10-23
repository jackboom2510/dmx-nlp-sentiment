# SKILLS.md — Danh mục Kỹ năng & Quy trình Tác nghiệp cho Agent (`dmx-nlp-sentiment`)

Tài liệu này định nghĩa **7 Kỹ năng chuẩn hóa (Agent Skills)** tương ứng với toàn bộ vòng đời thực hiện dự án Phân loại cảm xúc bình luận sản phẩm Điện Máy Xanh (5 bước bắt buộc + bổ sung Ý 4 mô hình hiện đại + xuất biểu đồ & viết báo cáo theo mẫu PTIT).

Khi người dùng yêu cầu thực hiện một phần việc cụ thể, Agent hãy tra cứu đúng **Skill ID** dưới đây và tuân thủ chặt chẽ phần **Đầu vào (Inputs)**, **Quy trình thực hiện (Procedure)**, **Đầu ra (Outputs)** và **Tiêu chí nghiệm thu (Verification)**.

---

## Danh mục Tóm tắt các Skills

| Skill ID | Tên kỹ năng | Bước tương ứng trong đề bài | Đầu ra chính |
| :--- | :--- | :--- | :--- |
| **`skill:01-crawl-dmx`** | Thu thập & Chuyển đổi dữ liệu Điện Máy Xanh | **Bước 1**: Thu thập dữ liệu | `data/raw/reviews.jsonl`, `data/raw/reviews.csv` |
| **`skill:02-label-and-split`** | Gán nhãn (Silver/Gold) & Chia tập dữ liệu | **Bước 1 & Bước 4**: Gán nhãn, Chia Train/Val/Test | `data/processed/train.csv`, `val.csv`, `test_gold.csv` |
| **`skill:03-preprocess-vi`** | Tiền xử lý văn bản tiếng Việt chuyên sâu | **Bước 2**: Tiền xử lý dữ liệu | Cột `text_segmented` (PhoBERT) & `text_clean` (ML) |
| **`skill:04-extract-features`** | Trích chọn đặc trưng (BoW, TF-IDF, Word2Vec) | **Bước 3**: Trích chọn đặc trưng (3 phương pháp) | Ma trận đặc trưng BoW, TF-IDF, Word2Vec |
| **`skill:05-train-classical`** | Huấn luyện Naïve Bayes & SVM + Xử lý mất cân bằng | **Bước 4A**: Xây dựng & huấn luyện NB, SVM | 6 mô hình ML truyền thống + 1 Baseline |
| **`skill:06-train-phobert`** | Xây dựng & Fine-tune mô hình hiện đại PhoBERT | **Bước 4B (Bổ sung Ý 4)**: Mô hình hiện đại | Checkpoint & dự đoán của `vinai/phobert-base-v2` |
| **`skill:07-eval-plot-report`** | Đánh giá chỉ số, Xuất biểu đồ & Viết báo cáo PTIT | **Bước 5**: Đánh giá, so sánh & Viết báo cáo | Biểu đồ mới trong `figures/` & Báo cáo hoàn chỉnh |

---

## Chi tiết từng Kỹ năng (Skill Specifications)

---

### 1. `skill:01-crawl-dmx` — Thu thập & Chuyển đổi dữ liệu Điện Máy Xanh

- **Mục tiêu**: Thu thập bình luận thực tế kèm số sao từ `dienmayxanh.com` một cách an toàn, tuân thủ `robots.txt` và chuyển đổi sang định dạng bảng CSV.
- **Đầu vào (Inputs)**:
  - Script có sẵn: `crawl_dmx_reviews.py` (đơn luồng), `protected_multi_thread_crawler.py` (đa luồng an toàn), `json_to_csv.py`.
  - Danh mục mục tiêu (`TARGET_CATEGORIES`): `may-giat`, `tu-lanh`, `may-lanh`, `tivi`, `noi-com-dien`, `may-loc-nuoc`.
- **Quy trình thực hiện (Procedure)**:
  1. Kiểm tra dữ liệu hiện có trong `data/raw/reviews.csv` (hiện đã có **12,189 bình luận**). Chỉ chạy crawl mới khi người dùng yêu cầu thu thập thêm.
  2. Nếu cần kiểm tra CSS selector của trang đánh giá mới:
     ```bash
     python3 crawl_dmx_reviews.py --inspect <url-san-pham>/danh-gia
     ```
  3. Nếu chạy thu thập bổ sung, ưu tiên dùng `protected_multi_thread_crawler.py --workers 2` (nghỉ `2–4s` giữa mỗi thao tác, tự động lưu trạng thái vào `data/raw/crawled_products.json` và dừng ngay nếu gặp `403/429/captcha`).
  4. Chạy `python3 json_to_csv.py` để đồng bộ từ `data/raw/reviews.jsonl` sang `data/raw/reviews.csv`.
- **Đầu ra (Outputs)**:
  - `data/raw/reviews.jsonl` và `data/raw/reviews.csv`.
- **Tiêu chí nghiệm thu (Verification)**:
  - Đảm bảo loại bỏ phần phản hồi của quản trị viên/nhân viên Điện Máy Xanh (`.reply`, `.cmt-reply`, `.qtv`) khỏi nội dung bình luận của khách hàng.

---

### 2. `skill:02-label-and-split` — Làm sạch sơ bộ, Gán nhãn & Chia tập dữ liệu

- **Mục tiêu**: Khử trùng lặp, ánh xạ số sao sang 3 nhãn cảm xúc (`-1`, `0`, `1`), trích xuất tập kiểm tra `test_gold` để hiệu chỉnh thủ công và chia tập `Train` / `Val` / `Test` không rò rỉ dữ liệu.
- **Đầu vào (Inputs)**:
  - `data/raw/reviews.csv` (12,189 dòng).
- **Quy trình thực hiện (Procedure)**:
  1. **Loại bỏ dòng lỗi & trùng lặp**:
     - Xóa các dòng có `content` rỗng hoặc chỉ chứa khoảng trắng/ký tự đặc biệt.
     - Khử trùng lặp theo cặp `(content.strip().lower(), rating)` để tránh việc cùng một nội dung xuất hiện ở cả tập Train và tập Test.
     - Lọc ngôn ngữ bằng `langdetect` (chỉ giữ lại các bình luận tiếng Việt `vi` như báo cáo mẫu Lazada).
  2. **Gán nhãn theo số sao (Silver Label – `label_star`)**:
     - `rating in {4, 5}` $\rightarrow$ `1` (**Positive**)
     - `rating == 3` $\rightarrow$ `0` (**Neutral**)
     - `rating in {1, 2}` $\rightarrow$ `-1` (**Negative**)
  3. **Tạo tập Kiểm tra hiệu chỉnh thủ công (`test_gold.csv`, $n = 500$)**:
     - Lấy mẫu phân tầng (hoặc ưu tiên lấy đủ đại diện cho cả 3 lớp cảm xúc và 6 ngành hàng) gồm **500 bình luận**.
     - Tạo cột `label_gold` (khởi tạo bằng `label_star`), sau đó kiểm tra và hiệu chỉnh các câu bị **lệch giữa số sao và nội dung** (ví dụ: chấm 5 sao nhưng chê máy kêu to, tốn điện $\rightarrow$ sửa `label_gold = -1`; chấm 1 sao nhưng khen dùng rất tốt $\rightarrow$ sửa `label_gold = 1`).
  4. **Chia tập `train.csv` và `val.csv`**:
     - Phần dữ liệu còn lại (sau khi tách riêng 500 mẫu `test_gold`) được chia theo tỷ lệ **85% Train / 15% Validation** có phân tầng (`stratify=df["label_star"]`, `random_state=42`).
- **Đầu ra (Outputs)**:
  - `data/processed/train.csv`
  - `data/processed/val.csv`
  - `data/processed/test_gold.csv`
- **Tiêu chí nghiệm thu (Verification)**:
  - Kiểm tra giao (intersection) của `review_id` và `content` giữa `train`, `val`, `test_gold` phải bằng `0`.

---

### 3. `skill:03-preprocess-vi` — Tiền xử lý Văn bản Tiếng Việt Chuyên sâu

- **Mục tiêu**: Chuẩn hóa văn bản bình luận TMĐT tiếng Việt và tạo 2 phiên bản văn bản đầu vào: một bản phục vụ PhoBERT và một bản phục vụ BoW / TF-IDF / Word2Vec.
- **Đầu vào (Inputs)**:
  - Các tập dữ liệu trong `data/processed/` (`train.csv`, `val.csv`, `test_gold.csv`).
- **Quy trình thực hiện (Procedure)**:
  1. **Chuẩn hóa bảng mã & ký tự**:
     - Đưa toàn bộ chuỗi về chuẩn Unicode dựng sẵn `unicodedata.normalize("NFC", text)`.
     - Chuyển chữ thường (`lowercase`), xóa URL, thẻ HTML, số điện thoại.
     - Rút gọn ký tự lặp kéo dài liên tiếp (`re.sub(r"(.)\1{2,}", r"\1", text)`, ví dụ: `"tốtttt"` $\rightarrow$ `"tốt"`).
  2. **Chuẩn hóa từ viết tắt / Teencode TMĐT**:
     - Thay thế các từ viết tắt phổ biến trên Điện Máy Xanh (`đmx` $\rightarrow$ `điện máy xanh`, `sp` $\rightarrow$ `sản phẩm`, `nv` $\rightarrow$ `nhân viên`, `ko`/`k`/`kh`/`hok` $\rightarrow$ `không`, `đc`/`dc` $\rightarrow$ `được`, `bh` $\rightarrow$ `bảo hành`, `sd` $\rightarrow$ `sử dụng`, `ok`/`oke` $\rightarrow$ `tốt`).
  3. **Tách từ tiếng Việt (Word Segmentation)**:
     - Dùng `pyvi.ViTokenizer.tokenize(text)` hoặc `underthesea.word_tokenize(text, format="text")` để ghép các từ nhiều âm tiết bằng dấu gạch dưới `_` (`máy_lạnh`, `tiết_kiệm`, `nhiệt_tình`).
     - Lưu kết quả này vào cột **`text_segmented`** (giữ nguyên dấu câu cơ bản và stop words để dùng cho **PhoBERT**).
  4. **Loại bỏ dấu câu, chữ số & Stop words (có bảo toàn từ phủ định)**:
     - Từ `text_segmented`, loại bỏ dấu câu, số và lọc bỏ các hư từ tiếng Việt (`thì`, `là`, `mà`, `và`, `của`, `những`, `các`, ...).
     - **BẮT BUỘC GIỮ LẠI** nhóm từ phủ định và mức độ cảm xúc: `{"không", "chưa", "chẳng", "chả", "đừng", "rất", "quá", "hơi", "lắm", "khá", "tốt", "tệ", "kém", "ổn", "lỗi", "hỏng", "ồn", "êm", "nhưng", "tuy_nhiên", "bị"}`.
     - Lưu kết quả vào cột **`text_clean`** (dùng cho **BoW, TF-IDF, Word2Vec**).
- **Đầu ra (Outputs)**:
  - Các file `train.csv`, `val.csv`, `test_gold.csv` được bổ sung 2 cột `text_segmented` và `text_clean`.

---

### 4. `skill:04-extract-features` — Trích chọn Đặc trưng (BoW, TF-IDF, Word Embeddings)

- **Mục tiêu**: Biến đổi văn bản `text_clean` thành vector đặc trưng số học bằng cả **3 phương pháp** theo đúng yêu cầu Bước 3.
- **Đầu vào (Inputs)**:
  - Cột `text_clean` của `train`, `val`, `test_gold`.
- **Quy trình thực hiện (Procedure)**:
  1. **Phương pháp 1 – Bag of Words (BoW)**:
     - Khởi tạo `CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=15000)`.
     - Gọi `.fit_transform(train["text_clean"])`, sau đó `.transform()` trên `val` và `test_gold`.
  2. **Phương pháp 2 – TF-IDF**:
     - Khởi tạo `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95, max_features=15000, norm="l2")`.
     - Gọi `.fit_transform(train["text_clean"])`, sau đó `.transform()` trên `val` và `test_gold`.
  3. **Phương pháp 3 – Word Embeddings (Word2Vec)**:
     - Huấn luyện `gensim.models.Word2Vec` (`vector_size=100`, `window=5`, `min_count=2`, `sg=1` Skip-gram, `epochs=15`, `seed=42`) **chỉ trên các câu của tập Train**.
     - Biểu diễn mỗi bình luận bằng vector trung bình cộng (**Mean Pooling**) của các từ xuất hiện trong `model.wv` (nếu câu rỗng hoặc toàn từ OOV thì trả về vector `np.zeros(100)`).
- **Đầu ra (Outputs)**:
  - 3 bộ ma trận đặc trưng `(X_train, X_val, X_test)` cho `BoW`, `TF-IDF`, và `Word2Vec`.

---

### 5. `skill:05-train-classical` — Huấn luyện Mô hình Học máy Truyền thống (Naïve Bayes & SVM)

- **Mục tiêu**: Huấn luyện 2 thuật toán **Naïve Bayes** và **SVM** trên cả 3 bộ đặc trưng (tạo thành 6 tổ hợp mô hình chính + 1 mô hình Baseline + 1 biến thể đối chứng không bỏ stopword).
- **Đầu vào (Inputs)**:
  - Các ma trận đặc trưng `BoW`, `TF-IDF`, `Word2Vec` và nhãn `y_train`, `y_val`, `y_test_gold`.
- **Quy trình thực hiện (Procedure)**:
  1. **Mô hình Baseline (`DummyClassifier`)**:
     - `DummyClassifier(strategy="most_frequent")` dự đoán toàn bộ mẫu thuộc lớp đa số (`Positive = 1`) để làm mốc so sánh tối thiểu.
  2. **Xử lý mất cân bằng dữ liệu cho Naïve Bayes (Hybrid Resampling)**:
     - Vì lớp `Positive` chiếm ~75%, áp dụng kết hợp `RandomUnderSampler` (giảm bớt lớp `1`) và `RandomOverSampler` (tăng số lượng mẫu lớp `-1` và `0`) trên tập `Train` (tương tự báo cáo mẫu Lazada trang 6).
     - Với `BoW` và `TF-IDF`: Huấn luyện `MultinomialNB(alpha=1.0)` (tối ưu `alpha` trong $\{0.1, 0.5, 1.0\}$ trên tập `Val`).
     - Với `Word2Vec`: Huấn luyện `GaussianNB()` (do vector Word2Vec có giá trị thực âm).
  3. **Xử lý mất cân bằng dữ liệu cho SVM (`class_weight="balanced"`)**:
     - Sử dụng `LinearSVC(class_weight="balanced", C=1.0, random_state=42, max_iter=5000)` cho cả 3 bộ đặc trưng `BoW`, `TF-IDF`, `Word2Vec` (tối ưu siêu tham số `C` trong $\{0.1, 0.5, 1.0, 2.0\}$ theo `Macro-F1` trên tập `Val`).
- **Đầu ra (Outputs)**:
  - Dự đoán `y_val_pred` và `y_test_pred` của tất cả các mô hình:
    1. `Baseline (lớp đông nhất)`
    2. `BoW + NB`
    3. `BoW + SVM`
    4. `TF-IDF + NB`
    5. `TF-IDF + SVM`
    6. `W2V + NB`
    7. `W2V + SVM`
    8. *(Tham khảo)* `TF-IDF + NB (không bỏ stopword)`

---

### 6. `skill:06-train-phobert` — Xây dựng & Huấn luyện Mô hình Hiện đại PhoBERT (Bổ sung Ý 4)

- **Mục tiêu**: Xây dựng mô hình Học sâu hiện đại dựa trên kiến trúc Transformer tiền huấn luyện cho tiếng Việt (**`vinai/phobert-base-v2`**) để giải quyết yêu cầu **Bổ sung Ý 4**.
- **Đầu vào (Inputs)**:
  - Cột `text_segmented` (đã tách từ bằng `pyvi`/`underthesea`, **không** bỏ stopword) của `train`, `val`, `test_gold`.
- **Quy trình thực hiện (Procedure)**:
  1. Ánh xạ nhãn từ $\{-1, 0, 1\}$ sang chỉ số $\{0, 1, 2\}$ (`0: Negative`, `1: Neutral`, `2: Positive`).
  2. Tokenize văn bản bằng `AutoTokenizer.from_pretrained("vinai/phobert-base-v2")` với `max_length=128`, `truncation=True`, `padding="max_length"`.
  3. Khởi tạo mô hình `AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=3)`.
  4. **Xử lý mất cân bằng lớp trong Deep Learning**:
     - Tính vector trọng số lớp `compute_class_weight("balanced", classes=[0, 1, 2], y=y_train_mapped)`.
     - Tuỳ chỉnh `WeightedTrainer` ghi đè hàm `compute_loss` sử dụng `nn.CrossEntropyLoss(weight=class_weights_tensor)`.
  5. Huấn luyện với `AdamW` (`learning_rate=2e-5`, `batch_size=16`, `epochs=3–5`, `warmup_ratio=0.1`, `weight_decay=0.01`, `metric_for_best_model="macro_f1"`).
  6. *(Phương án dự phòng trên máy chỉ có CPU yếu)*: Nếu môi trường không có GPU để fine-tune toàn mạng 135M tham số, có thể đóng băng (freeze) 10 tầng đầu của PhoBERT và chỉ fine-tune 2 tầng Transformer cuối + Classification Head, hoặc chạy notebook huấn luyện trên Google Colab / Kaggle T4 GPU rồi lưu file dự đoán `phobert_predictions.csv` về repo.
- **Đầu ra (Outputs)**:
  - Dự đoán `y_val_pred` và `y_test_pred` của mô hình `PhoBERT`.

---

### 7. `skill:07-eval-plot-report` — Đánh giá, Sinh Biểu đồ Mới & Viết Báo cáo theo Mẫu PTIT

- **Mục tiêu**: Tính toán toàn bộ chỉ số đánh giá, vẽ lại bộ biểu đồ thực tế của dữ liệu Điện Máy Xanh (thay thế cho các biểu đồ tham khảo trong `figures/`), và tổng hợp báo cáo theo đúng mẫu `docs/Bài Tập Lớn I - Lazada.pdf`.
- **Đầu vào (Inputs)**:
  - Kết quả dự đoán của toàn bộ mô hình ở `skill:05-train-classical` và `skill:06-train-phobert`.
  - Hàm vẽ biểu đồ chuẩn trong `INSTRUCTIONS.md` (Mục 5.1).
- **Quy trình thực hiện (Procedure)**:
  1. **Tính toán các chỉ số bắt buộc**:
     - `Accuracy`
     - `Macro-Precision`, `Macro-Recall`, `Macro-F1`
     - `Weighted-F1`
     - `Precision`, `Recall`, `F1-Score` riêng cho từng lớp (`Negative`, `Neutral`, `Positive`).
  2. **Xuất bộ biểu đồ mới của dữ liệu Điện Máy Xanh vào `figures/`**:
     - Sử dụng hàm `plot_dual_confusion_matrix()` để xuất các file Confusion Matrix 2 bảng (*Số đếm* + *Chuẩn hóa theo hàng Recall %*):
       - `figures/cm_baseline_lop_dong_nhat.png`
       - `figures/cm_bow_nb.png`, `figures/cm_bow_svm.png`
       - `figures/cm_tf_idf_nb.png`, `figures/cm_tf_idf_svm.png`
       - `figures/cm_w2v_nb.png`, `figures/cm_w2v_svm.png`
       - `figures/cm_phobert.png`
     - Sử dụng hàm `plot_macro_f1_comparison()` để xuất biểu đồ tổng hợp `figures/macro_f1_val_test.png`.
  3. **Soạn thảo Báo cáo theo đúng bố cục `docs/Bài Tập Lớn I - Lazada.pdf`**:
     - **Trang bìa + Mục lục + Bảng đóng góp thành viên (4 thành viên × 25%)**.
     - **I. Đề bài**: Bài toán phân loại cảm xúc bình luận Điện Máy Xanh (3 nhãn `-1, 0, 1`), 5 bước thực hiện + Bổ sung Ý 4 (PhoBERT).
     - **II. Cơ sở lý thuyết**: BoW, TF-IDF, Word Embeddings (Word2Vec), Naïve Bayes, SVM, Mô hình hiện đại PhoBERT (Ý 4), Xử lý mất cân bằng dữ liệu, Thư viện hỗ trợ.
     - **III. Thực nghiệm và kết quả**: Link code thực nghiệm, thống kê dữ liệu Điện Máy Xanh, hình ảnh Confusion Matrix, Bảng so sánh tổng hợp & Nhận xét đa chiều (giữa các phương pháp trích chọn đặc trưng, giữa NB và SVM, và sự vượt trội của PhoBERT).
     - **IV. Kết luận**: Kết quả đạt được, Hạn chế (nhiễu nhãn sao, lớp Neutral ít mẫu) & Hướng cải thiện.
- **Đầu ra (Outputs)**:
  - Bộ biểu đồ thực tế mới trong `figures/*.png`, bảng tổng hợp kết quả CSV/Markdown và tài liệu Báo cáo hoàn chỉnh.
