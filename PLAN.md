# KẾ HOẠCH THỰC HIỆN DỰ ÁN (PROJECT PLAN)
## Đề tài: Xây dựng hệ thống Phân loại Cảm xúc Bình luận Sản phẩm trên Điện Máy Xanh

> **Tài liệu tham chiếu chuẩn**: `docs/Bài Tập Lớn I - Lazada.pdf` (Báo cáo Bài tập lớn I – Bộ môn Phân tích và Khai phá Dữ liệu Văn bản – Học viện Công nghệ Bưu chính Viễn thông PTIT).

---

## 1. Tổng quan bài toán & Mục tiêu dự án

- **Bài toán**: Phân loại cảm xúc (Sentiment Analysis) các bình luận đánh giá sản phẩm tiếng Việt trên sàn thương mại điện tử **Điện Máy Xanh** (`dienmayxanh.com`).
- **Không gian nhãn (3 lớp)**:
  - **Tích cực (`Positive`, quy ước `1`)**: Bình luận khen chất lượng sản phẩm, dịch vụ giao hàng/lắp đặt tốt (thường tương ứng **4–5 sao**).
  - **Trung tính (`Neutral`, quy ước `0`)**: Bình luận mang tính hỏi đáp, nhận xét trung lập, vừa khen vừa chê nhẹ hoặc chưa đánh giá rõ sau khi mới mua (thường tương ứng **3 sao**).
  - **Tiêu cực (`Negative`, quy ước `-1`)**: Bình luận phàn nàn về lỗi sản phẩm, độ ồn, hao điện, bảo hành hoặc thái độ phục vụ kém (thường tương ứng **1–2 sao**).
- **Mục tiêu kỹ thuật**:
  1. Thu thập và chuẩn hóa bộ dữ liệu bình luận thực tế từ Điện Máy Xanh, kết hợp gán nhãn tự động theo số sao (Silver Labels) cho tập huấn luyện và hiệu chỉnh thủ công (Gold Labels) cho tập kiểm tra để khắc phục nhiễu *"lệch sao – nội dung"*.
  2. Xây dựng pipeline tiền xử lý văn bản tiếng Việt chuẩn (chuẩn hóa Unicode, tách từ ghép tiếng Việt, xử lý từ phủ định và loại bỏ hư từ/stop words).
  3. Thực nghiệm đầy đủ cả **3 phương pháp trích chọn đặc trưng**: **Bag of Words (BoW)**, **TF-IDF**, và **Word Embeddings (Word2Vec)**.
  4. Xây dựng và huấn luyện **2 thuật toán học máy truyền thống**: **Naïve Bayes (NB)** và **Support Vector Machine (SVM)** có tích hợp kỹ thuật xử lý mất cân bằng dữ liệu (Class Imbalance).
  5. **Mở rộng (Bổ sung ý 4)**: Xây dựng và huấn luyện **mô hình học sâu hiện đại dựa trên kiến trúc Transformer – PhoBERT (`vinai/phobert-base-v2`)** chuyên biệt cho tiếng Việt để giải quyết triệt để hạn chế về ngữ cảnh và từ phủ định của các mô hình truyền thống.
  6. Đánh giá và so sánh toàn diện bằng các chỉ số **Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Confusion Matrix** và hoàn thiện báo cáo theo đúng mẫu PTIT.

---

## 2. Đánh giá hiện trạng Kho mã nguồn (`dmx-nlp-sentiment`)

Qua khảo sát hiện trạng repository, dự án đã có sẵn các tài nguyên quan trọng sau:

| Hạng mục | Trạng thái hiện tại trong Repo | Ghi chú / Công việc cần làm tiếp |
| :--- | :--- | :--- |
| **1. Thu thập dữ liệu (Crawler)** | **Đã có sẵn**: `crawl_dmx_reviews.py`, `multi_thread_crawler.py`, `protected_multi_thread_crawler.py`, `json_to_csv.py` | Đã thu thập thành công **12,189 bình luận** thuộc 6 ngành hàng lớn (`data/raw/reviews.jsonl`, `data/raw/reviews.csv`, `data/raw/reviews_drop_na.csv`). |
| **2. Dữ liệu thô (`data/raw/`)** | **Đã có sẵn**: 12,189 mẫu (5★: 6,741; 4★: 2,407; 3★: 988; 2★: 706; 1★: 1,347) | Cần chuẩn hóa script làm sạch trùng lặp, tách tập `train`/`val` (nhãn sao) và lưu bộ `test_gold` (`n = 500` mẫu đã hiệu chỉnh thủ công) vào `data/processed/`. |
| **3. Tiền xử lý & Trích chọn đặc trưng** | **Chưa có code trong repo** | Cần xây dựng module/notebook (`src/preprocess.py`, `src/features.py` hoặc notebook tổng hợp) để xử lý 12,189 bình luận Điện Máy Xanh. |
| **4. Huấn luyện NB, SVM & PhoBERT** | **Chưa có code trong repo** | Cần xây dựng code huấn luyện (`src/train_ml.py`, `src/train_phobert.py` hoặc Jupyter Notebook) cho cả 2 thuật toán truyền thống (NB, SVM) và mô hình hiện đại (PhoBERT). |
| **5. Biểu đồ trong `figures/`** | **Hiện là 10 biểu đồ THAM KHẢO từ một repo khác cùng chủ đề** (`cm_*.png`, `macro_f1_val_test.png`) | Dùng làm **mẫu chuẩn định dạng trực quan hóa** (Confusion Matrix 2 panel Số đếm + Recall % theo hàng, và biểu đồ so sánh Macro-F1). Cần chạy thực nghiệm trên tập Điện Máy Xanh để **xuất và ghi đè bộ biểu đồ thực tế mới** vào `figures/`. |
| **6. Báo cáo Bài tập lớn** | Đã có file mẫu tham khảo `docs/Bài Tập Lớn I - Lazada.pdf` | Cần viết báo cáo chính thức cho bài toán Điện Máy Xanh bám sát cấu trúc mẫu PTIT và bổ sung phần mô hình hiện đại PhoBERT. |

---

## 3. Kiến trúc thư mục chuẩn hóa của dự án

Để dự án sạch sẽ, dễ chấm điểm và dễ tái lập thực nghiệm, cấu trúc thư mục được quy hoạch như sau:

```text
dmx-nlp-sentiment/
├── data/
│   ├── raw/                          # Dữ liệu thô crawl từ Điện Máy Xanh (12,189 reviews)
│   │   ├── reviews.jsonl
│   │   ├── reviews.csv
│   │   └── reviews_drop_na.csv
│   ├── processed/                    # Dữ liệu sau tiền xử lý & chia tập (cần tạo/lưu)
│   │   ├── train.csv                 # Tập huấn luyện (nhãn sao đã làm sạch)
│   │   ├── val.csv                   # Tập validation (nhãn sao)
│   │   └── test_gold.csv             # Tập kiểm tra (n = 500, gán nhãn thủ công Gold)
│   └── stopwords_vi_custom.txt       # Danh sách stopword tiếng Việt (đã giữ lại từ phủ định)
├── docs/
│   ├── Bài Tập Lớn I - Lazada.pdf    # Báo cáo mẫu tham khảo
│   └── Bao_Cao_BTL1_DMX.docx / .pdf  # Báo cáo hoàn chỉnh của nhóm
├── figures/                          # ⚠️ Hiện đang chứa 10 biểu đồ THAM KHẢO từ một repo khác cùng chủ đề;
│   │                                 # sẽ được ghi đè bằng biểu đồ thực tế của tập dữ liệu Điện Máy Xanh
│   ├── cm_baseline_lop_dong_nhat.png
│   ├── cm_bow_nb.png
│   ├── cm_bow_svm.png
│   ├── cm_tf_idf_nb.png
│   ├── cm_tf_idf_nb_khong_bo_stopword_tham_khao.png
│   ├── cm_tf_idf_svm.png
│   ├── cm_w2v_nb.png
│   ├── cm_w2v_svm.png
│   ├── cm_phobert.png
│   └── macro_f1_val_test.png
├── notebooks/                        # Notebook thực nghiệm (chạy trên Local / Colab / Kaggle)
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_ml_bow_tfidf_w2v_nb_svm.ipynb
│   └── 03_modern_model_phobert.ipynb
├── crawl_dmx_reviews.py              # Script crawl đơn luồng
├── protected_multi_thread_crawler.py # Script crawl đa luồng an toàn
├── json_to_csv.py                    # Script chuyển đổi JSONL -> CSV
├── PLAN.md                           # Kế hoạch thực hiện dự án (file này)
└── INSTRUCTIONS.md                   # Hướng dẫn thực thi chi tiết & khung viết báo cáo
```

---

## 4. Lộ trình thực hiện chi tiết theo 5 Bước + Mở rộng Ý 4

### Bước 1: Thu thập dữ liệu & Gán nhãn (Data Collection & Labeling)
1. **Nguồn dữ liệu**: Sàn TMĐT Điện Máy Xanh (`https://www.dienmayxanh.com`), tuân thủ `robots.txt`, giới hạn tốc độ tải (`DELAY = 2.0–4.0s`).
2. **Phạm vi**: 6 ngành hàng chính:
   - *Tủ lạnh* (3,366 mẫu), *Nồi cơm điện* (2,796 mẫu), *Máy lạnh* (2,030 mẫu), *Máy giặt* (1,725 mẫu), *Máy lọc nước* (1,724 mẫu), *Tivi* (548 mẫu).
   - Tổng cộng: **12,189 bình luận** (sau khi loại bỏ dòng trống và trùng lặp nội dung còn khoảng **9,716 bình luận duy nhất**).
3. **Quy tắc ánh xạ nhãn ban đầu theo số sao (Silver Label)**:
   - `4 sao, 5 sao` $\rightarrow$ **Positive (`1`)** (~75.05% tập dữ liệu thô – chiếm đa số áp đảo).
   - `3 sao` $\rightarrow$ **Neutral (`0`)** (~8.11% tập dữ liệu thô).
   - `1 sao, 2 sao` $\rightarrow$ **Negative (`-1`)** (~16.84% tập dữ liệu thô).
4. **Hiệu chỉnh thủ công tập Kiểm tra (`test_gold`, đề xuất $n = 500$)**:
   - Trên thực tế, nhiều khách hàng đánh giá 5 sao nhưng bình luận phàn nàn (*"mới mua đã kêu to"*), hoặc đánh giá 1–3 sao nhưng nội dung khen (*"dùng tốt, nhân viên nhiệt tình"*).
   - Do đó, tham khảo cách thiết kế thực nghiệm từ repo mẫu cùng chủ đề, nhóm trích riêng **500 mẫu** từ dữ liệu Điện Máy Xanh làm tập **Gold Test** được rà soát và gán nhãn thủ công theo đúng ngữ nghĩa thực tế của câu bình luận để đánh giá mô hình một cách khách quan nhất.

---

### Bước 2: Tiền xử lý dữ liệu văn bản tiếng Việt (Data Preprocessing)
Áp dụng chuỗi kỹ thuật tiền xử lý đặc thù cho bình luận TMĐT tiếng Việt:
1. **Lọc rác & Nhận diện ngôn ngữ**:
   - Loại bỏ các dòng có `content` rỗng, bình luận chỉ có ký tự đặc biệt/số, hoặc bình luận trùng lặp.
   - Sử dụng `langdetect` hoặc bộ lọc ký tự tiếng Việt để loại bỏ các bình luận rác ngoại ngữ gây nhiễu từ điển.
2. **Chuẩn hóa văn bản (Text Normalization)**:
   - Chuẩn hóa bảng mã Unicode dựng sẵn (**NFC**) để tránh lỗi hai từ giống hệt nhau nhưng khác mã tổ hợp/dựng sẵn.
   - Chuyển toàn bộ chữ hoa thành chữ thường (lowercasing).
   - Chuẩn hóa từ viết tắt, teencode phổ biến trên TMĐT (`"đmx"` $\rightarrow$ `"điện máy xanh"`, `"sp"` $\rightarrow$ `"sản phẩm"`, `"nv"` $\rightarrow$ `"nhân viên"`, `"ko"`/`"k"`/`"kh"` $\rightarrow$ `"không"`, `"ok"`/`"oke"` $\rightarrow$ `"tốt"`, `"đc"` $\rightarrow$ `"được"`, ...).
   - Kéo giãn các ký tự lặp (`"tốtttt"` $\rightarrow$ `"tốt"`, `"ồn quáaaa"` $\rightarrow$ `"ồn quá"`).
   - Loại bỏ URL, thẻ HTML, số điện thoại, dấu câu thừa và khoảng trắng thừa.
3. **Tách từ tiếng Việt (Vietnamese Word Segmentation)**:
   - Sử dụng thư viện `underthesea` (`word_tokenize(..., format="text")`) hoặc `pyvi` (`ViTokenizer.tokenize`) để nối các âm tiết của từ ghép tiếng Việt bằng dấu gạch dưới `_` (ví dụ: `"máy_giặt"`, `"nhân_viên"`, `"tiết_kiệm"`, `"bảo_hành"`).
4. **Loại bỏ hư từ (Stop words removal) có bảo toàn từ phủ định**:
   - Sử dụng bộ `vietnamese-stopwords`, **NHƯNG bắt buộc giữ lại** các từ mang sắc thái phủ định/cảm xúc như: `không`, `chưa`, `chẳng`, `chả`, `kém`, `tệ`, `tốt`, `rất`, `quá`, `hơi`, `nhưng`, `tuy_nhiên`, `bị`, `lỗi`.
   - *(Thực nghiệm đối chứng/Ablation study)*: So sánh thêm biến thể giữ nguyên stopword (`cm_tf_idf_nb_khong_bo_stopword_tham_khao.png`) để chứng minh vai trò của việc thiết kế danh sách stopword hợp lý.

---

### Bước 3: Trích chọn đặc trưng (Feature Extraction)
Thực nghiệm đầy đủ cả **3 phương pháp** để có bảng so sánh toàn diện như mẫu báo cáo Lazada:

1. **Phương pháp 1 – Bag of Words (BoW)**:
   - Sử dụng `CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=15000)`.
   - Kết hợp cả unigram và bigram (ví dụ `"không_tốt"`, `"chưa_hài_lòng"`) giúp giảm một phần nhược điểm mất ngữ cảnh cục bộ của BoW.
2. **Phương pháp 2 – TF-IDF (Term Frequency – Inverse Document Frequency)**:
   - Sử dụng `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95, max_features=15000, norm='l2')`.
   - Giảm trọng số các từ xuất hiện phổ biến ở mọi bình luận (*"mua"*, *"máy"*, *"dùng"*) và làm nổi bật các từ mang tính phân biệt cảm xúc cao (*"kêu_to"*, `"rỉ_nước"`, `"êm_ái"`).
3. **Phương pháp 3 – Word Embeddings (Word2Vec)**:
   - Huấn luyện mô hình `Word2Vec` (`gensim`, `vector_size=100` hoặc `200`, `window=5`, `min_count=2`, `sg=1` Skip-gram) trên tập bình luận đã tách từ (hoặc dùng pre-trained Word2Vec tiếng Việt).
   - Biểu diễn mỗi bình luận thành một vector dày đặc (dense vector) bằng phương pháp **Mean Pooling** (hoặc **TF-IDF Weighted Pooling**) trung bình cộng các vector từ trong câu.

---

### Bước 4: Xây dựng và Huấn luyện Mô hình (Model Building & Training)

#### 4.1. Chia tập dữ liệu (Data Splitting)
- **Tập Test (`test_gold`)**: Trích xuất cố định `n = 500` mẫu từ dữ liệu Điện Máy Xanh và hiệu chỉnh nhãn thủ công (Gold labels). Tuyệt đối không dùng tập này khi `fit` bộ trích chọn đặc trưng hay huấn luyện mô hình.
- **Tập Train & Validation**: Phần dữ liệu Điện Máy Xanh còn lại sau khi loại bỏ các mẫu thuộc `test_gold` được chia theo tỷ lệ **85% Train / 15% Validation** có phân tầng (`stratify=y`).

#### 4.2. Xử lý mất cân bằng dữ liệu (Handling Class Imbalance)
Vì nhãn `Positive` chiếm ~75% trong khi `Neutral` chỉ chiếm ~8% và `Negative` chiếm ~17%:
- **Baseline đối chiếu**: Mô hình `DummyClassifier(strategy="most_frequent")` (luôn dự đoán lớp đông nhất `Positive`) để làm mốc sàn (Macro-F1 chỉ đạt `0.165`).
- **Với Naïve Bayes**: Áp dụng **Hybrid Sampling** (kết hợp Undersampling lớp đa số `Positive` và Oversampling/RandomOverSampler cho lớp thiểu số `Neutral`, `Negative`) hoặc điều chỉnh `class_prior` trên tập Train.
  - *Lưu ý kỹ thuật*: Với BoW và TF-IDF (giá trị không âm), sử dụng `MultinomialNB(alpha=1.0)`. Với Word2Vec (vector số thực có giá trị âm), sử dụng `GaussianNB()` (hoặc `MinMaxScaler` đưa về $[0, 1]$ trước khi qua `MultinomialNB`).
- **Với Support Vector Machine (SVM)**: Sử dụng `LinearSVC(class_weight="balanced", C=1.0, random_state=42)`, tự động gán trọng số phạt tỷ lệ nghịch với tần suất xuất hiện của từng lớp.

#### 4.3. Bổ sung Ý 4: Xây dựng Mô hình Hiện đại – PhoBERT (`vinai/phobert-base-v2`)
- **Lý do lựa chọn PhoBERT**:
  - Trong báo cáo mẫu Lazada (trang 5 & trang 13), nhóm tác giả chỉ dùng PhoBERT để trích xuất đặc trưng tĩnh ghép với SVM hoặc dùng trung bình embedding, nên Macro-F1 chỉ đạt `0.4407` và chính báo cáo mẫu đã nêu ở phần *Hướng cải thiện (trang 13)*: *"Cải thiện bằng cách fine-tune PhoBERT... Sử dụng mô hình mạnh hơn, đặc biệt là các mô hình Transformers"*.
  - Do đó, ở dự án này, chúng ta triển khai **Fine-tuning End-to-End mô hình `vinai/phobert-base-v2`** với tầng phân loại `AutoModelForSequenceClassification(num_labels=3)`.
- **Điểm mạnh vượt trội của PhoBERT**:
  1. **Kiến trúc Transformer hai chiều (Bidirectional Self-Attention)**: Nắm bắt toàn bộ ngữ cảnh câu, phân biệt chính xác hiện tượng đảo cực cảm xúc do từ phủ định (*"không ồn"* vs *"rất ồn"*, *"chưa thấy lỗi"* vs *"bị lỗi"*).
  2. **Tiền huấn luyện chuyên biệt cho tiếng Việt**: Được huấn luyện trên 20GB văn bản tiếng Việt đã tách từ, xử lý hiệu quả từ ghép, từ láy và từ hiếm thông qua cơ chế **BPE (Byte-Pair Encoding) subword**.
  3. **Hàm mất mát có trọng số (`Weighted Cross-Entropy Loss`)**: Tích hợp vector trọng số lớp $w_c = \frac{N}{K \cdot N_c}$ trực tiếp vào hàm loss trong quá trình fine-tune, giúp mô hình học tốt cả lớp thiểu số `Neutral` và `Negative`.

---

### Bước 5: Đánh giá và So sánh Mô hình (Evaluation & Comparison)
1. **Các chỉ số đánh giá**:
   - **Accuracy**: Độ chính xác tổng thể.
   - **Precision, Recall, F1-Score theo từng lớp** (`Negative`, `Neutral`, `Positive`).
   - **Macro-F1** *(Chỉ số quan trọng nhất cho dữ liệu mất cân bằng)*: Trung bình cộng không trọng số của F1 ba lớp, phản ánh công bằng hiệu năng trên cả lớp ít mẫu (`Neutral`).
   - **Weighted-F1**: Trung bình F1 có trọng số theo số mẫu mỗi lớp.
   - **Confusion Matrix**: Biểu diễn dưới cả 2 dạng **Số đếm (Counts)** và **Chuẩn hóa theo hàng (Row-normalized Recall %)** để phân tích sâu các cặp nhầm lẫn.
2. **Các chiều so sánh bắt buộc trong báo cáo**:
   - **Chiều 1 – Giữa 3 phương pháp trích chọn đặc trưng**: BoW vs. TF-IDF vs. Word2Vec.
   - **Chiều 2 – Giữa 2 thuật toán ML truyền thống**: Naïve Bayes vs. SVM.
   - **Chiều 3 – Giữa ML truyền thống và Mô hình hiện đại (PhoBERT)**: So sánh trên cả tập Validation (nhãn sao) và tập Gold Test (nhãn thủ công), phân tích độ lệch giữa Val và Test.
   - **Chiều 4 – Phân tích lỗi (Error Analysis)**: Tại sao lớp `Neutral` dễ bị nhầm sang `Positive` và lớp `Negative` có một phần bị dự đoán thành `Neutral`.

---

## 5. Phân công công việc đề xuất (Theo mẫu bảng đóng góp 4 thành viên của PTIT)

| STT | Thành viên | Hạng mục phụ trách chính | Đầu ra (Deliverables) | Tỷ lệ đóng góp |
| :-: | :--- | :--- | :--- | :-: |
| **1** | **Thành viên 1** | Thu thập dữ liệu (Crawler DMX), Làm sạch dữ liệu thô, Gán nhãn thủ công tập Gold Test (`n=500`) | `crawl_dmx_reviews.py`, `data/raw/reviews.csv`, `data/processed/test_gold.csv`, Mục I & III.1 Báo cáo | 25% |
| **2** | **Thành viên 2** | Tiền xử lý văn bản tiếng Việt, Trích chọn đặc trưng (BoW, TF-IDF, Word2Vec), Huấn luyện Naïve Bayes | Notebook tiền xử lý & NB, `figures/cm_bow_nb.png`, `figures/cm_tf_idf_nb.png`, `figures/cm_w2v_nb.png`, Mục II.1–II.4 Báo cáo | 25% |
| **3** | **Thành viên 3** | Xử lý mất cân bằng dữ liệu, Huấn luyện & tối ưu SVM trên 3 bộ đặc trưng | Notebook SVM, `figures/cm_bow_svm.png`, `figures/cm_tf_idf_svm.png`, `figures/cm_w2v_svm.png`, Mục II.5–II.7 Báo cáo | 25% |
| **4** | **Thành viên 4** | Xây dựng & Fine-tune mô hình hiện đại **PhoBERT** (Ý 4 bổ sung), Tổng hợp bảng/biểu đồ so sánh & Biên tập Báo cáo | Notebook PhoBERT, `figures/cm_phobert.png`, `figures/macro_f1_val_test.png`, Mục III.2, III.3, IV Báo cáo | 25% |
