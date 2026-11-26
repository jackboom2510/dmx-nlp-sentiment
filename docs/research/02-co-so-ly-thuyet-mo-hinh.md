# Research 02 — Cơ sở lý thuyết các mô hình (v1.6, mục 2.3–2.6)

Nguồn sơ cấp: paper gốc (BERT, PhoBERT), tài liệu scikit-learn, gensim; đối chiếu code `src/features.py`, `src/train_classic.py`, `src/evaluate.py` của repo nguồn @ `0184366`.

## 2.3.1 Túi từ và n-gram

- **Báo cáo:** BoW đếm theo từ điển học trên train; bigram giữ một phần thứ tự ("không lạnh", "tốt nhưng"); min_df cắt đặc trưng hiếm.
- **Nguồn:** `src/features.py` (CountVectorizer), lưới chọn tại `results/metrics/val_grid.csv` (n-gram (1,2) thắng ở cả hai mô hình BoW).
- **Kết luận:** Khớp lý thuyết chuẩn và khớp thực nghiệm của chính dự án.

## 2.3.2 TF-IDF và độ tương tự cosine

- **Báo cáo:** tfidf = tf × idf, idf làm mịn dạng ln((1+N)/(1+df)) + 1; tf logarit; chuẩn hóa L2; cosine trên véc-tơ đã chuẩn hóa bằng tích vô hướng, nằm trong [0, 1] với TF-IDF.
- **Nguồn:** `src/features.py` (TfidfVectorizer, sublinear); cosine dùng trong `src/clean_label.py` để khử gần trùng (ngưỡng 0,9).
- **Kết luận:** Đúng chuẩn. **Điều kiện còn thiếu:** công thức tf = 1 + ln f chưa nêu trường hợp f = 0 (tf phải bằng 0); khoảng [0, 1] chỉ đúng vì TF-IDF không âm — báo cáo đã gắn ngữ cảnh này, chấp nhận được.

## 2.3.3 Word2Vec Skip-gram

- **Báo cáo:** CBOW vs Skip-gram; véc-tơ tĩnh, mỗi từ một véc-tơ; dự án dùng Skip-gram 200 chiều, cửa sổ 5, min_count 2, 20 epoch.
- **Nguồn:** `src/features.py` — `Word2Vec(sents, vector_size=200, window=5, min_count=2, epochs=20, sg=1)` (kèm hàm băm xác định để tái lập).
- **Kết luận:** Khớp tham số. Ở mức lý thuyết, bản v1.6 (như v1.3) trình bày định tính, không có hàm mục tiêu softmax/negative sampling — đủ cho báo cáo ứng dụng, thiếu nếu hội đồng hỏi sâu cơ chế huấn luyện.

## 2.3.4 Véc-tơ câu trung bình có trọng số TF-IDF

- **Báo cáo:** s(D) = trung bình véc-tơ từ theo trọng số TF-IDF (tf thô, từ đơn); từ ngoài từ điển bị bỏ qua; toàn bộ ngoài từ điển → véc-tơ 0.
- **Nguồn:** `src/features.py` (`TfidfWeightedW2V`).
- **Kết luận:** Khớp cài đặt. Đây cũng là mắt xích giải thích kết quả thấp của Word2Vec (mất phủ định khi trung bình) ở mục 3.6.2.

## 2.4.1 Naive Bayes

- **Báo cáo:** định lý Bayes + giả thiết độc lập; ước lượng có làm mịn α (α = 1 là Laplace); tiên nghiệm theo dữ liệu hoặc bằng nhau; Gaussian NB cho véc-tơ liên tục (mỗi chiều Gauss theo lớp).
- **Nguồn:** scikit-learn MultinomialNB/GaussianNB; `val_grid.csv` (α và tiên nghiệm được chọn theo val).
- **Kết luận:** Khớp lý thuyết chuẩn và tài liệu thư viện.

## 2.4.2 Máy hỗ trợ véc-tơ (SVM)

- **Báo cáo:** siêu phẳng lề cực đại, lề mềm qua C; trọng số lớp tỉ lệ nghịch số mẫu; LinearSVC (one-vs-rest) cho BoW/TF-IDF, SVC nhân RBF (one-vs-one) cho Word2Vec.
- **Nguồn:** `src/train_classic.py`; tài liệu scikit-learn (LinearSVC dùng OvR, SVC dùng OvO — khớp).
- **Kết luận:** Đúng ở mức mô tả. **Thiếu:** công thức hàm nhân RBF và vai trò của γ (báo cáo chỉ ghi "γ tự động theo phương sai dữ liệu" ở Bảng 3.5) — điểm yếu lý thuyết còn lại, đã có đoạn soạn sẵn tại `BO_SUNG_BAO_CAO_BAN_CUOI.md` mục II.2.

## 2.5 PhoBERT

- **Báo cáo:** mô hình ngôn ngữ tiếng Việt của VinAI, kiến trúc RoBERTa (12 tầng, 768 chiều ẩn, 12 đầu, ~135 triệu tham số), tiền huấn luyện bằng nhiệm vụ che từ; làm việc ở mức từ nên tách bằng RDRSegmenter rồi mã hóa BPE; biểu diễn [CLS] cho phân loại; fine-tune toàn bộ với tốc độ học nhỏ.
- **Nguồn:** paper PhoBERT (Nguyen & Nguyen, 2020) và model card `vinai/phobert-base-v2`; chi tiết vận hành tại `docs/research-phobert-finetuning-pipeline.md`.
- **Kết luận:** Các thông số kiến trúc và quy trình khớp nguồn chính chủ. **Thiếu:** công thức attention và vai trò √dₖ (chỉ mô tả định tính cơ chế Q/K/V); đoạn soạn sẵn tại file bổ sung mục II.3.

## 2.6 Đánh giá mô hình

- **Báo cáo:** ma trận nhầm lẫn K×K; Precision/Recall/F1 theo lớp; Accuracy; macro-F1; lập luận chọn macro-F1 vì lệch lớp; ba tập train/val/test với vai trò tách bạch, test dùng một lần.
- **Nguồn:** `src/evaluate.py` tính đúng các chỉ số này trên file predictions; ADR-0002 (repo đích) ghi lại quyết định macro-F1.
- **Kết luận:** Khớp. Công thức Cohen's kappa không nằm ở 2.6 mà được dùng tại 3.3.2 (κ = 0,798, khớp `gold_agreement.json`: 0,7975).
