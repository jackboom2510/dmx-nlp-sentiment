# dmx-nlp-sentiment
**Hệ thống Phân loại Cảm xúc Bình luận Sản phẩm trên Điện Máy Xanh (Vietnamese Product Review Sentiment Analysis)**

Dự án thuộc học phần **Phân tích và Khai phá Dữ liệu Văn bản** (Học viện Công nghệ Bưu chính Viễn thông – PTIT), thực hiện bài toán phân loại cảm xúc bình luận sản phẩm tiếng Việt theo 3 nhãn:
- **Tích cực (`Positive`, quy ước `1`)**: Bình luận khen chất lượng sản phẩm, dịch vụ giao hàng/lắp đặt tốt (thường tương ứng **4–5 sao**).
- **Trung tính (`Neutral`, quy ước `0`)**: Bình luận nhận xét trung lập, hỏi đáp hoặc vừa khen vừa góp ý nhẹ (thường tương ứng **3 sao**).
- **Tiêu cực (`Negative`, quy ước `-1`)**: Bình luận phàn nàn về lỗi kỹ thuật, độ ồn, hao điện, bảo hành hoặc dịch vụ kém (thường tương ứng **1–2 sao**).

---

## Yêu cầu thực hiện dự án

Dự án được triển khai theo 5 bước chuẩn cùng yêu cầu mở rộng ở Bước 4 (tham khảo bố cục trình bày của báo cáo mẫu tại [`docs/Bài Tập Lớn I - Lazada.pdf`](docs/Bài%20Tập%20Lớn%20I%20-%20Lazada.pdf)):

1. **Bước 1 – Thu thập dữ liệu**: Tìm kiếm, thu thập bộ dữ liệu bình luận sản phẩm từ Điện Máy Xanh và thực hiện gán nhãn (kết hợp nhãn sao tự động và hiệu chỉnh thủ công).
2. **Bước 2 – Tiền xử lý dữ liệu**: Áp dụng các kỹ thuật xử lý ngôn ngữ tự nhiên tiếng Việt (chuẩn hóa Unicode, xử lý từ viết tắt/teencode, tách từ ghép tiếng Việt, loại bỏ stop words có bảo toàn từ phủ định, lọc ngôn ngữ).
3. **Bước 3 – Trích chọn đặc trưng**: Thử nghiệm với các phương pháp biểu diễn văn bản:
   - **Bag of Words (BoW)**
   - **TF-IDF (Term Frequency – Inverse Document Frequency)**
   - **Word Embeddings (Word2Vec / FastText)**
4. **Bước 4 – Xây dựng và huấn luyện mô hình**:
   - Chia dữ liệu thành tập huấn luyện (`Train`), tập kiểm định (`Validation`) và tập kiểm tra (`Test`).
   - Sử dụng 2 thuật toán phân loại học máy truyền thống: **Naïve Bayes** và **Support Vector Machine (SVM)** kết hợp xử lý mất cân bằng dữ liệu.
   - **Bổ sung Ý 4 (Mô hình hiện đại)**: Xây dựng và huấn luyện mô hình học sâu hiện đại dựa trên kiến trúc Transformer cho tiếng Việt (**PhoBERT** – `vinai/phobert-base-v2`) để giải quyết bài toán.
5. **Bước 5 – Đánh giá mô hình**:
   - Sử dụng các chỉ số: **Accuracy, Precision, Recall, F1-Score (Macro-F1, Weighted-F1), Confusion Matrix**.
   - So sánh hiệu suất giữa các phương pháp trích chọn đặc trưng, các thuật toán học máy truyền thống và mô hình hiện đại.

---

## Tài liệu Kế hoạch & Hướng dẫn thực hiện

- **[`PLAN.md`](PLAN.md)**: Kế hoạch tổng thể thực hiện dự án, đánh giá hiện trạng kho mã nguồn, quy hoạch kiến trúc thư mục, thiết kế thực nghiệm cho từng bước và bảng phân công công việc nhóm.
- **[`INSTRUCTIONS.md`](INSTRUCTIONS.md)**: Hướng dẫn thực thi chi tiết từng bước kèm mã nguồn mẫu (tiền xử lý tiếng Việt, trích chọn đặc trưng BoW/TF-IDF/Word2Vec, huấn luyện Naïve Bayes/SVM/PhoBERT, hàm vẽ Confusion Matrix & biểu đồ so sánh) và dàn ý viết báo cáo bám sát mẫu PTIT.

---

## Cấu trúc thư mục hiện tại & Lưu ý quan trọng

```text
dmx-nlp-sentiment/
├── data/
│   ├── raw/                              # Dữ liệu thực tế đã crawl từ Điện Máy Xanh
│   │   ├── reviews.jsonl                 # 12,189 bình luận gốc dạng JSON Lines
│   │   ├── reviews.csv                   # 12,189 bình luận đã chuyển sang CSV
│   │   ├── reviews_drop_na.csv           # Bản rút gọn các trường chính
│   │   ├── crawled_products.json         # Trạng thái các sản phẩm đã crawl (hỗ trợ resume)
│   │   ├── product_urls.json             # Danh sách URL sản phẩm theo ngành hàng
│   │   └── sitemap_report.json           # Báo cáo khảo sát sitemap Điện Máy Xanh
│   └── debug/                            # Ảnh chụp & HTML phục vụ kiểm tra CSS selector khi crawl
├── docs/
│   └── Bài Tập Lớn I - Lazada.pdf        # Mẫu báo cáo Bài tập lớn I (tham khảo cấu trúc trình bày)
├── figures/                              # ⚠️ LƯU Ý: Thư mục hiện chứa các biểu đồ THAM KHẢO từ một repo khác cùng chủ đề
│   ├── cm_*.png                          # Mẫu trình bày Confusion Matrix (Số đếm + Chuẩn hóa theo hàng)
│   └── macro_f1_val_test.png             # Mẫu trình bày biểu đồ so sánh Macro-F1 giữa các mô hình
├── crawl_dmx_reviews.py                  # Script thu thập bình luận Điện Máy Xanh (đơn luồng)
├── multi_thread_crawler.py               # Script thu thập bình luận (đa luồng)
├── protected_multi_thread_crawler.py     # Script thu thập đa luồng an toàn (tự dừng khi gặp 403/429/captcha)
├── json_to_csv.py                        # Script chuyển đổi dữ liệu từ JSONL sang CSV
├── PLAN.md                               # Kế hoạch thực hiện dự án
├── INSTRUCTIONS.md                       # Hướng dẫn chi tiết cách code, chạy thực nghiệm & viết báo cáo
└── README.md                             # Tài liệu giới thiệu dự án (file này)
```

> ### ⚠️ Lưu ý quan trọng về thư mục `figures/` và `docs/`
> 1. **Về `figures/`**: Các file ảnh hiện có trong thư mục `figures/` (`cm_bow_nb.png`, `cm_tf_idf_svm.png`, `cm_phobert.png`, `macro_f1_val_test.png`, ...) **chỉ là các biểu đồ mẫu tham khảo từ một repository khác có cùng chủ đề** (dùng để tham khảo quy chuẩn trực quan hóa: ma trận nhầm lẫn gồm 2 bảng *Số đếm* & *Chuẩn hóa theo hàng*, cùng biểu đồ thanh ngang so sánh *Macro-F1*). Khi chạy thực nghiệm trên bộ dữ liệu Điện Máy Xanh (`data/raw/reviews.csv`), nhóm cần **chạy lại toàn bộ pipeline đánh giá để sinh ra bộ biểu đồ mới của riêng dữ liệu Điện Máy Xanh** (xem code vẽ biểu đồ chuẩn trong `INSTRUCTIONS.md`).
> 2. **Về `docs/Bài Tập Lớn I - Lazada.pdf`**: Đây là báo cáo mẫu của đề tài Lazada dùng để tham khảo bố cục trình bày báo cáo (Mục lục, Bảng đóng góp thành viên, Cơ sở lý thuyết, Thực nghiệm & Kết quả, Kết luận).

---

## Thống kê bộ dữ liệu Điện Máy Xanh đã thu thập (`data/raw/reviews.csv`)

Bộ dữ liệu thực tế của dự án gồm **12,189 bình luận** (`9,716` bình luận duy nhất theo cặp nội dung – số sao) thuộc **6 ngành hàng**:

| Ngành hàng | Slug danh mục | Số lượng bình luận | Tỷ lệ (%) |
| :--- | :--- | ---: | ---: |
| **Tủ lạnh** | `tu-lanh` | 3,366 | 27.62% |
| **Nồi cơm điện** | `noi-com-dien` | 2,796 | 22.94% |
| **Máy lạnh** | `may-lanh` | 2,030 | 16.65% |
| **Máy giặt** | `may-giat` | 1,725 | 14.15% |
| **Máy lọc nước** | `may-loc-nuoc` | 1,724 | 14.14% |
| **Tivi** | `tivi` | 548 | 4.50% |
| **Tổng cộng** | | **12,189** | **100.00%** |

**Phân bố theo số sao và nhãn cảm xúc sơ bộ**:
- **Tích cực (`1`) – 4 & 5 sao**: `6,741` (5★) + `2,407` (4★) = **9,148 mẫu (75.05%)**
- **Trung tính (`0`) – 3 sao**: **988 mẫu (8.11%)**
- **Tiêu cực (`-1`) – 1 & 2 sao**: `706` (2★) + `1,347` (1★) = **2,053 mẫu (16.84%)**

---

## Hướng dẫn bắt đầu nhanh (Quickstart)

### 1. Cài đặt môi trường
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install numpy pandas scikit-learn imbalanced-learn gensim
pip install underthesea pyvi langdetect matplotlib seaborn tqdm
pip install torch transformers accelerate datasets
pip install requests lxml playwright
```

### 2. Thu thập dữ liệu (Nếu cần crawl bổ sung)
```bash
# Chạy thử nghiệm trên 3 sản phẩm
python crawl_dmx_reviews.py --test 3

# Chạy crawler đa luồng có bảo vệ chống chặn IP và tự động resume
python protected_multi_thread_crawler.py --workers 2

# Chuyển đổi file reviews.jsonl sang reviews.csv
python json_to_csv.py
```

### 3. Thực hiện Tiền xử lý, Huấn luyện & Đánh giá mô hình
Xem hướng dẫn chi tiết từng bước và mã nguồn đầy đủ tại **[`INSTRUCTIONS.md`](INSTRUCTIONS.md)** và lộ trình phân công tại **[`PLAN.md`](PLAN.md)**.
