# AGENTS.md — Hướng dẫn dành cho AI Coding Agents (`dmx-nlp-sentiment`)

Tài liệu này định nghĩa bối cảnh dự án, quy tắc bất biến (invariants), kiến trúc dữ liệu và tiêu chuẩn kỹ thuật dành cho mọi AI Agent làm việc trên kho mã nguồn **`dmx-nlp-sentiment`**.

---

## 1. Bối cảnh & Mục tiêu Dự án

- **Tên dự án**: Hệ thống Phân loại Cảm xúc Bình luận Sản phẩm trên Điện Máy Xanh (`dmx-nlp-sentiment`).
- **Môn học**: Phân tích và Khai phá Dữ liệu Văn bản – Học viện Công nghệ Bưu chính Viễn thông (PTIT).
- **Bài toán**: Phân loại văn bản tiếng Việt thành **3 lớp cảm xúc**:
  - `1` (**Positive / Tích cực**): Bình luận khen sản phẩm/dịch vụ (ánh xạ sơ bộ từ `4–5 sao`).
  - `0` (**Neutral / Trung tính**): Bình luận trung lập, hỏi đáp, vừa khen vừa góp ý nhẹ (ánh xạ sơ bộ từ `3 sao`).
  - `-1` (**Negative / Tiêu cực**): Bình luận phàn nàn chất lượng, lỗi, độ ồn, bảo hành kém (ánh xạ sơ bộ từ `1–2 sao`).
- **5 Bước bắt buộc + Mở rộng Ý 4**:
  1. **Thu thập dữ liệu**: Crawl bình luận từ Điện Máy Xanh (`12,189` mẫu thuộc 6 ngành hàng trong `data/raw/`).
  2. **Tiền xử lý dữ liệu**: Chuẩn hóa Unicode NFC, xử lý teencode, tách từ tiếng Việt (`pyvi`/`underthesea`), loại bỏ stop words có bảo toàn từ phủ định.
  3. **Trích chọn đặc trưng**: Thực nghiệm cả 3 phương pháp: **Bag of Words (BoW)**, **TF-IDF**, và **Word Embeddings (Word2Vec)**.
  4. **Xây dựng & huấn luyện mô hình**:
     - Học máy truyền thống: **Naïve Bayes** (`MultinomialNB`, `GaussianNB`) và **SVM** (`LinearSVC`).
     - **Bổ sung Ý 4 (Mô hình hiện đại)**: Fine-tune mô hình ngôn ngữ tiền huấn luyện dựa trên Transformer cho tiếng Việt — **PhoBERT (`vinai/phobert-base-v2`)**.
  5. **Đánh giá & so sánh**: Tính toán **Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Confusion Matrix** và viết báo cáo theo mẫu `docs/Bài Tập Lớn I - Lazada.pdf`.

---

## 2. Quy tắc Bất biến (Critical Guardrails & Invariants)

Mọi Agent khi đọc, sửa hoặc sinh mã nguồn/báo cáo trong repository này **BẮT BUỘC** tuân thủ 5 quy tắc sau:

### Quy tắc 1: Bảo toàn dữ liệu gốc trong `data/raw/`
- Các file `data/raw/reviews.jsonl`, `data/raw/reviews.csv`, `data/raw/reviews_drop_na.csv` chứa **12,189 bình luận thực tế của Điện Máy Xanh** đã tốn nhiều thời gian crawl.
- **KHÔNG BAO GIỜ** ghi đè, xóa hoặc làm hỏng các file trong `data/raw/`.
- Mọi dữ liệu đã làm sạch, tách từ hoặc chia tập (`train.csv`, `val.csv`, `test_gold.csv`) phải được lưu vào thư mục `data/processed/`.

### Quy tắc 2: Hiểu đúng bản chất thư mục `figures/`
- ⚠️ **CẢNH BÁO QUAN TRỌNG**: Các file ảnh hiện có sẵn trong thư mục `figures/` (`cm_baseline_lop_dong_nhat.png`, `cm_bow_nb.png`, `cm_bow_svm.png`, `cm_tf_idf_nb.png`, `cm_tf_idf_svm.png`, `cm_w2v_nb.png`, `cm_w2v_svm.png`, `cm_phobert.png`, `macro_f1_val_test.png`, ...) **CHỈ LÀ CÁC BIỂU ĐỒ MẪU THAM KHẢO TỪ MỘT REPOSITORY KHÁC CÙNG CHỦ ĐỀ**.
- Agent **không được** coi các con số trong những ảnh hiện tại ở `figures/` là kết quả thực nghiệm cuối cùng của bộ dữ liệu Điện Máy Xanh.
- Thay vào đó, Agent sử dụng các ảnh này làm **quy chuẩn thiết kế hình ảnh (Visual Specification)**:
  1. Mỗi Confusion Matrix gồm **2 biểu đồ nhiệt (heatmap) đặt cạnh nhau**: bên trái là *Số đếm (Counts)*, bên phải là *Chuẩn hóa theo hàng (Recall %)*.
  2. Biểu đồ tổng hợp `macro_f1_val_test.png` là biểu đồ thanh ngang kép so sánh *Macro-F1* giữa tập `Val (nhãn sao)` và tập `Test (nhãn gold)`.
- Khi chạy thực nghiệm trên tập dữ liệu Điện Máy Xanh, Agent cần xuất biểu đồ mới theo đúng định dạng này để cập nhật vào `figures/`.

### Quy tắc 3: Chống rò rỉ dữ liệu (Zero Data Leakage)
- Tập kiểm tra `test_gold` (đề xuất $n = 500$ mẫu được rà soát và hiệu chỉnh nhãn thủ công) và tập `val` phải được tách riêng **TRƯỚC KHI**:
  - `fit` bộ trích chọn đặc trưng (`CountVectorizer`, `TfidfVectorizer`, `Word2Vec`).
  - Thực hiện lấy mẫu lại dữ liệu mất cân bằng (`RandomOverSampler`, `RandomUnderSampler`, `SMOTE`).
- Tuyệt đối chỉ gọi `.fit()` hoặc `.fit_resample()` trên tập **Train**; tập **Val** và **Test** chỉ được gọi `.transform()`.

### Quy tắc 4: Quy tắc Tiền xử lý Tiếng Việt (Negation & PhoBERT Input)
- **Khi lọc Stop words cho BoW / TF-IDF / Word2Vec**: Bắt buộc **GIỮ LẠI** các từ phủ định, từ chỉ mức độ và liên từ chuyển ý (`không`, `chưa`, `chẳng`, `chả`, `đừng`, `rất`, `quá`, `hơi`, `lắm`, `tốt`, `tệ`, `kém`, `ổn`, `lỗi`, `hỏng`, `ồn`, `êm`, `nhưng`, `tuy_nhiên`, `bị`). Việc xóa từ `"không"` sẽ biến `"không tốt"` thành `"tốt"` và làm sai lệch hoàn toàn nhãn cảm xúc.
- **Khi chuẩn bị đầu vào cho PhoBERT (`vinai/phobert-base-v2`)**:
  - Văn bản đầu vào **BẮT BUỘC phải được tách từ ghép tiếng Việt** bằng dấu gạch dưới `_` (sử dụng `pyvi.ViTokenizer.tokenize` hoặc `underthesea.word_tokenize(..., format="text")`), vì từ điển BPE của PhoBERT được huấn luyện trên văn bản đã tách từ.
  - **KHÔNG xóa stop words** đối với đầu vào của PhoBERT để giữ nguyên cấu trúc ngữ pháp cho cơ chế Self-Attention.

### Quy tắc 5: Tương thích thuật toán Naïve Bayes với từng loại đặc trưng
- Với đặc trưng **BoW** và **TF-IDF** (ma trận thưa, giá trị $\ge 0$): sử dụng `MultinomialNB` (hoặc `ComplementNB`).
- Với đặc trưng **Word2Vec** (vector dày đặc số thực có cả giá trị âm $< 0$): `MultinomialNB` sẽ báo lỗi `ValueError: Negative values in data passed to MultinomialNB`. Agent phải sử dụng `GaussianNB()` hoặc chuẩn hóa qua `MinMaxScaler(feature_range=(0, 1))` trước khi đưa vào `MultinomialNB`.

---

## 3. Bản đồ Tài liệu trong Repository

| File | Vai trò | Khi nào Agent cần đọc |
| :--- | :--- | :--- |
| `README.md` | Tổng quan dự án, cấu trúc thư mục, thống kê dữ liệu Điện Máy Xanh | Khi bắt đầu phiên làm việc mới |
| `PLAN.md` | Kế hoạch tổng thể 5 bước + Mở rộng Ý 4, thiết kế thực nghiệm, phân công nhóm | Khi lập kế hoạch hoặc tổ chức kiến trúc module |
| `INSTRUCTIONS.md` | Hướng dẫn code chi tiết từng bước, hàm vẽ biểu đồ và dàn ý viết báo cáo | Khi viết code thực nghiệm hoặc soạn thảo báo cáo |
| `SKILLS.md` | Danh mục các kỹ năng (Skills/Workflows) chuẩn hóa cho Agent | Khi thực thi một tác vụ cụ thể (crawl, preprocess, train, plot, report) |
| `docs/Bài Tập Lớn I - Lazada.pdf` | Báo cáo mẫu của PTIT (đề tài Lazada) | Khi đối chiếu bố cục, văn phong và trình bày bảng biểu báo cáo |

---

## 4. Lệnh thực thi chuẩn (Standard Commands)

```bash
# 1. Kiểm tra thống kê dữ liệu thô Điện Máy Xanh
python3 -c "
import csv
from collections import Counter
with open('data/raw/reviews.csv', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
print('Total:', len(rows), 'Ratings:', Counter(r['rating'] for r in rows))
"

# 2. Chuyển đổi JSONL sang CSV (nếu có dữ liệu crawl mới)
python3 json_to_csv.py

# 3. Chạy crawler thử nghiệm (3 sản phẩm)
python3 crawl_dmx_reviews.py --test 3
```

---

## 5. Checklist Kiểm tra Chất lượng trước khi Hoàn thành Tác vụ (Definition of Done)

- [ ] Code chạy không lỗi syntax/import, có cố định `random_state=42` / `seed=42` để tái lập kết quả.
- [ ] Đã kiểm tra không có hiện tượng rò rỉ dữ liệu (Data Leakage) từ tập `val`/`test_gold` sang tập `train`.
- [ ] Đã đánh giá đầy đủ 5 nhóm chỉ số: `Accuracy`, `Precision`, `Recall`, `F1-Score (Macro & Weighted)`, và `Confusion Matrix`.
- [ ] Biểu đồ xuất ra có tiêu đề tiếng Việt rõ ràng, font chữ không lỗi Unicode, lưu đúng tên file trong `figures/`.
- [ ] Không commit các file checkpoint mô hình nặng (`*.bin`, `*.pt`, `*.safetensors`, `results_phobert/`) vào Git (giữ `.gitignore` sạch sẽ).
