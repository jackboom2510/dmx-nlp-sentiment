# Phân tích cảm xúc bình luận sản phẩm Điện Máy Xanh

Dự án xây dựng pipeline phân loại bình luận tiếng Việt thành **Negative (-1), Neutral (0), Positive (1)**, phục vụ học phần Phân tích và Khai phá Dữ liệu Văn bản (PTIT). Repo gồm thu thập dữ liệu, chuẩn hóa tiếng Việt, gán nhãn hybrid, thí nghiệm ML truyền thống, fine-tuning Transformer và notebook báo cáo có thể tái lập.

> **Tình trạng nhãn:** các nhãn hiện tại trong `data/processed/{train,val,test}.csv` là `label_target` được sinh tự động từ rating + lexicon/quy tắc. Tập `test.csv` là held-out pseudo-labeled set, **không phải gold test do người gán**. Các chỉ số hiện tại phản ánh độ khớp với nhãn quy tắc; muốn đưa ra khẳng định chất lượng thực tế cần gán nhãn độc lập mẫu kiểm tra. Dự án cung cấp `scripts/export_annotation_sample.py` để tạo mẫu mù.

## Cấu trúc chính

```text
dmx-nlp-sentiment/
├── configs/config.yaml
├── data/
│   ├── raw/                  # Bình luận gốc (không chỉnh sửa)
│   ├── external/             # Stopword tiếng Việt
│   ├── interim/              # Nhãn hybrid, xung đột rating-văn bản, thống kê
│   └── processed/            # train.csv, val.csv, test.csv (pseudo-label)
├── docs/                     # PDF tham khảo đề bài / báo cáo
├── figures/
│   └── reference/            # Hình tham khảo từ repo khác, không phải kết quả DMX
├── notebooks/
│   └── 01_sentiment_analysis_report.ipynb  # Notebook phân tích + báo cáo
├── outputs/
│   ├── metrics/              # JSON metrics
│   ├── predictions/          # Dự đoán từng mẫu
│   ├── models/               # Transformer checkpoints (sinh khi train)
│   └── annotation/           # Mẫu gán nhãn độc lập (sinh theo yêu cầu)
├── scripts/
│   ├── crawl/                # Crawler và chuyển JSONL → CSV
│   ├── auto_label_review.py
│   ├── train_classical.py
│   ├── train_transformer.py  # PhoBERT / XLM-RoBERTa
│   ├── train_phobert.py      # Wrapper tương thích ngược
│   └── export_annotation_sample.py
├── src/{data,features,models,evaluation}/
└── auto_label_review.py      # Entrypoint từ thư mục gốc
```

### Minh bạch về biểu đồ tham khảo
Các ảnh `figures/reference/cm_*.png` và `macro_f1_val_test.png` là **biểu đồ tham khảo từ một repository khác cùng chủ đề**, không phải kết quả thực nghiệm của Điện Máy Xanh. Kết quả thật được sinh vào `figures/` khi chạy các script huấn luyện.

## Cài đặt và chạy

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 1. Tạo nhãn và các tập dữ liệu

```bash
python auto_label_review.py
```

Pipeline mặc định lọc bình luận lỗi/trùng, chuẩn hóa và tách từ, chấm cảm xúc lexicon có xử lý phủ định/cường độ/liên từ chuyển ý, đối chiếu với rating và sinh stratified train/validation/test. Chế độ mở rộng `embedding-prototype`, `zero-shot`, `pretrained` có thể chạy với `--mode`; chúng cần tải checkpoint ngoài và tài nguyên bổ sung.

Mẫu thống kê lần chạy đã có trong repo: 12,189 dòng thô → 9,516 bình luận hợp lệ duy nhất; 490 xung đột sao-văn bản được đánh dấu; các tập có 7,664 / 1,352 / 500 mẫu. Đây là thống kê nhãn tự động, không phải kết quả đánh giá mô hình.

### 2. Chạy baseline ML truyền thống

```bash
python scripts/train_classical.py --bootstrap 2000
```

So sánh majority baseline, BoW/TF-IDF/Word2Vec với Multinomial/Gaussian Naïve Bayes và LinearSVC. Tất cả dùng cùng `label_target`; các vectorizer và Word2Vec fit trên train, validation dùng để theo dõi, test chỉ đánh giá sau cùng. Script xuất per-class metrics, confusion matrix, bootstrap CI Macro-F1 và file dự đoán.

### 3. Fine-tune Transformer hiện đại

PhoBERT tiếng Việt:

```bash
python scripts/train_transformer.py --model-name vinai/phobert-base-v2 --epochs 5
```

Đối chứng đa ngôn ngữ mạnh hơn về dung lượng (cần GPU VRAM lớn hơn; văn bản gốc không cần word segmentation):

```bash
python scripts/train_transformer.py \
  --model-name FacebookAI/xlm-roberta-large \
  --batch-size 2 --gradient-accumulation-steps 8 --max-length 160 --epochs 5
```

XLM-R Large là đối chứng Transformer hiện đại, không tự động được gọi là SOTA nếu chưa đánh giá công bằng trên cùng protocol và nhãn người gán. Hai mô hình dùng weighted cross-entropy, validation Macro-F1 để chọn checkpoint và early stopping. Có thể dùng `python scripts/train_phobert.py` như entrypoint cũ.

### Kết quả đã chạy: baseline cổ điển

Đã chạy `scripts/train_classical.py --bootstrap 1000` trên split hiện có. Kết quả cao nhất trong các baseline là **Word2Vec + LinearSVC** (Accuracy `0.828`, Macro-F1 `0.695`, bootstrap CI 95% `[0.647, 0.745]`); lớp Neutral vẫn khó (`F1=0.370`, support `60`). Đây là đánh giá đối với **pseudo-labels** trên test giữ lại, không phải ước lượng hiệu năng trên gold labels. Chi tiết nằm ở `outputs/metrics/classical_metrics.json`, dự đoán theo mẫu tại `outputs/predictions/` và confusion matrices mới trong `figures/`. PhoBERT/XLM-R chưa fine-tune trong lần chạy này; mã runner đã sẵn sàng.

### 4. Chạy notebook báo cáo

```bash
jupyter lab notebooks/01_sentiment_analysis_report.ipynb
```

Notebook trình bày mục tiêu/câu hỏi nghiên cứu, EDA, chất lượng nhãn, integrity checks, mô tả mô hình, bảng so sánh, per-class scores, confusion matrix, phân tích lỗi và giới hạn. Mặc định không tự huấn luyện; bật `RUN_CLASSICAL = True` / điền `TRANSFORMER_MODELS` trong notebook để chạy. Các điểm số không có artifact metrics sẽ không bị điền giả.

### 5. Tạo nhãn kiểm tra độc lập (khuyến nghị trước khi chốt báo cáo)

```bash
python scripts/export_annotation_sample.py --n 300 --seed 42
```

Hai annotator gán độc lập `-1/0/1`, không nhìn pseudo-label, adjudicate bất đồng; sau đó dùng cell annotation trong notebook tính Cohen’s kappa và đánh giá mô hình với nhãn đã kiểm tra.

## Đánh giá đúng cách

- Dữ liệu mất cân bằng: báo cáo **Macro-F1** làm chỉ số chính, đồng thời nêu Accuracy, macro precision/recall, weighted-F1 và metrics từng lớp.
- Chọn mô hình/siêu tham số bằng validation; không chọn dựa trên held-out test.
- Confusion matrix nêu rõ thứ tự lớp Negative, Neutral, Positive.
- Bootstrap CI trên test đo bất định lấy mẫu đối với nhãn đang có, không bù được nhiễu/sai lệch pseudo-label.
- Split hiện chưa group theo `product_code`; cần đánh giá group split nếu mục tiêu là tổng quát hóa sang sản phẩm chưa xuất hiện trong train.
- Không dùng biểu đồ `figures/reference/` làm bằng chứng cho kết quả dataset này.

## Tài liệu

- [`PLAN.md`](PLAN.md): kế hoạch và kiến trúc.
- [`INSTRUCTIONS.md`](INSTRUCTIONS.md): hướng dẫn pipeline và cấu trúc viết báo cáo.
- [`notebooks/01_sentiment_analysis_report.ipynb`](notebooks/01_sentiment_analysis_report.ipynb): notebook báo cáo chi tiết, có outputs baseline.
- [`docs/report_latex/`](docs/report_latex/): báo cáo LaTeX nhiều chương cho nhóm 4 thành viên; biên dịch XeLaTeX theo README riêng.
- `docs/Bài Tập Lớn I - Lazada.pdf`: mẫu báo cáo tham khảo.
