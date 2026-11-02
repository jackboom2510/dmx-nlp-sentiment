# Báo cáo LaTeX (nhóm 4 thành viên)

Nguồn báo cáo được tách theo chương để nhóm dễ phân công biên tập. Cách tổ chức phần lý thuyết, dữ liệu, triển khai, đánh giá/so sánh và phân công có tham khảo báo cáo Clustering do người dùng cung cấp; nội dung chuyên môn và biểu đồ được thay hoàn toàn cho bài toán sentiment. Bố cục học phần cũng đối chiếu mẫu `docs/Bài Tập Lớn I - Lazada.pdf`.

```text
report_latex/
├── main.tex
├── README.md
└── chapters/
    ├── 01_introduction.tex
    ├── 02_theory.tex
    ├── 03_data_method.tex
    ├── 04_experiments_results.tex
    └── 05_conclusion.tex
```

## Biên dịch

Cần cài TeX Live/MiKTeX với XeLaTeX và các package thường dùng (`fontspec`, `polyglossia`, `booktabs`, `tabularx`, `siunitx`, `graphicx`, ...). Từ chính thư mục này, chạy hai lượt để cập nhật mục lục/danh sách hình:

```bash
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Hoặc mở `main.tex` bằng Overleaf và chọn **XeLaTeX** làm compiler. Hình được lấy tương đối từ `../../figures/` ở repo root. Báo cáo chỉ nhúng các biểu đồ mới sinh từ dataset DMX; ảnh tại `figures/reference/` không được dùng làm kết quả.

## Việc nhóm cần điền trước khi nộp

1. Điền khoa/bộ môn, giảng viên, họ tên và MSSV của đủ **4 thành viên** trong `main.tex`.
2. Xác nhận phần việc/tỷ lệ đóng góp ở trang bìa và cuối báo cáo; tổng tỷ lệ phải bằng 100%.
3. Chạy lại dữ liệu/mô hình sau khi thay đổi pipeline; kiểm tra các số liệu trong chương 3–4 khớp artifacts `data/interim/labeling_summary.json` và `outputs/metrics/classical_metrics.json`.
4. Hoàn tất đánh giá Transformer trước khi ghi nhận hiệu quả PhoBERT/XLM-R. Hiện báo cáo ghi rõ các mô hình này mới có code runner, chưa có metrics fine-tuning.
5. Tạo nhãn kiểm tra độc lập nếu muốn khẳng định hiệu năng thực tế. `data/processed/test.csv` hiện là tập held-out pseudo-label, không phải gold test.
6. Rà soát lỗi LaTeX / ngắt dòng bảng / chất lượng ảnh sau khi biên dịch và thêm ngày, thông tin lớp theo yêu cầu giảng viên.

## Kết quả thực nghiệm trong bản hiện tại

`Word2Vec + LinearSVC`: Accuracy 0,828; Macro-F1 0,6949; bootstrap CI 95% [0,6467; 0,7445] (1.000 bootstrap resamples). Các số này được tính trên pseudo-label. Kết quả khác có thể xảy ra nếu seed, phiên bản thư viện, dữ liệu hoặc cấu hình thay đổi.
