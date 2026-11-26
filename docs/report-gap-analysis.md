# Phân tích thiếu sót của báo cáo bản cuối

Tài liệu này rà soát báo cáo `BAO_CAO_ban_cuoi.pdf` (29 trang) đối chiếu với mã nguồn đã sinh ra nó, nhằm chỉ ra những gì báo cáo đã thực hiện được và những thiếu sót còn lại. Mọi khẳng định về code đều trích dẫn repo nguồn `sannguyen125/sentiment-classification` tại commit ghim cứng `01843665a5e31f64e6dce1adf66a0116ece3fbc0`, và repo đích `jackboom2510/dmx-nlp-sentiment` tại commit `6c273fc52ab4817d7c9580fa1377880b8c8640fc`. Mức độ nghiêm trọng: **Cao** = ảnh hưởng tính đúng đắn hoặc khả năng tái lập; **Trung bình** = làm yếu bằng chứng hoặc lập luận; **Thấp** = trình bày, nhất quán.

## 1. Báo cáo bản cuối đã thực hiện được gì

- Pipeline đầy đủ từ thu thập (9.481 bình luận, 72 sản phẩm, 19 ngành hàng) → làm sạch còn 6.540 (Bảng 3.1) → gán nhãn theo nội dung → tiền xử lý → 6 cấu hình cổ điển + PhoBERT → đánh giá (Bảng 3.6–3.7) → phân tích lỗi (Bảng 3.8–3.10). Số liệu khớp với repo nguồn (`results/metrics/clean_stats.json`: raw = 9.481; `results/comparison.csv`: PhoBERT macro-F1 = 0,9023, Accuracy = 0,948).
- Phát hiện cốt lõi có bằng chứng: nhãn theo sao lệch nhãn theo nội dung 31,2% trên tập kiểm tra, riêng 3 sao lệch 83,3% (mục 3.3.2, Hình 3.2).
- Kỷ luật đánh giá đúng: chọn siêu tham số và checkpoint bằng macro-F1 trên tập kiểm định, tập kiểm tra dùng một lần (mục 2.6, 3.5); khử trùng toàn cục trước khi chia tập và loại thêm 16 bình luận trùng sau tiền xử lý (mục 3.5).
- So với v1.2, bản cuối đã: sửa sạch nhóm lỗi chính tả ở mục 2.5/3.5 (PhoBFRT, Fpsilon, Icarning… không còn); bổ sung đoạn giải thích các siêu tham số sau Bảng 3.4; bổ sung Bảng 3.5 kết quả PhoBERT theo từng epoch kèm lập luận dừng sớm (train loss giảm 0,356 → 0,167 trong khi macro-F1 kiểm định không tăng sau epoch 2).

## 2. Nhóm A — Thiếu sót nội dung và giải thích

| # | Mức | Thiếu sót | Bằng chứng | Gợi ý sửa |
|---|---|---|---|---|
| A1 | Trung bình | Giải thích siêu tham số mới dừng ở định nghĩa. Chưa giải thích vì sao giá trị chọn lại khác nhau giữa các mô hình (α = 1,0 với BoW nhưng 0,1 với TF-IDF; min_df = 5 chỉ ở TF-IDF + SVM), vì sao C không so sánh trực tiếp được giữa các loại đặc trưng, và γ của RBF thực chất nhận giá trị nào | Bảng 3.4 và đoạn sau bảng (trang 18) | Thêm 1 đoạn diễn giải theo từng cặp mô hình; ghi giá trị γ thực tế từ `results/metrics/val_grid.csv` của repo nguồn |
| A2 | Trung bình | Lý thuyết bị rút gọn quá tay so với v1.2: công thức Attention và vai trò √dₖ, bài toán tối ưu SVM và công thức hàm nhân RBF đều đã bị bỏ; mục 2.4.2 chỉ nói định tính "C lớn thì phạt lỗi nặng" | Mục 2.4.2, 2.5 (trang 10–11); đối chiếu v1.2 mục 2.4.2/2.5 | Khôi phục tối thiểu công thức RBF kèm giải thích γ, và một câu về √dₖ |
| A3 | Thấp | Công thức tf logarit vẫn thiếu điều kiện f = 0: tf(t,d) = 1 + ln f không xác định khi từ không xuất hiện trong văn bản | Mục 2.3.2 (trang 8) | Viết dạng hai trường hợp: 0 nếu f = 0, 1 + ln f nếu f > 0 |
| A4 | Trung bình | Không có mục EDA riêng, dù repo nguồn đã có `notebooks/01_eda.ipynb` và 7 biểu đồ `results/figures/eda_*.png`; độ dài văn bản, từ vựng, phân bố theo ngành hàng chỉ được nhắc rải rác | Mục 3.2–3.3; repo nguồn `results/figures/eda_categories.png`, `eda_length_by_label.png`, `eda_top_terms.png` | Bổ sung mục EDA ngắn (0,5–1 trang) tái dùng các biểu đồ sẵn có |
| A5 | Thấp | Mục 3.5 nói "mã hóa theo bộ từ vựng của PhoBERT" nhưng không nêu tên thuật toán BPE cũng như lý do phải tách từ trước bằng RDRSegmenter (đã có ở mục 2.5 nhưng không nối sang 3.5) | Mục 3.5 (trang 18) | Thêm một câu nối 2.5 → 3.5 |

## 3. Nhóm B — Thiếu sót phương pháp và bằng chứng

| # | Mức | Thiếu sót | Bằng chứng | Gợi ý sửa |
|---|---|---|---|---|
| B1 | Cao | Ba bằng chứng phương pháp của v1.2 đã bị cắt khỏi bản cuối trong khi kết luận giữ nguyên: (i) ablation huấn luyện bằng nhãn sao vs nhãn nội dung, (ii) ablation bỏ/không bỏ từ dừng, (iii) chấm lại bằng nhãn người. Kết luận "gán nhãn theo nội dung là đúng" nay thiếu thí nghiệm đối chứng trong cùng tài liệu | Mục lục chỉ còn 3.6.1–3.6.2; dữ liệu vẫn còn tại repo nguồn `results/archive_star/comparison.csv` và `results/robustness.csv` | Khôi phục ít nhất bảng (i) và (iii) dạng rút gọn, hoặc hạ giọng kết luận |
| B2 | Trung bình | Độ tin cậy nhãn bị trình bày yếu đi: hệ số Cohen's kappa = 0,798 (có trong `results/metrics/gold_agreement.json`: 0,7975) đã bị bỏ, chỉ còn "giống nhau 88/100"; chi tiết 191 bình luận bị gắn cờ và quy trình duyệt lại cũng bị lược | Mục 3.3.2 (trang 14–15) | Bổ sung kappa + một câu về quy trình xử lý bình luận gắn cờ |
| B3 | Trung bình | Không báo cáo khoảng tin cậy dù repo nguồn đã tính bootstrap CI (`results/robustness.csv`, PhoBERT test macro-F1 CI95 [0,864; 0,935]). Chênh lệch 0,005 giữa TF-IDF + NB và TF-IDF + SVM được kết luận "chưa nói được" là đúng, nhưng thiếu CI khiến các chênh lệch khác (0,711 vs 0,760…) cũng không có căn cứ định lượng | Bảng 3.6–3.7; repo nguồn `results/robustness.csv` | Thêm cột CI95 hoặc câu chú thích khoảng tin cậy cho các mô hình chính |
| B4 | Trung bình | Chia tập huấn luyện/kiểm định phân tầng theo **số sao** chứ không theo nhãn nội dung cuối cùng, và không tách theo sản phẩm — cùng một sản phẩm xuất hiện ở cả ba tập, nên kết quả chưa đo khả năng tổng quát sang sản phẩm mới | Repo nguồn `src/split.py`: `stratify=pool["label_star"]`; mục 3.5 chỉ nói "giữ tỷ lệ các mức sao như nhau" | Ghi rõ đây là giới hạn; cân nhắc thí nghiệm bổ sung chia theo sản phẩm |
| B5 | Trung bình | Danh tính mô hình LLM gán nhãn (tên, phiên bản, thời điểm chạy) không được nêu trong báo cáo, làm giảm khả năng tái lập của toàn bộ nhãn — tài sản quan trọng nhất của đề tài | Mục 3.3.2 chỉ nói "một mô hình ngôn ngữ lớn"; `src/gold_llm.py`, `src/pool_llm.py` | Bổ sung tên/phiên bản LLM và ngày chạy vào mục 3.1 hoặc 3.3.2 |
| B6 | Thấp | Nhãn tập huấn luyện/kiểm định do LLM gán không được con người duyệt lại (886 bình luận bị gắn cờ "chưa chắc chắn" theo v1.2); bản cuối không còn nhắc con số này | Mục 3.3.2; `data/pool_llm/pool_llm.csv` | Nêu rõ giới hạn và hướng kiểm chứng mẫu ngẫu nhiên |
| B7 | Thấp | Chênh lệch macro-F1 PhoBERT giữa kiểm định (0,875) và kiểm tra (0,902) được giải thích bằng phân bố nhãn là hợp lý, nhưng mô hình được chọn theo hạt giống 42 trong khi ba hạt giống cho độ lệch chuẩn 0,015 (số liệu có trong `results/metrics/phobert_seeds.json`) không được báo cáo ở bản cuối | Mục 3.5, Bảng 3.5 | Thêm một câu về độ biến thiên theo hạt giống |

## 4. Nhóm C — Trình bày và tầng repo

| # | Mức | Thiếu sót | Bằng chứng | Gợi ý sửa |
|---|---|---|---|---|
| C1 | Trung bình | Mục lục còn lỗi "3.6.3 Có và không bỏ từ dừng … Error! Bookmark not defined." — dấu vết của mục đã xóa nhưng trường tham chiếu chéo chưa cập nhật | Mục lục (trang 5) của PDF bản cuối | Cập nhật toàn bộ field (Ctrl+A → F9) trước khi xuất PDF; xóa mục 3.6.3 khỏi mục lục |
| C2 | Cao | Repo đích theo dõi `sentiment-classification` dạng gitlink (mode 160000, commit `0184366…`) nhưng **không có `.gitmodules`**: clone mới nhận thư mục rỗng, không tái lập được nguồn phát sinh báo cáo | `git ls-files -s sentiment-classification`; không tồn tại `.gitmodules` | Thêm `.gitmodules` trỏ tới `https://github.com/sannguyen125/sentiment-classification`, hoặc gỡ gitlink và ghi đường dẫn nguồn trong README |
| C3 | Trung bình | Hai pipeline gán nhãn cùng tồn tại mà tài liệu dễ gây nhầm: repo đích (`README.md`, `SKILLS.md`) cảnh báo test.csv là pseudo-label của pipeline hybrid rating+lexicon, trong khi báo cáo dùng tập gold của repo nguồn. Người đọc không được chỉ rõ báo cáo thuộc pipeline nào | `README.md` dòng 5; `SKILLS.md` quy tắc 2; so với mục 3.3 của báo cáo | Thêm đoạn "Nguồn gốc báo cáo" trong README của repo đích, phân biệt hai pipeline |
| C4 | Thấp | Số đếm tài nguyên trong báo cáo (65 teencode, 111 từ dừng, 39 từ giữ lại) khớp repo nguồn khi loại dòng chú thích (`resources/teencode.txt`, `stopwords_vi.txt`, `keep_words.txt`) — điểm này đúng, chỉ cần giữ nguyên khi tái bản | Kiểm đếm trực tiếp ba file resources | Không cần sửa |

## 5. Ưu tiên sửa trước khi bảo vệ

1. **C1** (5 phút): cập nhật field mục lục, xóa lỗi Bookmark.
2. **B1 + B2** (nửa giờ): khôi phục rút gọn bảng ablation nhãn sao/nội dung và con số kappa — đây là hai bằng chứng dễ bị hội đồng hỏi nhất.
3. **A1 + A2** (nửa giờ): bổ sung diễn giải siêu tham số theo mô hình và công thức RBF/γ.
4. **C2** (15 phút): thêm `.gitmodules` để repo tự tái lập.
5. **B3 + B5** (nửa giờ): bổ sung CI95 từ `results/robustness.csv` và danh tính LLM gán nhãn.
