# Research 04 — Huấn luyện, kết quả, phân tích lỗi, kết luận (v1.6, mục 3.5–3.8)

Nguồn sơ cấp: repo nguồn @ `0184366`. Quy trình PhoBERT chi tiết tại `docs/research-phobert-finetuning-pipeline.md`.

## 3.5 Trích chọn đặc trưng và huấn luyện — Bảng 3.5 (siêu tham số)

- **Báo cáo:** đặc trưng chỉ học trên train; BoW (1,1)/(1,2); TF-IDF (1,2) tf logarit, idf làm mịn, chuẩn L2; Word2Vec Skip-gram 200 chiều; Multinomial NB + LinearSVC cho BoW/TF-IDF; Gaussian NB + SVM RBF (chuẩn hóa mean 0, var 1) cho Word2Vec; SVM đặt trọng số lớp tỉ lệ nghịch; chọn theo macro-F1 trên kiểm định; kèm đoạn diễn giải các siêu tham số.
- **Nguồn:** `src/features.py`, `src/train_classic.py`; toàn bộ lưới và giá trị chọn tại `results/metrics/val_grid.csv` (ví dụ TF-IDF + SVM: min_df = 5, C = 0,1; Word2Vec + SVM: C = 0,3, γ tự động).
- **Kết luận:** Khớp. Đoạn diễn giải sau bảng (mới có từ bản "cuối", giữ lại ở v1.6) vẫn dừng ở định nghĩa; phần "vì sao giá trị khác nhau giữa các mô hình" đã có văn bản soạn sẵn (file bổ sung, mục II.4).

## Bảng 3.6 — PhoBERT theo từng epoch

- **Báo cáo:** train loss 0,649/0,356/0,243/0,167; val macro-F1 0,843/0,875/0,861/0,862; lưu checkpoint ở epoch 1 và 2; lập luận dừng sớm.
- **Nguồn:** `results/metrics/phobert.json` (`history`) — khớp từng số; độ lệch chuẩn 3 hạt giống 0,881 ± 0,015 tại `phobert_seeds.json` cũng được v1.6 nêu.
- **Kết luận:** Khớp hoàn toàn.

## 3.6.1 Kết quả chung — Bảng 3.7, Hình 3.5

- **Báo cáo:** 8 mô hình trên 500 bình luận; PhoBERT Accuracy 0,948, macro-F1 0,902; mốc đoán lớp đông nhất 0,328/0,165; cột cuối là macro-F1 kiểm định.
- **Nguồn:** `results/comparison.csv` và `results/metrics/test_metrics.json` (PhoBERT test_macro_f1 = 0,9023; per-class F1 khớp: Neutral precision 0,6774, recall 0,8936, F1 0,7706); Hình 3.5 tương ứng `results/figures/macro_f1_val_test.png`.
- **Kết luận:** Khớp. Lưu ý diễn giải: macro-F1 kiểm tra cao hơn kiểm định ở TF-IDF/PhoBERT được báo cáo giải thích bằng lệch phân bố nhãn (Bảng 3.2) — nhất quán với `split_stats.json`.

## 3.6.2 So sánh đặc trưng × thuật toán — Bảng 3.8, Hình 3.6

- **Báo cáo:** TF-IDF tốt nhất (0,799/0,804); Word2Vec thấp nhất (0,623/0,718); SVM hơn NB rõ ở BoW và Word2Vec, chênh 0,005 ở TF-IDF nên không xếp hạng; ma trận nhầm lẫn: TF-IDF + SVM đúng 23/47 Trung tính (15 → Tiêu cực, 9 → Tích cực), PhoBERT đúng 42/47 nhưng precision Trung tính 0,677.
- **Nguồn:** `results/comparison.csv` (bảng chéo đặc trưng × thuật toán); `results/figures/cm_tf_idf_svm.png`, `cm_phobert.png`; `test_metrics.json`.
- **Kết luận:** Khớp từng con số, kể cả cách kết luận thận trọng về chênh 0,005.

## 3.6.3 Có và không bỏ từ dừng

- **Báo cáo:** macro-F1 kiểm định 0,793 (không bỏ) vs 0,791 (bỏ) — gần như không đổi; kiểm tra 0,812 chỉ để tham khảo, không dùng chọn mô hình.
- **Nguồn:** biến thể được lưu tại `results/figures/cm_tf_idf_svm_khong_bo_stopword_tham_khao.png` và metrics tương ứng trong `results/`.
- **Kết luận:** Khớp; cách xử lý "chỉ tham khảo" đúng kỷ luật chống rò rỉ qua tập kiểm tra.

## 3.6.4 Nhãn theo số sao và nhãn theo nội dung — Bảng 3.9

- **Báo cáo:** huấn luyện theo sao làm macro-F1 kiểm tra (chấm theo nội dung) tụt mạnh (PhoBERT 0,781 kiểm định → 0,632 kiểm tra); đổi sang nhãn nội dung tăng 0,17–0,27; lưu ý mức tăng có thể không hoàn toàn do đổi nhãn (đồng thời sửa từ điển và loại 16 bình luận trùng).
- **Nguồn:** thí nghiệm A lưu tại `results/archive_star/comparison.csv` và notebook `results/archive_star/02_phobert_colab_star.ipynb`.
- **Kết luận:** Khớp, và câu lưu ý về yếu tố nhiễu đồng thời là diễn đạt trung thực cần giữ.

## 3.6.5 Chấm lại bằng nhãn do thành viên gán — Bảng 3.10

- **Báo cáo:** PhoBERT vẫn đầu (0,850), cao hơn mốc đồng thuận nhãn kiểm tra vs thành viên (0,818); TF-IDF + NB (0,770) vượt TF-IDF + SVM (0,705); chỉ coi là kiểm chứng bổ sung vì n = 100.
- **Nguồn:** `results/robustness.csv` (cột `human2_macro_f1`: PhoBERT 0,8497); `results/metrics/gold_agreement.json`.
- **Kết luận:** Khớp. Cùng file `robustness.csv` còn chứa CI95 bootstrap (PhoBERT [0,864; 0,935]) mà **cả v1.6 lẫn bản cuối đều chưa đưa vào báo cáo** — thiếu sót còn lại duy nhất ở nhóm bằng chứng.

## 3.7 Phân tích lỗi — Bảng 3.11, 3.12, 3.13; mục 3.7.3

- **Báo cáo:** TF-IDF + SVM sai 55, PhoBERT sai 26, cùng sai 12; sai nhiều nhất ở 3–4 sao; nhóm lỗi lớn nhất là khen chê lẫn lộn (14/30), phủ định chỉ ở TF-IDF + SVM (4 câu); lỗi tiền xử lý "ăn_không" và mất chữ "điều" trong "điều hòa"; câu thử "máy giặt chạy êm nhưng vắt không khô": PhoBERT đoán Tiêu cực p = 0,62, bỏ "giặt" thì thành Trung tính p = 0,68.
- **Nguồn:** `results/errors.csv`, `results/error_groups.csv` (Phase 6 lấy 15 lỗi của mô hình cổ điển tốt nhất + 15 lỗi PhoBERT); câu thử ở 3.7.3 chính là câu mặc định đầu tiên của `src/predict.py`.
- **Kết luận:** Khớp. Phân tích lỗi truy được về file predictions, không phải chọn tay tự do.

## 3.8 Kết luận và hướng phát triển

- **Báo cáo:** tóm tắt pipeline và kết quả chính (TF-IDF tốt nhất trong nhóm cổ điển ~0,80; Word2Vec thấp nhất; PhoBERT 0,902, thắng chủ yếu ở lớp Trung tính); hướng phát triển về dữ liệu và mô hình.
- **Nguồn:** tổng hợp từ các mục 3.5–3.7 ở trên.
- **Kết luận:** Mọi con số trong kết luận đều truy được về artifacts. Phần hướng phát triển là định hướng, không cần nguồn; khi viết nên gắn với các giới hạn đã nêu (nhãn train/val chưa duyệt toàn bộ; chưa tách theo sản phẩm) để tránh hứa hẹn chung chung.
