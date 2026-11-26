# Research 03 — Dữ liệu, gán nhãn và kiểm tra sau chia tập (v1.6, mục 3.1–3.4)

Nguồn sơ cấp: repo nguồn @ `0184366` — `src/crawl/`, `src/clean_label.py`, `src/gold_llm.py`, `src/pool_llm.py`, `src/split.py`, các file `results/metrics/*.json` và `results/figures/eda_*.png`.

## 3.1 Môi trường và công cụ

- **Báo cáo:** Python; Playwright (thu thập), Underthesea/VnCoreNLP (tách từ), scikit-learn (mô hình cổ điển), PyTorch + transformers (PhoBERT); NB/SVM chạy CPU, PhoBERT trên GPU; seed 42.
- **Nguồn:** `requirements.txt`; `results/metrics/phobert.json` (GPU RTX 3060 local).
- **Kết luận:** Khớp. **Điều kiện:** mô hình ngôn ngữ dùng để gán nhãn vẫn chỉ được gọi là "một mô hình ngôn ngữ lớn" — không nêu tên/phiên bản ở cả báo cáo lẫn cấu hình đọc được, làm giảm khả năng tái lập của nhãn (thiếu sót B5 đã ghi ở `docs/report-gap-analysis.md`).

## 3.2 Thu thập dữ liệu + Hình 3.1

- **Báo cáo:** sitemap 14.841 trang sản phẩm thuộc 195 ngành hàng; lấy xen kẽ bán chạy nhất và điểm thấp nhất; mỗi sản phẩm lấy hết 1–3 sao, tối đa 60 bình luận 4 sao, 100 bình luận 5 sao, không quá 300; kết quả 9.481 bình luận, 72 sản phẩm (45 bán chạy, 27 điểm thấp), 19 ngành hàng; thừa nhận phân bố sao không giống thực tế website.
- **Nguồn:** `results/metrics/crawl_stats.json` — `sitemap_products_allowed = 14841`, `sitemap_categories_allowed = 195`, `products_crawled = 72`, `categories_crawled = 19`, `products_by_pick = {top_sold: 45, low_rank: 27}`; Hình 3.1 tương ứng `results/figures/eda_categories.png`.
- **Kết luận:** Khớp từng con số; việc tự nêu thiên lệch lấy mẫu là điểm mạnh phương pháp.

## 3.3.1 Làm sạch dữ liệu — Bảng 3.1

- **Báo cáo:** 9.481 → 6.540 qua: rỗng 17; dưới 3 từ 1.976; không phải tiếng Việt 3; phản hồi nhân viên 0; trùng hoàn toàn 690; gần trùng (cosine > 0,9) 255; trung vị 13 từ; phân bố sao sau làm sạch 1.057/612/1.068/1.306/2.497.
- **Nguồn:** `src/clean_label.py` (hằng số `MIN_WORDS = 3`, `NEAR_DUP = 0.9`); `results/metrics/clean_stats.json` (`raw = 9481`, `clean = 6540`, `clean_by_star` khớp); biểu đồ phễu `results/figures/eda_cleaning.png`.
- **Kết luận:** Khớp toàn bộ. Chi tiết "75 cặp gần trùng khác số sao" cũng truy được từ logic đếm cặp lệch sao trong `clean_label.py`.

## 3.3.2 Gán nhãn — Hình 3.2, Bảng 3.2

- **Báo cáo:** test 500 theo hạn ngạch sao 100/70/150/80/100, ≤15/sản phẩm, 71 sản phẩm; guideline 3 định nghĩa + 9 quy tắc + 10 ví dụ; LLM gán không xem sao, xử lý 50 bình luận/lần kèm lý do và cờ chắc chắn; 191 bình luận gắn cờ, duyệt kỹ 2 trường hợp, chấp nhận 189; kiểm chứng mù 100 bình luận: đồng thuận 88%, kappa 0,798; lệch sao–nội dung 31,2% (3 sao 83,3%: 112 Tiêu cực / 25 Trung tính / 13 Tích cực); train+dev 6.024 bình luận cũng gán theo nội dung, 886 bị gắn cờ chưa chắc chắn; Bảng 3.2 phân bố 5.120/904/500.
- **Nguồn:** `src/clean_label.py` (`GOLD_QUOTA`, `GOLD_MAX_PER_PRODUCT = 15`); `docs/HUONG_DAN_GAN_NHAN.md`; `results/metrics/gold_agreement.json` — `n_flagged_for_review = 191`, `n_reviewed_row_by_row = 2`, `n_llm_accepted_in_bulk = 189`, `label_source = {llm: 309, llm_accepted: 189, human: 2}`, `kappa_llm_vs_member2 = 0,7975`, `agree = 0,88`, `pct_star_label_differs_from_final = 0,312`; `results/metrics/split_stats.json` khớp từng ô của Bảng 3.2; Hình 3.2 tương ứng `results/figures/eda_text_label_by_star.png`.
- **Kết luận:** Khớp toàn bộ, đây là mục truy vết tốt nhất của báo cáo. Hai lưu ý phương pháp (không phải sai số): chia train/val được `src/split.py` phân tầng theo **nhãn sao** chứ không theo nhãn nội dung; sản phẩm không được tách rời giữa các tập.

## Bảng 3.4 + Hình 3.4 — Kiểm tra sau chia tập (mục mới ở v1.6)

- **Báo cáo:** trung vị số từ trước tiền xử lý 13/13/15; token sau tiền xử lý 9/9/10; token PhoBERT 15/15/17; vượt 128 token: train 78 (1,5%), val 14 (1,5%), test 10 (2,0%); từ điển train 6.079 từ; lượt từ ngoài từ điển val 4,6%, test 5,6%; không bình luận nào mất toàn bộ từ.
- **Nguồn:** chỉ truy được một phần — tỷ lệ cắt token train 1,5% khớp `truncated_train_pct = 0,0152` trong `results/metrics/phobert.json`. Các số còn lại (trung vị token PhoBERT theo tập, 78/14/10 bình luận vượt ngưỡng, từ điển 6.079, OOV 4,6%/5,6%) **không có artifact lưu sẵn** trong repo nguồn @ `0184366` (`preprocess_stats.json` chỉ có thống kê gộp trên 6.540 bình luận, ví dụ vocab text_clean = 7.086 của toàn bộ dữ liệu đã làm sạch — khác khái niệm với từ điển train 6.079 sau min_df).
- **Kết luận:** **Chênh về truy vết** (không khẳng định sai): các số này nhất quán nội bộ và hợp lý, nhưng muốn kiểm chứng độc lập phải chạy lại tokenizer trên `data/processed/*.csv`. Đây là bảng duy nhất của v1.6 chưa có artifact sơ cấp đi kèm; nên bổ sung script/thống kê `results/metrics/token_stats.json` vào repo nguồn hoặc ghi rõ cách tính.
