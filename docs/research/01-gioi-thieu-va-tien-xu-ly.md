# Research 01 — Giới thiệu và tiền xử lý (v1.6)

Nguồn sơ cấp: repo `sannguyen125/sentiment-classification` @ `0184366`. Mỗi mục: khẳng định trong báo cáo → nguồn → kết luận.

## 1.1 Bối cảnh

- **Báo cáo:** bình luận Điện Máy Xanh chủ yếu về điện máy/gia dụng; số sao "về nguyên tắc" có thể làm nhãn nhưng khảo sát cho thấy không nhất quán với nội dung.
- **Nguồn:** `results/metrics/crawl_stats.json` (19 ngành hàng đã thu thập); `gold_agreement.json` (`pct_star_label_differs_from_final = 0,312`).
- **Kết luận:** Khớp. Khẳng định "một trong những hệ thống bán lẻ điện máy lớn tại Việt Nam" là diễn đạt bối cảnh, không cần số liệu.

## 1.2 Phát biểu bài toán

- **Báo cáo:** đầu vào một bình luận, đầu ra một trong ba nhãn; định nghĩa Trung tính gồm "hỏi thông tin, mới mua chưa dùng, hoặc khen chê ngang nhau".
- **Nguồn:** `docs/HUONG_DAN_GAN_NHAN.md` — định nghĩa ba nhãn và 9 quy tắc (khen chê lẫn lộn lấy ý nặng hơn; hỏi vì sự cố là Tiêu cực; mỉa mai theo nghĩa thật).
- **Kết luận:** Khớp; định nghĩa ở 1.2 là bản rút gọn trung thành của guideline.

## 1.3 Thách thức của bài toán

- **Báo cáo:** teencode/viết tắt, không dấu, câu ngắn; tiếng Việt tách theo âm tiết; khen chê lẫn lộn; số sao lệch nội dung; lệch lớp.
- **Nguồn:** `resources/teencode.txt`; `results/metrics/preprocess_stats.json` (trung vị 13 từ/bình luận); Bảng 3.2 (Trung tính 7,2–9,4%).
- **Kết luận:** Khớp về nội dung. Còn typo "thao tác **vôi** trên điện thoại" (đúng: "vội").

## 1.4 Mục tiêu, phạm vi, bố cục

- **Báo cáo:** 3 đặc trưng × 2 thuật toán = 6 mô hình + PhoBERT; chỉ số chính macro-F1; một nhãn/bình luận, không tách khía cạnh.
- **Nguồn:** `src/train_classic.py`, notebook PhoBERT; ADR-0002 của repo đích.
- **Kết luận:** Khớp phạm vi. **Còn câu lặp** "tạo thành, tổng cộng là sáu mô hình" tồn tại từ v1.2 đến v1.6 chưa sửa.

## 2.1 Quy trình phân loại văn bản

- **Báo cáo:** 4 bước (tiền xử lý → trích chọn đặc trưng → xây dựng mô hình → đánh giá); cảnh báo từ điển/IDF/Word2Vec chỉ học trên tập huấn luyện, học trên cả tập kiểm tra là rò rỉ dữ liệu.
- **Nguồn:** `src/features.py` (fit trên train), `src/split.py` (chia tập trước khi trích chọn đặc trưng).
- **Kết luận:** Khớp với thiết kế code.

## 2.2.1 Làm sạch và chuẩn hóa văn bản

- **Báo cáo:** bỏ HTML/URL/email/emoji/ký tự đặc biệt, chuyển chữ thường; chuẩn hóa Unicode NFC; chuẩn hóa vị trí dấu thanh (hoà → hòa); rút gọn ký tự kéo dài ("tốtttt" → "tốt").
- **Nguồn:** `src/preprocess.py`: `clean()` (dòng 63), `normalize()` (dòng 84) gồm NFC + đổi dấu thanh kiểu cũ sang kiểu mới (comment dòng 74: "hoà -> hòa, thuỷ -> thủy") + rút gọn ký tự lặp.
- **Kết luận:** Khớp, kể cả bước đổi vị trí dấu thanh — có cài đặt thật. **Điều kiện:** "chuyển về chữ thường" chỉ đúng với đường mô hình cổ điển (`to_text_clean` có `.lower()`); đường PhoBERT (`to_text_seg`) giữ nguyên chữ hoa — mục 2.2.1 nói chung chung, dễ gây hiểu nhầm là mọi mô hình đều hạ chữ thường (mục 3.4 đã nói đúng phần này).

## 2.2.2 Chuẩn hóa từ viết tắt (Word Correction)

- **Báo cáo:** từ điển ánh xạ tự soạn; từ đa nghĩa không đưa vào vì sửa sai gây lệch hơn giữ nguyên.
- **Nguồn:** `resources/teencode.txt` — 65 mục (không tính comment); header ghi rõ các từ bị loại vì đa nghĩa: `bt, bh, kg, kt, nc, a, kh` ("kh" bị gỡ ngày 2026-10-02 vì còn nghĩa "khách hàng").
- **Kết luận:** Khớp hoàn toàn, kể cả lý do loại từ đa nghĩa. Mục lục còn typo "Word **Corection**" (thân bài đúng "Correction").

## 2.2.3 Tách từ (Tokenization)

- **Báo cáo:** khoảng trắng chỉ ngăn âm tiết; hai hướng tiếp cận (từ điển/khớp dài nhất và học máy); nhóm dùng Underthesea và VnCoreNLP; âm tiết nối bằng `_`.
- **Nguồn:** `src/preprocess.py: segment()` (Underthesea); VnCoreNLP RDRSegmenter (Ripple Down Rules — dựa trên luật học từ dữ liệu) dùng cho PhoBERT.
- **Kết luận:** Khớp. Phân loại "hai hướng tiếp cận" là đơn giản hóa chấp nhận được (RDRSegmenter thuộc nhóm luật học được, không phải khớp từ điển thuần túy).

## 2.2.4 Xử lý từ dừng

- **Báo cáo:** bỏ từ dừng giảm chiều/nhiễu nhưng phải giữ từ phủ định, mức độ, tương phản; ví dụ "không lạnh" → "lạnh" đảo nghĩa.
- **Nguồn:** `resources/stopwords_vi.txt` (111 từ), `keep_words.txt` (39 từ luôn giữ); `drop_special_and_stopwords()` trong `src/preprocess.py`; `preprocess_stats.json` (`pct_tokens_removed_as_stopwords = 0,2245`).
- **Kết luận:** Khớp cả về danh sách lẫn lập luận (Hình 3.3 là bằng chứng thực nghiệm).

## 2.2.5 Đưa từ về dạng gốc (Stemming, Lemmatization)

- **Báo cáo:** tiếng Việt là ngôn ngữ đơn lập, từ không đổi dạng nên không cần stemming/lemmatization.
- **Nguồn:** toàn bộ pipeline `src/preprocess.py` không có bước stemming — đúng như tuyên bố.
- **Kết luận:** Khớp; lập luận ngôn ngữ học chuẩn.

## 3.4 Tiền xử lý + Bảng 3.3 + Hình 3.3

- **Báo cáo:** 6 bước cho mô hình cổ điển; PhoBERT chỉ làm sạch/chuẩn hóa/sửa từ/tách từ, giữ chữ hoa, dấu câu, từ dừng; từ điển sửa 3.745 lượt trong 2.057 bình luận; bỏ từ dừng loại 22,5% token; độ dài trung bình giảm 19,1 → 14,8 token.
- **Nguồn:** `results/metrics/preprocess_stats.json` — `word_corrections_total = 3745`, `reviews_with_word_correction = 2057`, `pct_tokens_removed_as_stopwords = 0,2245`, `tokens_mean`: text_nostop 19,14 → text_clean 14,84; `segmented_multi_syllable_tokens_pct = 0,2349` (khớp "23,5% token là từ nhiều âm tiết"); ví dụ Bảng 3.3 truy về `results/preprocess_examples.csv`.
- **Kết luận:** Khớp toàn bộ số liệu. Hình 3.3 (12 n-gram đặc trưng/lớp, tính trên train) nhất quán với kỷ luật chỉ học trên tập huấn luyện.
