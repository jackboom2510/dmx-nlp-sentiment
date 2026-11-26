# ADR-0002: Dùng macro-F1 làm chỉ số chính để chọn cấu hình và xếp hạng mô hình

- **Status**: Accepted
- **Date**: 2026-11-13

## Context

Phân bố nhãn lệch mạnh (Trung tính chỉ 7,2–9,4%). Nếu chọn theo Accuracy, mô hình luôn đoán lớp đa số vẫn đạt 0,328 trên tập kiểm tra trong khi không nhận diện được bất kỳ bình luận Trung tính nào (macro-F1 = 0,165, F1 Trung tính = 0). Các phương án đã cân nhắc: (a) Accuracy; (b) F1 trọng số theo số mẫu (weighted-F1); (c) macro-F1.

## Decision

Dùng macro-F1 — trung bình cộng F1 của ba lớp, mỗi lớp ngang nhau — làm tiêu chí duy nhất để chọn siêu tham số (Bảng 3.4), chọn checkpoint PhoBERT (Bảng 3.5) và xếp hạng mô hình (Bảng 3.6). Accuracy và F1 từng lớp chỉ đóng vai trò báo cáo bổ trợ.

## Consequences

- Các quyết định chọn mô hình ưu tiên lớp Trung tính thiểu số; đây là lý do PhoBERT với trọng số lớp Trung tính 4,267 và các cấu hình tiên nghiệm bằng nhau của Naive Bayes được chọn.
- Chênh lệch macro-F1 nhỏ (ví dụ 0,005 giữa TF-IDF + NB và TF-IDF + SVM) không đủ căn cứ xếp hạng khi chưa có khoảng tin cậy — kết luận phải giữ ở mức "tương đương" (xem phân tích thiếu sót B3).
- Mọi thí nghiệm sau này phải báo cáo macro-F1 kèm F1 lớp Trung tính để giữ tính so sánh được.
