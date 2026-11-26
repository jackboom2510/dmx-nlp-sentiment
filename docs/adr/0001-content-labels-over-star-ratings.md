# ADR-0001: Dùng nhãn theo nội dung làm nhãn chuẩn thay cho nhãn suy từ số sao

- **Status**: Accepted
- **Date**: 2026-11-13

## Context

Cách gán nhãn rẻ nhất là suy từ số sao (1–2 Tiêu cực, 3 Trung tính, 4–5 Tích cực), nhưng đo trên tập kiểm tra 500 bình luận cho thấy 31,2% bình luận có nhãn theo sao khác nhãn theo nội dung; riêng mức 3 sao lệch 83,3% (112/150 bình luận là Tiêu cực). Huấn luyện bằng nhãn sao làm macro-F1 trên tập kiểm tra gán theo nội dung giảm mạnh (PhoBERT 0,781 → 0,632 trong thí nghiệm đối chứng lưu tại `results/archive_star/` của repo nguồn). Các phương án đã cân nhắc: (a) giữ nhãn sao cho toàn bộ dữ liệu; (b) chỉ gán nhãn nội dung cho tập kiểm tra, huấn luyện vẫn bằng nhãn sao; (c) gán nhãn nội dung cho cả ba tập.

## Decision

Gán nhãn theo nội dung cho toàn bộ dữ liệu (phương án c): LLM gán theo hướng dẫn gán nhãn gồm 3 định nghĩa, 9 quy tắc, 10 ví dụ, không được xem số sao; độ tin cậy được kiểm chứng bằng gán nhãn mù độc lập của một thành viên trên 100 bình luận (đồng thuận 88%, Cohen's kappa = 0,798).

## Consequences

- Nhãn huấn luyện nhất quán với nhãn đánh giá; hiện tượng sụt macro-F1 từ kiểm định sang kiểm tra biến mất.
- Chi phí gán nhãn và kiểm chứng tăng; chất lượng nhãn phụ thuộc hướng dẫn gán nhãn và LLM được chọn, nên danh tính LLM phải được ghi lại khi tái bản (xem phân tích thiếu sót B5).
- Lớp Trung tính co lại còn 7,2–9,4% (so với 15,2% nếu theo sao), làm bài toán lệch lớp hơn và buộc dùng trọng số lớp cùng macro-F1.
