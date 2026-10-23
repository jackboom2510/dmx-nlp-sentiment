import csv
import json
import os

json_file_path = "data/raw/reviews.jsonl"
csv_file_path = "data/raw/reviews.csv"

os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

rows = []
all_keys = set()

# Bước 1: Đọc toàn bộ file để gom tất cả các keys xuất hiện ở mọi dòng
with open(json_file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
                all_keys.update(item.keys())
        except json.JSONDecodeError:
            continue  # Bỏ qua dòng lỗi định dạng

if not rows:
    print("Không có dữ liệu hợp lệ trong file JSONL.")
else:
    # Sắp xếp lại danh sách header cho đẹp (hoặc giữ nguyên thứ tự tùy ý)
    headers = list(all_keys)

    # Bước 2: Ghi ra file CSV với đầy đủ các trường
    with open(csv_file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=headers, restval=""
        )  # restval="" tự động điền trống nếu thiếu trường
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"Đã chuyển đổi thành công file bất quy tắc sang: {csv_file_path}")
