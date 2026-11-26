# Nghiên cứu: Quy trình fine-tuning và áp dụng PhoBERT hiện tại

**Phạm vi:** chỉ đọc và mô tả quy trình sẵn có, không chạy lại huấn luyện, không thay đổi cấu trúc. Mọi khẳng định truy về nguồn sơ cấp là mã nguồn và artifacts của repo `sannguyen125/sentiment-classification` tại commit `01843665a5e31f64e6dce1adf66a0116ece3fbc0` (bản clone tham chiếu: `refs/sentiment-classification-src/`).

## 1. Tổng quan quy trình

```
data/processed/{train,val,test}.csv
  → [notebook 02] tách lại từ bằng RDRSegmenter (trên nền text_seg của Underthesea)
  → tokenize (max_len 128) → fine-tune phobert-base-v2 (≤4 epoch, weighted CE)
  → lưu checkpoint có val macro-F1 cao nhất vào models/phobert_best
  → nạp lại checkpoint, dự đoán val/test, lưu CSV kèm xác suất từng lớp
  → [Phase 6: src/evaluate.py] tính metrics trên các file predictions
Áp dụng: src/predict.py nạp models/phobert_best (nếu có) để dự đoán bình luận mới.
```

## 2. Nguồn sự thật của quy trình

- Toàn bộ logic huấn luyện PhoBERT nằm trong `notebooks/02_phobert_colab.ipynb` (15 ô); trong `src/` không có module huấn luyện PhoBERT tương đương — `src/` chỉ có `predict.py` (áp dụng) và `evaluate.py` (đánh giá).
- Notebook được đặt tên "colab" và có nhánh chạy trên Colab (cài `transformers`, `py_vncorenlp`, upload CSV thủ công qua `files.upload()` — ô 2, 3), nhưng lần chạy sinh ra kết quả trong báo cáo thực tế chạy **local**: `results/metrics/phobert.json` ghi `"in_colab": false`, GPU "NVIDIA GeForce RTX 3060 Laptop GPU", ~115 giây/epoch.
- Biến thể huấn luyện bằng nhãn sao được lưu riêng tại `results/archive_star/02_phobert_colab_star.ipynb`, tương ứng thí nghiệm đối chứng A.

## 3. Đầu vào dữ liệu và nhãn

- Notebook đọc trực tiếp `data/processed/{train,val,test}.csv` (ô 6); cột nhãn là `label`, cột văn bản đầu vào là `text_seg`.
- Loại nhãn (theo nội dung hay theo sao) được notebook tự nhận diện từ `results/metrics/split_stats.json` (trường `train_val_labels`), hoặc suy từ cột `label_text` khi chạy trên Colab, và ghi vào metadata (`note` trong `phobert.json`) — cơ chế này ngăn nhầm lẫn giữa hai thí nghiệm A/B.
- Hạt giống đọc từ biến môi trường `PHOBERT_SEED` (ô 2): seed 42 là mô hình chính thức và là seed duy nhất được dự đoán trên test; seed 43/44 chỉ dự đoán val, phục vụ đo độ biến thiên.

## 4. Tiền xử lý và tách từ

Quy trình tách từ cho PhoBERT gồm **hai tầng**:

1. Ở khâu tiền xử lý thượng nguồn (`src/preprocess.py`), văn bản đã được tách bằng **Underthesea** thành cột `text_seg`.
2. Trong notebook (ô 7), `resegment()` gỡ dấu gạch dưới của Underthesea để khôi phục câu, rồi tách lại bằng **RDRSegmenter** của VnCoreNLP (`py_vncorenlp`, tải `VnCoreNLP-1.2.jar` vào `models/vncorenlp` nếu chưa có). Kết quả là cột `x` — đầu vào thực tế của mô hình.

Nếu VnCoreNLP không khả dụng, notebook **âm thầm giữ nguyên tách từ Underthesea** (chỉ in cảnh báo), nhưng tên bộ tách từ thực tế luôn được ghi vào metadata (`"segmenter": "RDRSegmenter (VnCoreNLP)"` trong lần chạy chính thức). VnCoreNLP đổi thư mục làm việc khi khởi tạo; code đã lưu và khôi phục `cwd` (ô 7).

Đường áp dụng trong `src/predict.py` tái hiện đúng logic này: `to_text_seg()` (Underthesea) → gỡ `_` → RDRSegmenter nếu có jar, ngược lại dùng thẳng bản Underthesea.

## 5. Token hóa và giới hạn độ dài

- Tokenizer `AutoTokenizer` của `vinai/phobert-base-v2`; cắt tại `max_len = 128`, đệm theo lô bằng `DataCollatorWithPadding` (ô 9).
- Tỷ lệ bình luận huấn luyện bị cắt được đo và ghi lại: `truncated_train_pct = 0,0152` (1,52%) — khớp con số "1,5%" trong báo cáo.

## 6. Cấu hình huấn luyện (ô 4, 11)

| Tham số | Giá trị |
|---|---|
| Mô hình nền | `vinai/phobert-base-v2` |
| max_len / batch / grad accum | 128 / 16 / 1 |
| Learning rate / weight decay | 2e-5 / 0,01 (AdamW của PyTorch) |
| Lịch tốc độ học | Tuyến tính, warmup 10% tổng số bước |
| Epoch tối đa | 4 |
| fp16 | Bật khi có CUDA (autocast + GradScaler) |
| Cắt gradient | Chuẩn 1,0 |
| Hàm mất mát | CrossEntropyLoss có trọng số lớp |

Trọng số lớp tính trực tiếp từ phân bố train, $w_k = n/(3 \cdot n_k)$: Negative 0,8432; Neutral 4,2667; Positive 0,6330 (`results/metrics/phobert.json`).

## 7. Vòng huấn luyện và chọn checkpoint

- Sau mỗi epoch, mô hình được đánh giá trên val (macro-F1, accuracy, F1 từng lớp); nếu macro-F1 cao hơn mọi epoch trước, toàn bộ mô hình + tokenizer được lưu vào `models/phobert_best{SUFFIX}` bằng `save_pretrained` (ô 11). Đây là cơ chế chọn checkpoint theo kiểm định, tương đương dừng sớm nhưng không có patience — luôn chạy đủ 4 epoch.
- Nếu gặp lỗi hết bộ nhớ CUDA ở batch 16, notebook tự khởi động lại từ đầu với batch 8 + tích lũy gradient 2 (ô 11); lần chạy chính thức dùng batch 16 (`batch_used: 16`).
- Lịch sử seed 42 (khớp Bảng 3.5 của báo cáo): epoch 1 — loss 0,6488, val macro-F1 0,8427; epoch 2 — 0,3558 / **0,8749** (được lưu); epoch 3 — 0,2433 / 0,8609; epoch 4 — 0,167 / 0,8621.
- Độ biến thiên theo hạt giống (`results/metrics/phobert_seeds.json`): seed 43 đạt đỉnh 0,8694 và seed 44 đạt 0,8978 — **cả hai đều đạt đỉnh ở epoch 4**, khác với seed 42 (epoch 2). Quy tắc chọn checkpoint vì vậy ảnh hưởng thực chất tới mô hình cuối, không chỉ hình thức.

## 8. Đầu ra và đánh giá

- Sau huấn luyện, checkpoint tốt nhất được nạp lại và dự đoán val/test (ô 13); kết quả lưu tại `results/predictions/phobert_val.csv`, `phobert_test.csv` gồm nhãn thật, nhãn dự đoán và xác suất từng lớp (làm tròn 4 chữ số).
- Notebook có assert kiểm tra macro-F1 tính lại từ file predictions khớp với giá trị lúc chọn checkpoint (sai lệch < 1e-3) — một chốt kiểm soát tính nhất quán giữa huấn luyện và suy luận.
- Notebook **không** tính metrics trên test; việc này thuộc Phase 6 (`src/evaluate.py`, đọc các file trong `results/predictions/`), như ghi chú trong metadata: "test predictions only, metrics in Phase 6".

## 9. Áp dụng (inference)

- `src/predict.py` là điểm áp dụng duy nhất: chạy `python -m src.predict "câu bình luận"`; luôn dự đoán bằng mô hình cổ điển tốt nhất trên val (nạp từ `models/*.joblib`), và dự đoán thêm bằng PhoBERT **chỉ khi** tồn tại `models/phobert_best` và máy có `torch`/`transformers` — thiếu một trong hai điều kiện, phần PhoBERT bị bỏ qua lặng lẽ (trả về `None`), không báo lỗi.
- Suy luận cắt tại 128 token, lấy nhãn có xác suất softmax cao nhất kèm xác suất đó; không có ngưỡng từ chối dự đoán (abstention) cho trường hợp xác suất thấp.
- Thư mục `models/` (chứa checkpoint ~540MB và VnCoreNLP jar) **không được commit** theo `src/config.py` ("fitted models (not committed)"). Hệ quả: clone repo về không áp dụng được PhoBERT ngay; phải chạy lại notebook (khoảng 8 phút trên GPU như RTX 3060) để tái tạo `models/phobert_best`.

## 10. Đối chiếu với báo cáo

| Khẳng định trong báo cáo | Thực tế trong code |
|---|---|
| Tách từ bằng RDRSegmenter của VnCoreNLP | Đúng cho lần chạy chính thức, nhưng là tách **lại** trên nền Underthesea và có fallback âm thầm về Underthesea |
| Chỉ 1,5% bình luận dài hơn 128 token | Khớp: 1,52% (`phobert.json`) |
| Bảng kết quả theo epoch, tốt nhất ở epoch 2 | Khớp hoàn toàn với `history` trong `phobert.json` |
| Trọng số lớp 0,843 / 4,267 / 0,633 | Khớp, tính từ phân bố train |
| Huấn luyện trên GPU | Đúng, nhưng là GPU local (RTX 3060), không phải Colab như tên notebook gợi ý |
| Không nhắc trong báo cáo | Upload CSV thủ công khi chạy Colab; cơ chế fallback batch 8 khi OOM; assert tái lập val macro-F1; seed 43/44 đạt đỉnh ở epoch 4 |

## 11. Điểm cần lưu ý khi vận hành (mô tả, không thay đổi)

1. **Fallback tách từ âm thầm**: nếu VnCoreNLP hỏng, đầu vào huấn luyện/áp dụng đổi sang Underthesea mà pipeline vẫn chạy; chỉ có trường `segmenter` trong metadata cho biết điều gì đã xảy ra.
2. **Áp dụng phụ thuộc artifact ngoài git**: checkpoint không nằm trong repo, nên "áp dụng PhoBERT" đồng nghĩa "đã từng chạy notebook trên máy này".
3. **Hai đường tiền xử lý phải khớp nhau**: notebook (huấn luyện) và `predict.py` (áp dụng) cùng cài lại logic resegment ở hai nơi; lệch nhau sẽ làm giảm chất lượng mà không có tín hiệu lỗi.
4. **Tên notebook gây hiểu nhầm môi trường**: quy trình chuẩn trên thực tế là chạy local; Colab chỉ là nhánh dự phòng.
