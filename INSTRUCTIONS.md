# HƯỚNG DẪN THỰC THI CHI TIẾT & VIẾT BÁO CÁO (STEP-BY-STEP INSTRUCTIONS)

## Dự án: Phân loại Cảm xúc Bình luận Sản phẩm Điện Máy Xanh (`dmx-nlp-sentiment`)

Tài liệu này cung cấp hướng dẫn chi tiết từng bước từ cài đặt môi trường, chạy quy trình thu thập dữ liệu, tiền xử lý tiếng Việt, trích chọn đặc trưng (BoW, TF-IDF, Word2Vec), huấn luyện mô hình truyền thống (Naïve Bayes, SVM), xây dựng mô hình hiện đại (**PhoBERT** – bổ sung ý 4), cho tới khung nội dung chi tiết để viết báo cáo theo đúng mẫu **`docs/Bài Tập Lớn I - Lazada.pdf`**.

---

## PHẦN A: CÀI ĐẶT MÔI TRƯỜNG (ENVIRONMENT SETUP)

### 1. Tạo môi trường ảo và cài đặt thư viện

Chạy các lệnh sau tại thư mục gốc của dự án:

```bash
# Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Trên Windows: .venv\Scripts\activate

# Cài đặt toàn bộ thư viện từ file requirements.txt chuẩn của dự án
pip install --upgrade pip
pip install -r requirements.txt
```

---

## PHẦN B: HƯỚNG DẪN THỰC HIỆN 5 BƯỚC + BỔ SUNG Ý 4

---

### BƯỚC 1: Thu thập dữ liệu & Tự động gán nhãn cảm xúc (`auto_label_review.py`)

#### 1.1. Chạy công cụ thu thập dữ liệu (`scripts/crawl/`)

Dự án sử dụng Playwright kết hợp trình duyệt Chromium để thu thập bình luận từ 6 ngành hàng trên Điện Máy Xanh (`Tủ lạnh`, `Nồi cơm điện`, `Máy lạnh`, `Máy giặt`, `Máy lọc nước`, `Tivi`):

```bash
# Kiểm tra thử với 3 sản phẩm
python3 scripts/crawl/crawl_dmx_reviews.py --test 3

# Chạy thu thập đa luồng có cơ chế tự dừng khi gặp chặn (403/429/captcha) và tự resume
python3 scripts/crawl/protected_multi_thread_crawler.py --workers 2

# Chuyển đổi dữ liệu từ JSONL sang CSV
python3 scripts/crawl/json_to_csv.py
```

#### 1.2. Thống kê tập dữ liệu thu thập được (`data/raw/reviews.csv`)

- Tổng số dòng thu thập: **12,189 bình luận** (`9,516` bình luận duy nhất hợp lệ sau khi khử 49 dòng rỗng/rác và 2,624 dòng trùng lặp).
- Phân bố theo ngành hàng:
  - **Tủ lạnh**: 3,366 bình luận (27.6%)
  - **Nồi cơm điện**: 2,796 bình luận (22.9%)
  - **Máy lạnh**: 2,030 bình luận (16.7%)
  - **Máy giặt**: 1,725 bình luận (14.2%)
  - **Máy lọc nước**: 1,724 bình luận (14.1%)
  - **Tivi**: 548 bình luận (4.5%)
- Phân bố theo số sao gốc (Rating):
  - **5 sao**: 6,741 | **4 sao**: 2,407 $\rightarrow$ Nhãn **Positive (`1`)**: **9,148 mẫu (75.05%)**
  - **3 sao**: 988 $\rightarrow$ Nhãn **Neutral (`0`)**: **988 mẫu (8.11%)**
  - **2 sao**: 706 | **1 sao**: 1,347 $\rightarrow$ Nhãn **Negative (`-1`)**: **2,053 mẫu (16.84%)**

#### 1.3. Quy trình tạo nhãn `label_target` bằng `auto_label_review.py` (Unsupervised + Star Reconciliation)

Thay vì chỉ gán nhãn máy móc theo số sao hoặc phải đọc thủ công toàn bộ 12,189 dòng, dự án cung cấp script **`auto_label_review.py`** (mã nguồn lõi tại `src/data/auto_labeler.py` & `scripts/auto_label_review.py`) hỗ trợ 4 chế độ đánh giá cảm xúc không giám sát / bán giám sát:

1. **`--mode hybrid-lexicon` *(Mặc định – khuyên dùng)***: Kết hợp từ điển cảm xúc miền Điện máy + Quy tắc đảo cực từ phủ định (`không`, `chưa`, `chẳng`) + Trọng số mệnh đề sau liên từ chuyển ý (`nhưng`, `tuy_nhiên`, `mỗi_tội`) để đối chiếu với số sao (`rating`).
2. **`--mode embedding-prototype`**: Dùng Sentence Embeddings (`keepitreal/vietnamese-sbert`) tính Cosine Similarity với 3 câu hạt giống (Prototypes) đại diện cho 3 nhãn.
3. **`--mode zero-shot`**: Dùng mô hình Multilingual NLI (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`).
4. **`--mode pretrained`**: Dùng Teacher Model tiếng Việt có sẵn (`wonrax/phobert-base-vietnamese-sentiment`).

**Cách chạy**:

```bash
# Chạy chế độ mặc định (Hybrid Unsupervised Lexicon + Star Reconciliation)
python3 auto_label_review.py

# Hoặc chạy với chế độ mở rộng:
python3 auto_label_review.py --mode embedding-prototype
```

**Kết quả sinh ra tự động**:

- `data/interim/reviews_labeled.csv`: Toàn bộ **9,516** bình luận duy nhất đã tách từ và gán nhãn `label_target` (`-1: 2,415`, `0: 633`, `1: 6,468`).
- `data/interim/rating_text_conflicts.csv`: Danh sách **490 bình luận bị lệch giữa số sao và nội dung câu chữ** đã được phát hiện và hiệu chỉnh (rất hữu ích để trích dẫn ví dụ thực tế vào Báo cáo!).
- `data/processed/train.csv` (**7,664 mẫu**), `data/processed/val.csv` (**1,352 mẫu**), `data/processed/test_gold.csv` (**500 mẫu**).

---

### BƯỚC 2: Tiền xử lý dữ liệu văn bản tiếng Việt (Text Preprocessing)

Dưới đây là mã nguồn chuẩn hóa quy trình tiền xử lý tiếng Việt (có thể lưu vào `src/preprocess.py` hoặc chạy trong Notebook):

```python
import re
import unicodedata
from pyvi import ViTokenizer
from langdetect import detect, LangDetectException

# 1. Từ điển chuẩn hóa viết tắt / teencode phổ biến trên Điện Máy Xanh
TEENCODE_MAP = {
    "đmx": "điện_máy_xanh",
    "dmx": "điện_máy_xanh",
    "sp": "sản_phẩm",
    "nv": "nhân_viên",
    "ko": "không",
    "k": "không",
    "kh": "không",
    "hok": "không",
    "đc": "được",
    "dc": "được",
    "ok": "tốt",
    "oke": "tốt",
    "okay": "tốt",
    "good": "tốt",
    "bh": "bảo_hành",
    "sd": "sử_dụng",
    "ae": "anh_em",
    "mn": "mọi_người",
    "mk": "mình",
    "sz": "kích_cỡ",
    "vs": "với",
    "cx": "cũng",
    "r": "rồi",
    "wa": "quá",
    "qá": "quá",
}

# 2. Danh sách từ phủ định & từ chỉ mức độ cảm xúc BẮT BUỘC GIỮ LẠI khi lọc stopword
NEGATION_AND_SENTIMENT_WORDS = {
    "không", "chưa", "chẳng", "chả", "đừng", "chớ",
    "rất", "quá", "lắm", "hơi", "khá", "cực", "cực_kỳ",
    "tốt", "tệ", "kém", "ổn", "đẹp", "xấu", "lỗi", "hỏng", "ồn", "êm",
    "nhưng", "tuy_nhiên", "bị", "được", "nên", "Đáng"
}

def load_stopwords(filepath="data/stopwords_vi.txt"):
    """Đọc danh sách stopword và loại bỏ các từ phủ định/cảm xúc khỏi danh sách cần xóa."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            stopwords = set(line.strip().replace(" ", "_") for line in f if line.strip())
    except FileNotFoundError:
        # Danh sách hư từ cơ bản nếu chưa tải file ngoài
        stopwords = {
            "thì", "là", "mà", "và", "của", "những", "các", "một", "này", "đó",
            "cho", "với", "trong", "khi", "để", "từ", "về", "như", "theo", "do",
            "tại", "vì", "nếu", "hay", "hoặc", "vào", "ra", "lên", "xuống"
        }
    return stopwords - NEGATION_AND_SENTIMENT_WORDS

def is_vietnamese(text: str) -> bool:
    """Lọc giữ lại bình luận tiếng Việt bằng langdetect (như báo cáo mẫu)."""
    try:
        return detect(text) == "vi"
    except LangDetectException:
        return False

def preprocess_vietnamese_text(text: str, stopwords: set = None, remove_stopwords: bool = True) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    # B1: Chuẩn hóa Unicode dựng sẵn NFC
    text = unicodedata.normalize("NFC", text)
    # B2: Chuyển về chữ thường
    text = text.lower()
    # B3: Xóa URL, HTML tags, email, số điện thoại
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    # B4: Rút gọn các ký tự kéo dài liên tiếp (vd: "tốtttt" -> "tốt")
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    # B5: Chỉ giữ lại chữ cái tiếng Việt và khoảng trắng
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text)
    text = re.sub(r"\d+", " ", text)
    # B6: Chuẩn hóa từ viết tắt / teencode
    tokens = [TEENCODE_MAP.get(tok, tok) for tok in text.split()]
    text = " ".join(tokens)
    # B7: Tách từ tiếng Việt (Word Segmentation) nối từ ghép bằng dấu "_"
    segmented_text = ViTokenizer.tokenize(text)
    # B8: Lọc stop words (có bảo toàn từ phủ định)
    words = segmented_text.split()
    if remove_stopwords and stopwords:
        words = [w for w in words if w not in stopwords and len(w) > 1]
    return " ".join(words)
```

> **Lưu ý thiết kế thí nghiệm đối chứng (Ablation Study)**:
> Tham khảo cách làm từ bộ biểu đồ mẫu trong `figures/` (`cm_tf_idf_nb_khong_bo_stopword_tham_khao.png`), nhóm nên chạy thêm 1 thí nghiệm đối chứng giữa `TF-IDF + NB` **có bỏ stopword** và **không bỏ stopword**. Việc bỏ stopword có chọn lọc (giữ lại từ phủ định) thường giúp tăng độ chính xác tổng thể (Accuracy) và giảm kích thước từ điển, trong khi nếu xóa nhầm các từ như *"không"*, *"chưa"*, *"hơi"* sẽ làm giảm mạnh Recall của lớp `Neutral` và `Negative`.

---

### BƯỚC 3: Trích chọn đặc trưng (Feature Extraction: BoW, TF-IDF, Word Embeddings)

Triển khai cả 3 phương pháp trích chọn đặc trưng theo đúng yêu cầu:

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# ==========================================================
# 1. Phương pháp Bag of Words (BoW)
# ==========================================================
bow_vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=15000
)
X_train_bow = bow_vectorizer.fit_transform(train_df["clean_text"])
X_val_bow   = bow_vectorizer.transform(val_df["clean_text"])
X_test_bow  = bow_vectorizer.transform(test_gold_df["clean_text"])

# ==========================================================
# 2. Phương pháp TF-IDF
# ==========================================================
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    max_features=15000,
    norm="l2"
)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["clean_text"])
X_val_tfidf   = tfidf_vectorizer.transform(val_df["clean_text"])
X_test_tfidf  = tfidf_vectorizer.transform(test_gold_df["clean_text"])

# ==========================================================
# 3. Phương pháp Word Embeddings (Word2Vec Mean Pooling)
# ==========================================================
train_sentences = [doc.split() for doc in train_df["clean_text"]]
w2v_model = Word2Vec(
    sentences=train_sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,          # Skip-gram phù hợp với từ hiếm tiếng Việt
    epochs=15,
    seed=42
)

def sentence_to_w2v(doc: str, model: Word2Vec, dim: int = 100) -> np.ndarray:
    words = [w for w in doc.split() if w in model.wv]
    if not words:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(model.wv[words], axis=0)

X_train_w2v = np.vstack([sentence_to_w2v(d, w2v_model) for d in train_df["clean_text"]])
X_val_w2v   = np.vstack([sentence_to_w2v(d, w2v_model) for d in val_df["clean_text"]])
X_test_w2v  = np.vstack([sentence_to_w2v(d, w2v_model) for d in test_gold_df["clean_text"]])
```

---

### BƯỚC 4A: Xây dựng và Huấn luyện Mô hình Học máy Truyền thống (Naïve Bayes & SVM)

Kết hợp kỹ thuật xử lý mất cân bằng dữ liệu (`Hybrid Sampling` cho Naïve Bayes và `class_weight="balanced"` cho SVM):

```python
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# 0. Mô hình Baseline (Dự đoán lớp đông nhất - Positive)
baseline_clf = DummyClassifier(strategy="most_frequent")
baseline_clf.fit(X_train_bow, y_train)

# Xử lý mất cân bằng bằng Hybrid Sampling cho Naive Bayes (như báo cáo mẫu trang 6):
# Giảm bớt nhãn đa số (1) và tăng cường nhãn thiểu số (-1, 0)
sampler = ImbPipeline([
    ("under", RandomUnderSampler(sampling_strategy={1: 2500}, random_state=42)),
    ("over", RandomOverSampler(sampling_strategy={-1: 1500, 0: 1200}, random_state=42))
])

# 1. BoW + Naive Bayes & BoW + SVM
X_tr_bow_res, y_tr_bow_res = sampler.fit_resample(X_train_bow, y_train)
nb_bow = MultinomialNB(alpha=1.0).fit(X_tr_bow_res, y_tr_bow_res)
svm_bow = LinearSVC(class_weight="balanced", C=0.5, random_state=42).fit(X_train_bow, y_train)

# 2. TF-IDF + Naive Bayes & TF-IDF + SVM
X_tr_tfidf_res, y_tr_tfidf_res = sampler.fit_resample(X_train_tfidf, y_train)
nb_tfidf = MultinomialNB(alpha=0.5).fit(X_tr_tfidf_res, y_tr_tfidf_res)
svm_tfidf = LinearSVC(class_weight="balanced", C=1.0, random_state=42).fit(X_train_tfidf, y_train)

# 3. Word2Vec + Naive Bayes (GaussianNB cho đặc trưng liên tục âm/dương) & Word2Vec + SVM
X_tr_w2v_res, y_tr_w2v_res = sampler.fit_resample(X_train_w2v, y_train)
nb_w2v = GaussianNB().fit(X_tr_w2v_res, y_tr_w2v_res)
svm_w2v = LinearSVC(class_weight="balanced", C=1.0, random_state=42).fit(X_train_w2v, y_train)
```

---

### BƯỚC 4B (BỔ SUNG Ý 4): Xây dựng Mô hình Hiện đại – Fine-tune PhoBERT (`vinai/phobert-base-v2`)

#### 1. Tại sao chọn Fine-tune PhoBERT làm mô hình hiện đại?

- **Khắc phục nhược điểm của BoW / TF-IDF / Word2Vec trung bình**:
  - Trong câu *"Máy giặt chạy không êm nhưng vắt rất khô"*, BoW/TF-IDF tách rời các từ nên dễ bị nhiễu bởi từ *"êm"* và *"rất"*, còn Word2Vec trung bình cộng làm triệt tiêu thông tin vị trí.
  - **PhoBERT** (Pre-trained RoBERTa cho tiếng Việt của VinAI Research) sử dụng **12 tầng Transformer Encoder** với cơ chế **Multi-Head Self-Attention** giúp biểu diễn từng từ phụ thuộc vào ngữ cảnh toàn câu.
- **Khác biệt so với báo cáo mẫu Lazada**:
  - Báo cáo mẫu chỉ lấy PhoBERT làm bộ trích chọn đặc trưng cố định rồi đưa vào SVM (trang 5), dẫn đến kết quả thấp (`Macro-F1 = 0.4407`).
  - Trong dự án này, chúng ta **Fine-tune toàn bộ mạng PhoBERT (End-to-End)** cùng tầng phân loại tuyến tính và hàm mất mát **Weighted Cross-Entropy Loss** chống mất cân bằng lớp.

#### 2. Mã nguồn huấn luyện PhoBERT chuẩn

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = "vinai/phobert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ánh xạ nhãn {-1: 0 (Negative), 0: 1 (Neutral), 1: 2 (Positive)}
LABEL2ID = {-1: 0, 0: 1, 1: 2}
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

class DMXReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        # Lưu ý: Đầu vào cho PhoBERT phải là câu ĐÃ TÁCH TỪ bằng pyvi/underthesea
        # nhưng KHÔNG xóa stopword để giữ nguyên cấu trúc ngữ pháp cho Transformer
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
        self.labels = [LABEL2ID[y] for y in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Tính trọng số lớp ngược với tần suất xuất hiện để xử lý mất cân bằng dữ liệu
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=np.array([LABEL2ID[y] for y in y_train])
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
    }

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id={v: k for k, v in ID2LABEL.items()}
)

training_args = TrainingArguments(
    output_dir="./results_phobert",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=torch.cuda.is_available(),
    seed=42
)
```

---

### BƯỚC 5: Đánh giá Mô hình, Vẽ Biểu đồ & Bảng kết quả Tham khảo (Evaluation & Visualization)

> ⚠️ **LƯU Ý QUAN TRỌNG VỀ THƯ MỤC `figures/`**:
> Các biểu đồ hiện có trong thư mục `figures/` (`cm_*.png` và `macro_f1_val_test.png`) **chỉ là các biểu đồ mẫu tham khảo từ một repository khác có cùng chủ đề**. Khi thực hiện dự án trên bộ dữ liệu Điện Máy Xanh (`data/raw/reviews.csv`), nhóm sử dụng đoạn code dưới đây để **sinh lại toàn bộ các biểu đồ mới từ kết quả thực tế của Điện Máy Xanh** theo đúng chuẩn trình bày của bộ hình tham khảo.

#### 5.1. Mã nguồn xuất biểu đồ Confusion Matrix (2 panel) và Biểu đồ so sánh Macro-F1

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
LABELS_ORDER = [-1, 0, 1]

def plot_dual_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    """
    Vẽ Confusion Matrix gồm 2 bảng cạnh nhau giống mẫu tham khảo trong /figures:
    - Trái: Số đếm mẫu tuyệt đối (Counts)
    - Phải: Chuẩn hóa theo hàng (Recall %)
    """
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDER)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(title, fontsize=14, y=1.02)

    # Bảng 1: Số đếm
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0],
        annot_kws={"size": 12}
    )
    axes[0].set_title("Số đếm", fontsize=11)
    axes[0].set_xlabel("Dự đoán")
    axes[0].set_ylabel("Nhãn gold")

    # Bảng 2: Chuẩn hóa theo hàng (Recall %)
    annot_pct = np.array([[f"{val*100:.0f}%" for val in row] for row in cm_norm])
    sns.heatmap(
        cm_norm, annot=annot_pct, fmt="", cmap="Blues", cbar=False,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1],
        vmin=0.0, vmax=1.0, annot_kws={"size": 12}
    )
    axes[1].set_title("Chuẩn hóa theo hàng (recall)", fontsize=11)
    axes[1].set_xlabel("Dự đoán")
    axes[1].set_ylabel("Nhãn gold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_macro_f1_comparison(model_names, val_scores, test_scores, save_path="figures/macro_f1_val_test.png"):
    """
    Vẽ biểu đồ thanh ngang so sánh Macro-F1 trên tập Val (nhãn sao) và Test (nhãn gold).
    """
    y = np.arange(len(model_names))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.8))
    rects1 = ax.barh(y - height/2, val_scores, height, label="Val (nhãn sao)", color="#2b7bba")
    rects2 = ax.barh(y + height/2, test_scores, height, label="Test (nhãn gold)", color="#eb6b34")

    ax.set_xlabel("Macro-F1")
    ax.set_title("Macro-F1 theo mô hình: val (nhãn sao) và test (nhãn gold)", loc="left", fontsize=13)
    ax.set_yticks(y)
    ax.set_yticklabels(model_names)
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)

    for rect in rects2:
        w = rect.get_width()
        ax.annotate(f"{w:.3f}", xy=(w + 0.01, rect.get_y() + rect.get_height()/2),
                    va="center", ha="left", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

---

#### 5.2. Bảng số liệu mẫu (Trích xuất từ bộ biểu đồ tham khảo trong `figures/` của repo cùng chủ đề)

> *Ghi chú: Hai bảng dưới đây được tính toán từ các ma trận nhầm lẫn tham khảo hiện có trong `figures/` (`n = 500$: `289 Negative`, `47 Neutral`, `164 Positive`) để minh họa chuẩn trình bày bảng số liệu trong Báo cáo. Sau khi chạy thực nghiệm trên dữ liệu Điện Máy Xanh, nhóm chỉ cần điền kết quả mới vào đúng cấu trúc bảng này.*

##### Bảng A: Tổng hợp hiệu năng các mô hình (Mẫu tham khảo từ `figures/`)

| Mô hình thực nghiệm                             |     Accuracy     |     Macro-F1     |   Weighted-F1   | Macro-Precision |   Macro-Recall   | File biểu đồ mẫu trong `figures/`                  |
| :-------------------------------------------------- | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :------------------------------------------------------- |
| **0. Baseline (Lớp đông nhất)**           |      0.3280      |      0.1647      |      0.1620      |      0.1093      |      0.3333      | `figures/cm_baseline_lop_dong_nhat.png`                |
| **1. BoW + Naïve Bayes**                     |      0.6540      |      0.5677      |      0.6924      |      0.6077      |      0.6016      | `figures/cm_bow_nb.png`                                |
| **2. BoW + SVM**                              |      0.6820      |      0.5812      |      0.7000      |      0.6092      |      0.6281      | `figures/cm_bow_svm.png`                               |
| **3. TF-IDF + Naïve Bayes**                  | **0.7420** | **0.6329** | **0.7639** |      0.6381      |      0.6660      | `figures/cm_tf_idf_nb.png`                             |
| **4. TF-IDF + SVM**                           |      0.7340      |      0.6255      |      0.7531      |      0.6321      |      0.6623      | `figures/cm_tf_idf_svm.png`                            |
| **5. Word2Vec + Naïve Bayes**                |      0.6460      |      0.5829      |      0.6890      |      0.6118      |      0.6318      | `figures/cm_w2v_nb.png`                                |
| **6. Word2Vec + SVM**                         |      0.6160      |      0.5710      |      0.6670      |      0.6384      |      0.6362      | `figures/cm_w2v_svm.png`                               |
| **7. PhoBERT (Mô hình hiện đại - Ý 4)** |      0.7140      | **0.6320** |      0.7515      | **0.6623** | **0.6831** | `figures/cm_phobert.png`                               |
| *(Tham khảo) TF-IDF + NB (Không bỏ stopword)*  |    *0.7140*    |    *0.6389*    |    *0.7497*    |    *0.6625*    |    *0.6965*    | `figures/cm_tf_idf_nb_khong_bo_stopword_tham_khao.png` |

##### Bảng B: Chi tiết Precision / Recall / F1 theo từng lớp cảm xúc (Mẫu tham khảo từ `figures/`)

| Mô hình                |   Negative (`-1`, $n=289$) P / R / F1   |     Neutral (`0`, $n=47$) P / R / F1     |         Positive (`1`, $n=164$) P / R / F1         |
| :----------------------- | :------------------------------------------: | :------------------------------------------: | :----------------------------------------------------: |
| **BoW + NB**       |           0.9461 / 0.5467 / 0.6930           |           0.1145 / 0.3191 / 0.1685           |                0.7624 / 0.9390 / 0.8415                |
| **BoW + SVM**      |      **0.9940** / 0.5744 / 0.7281      |           0.1739 / 0.3404 / 0.2302           |                0.6598 / 0.9695 / 0.7852                |
| **TF-IDF + NB**    | 0.9474 / **0.6851** / **0.7952** |           0.1868 / 0.3617 / 0.2464           |                0.7800 / 0.9512 / 0.8571                |
| **TF-IDF + SVM**   |           0.9602 / 0.6678 / 0.7878           | **0.1954** / 0.3617 / **0.2537** |                0.7406 / 0.9573 / 0.8351                |
| **Word2Vec + NB**  |           0.8731 / 0.5952 / 0.7078           |      0.1748 / 0.5319 / **0.2632**      |                0.7875 / 0.7683 / 0.7778                |
| **Word2Vec + SVM** |           0.9517 / 0.4775 / 0.6359           |      0.1453 / **0.5532** / 0.2301      |                0.8182 / 0.8780 / 0.8471                |
| **PhoBERT (Ý 4)** |           0.9887 / 0.6055 / 0.7511           |           0.1692 / 0.4681 / 0.2486           | **0.8290** / **0.9756** / **0.8964** |

---

## PHẦN C: HƯỚNG DẪN VIẾT BÁO CÁO BÁM SÁT MẪU `Bài Tập Lớn I - Lazada.pdf`

Khi soạn thảo file Word/PDF báo cáo nộp cho giảng viên, nhóm hãy trình bày theo đúng bố cục 4 Phần lớn dưới đây:

### Trang bìa & Mục lục + Bảng đóng góp thành viên

- **Tiêu đề**: HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG – BỘ MÔN PHÂN TÍCH VÀ KHAI PHÁ DỮ LIỆU VĂN BẢN – BÁO CÁO BÀI TẬP LỚN I.
- **Bảng đóng góp thành viên**: Đặt ngay dưới mục lục (trang 2) gồm 4 cột: `STT | Họ tên | Mã sinh viên | Công việc | Mức độ đóng góp (25%)`.

### Phần I. Đề bài

- Xây dựng hệ thống phân loại cảm xúc bình luận sản phẩm trên sàn thương mại điện tử **Điện Máy Xanh**.
- Quy ước 3 nhãn: **Tích cực (`1`)**, **Trung tính (`0`)**, **Tiêu cực (`-1`)**.
- Liệt kê 5 bước thực hiện chuẩn + **Nội dung mở rộng ở Bước 4**: Xây dựng mô hình học sâu hiện đại **PhoBERT** giải quyết bài toán.

### Phần II. Cơ sở lý thuyết

Trình bày ngắn gọn, súc tích theo đúng phong cách báo cáo mẫu (khoảng 3–4 trang):

1. **Phương pháp Bag of Words (BoW)**: Khái niệm, Ý tưởng, Cách hoạt động, Ưu điểm, Nhược điểm (mất ngữ cảnh, ma trận thưa, OOV).
2. **Phương pháp TF-IDF**: Khái niệm, Công thức $TF(t,d)$, công thức làm mượt $IDF(t, D) = \ln\frac{1 + |D|}{1 + df(d, t)} + 1$, công thức chuẩn hóa $L_2$ của `scikit-learn`, Ưu/Nhược điểm.
3. **Phương pháp Word Embeddings (Word2Vec)**: Biểu diễn vector dày đặc (dense vectors) khoảng 100–300 chiều, kiến trúc Skip-gram/CBOW, cách lấy trung bình vector từ (Mean Pooling) cho văn bản, Ưu/Nhược điểm.
4. **Thuật toán phân loại Naïve Bayes**: Định lý Bayes $P(y|x_1,\dots,x_n) \propto P(y)\prod_{i=1}^n P(x_i|y)$, biến thể `MultinomialNB` (dùng cho BoW, TF-IDF) và `GaussianNB` (dùng cho Word2Vec).
5. **Thuật toán Support Vector Machine (SVM)**: Nguyên tắc tìm siêu phẳng (hyperplane) có lề (margin) cực đại, biến thể `LinearSVC` kết hợp `class_weight="balanced"`.
6. **Mô hình hiện đại PhoBERT (`vinai/phobert-base-v2`) *(Phần bổ sung Ý 4)***:
   - Giới thiệu kiến trúc Transformer Encoder hai chiều (RoBERTa) huấn luyện trước trên 20GB dữ liệu tiếng Việt đã tách từ.
   - Cơ chế Self-Attention giúp nắm bắt ngữ cảnh dài, từ phủ định và cấu trúc câu phức tạp.
   - Kỹ thuật Fine-tuning End-to-End với Classification Head và hàm mất mát có trọng số (`Weighted Cross-Entropy`).
7. **Xử lý vấn đề mất cân bằng dữ liệu**: Nêu thống kê thực tế của bộ dữ liệu Điện Máy Xanh (nhãn 4–5 sao chiếm 75.05%, nhãn 3 sao chỉ chiếm 8.11%, nhãn 1–2 sao chiếm 16.84%) và giải pháp áp dụng cho từng mô hình.
8. **Thư viện hỗ trợ**: `playwright`, `pyvi`/`underthesea`, `langdetect`, `scikit-learn`, `gensim`, `imbalanced-learn`, `transformers` & `pytorch`.

### Phần III. Thực nghiệm và Kết quả

1. **Code thực nghiệm & Thiết kế tập dữ liệu**:
   - Đưa link GitHub repository chứa code thu thập dữ liệu, tiền xử lý và huấn luyện mô hình.
   - Giải thích rõ quy trình chia tập dữ liệu: Tập **Validation** đánh giá trên nhãn sao (Silver label) và Tập **Test (`n=500`)** được gán nhãn thủ công (**Gold label**) để phản ánh trung thực chất lượng mô hình.
2. **Kết quả thực nghiệm**:
   - Chèn lần lượt các hình ảnh Confusion Matrix **đã được sinh mới từ tập dữ liệu Điện Máy Xanh** (sau khi chạy hàm `plot_dual_confusion_matrix` và `plot_macro_f1_comparison` ở Mục 5.1 để ghi đè lên các hình mẫu tham khảo trong `figures/`):
     - `figures/cm_baseline_lop_dong_nhat.png`
     - `figures/cm_bow_nb.png` & `figures/cm_bow_svm.png`
     - `figures/cm_tf_idf_nb.png` & `figures/cm_tf_idf_svm.png`
     - `figures/cm_w2v_nb.png` & `figures/cm_w2v_svm.png`
     - `figures/cm_phobert.png`
     - Biểu đồ tổng hợp `figures/macro_f1_val_test.png`
3. **So sánh và nhận xét** *(Điền số liệu thực tế của Điện Máy Xanh vào cấu trúc Bảng A & B ở Mục 5.2 và phân tích theo 4 luận điểm mẫu dưới đây)*:
   - **Về phương pháp trích chọn đặc trưng**: **TF-IDF** vượt trội rõ rệt so với cả **BoW** và **Word2Vec Mean Pooling** ở cả 2 thuật toán NB và SVM (`Macro-F1` đạt `0.633` với NB và `0.625` với SVM, so với `0.568`–`0.581` của BoW và `0.571`–`0.583` của Word2Vec). Lý do là TF-IDF giữ được các cụm từ khóa phân biệt trực tiếp (nhờ bigram + trọng số IDF), trong khi phép lấy trung bình cộng (Mean Pooling) của Word2Vec làm mờ đi các từ khóa cảm xúc khi câu dài.
   - **Về thuật toán truyền thống (NB vs SVM)**: Với đặc trưng BoW, SVM tốt hơn NB (`0.581` vs `0.568`). Với đặc trưng TF-IDF đã chuẩn hóa và xử lý mất cân bằng, cả NB (`0.633`) và SVM (`0.625`) đều đạt hiệu năng rất cạnh tranh và vượt xa Baseline (`0.165`).
   - **Về mô hình hiện đại PhoBERT (Ý 4 bổ sung)**:
     - Trên tập **Validation (nhãn sao)**, **PhoBERT đạt Macro-F1 cao nhất tuyệt đối (~`0.78`)**, bỏ xa mọi mô hình truyền thống (xem cột màu xanh trên hình `figures/macro_f1_val_test.png`).
     - Trên tập **Test Gold (`n=500`)**, PhoBERT đạt **Macro-F1 = `0.6320`**, **Macro-Recall cao nhất (`0.6831`)**, **Precision lớp Negative đạt `98.87%` (175/177)** và **F1 lớp Positive cao nhất toàn bộ các mô hình (`0.8964`, Recall `97.56%`)**.
   - **Phân tích hiện tượng chênh lệch giữa Val (nhãn sao) và Test (nhãn gold)**:
     - Mọi mô hình đều có độ sụt giảm khoảng `0.10 - 0.15` điểm Macro-F1 khi chuyển từ `Val (nhãn sao)` sang `Test (nhãn gold)`. Nguyên nhân là do các mô hình được huấn luyện trên nhãn sao (nơi nhiều bình luận chê nhẹ hoặc góp ý vẫn được người dùng chấm 4–5 sao hoặc 3 sao), dẫn đến ranh giới giữa `Negative` và `Neutral` bị dịch chuyển (ví dụ ở PhoBERT: 105 mẫu `Negative` được dự đoán sang `Neutral`, nhưng chỉ có 9 mẫu bị nhầm sang `Positive`).

### Phần IV. Kết luận

1. **Kết quả đạt được**: Tổng kết việc hoàn thành toàn bộ quy trình từ thu thập 12,189 bình luận Điện Máy Xanh, tiền xử lý tiếng Việt, thử nghiệm 3 bộ đặc trưng (BoW, TF-IDF, Word2Vec) với 2 thuật toán (NB, SVM) đến triển khai thành công mô hình Transformer hiện đại **PhoBERT**.
2. **Các hạn chế & Hướng cải thiện**:
   - **Nhiễu nhãn huấn luyện (Noisy Silver Labels)**: Tập Train vẫn dựa vào số sao người dùng chấm nên chứa nhiễu giữa lớp `Neutral` (3 sao) và `Negative`/`Positive`. $\rightarrow$ *Hướng cải thiện*: Áp dụng kỹ thuật **Confident Learning (`cleanlab`)** hoặc dùng LLM để lọc/gán lại nhãn tập huấn luyện trước khi train.
   - **Mất cân bằng nghiêm trọng ở lớp `Neutral` (chỉ ~8%)**: Khiến Precision của lớp `Neutral` còn thấp ở mọi mô hình. $\rightarrow$ *Hướng cải thiện*: Tăng cường dữ liệu (Data Augmentation / Back-translation / Paraphrasing) cho lớp `Neutral` hoặc dùng hàm mất mát **Focal Loss**.
   - **Cải tiến biểu diễn Word Embeddings**: Thay Mean Pooling đơn giản bằng **TF-IDF Weighted Word2Vec** hoặc kết hợp **BiLSTM/CNN** trên đỉnh tầng Embedding.
