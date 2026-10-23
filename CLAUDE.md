# CLAUDE.md — Quick Reference for AI Coding Agents (`dmx-nlp-sentiment`)

This file provides a fast entry point for Claude Code and other AI coding agents working in this repository. For full rules and step-by-step workflows, read:

- **[`AGENTS.md`](AGENTS.md)**: Project context, 5 critical guardrails/invariants, and quality checklist.
- **[`SKILLS.md`](SKILLS.md)**: 7 modular agent skills (`skill:01-crawl-dmx` through `skill:07-eval-plot-report`).
- **[`PLAN.md`](PLAN.md)**: Project roadmap, directory architecture, and 4-member PTIT team task division.
- **[`INSTRUCTIONS.md`](INSTRUCTIONS.md)**: Complete Python templates for Vietnamese NLP preprocessing, feature extraction (BoW, TF-IDF, Word2Vec), Naïve Bayes & SVM training, end-to-end **PhoBERT** fine-tuning, visualization functions, and PTIT report template.

---

## Key Repository Facts & Guardrails

1. **Dataset (`data/raw/reviews.csv` & `reviews.jsonl`)**:
   - Real crawled dataset of **12,189 Điện Máy Xanh product reviews** across 6 categories (`Tủ lạnh`, `Nồi cơm điện`, `Máy lạnh`, `Máy giặt`, `Máy lọc nước`, `Tivi`).
   - **Never overwrite or delete files in `data/raw/`**. Write all processed/split datasets (`train.csv`, `val.csv`, `test_gold.csv`) to `data/processed/`.
2. **Reference Figures (`figures/*.png`)**:
   - ⚠️ **Important**: The existing `.png` files in `figures/` are **reference charts from another repository on the same topic** (biểu đồ tham khảo từ một repo khác cùng chủ đề).
   - Use them only as the **visual format specification** (2-panel Confusion Matrix: raw counts + row-normalized recall %, and horizontal Macro-F1 comparison bar chart). Overwrite them with newly generated charts after running the experimental pipeline on the Điện Máy Xanh dataset.
3. **Reference Report (`docs/Bài Tập Lớn I - Lazada.pdf`)**:
   - Official PTIT report template (for the Lazada sentiment analysis topic). Follow its exact 4-section structure (`I. Đề bài`, `II. Cơ sở lý thuyết`, `III. Thực nghiệm và kết quả`, `IV. Kết luận`) while adding the **Requirement 4 Extension: Modern Transformer Model (`PhoBERT`)**.
4. **Vietnamese NLP Rules**:
   - Always normalize Unicode to `NFC` first.
   - Always segment Vietnamese compound words with `_` using `pyvi` or `underthesea`.
   - **Never remove negation or sentiment-intensity words** (`không`, `chưa`, `chẳng`, `rất`, `quá`, `hơi`, `kém`, `tệ`, `tốt`, `lỗi`, `nhưng`) during stopword filtering.
   - For `PhoBERT` (`vinai/phobert-base-v2`), pass word-segmented text **with stopwords intact**.
5. **Zero Data Leakage**:
   - Fit `CountVectorizer`, `TfidfVectorizer`, `Word2Vec`, and class-imbalance resamplers (`RandomOverSampler`/`RandomUnderSampler`) **exclusively on `train`**, never on `val` or `test_gold`.
