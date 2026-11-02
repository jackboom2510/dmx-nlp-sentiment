# Agent guidance — DMX sentiment project

## Project goal
Build and evaluate a Vietnamese three-way sentiment classifier for Điện Máy Xanh reviews. Read `README.md`, `PLAN.md`, `INSTRUCTIONS.md`, and `notebooks/01_sentiment_analysis_report.ipynb` before changing pipeline/reporting behavior.

## Non-negotiable data and evaluation rules
- Never overwrite or delete `data/raw/` content unless the user explicitly requests it.
- Current `label_target` values are hybrid pseudo-labels from ratings + lexicon/rules. `data/processed/test.csv` is **not** human-gold. Do not call its metrics gold accuracy or claim they measure true sentiment accuracy.
- Keep train/validation/test target definitions and preprocessing consistent. Select models on validation; do not tune on the held-out test.
- Fit vocabularies, TF-IDF, embeddings, sampling and learned preprocessing on train only.
- Keep `-1=Negative`, `0=Neutral`, `1=Positive`. Preserve negation/intensity/contrast words in cleaning.
- Reference plots in `figures/reference/` are from another repository. Never present them as results from the DMX dataset.
- Report uncertainty and class-level behavior (Macro-F1, per-class scores, support, confusion matrix); Accuracy alone is insufficient.

## Model guidance
- Classical comparisons: majority baseline, BoW/TF-IDF/Word2Vec with NB and LinearSVC (`scripts/train_classical.py`).
- Transformer runner: `scripts/train_transformer.py`; PhoBERT consumes segmented text, XLM-R uses raw review content by default. XLM-R Large is a strong modern comparator, not automatically “SOTA”.
- Use validation Macro-F1 for checkpoint selection and early stopping. Record model ID, seed, max length, hyperparameters and hardware.

## Reproducibility and reports
- Regenerate data with `python auto_label_review.py`; train with documented commands in `INSTRUCTIONS.md`.
- Keep generated large checkpoints out of Git. Metrics/predictions are lightweight artifacts under `outputs/`.
- For real-world performance claims, use independently annotated labels. `scripts/export_annotation_sample.py` exports a blinded, stratified sheet; the notebook computes inter-annotator agreement after labels are filled.
- Update README / plan / instructions when behavior, filenames, assumptions or outputs change.
- Run syntax and notebook JSON checks after code edits; do not claim GPU training or results unless actually executed.
