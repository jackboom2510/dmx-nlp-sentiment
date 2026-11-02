# Claude Code quick reference

Start with `README.md`, `INSTRUCTIONS.md`, and `notebooks/01_sentiment_analysis_report.ipynb` for the project overview, run commands and report flow.

## Data and label caveat
- Raw reviews live in `data/raw/`; do not overwrite them.
- `label_target` and all current split labels are hybrid auto-labels. `data/processed/test.csv` is held-out pseudo-label data, **not a human-gold test**. State this prominently in analysis and reports.
- Label IDs: `-1=Negative`, `0=Neutral`, `1=Positive`.
- `figures/reference/` contains charts from another repository; never cite these as DMX results.

## Key entrypoints
```bash
python auto_label_review.py
python scripts/train_classical.py --bootstrap 2000
python scripts/train_transformer.py --model-name vinai/phobert-base-v2
python scripts/train_transformer.py --model-name FacebookAI/xlm-roberta-large --batch-size 2 --gradient-accumulation-steps 8
python scripts/export_annotation_sample.py --n 300
jupyter lab notebooks/01_sentiment_analysis_report.ipynb
```

Use validation Macro-F1 for selection, evaluate the held-out test once, and keep label protocol identical across models. Fit vectorizers and learned features on train only. Preserve negation/intensity/contrast tokens. Do not call XLM-R Large SOTA without a fair human-labeled benchmark.
