# AI Agent Skills — sentiment classification

The following workflow skills describe the supported project procedures. Current filenames and evaluation assumptions in `README.md` / `INSTRUCTIONS.md` take precedence over outdated generated outputs.

| Skill | Purpose | Entrypoint / output |
|---|---|---|
| `data-audit` | Inspect raw input, duplicates, class/category/rating distribution | `data/raw/`, `data/interim/labeling_summary.json` |
| `hybrid-label-and-split` | Normalize text, reconcile rating and lexicon sentiment, create splits | `python auto_label_review.py` → `train.csv`, `val.csv`, `test.csv` |
| `preprocess-vietnamese` | NFC, teencode, Vietnamese segmentation, negation-safe stopword handling | `src/data/preprocess.py` |
| `classical-benchmark` | Compare sparse/dense representations and NB/SVM | `python scripts/train_classical.py` |
| `transformer-finetuning` | Fine-tune PhoBERT or XLM-R with weighted loss and validation early stopping | `python scripts/train_transformer.py --model-name ...` |
| `evaluation-and-error-analysis` | Report Macro-F1/per-class, CI, confusion and row-level errors | `src/evaluation/`, `outputs/metrics/`, `outputs/predictions/` |
| `report-notebook` | Generate exploratory analysis and evidence-based report | `notebooks/01_sentiment_analysis_report.ipynb` |
| `human-label-audit` | Export blinded stratified examples for independent annotation | `scripts/export_annotation_sample.py` |

## Rules shared by all skills

1. Preserve raw data and document preprocessing/label source.
2. **The existing `label_target` is a pseudo-label. The current `test.csv` is not gold or human-audited.** Do not make claims of true accuracy until a separate independently labeled test has been created.
3. Keep label IDs consistent: `-1 Negative`, `0 Neutral`, `1 Positive`.
4. Fit learned transforms on train only; validation selects configurations; held-out test is not used for tuning.
5. Use Macro-F1 as a primary score with per-class metrics, support, confusion matrix and uncertainty intervals.
6. Preserve negation, sentiment intensifiers and contrast markers while cleaning Vietnamese text.
7. Reference charts under `figures/reference/` came from another repository and are not DMX experimental results.
8. Do not call XLM-R Large SOTA without a fair benchmark on independently annotated labels.

## Suggested annotation protocol
Export `python scripts/export_annotation_sample.py --n 300 --seed 42`. Hide rating and model labels from annotators, independently label each review with the shared 3-class guide, adjudicate disagreements, then compute Cohen’s kappa and report model metrics on the audited labels separately from pseudo-label metrics.
