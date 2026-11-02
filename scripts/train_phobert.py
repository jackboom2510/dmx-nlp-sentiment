#!/usr/bin/env python3
"""Backward-compatible entry point for fine-tuning PhoBERT.

For a broader modern-model benchmark, see `scripts/train_transformer.py`, which also
supports multilingual XLM-RoBERTa (including the large checkpoint).
"""

from train_transformer import main


if __name__ == "__main__":
    main()
