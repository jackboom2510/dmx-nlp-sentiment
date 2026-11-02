#!/usr/bin/env python3
"""Export a blinded, stratified review sample for independent human gold annotation.

Example: python scripts/export_annotation_sample.py --n 300 --seed 42
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/test.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/annotation/annotation_sheet.csv")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--sampling", choices=["stratified", "random"], default="stratified",
                        help="Stratified gives rare classes coverage; random better preserves population prevalence")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.n > len(rows):
        raise ValueError(f"Requested {args.n} rows, but only {len(rows)} available")
    by_label = defaultdict(list)
    for row in rows:
        by_label[int(row["label_target"])].append(row)
    rng = random.Random(args.seed)
    if args.sampling == "random":
        selected = rng.sample(rows, args.n)
    else:
        selected = []
        if args.n < len(by_label):
            raise ValueError(f"Stratified sampling needs at least {len(by_label)} rows to cover every class")
        # Allocate proportionally with at least one item per observed class.
        exact = {label: args.n * len(group) / len(rows) for label, group in by_label.items()}
        counts = {label: min(len(group), max(1, int(exact[label]))) for label, group in by_label.items()}
        while sum(counts.values()) < args.n:
            candidates = [label for label, group in by_label.items() if counts[label] < len(group)]
            label = max(candidates, key=lambda key: exact[key] - counts[key])
            counts[label] += 1
        while sum(counts.values()) > args.n:
            candidates = [label for label in counts if counts[label] > 1]
            label = min(candidates, key=lambda key: exact[key] - counts[key])
            counts[label] -= 1
        for label, group in by_label.items():
            selected.extend(rng.sample(group, counts[label]))
        rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["review_id", "content", "annotator_1_label", "annotator_2_label", "adjudicated_label", "notes"]
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({"review_id": row.get("review_id", ""), "content": row.get("content", ""),
                             "annotator_1_label": "", "annotator_2_label": "", "adjudicated_label": "", "notes": ""})
    print(f"Exported {len(selected)} blinded rows to {args.output}")
    print("Annotators: use -1=Negative, 0=Neutral, 1=Positive; adjudicate disagreements independently.")


if __name__ == "__main__":
    main()
