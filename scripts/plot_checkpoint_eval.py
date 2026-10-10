"""Plot checkpoint-level downstream benchmark metrics.

Input is JSONL with at least ``step``, ``method`` and ``mean_auc``.  An
evaluation runner can append one record per checkpoint; this script turns the
records into the multi-run curve used for model selection.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="JSONL checkpoint evaluation records")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metric", default="mean_auc", help="Y-axis JSON field")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No records found in {args.metrics}")
    required = {"step", "method", args.metric}
    missing = required - rows[0].keys()
    if missing:
        raise SystemExit(f"Missing fields: {', '.join(sorted(missing))}")
    output = args.output or args.metrics.with_suffix(".png")
    fig, ax = plt.subplots(figsize=(12, 7))
    for method in sorted({row["method"] for row in rows}):
        values = sorted((row for row in rows if row["method"] == method), key=lambda row: row["step"])
        ax.plot([row["step"] for row in values], [row[args.metric] for row in values], marker="o", label=method)
    ax.set(xlabel="Step", ylabel=args.metric.replace("_", " ").title(), title="Downstream checkpoint evaluation")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
