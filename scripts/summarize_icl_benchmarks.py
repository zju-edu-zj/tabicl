"""Export benchmark records to CSV and plot the five-branch scaling matrix."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    records = []
    for path in sorted(args.results.glob("*.json")):
        result = json.loads(path.read_text())
        if isinstance(result, dict) and "median_total_s" in result:
            samples = result["records"]
            records.append(
                dict(
                    file=path.name,
                    attention=result["attention"],
                    global_type=result.get("global_type", "elu"),
                    partition=result.get("partition", "projection"),
                    scope=result.get("scope", "encoder"),
                    phase=result.get("phase", "train"),
                    dtype=result.get("dtype", "bfloat16"),
                    recompute=result.get("recompute", False),
                    portable=result.get("portable", False),
                    rows=result["rows"],
                    blocks=result["blocks"],
                    router=result.get("router", result.get("fused_router", "")),
                    median_ms=1000 * result["median_total_s"],
                    min_ms=1000 * min(row["total_s"] for row in samples),
                    max_ms=1000 * max(row["total_s"] for row in samples),
                    peak_allocated_gib=result["peak_allocated_gib"],
                    peak_reserved_gib=result["peak_reserved_gib"],
                )
            )
    if not records:
        parser.error("No successful benchmark records found")
    with (args.results / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    variants = sorted(set((r["attention"], r["global_type"], r["partition"]) for r in records))
    for attention, global_type, partition in variants:
        selected = sorted(
            (
                row
                for row in records
                if row["file"].startswith(("matrix_", "train_")) and row["attention"] == attention
                and row["global_type"] == global_type
                and row["partition"] == partition and row["scope"] == "encoder" and row["phase"] == "train"
                and row["blocks"] == 12 and row["dtype"] == "bfloat16" and row["recompute"] and not row["portable"]
            ),
            key=lambda row: row["rows"],
        )
        if not selected:
            continue
        x = [row["rows"] for row in selected]
        y = [row["median_ms"] for row in selected]
        label = attention + "/" + global_type + ("/" + partition if attention == "global_sparse" else "")
        axes[0].plot(x, y, ".-", label=label)
        axes[0].fill_between(x, [r["min_ms"] for r in selected], [r["max_ms"] for r in selected], alpha=0.12)
        axes[1].plot(x, [row["peak_allocated_gib"] for row in selected], ".-", label=label)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Total rows (80% support)")
        ax.grid(alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Forward + backward (ms, median and range)")
    axes[1].set_ylabel("Peak allocated GPU memory (GiB)")
    axes[0].legend(fontsize=8)
    fig.suptitle("ICL encoder: 12 layers, BF16, recompute; shared RTX 5090")
    fig.savefig(args.results / "scaling.png", dpi=180)
    fig.savefig(args.results / "scaling.svg")
    print(f"Exported {len(records)} records and scaling plots to {args.results}")


if __name__ == "__main__":
    main()
