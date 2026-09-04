"""Generates the final LoRA sweep report from results/sweep_kaggle.csv.

Run after downloading sweep_kaggle.csv and per_item/*.npy from the Kaggle
notebook (see kaggle/README.md). This is the report that answers the
project's actual question -- when is fine-tuning worth it -- with every
claim backed by the same paired-bootstrap / Holm-correction machinery used
throughout the rest of the project, not eyeballed off a table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapt.analyze import (
    cost_quality_frontier,
    data_efficiency_curve,
    load_sweep,
    pareto_frontier,
    per_item,
    rank_sweep_table,
    rank_significance_matrix,
)
from adapt.stats import paired_delta_ci


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    sweep_path = Path("results/sweep_kaggle.csv")
    if not sweep_path.exists():
        # A raw FileNotFoundError traceback here is actively unhelpful --
        # this is the EXPECTED state before the Kaggle (or Colab, see
        # colab/README.md) sweep has been run and its output downloaded,
        # not a bug. Same discipline as sweep_report.py's own guard
        # against an empty `table` from a zero-LoRA-rows CSV: tell the
        # user what to do next, don't just crash.
        print("No results/sweep_kaggle.csv yet -- the LoRA sweep hasn't been run and downloaded.")
        print("Run kaggle/run_sweep.py (or colab/run_sweep.py) and download its output first;")
        print("see kaggle/README.md or colab/README.md for the exact steps.")
        sys.exit(1)
    rows = load_sweep(str(sweep_path))

    section("RANK SWEEP -- accuracy vs. LoRA rank")
    table = rank_sweep_table(rows)
    print(f"{'rank':<8}{'mean acc':<12}{'std':<10}{'trainable params':<20}{'vs r_max':<12}")
    print("-" * 62)
    for row in table:
        ratio = f"{row['param_ratio_vs_r64']:.1%}" if row["param_ratio_vs_r64"] is not None else "-"
        print(f"{row['rank']:<8}{row['mean_accuracy']:<12.4f}{row['std_accuracy']:<10.4f}"
              f"{row['trainable_params']:<20,}{ratio:<12}")

    section("RANK SIGNIFICANCE -- adjacent comparisons, Holm-corrected")
    sig = rank_significance_matrix(rows)
    for name, (survives, interval) in sig.items():
        tag = "SIGNIFICANT" if survives else "not significant"
        print(f"  {name:<16} delta={interval}  [{tag}]")

    section("DATA-EFFICIENCY CURVE -- accuracy vs. training-set fraction")
    if not table:
        # Real condition hit while this was still being written: with only
        # a partial download (baseline configs but zero completed LoRA
        # runs yet), `table` is empty and min() on it raised ValueError --
        # this must degrade to "nothing to report yet", not crash before
        # the sections above it even get a chance to print.
        print("(skipped -- no completed LoRA rank-sweep rows yet)")
    else:
        # Prefer whichever rank the significance matrix shows as not
        # meaningfully worse than the largest -- report against the
        # smallest rank that isn't a real regression, not just "rank 8" by
        # convention. Falls back to the smallest available rank if the
        # significance matrix has nothing to say yet (e.g. only one rank
        # completed so far, so there's no adjacent comparison at all).
        non_regressions = [
            int(name.split("_vs_")[1][1:]) for name, (survives, interval) in sig.items()
            if not (survives and interval.point < 0)  # a SIGNIFICANT negative delta is a real regression
        ]
        best_rank = min(non_regressions) if non_regressions else min(r["rank"] for r in table)
        curve = data_efficiency_curve(rows, rank=best_rank)
        print(f"(against rank={best_rank})")
        print(f"{'fraction':<12}{'mean acc':<12}{'std':<10}")
        print("-" * 34)
        for row in curve:
            print(f"{row['train_fraction']:<12}{row['mean_accuracy']:<12.4f}{row['std_accuracy']:<10.4f}")
        if not curve:
            print(f"(no data-efficiency rows yet for rank={best_rank})")

    section("DEPLOY FRONTIER -- accuracy vs. p99 latency, dominated configs dropped")
    points = cost_quality_frontier(rows)
    frontier = pareto_frontier(points, maximize="accuracy", minimize="p99_latency_ms")
    print(f"{len(points)} configs total -> {len(frontier)} on the frontier")
    print(f"{'run_id':<24}{'accuracy':<12}{'p99 (ms)':<12}{'train (s)':<12}")
    print("-" * 60)
    for p in sorted(frontier, key=lambda x: x["accuracy"]):
        print(f"{p['run_id']:<24}{p['accuracy']:<12.4f}{p['p99_latency_ms']:<12.1f}{p['train_wall_time_s']:<12.0f}")
    print("Every config not listed above is dominated: some other config is at")
    print("least as accurate AND at least as fast. These are the only configs")
    print("worth considering for deployment; picking between them is a matter")
    print("of the actual latency budget, not more analysis.")

    section("FINE-TUNING vs. THE BEST PROMPTING BASELINE")
    # This is the project's actual headline comparison: does training beat
    # the best few-shot result from the local baseline sweep, by a margin
    # that survives the same statistical bar everything else in this
    # project is held to?
    try:
        best_lora_id = max(
            (r for r in rows if r["method"] == "lora"),
            key=lambda r: r["accuracy"],
        )["run_id"]
        import csv
        with open("results/sweep_local.csv") as f:
            baseline_rows = list(csv.DictReader(f))
        best_baseline = max(
            (r for r in baseline_rows if r["method"] in ("zero_shot", "few_shot")),
            key=lambda r: float(r["accuracy"]),
        )
        a = per_item(best_lora_id).astype(float)
        b = per_item(best_baseline["run_id"]).astype(float)
        n = min(len(a), len(b))
        ci = paired_delta_ci(a[:n], b[:n])
        print(f"best LoRA config: {best_lora_id}")
        print(f"best prompting baseline: {best_baseline['run_id']} ({best_baseline['method']}, "
              f"{best_baseline['n_shots']} shots)")
        print(f"delta (LoRA - best prompting): {ci}")
        if ci.excludes_zero and ci.point > 0:
            print("-> Fine-tuning wins by a real, statistically confirmed margin.")
        elif ci.excludes_zero and ci.point < 0:
            print("-> Prompting wins. Fine-tuning did not clear the baseline here --")
            print("   this is a legitimate, reportable finding, not a failed experiment.")
        else:
            print("-> No statistically distinguishable difference at this eval-set size.")
            print("   The honest conclusion: fine-tuning bought nothing measurable over")
            print("   a well-engineered prompt for this task.")
    except (FileNotFoundError, ValueError) as e:
        print(f"(skipped -- {e}; run the local baseline sweep first if this is missing)")


if __name__ == "__main__":
    main()
