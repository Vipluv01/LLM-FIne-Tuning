"""Turns results/sweep.csv into the claims the project actually makes.

Nothing in here reports a bare number. Every comparison goes through
adapt.stats so a delta always comes with the interval or test that says
whether it's real -- that is the entire discipline this project is built
around, and this module is where it gets applied to the actual sweep output.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from adapt.stats import Interval, holm_bonferroni, paired_delta_ci

RESULTS_DIR = Path("results")


def load_sweep(path: str = "results/sweep.csv") -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("accuracy", "unparseable_rate", "mean_latency_ms", "p99_latency_ms",
                   "peak_memory_mb", "train_wall_time_s", "final_train_loss"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
        for k in ("lora_rank", "n_shots", "seed", "trainable_params", "total_params", "eval_n_items"):
            if r.get(k) not in (None, ""):
                r[k] = int(float(r[k]))
    return rows


def per_item(run_id: str) -> np.ndarray:
    return np.load(RESULTS_DIR / "per_item" / f"{run_id}.npy")


def compare_runs(run_id_a: str, run_id_b: str) -> Interval:
    """Paired comparison between two specific runs, using their saved per-item scores."""
    a, b = per_item(run_id_a), per_item(run_id_b)
    n = min(len(a), len(b))
    return paired_delta_ci(a[:n].astype(float), b[:n].astype(float))


def rank_sweep_table(rows: list[dict]) -> list[dict]:
    """Mean accuracy (across seeds) per LoRA rank, plus the parameter ratio
    that makes 'rank 8 matches rank 64 at 1/8th the parameters' a computed
    fact instead of an eyeballed one."""
    by_rank: dict[int, list[dict]] = {}
    for r in rows:
        if r["method"] != "lora" or r.get("lora_rank") is None:
            continue
        by_rank.setdefault(r["lora_rank"], []).append(r)

    out = []
    for rank in sorted(by_rank):
        group = by_rank[rank]
        accs = np.array([g["accuracy"] for g in group])
        out.append({
            "rank": rank,
            "n_seeds": len(group),
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "trainable_params": group[0]["trainable_params"],
            "param_ratio_vs_r64": group[0]["trainable_params"] / by_rank[max(by_rank)][0]["trainable_params"]
            if max(by_rank) in by_rank else None,
        })
    return out


def rank_significance_matrix(rows: list[dict]) -> dict[str, tuple[bool, Interval]]:
    """Every adjacent-rank comparison, Holm-corrected across the whole family.

    Adjacent ranks specifically, not every pair: that's the comparison anyone
    actually cares about ("is going from 8 to 16 worth it"), and keeping the
    family small keeps the correction from being needlessly conservative.

    Returns, per comparison, BOTH whether it survives correction and the
    actual delta interval -- "significant" alone doesn't say whether the
    gap is worth training a bigger adapter for; the interval does.
    """
    by_rank: dict[int, list[str]] = {}
    for r in rows:
        if r["method"] == "lora" and r.get("lora_rank") is not None:
            by_rank.setdefault(r["lora_rank"], []).append(r["run_id"])

    ranks = sorted(by_rank)
    p_values: dict[str, float] = {}
    deltas: dict[str, Interval] = {}

    from scipy import stats as sstats

    for lo, hi in zip(ranks, ranks[1:]):
        # One representative seed pair per adjacent comparison; the seeds
        # loop is handled by the caller aggregating across repeated calls if
        # a fuller test is wanted. Kept simple: paired bootstrap per seed-0 run.
        lo_ids = by_rank[lo]
        hi_ids = by_rank[hi]
        if not lo_ids or not hi_ids:
            continue
        a = per_item(hi_ids[0])
        b = per_item(lo_ids[0])
        n = min(len(a), len(b))
        _, p = sstats.ttest_rel(a[:n].astype(float), b[:n].astype(float))
        name = f"r{lo}_vs_r{hi}"
        p_values[name] = float(p)
        deltas[name] = paired_delta_ci(a[:n].astype(float), b[:n].astype(float))

    survives = holm_bonferroni(p_values, alpha=0.05)
    return {name: (survives[name], deltas[name]) for name in deltas}


def data_efficiency_curve(rows: list[dict], *, rank: int) -> list[dict]:
    by_frac: dict[float, list[dict]] = {}
    for r in rows:
        if r["method"] == "lora" and r.get("lora_rank") == rank:
            by_frac.setdefault(r["train_fraction"], []).append(r)

    out = []
    for frac in sorted(by_frac):
        accs = np.array([g["accuracy"] for g in by_frac[frac]])
        out.append({
            "train_fraction": frac,
            "n_train_examples": None,  # filled by caller with dataset size context
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        })
    return out


def cost_quality_frontier(rows: list[dict]) -> list[dict]:
    """One row per method/config with the three axes deployment decisions
    actually turn on: accuracy, latency, and an approximate training cost."""
    out = []
    for r in rows:
        # A row missing p99_latency_ms (not measured for that run) can't be
        # placed on a latency axis at all -- skip it rather than let a blank
        # string reach pareto_frontier's numeric comparison.
        if r.get("p99_latency_ms") in (None, ""):
            continue
        # Rough training cost proxy: wall time is what a rented GPU bills for.
        train_cost_s = r.get("train_wall_time_s") or 0.0
        out.append({
            "run_id": r["run_id"],
            "method": r["method"],
            "accuracy": r["accuracy"],
            "p99_latency_ms": r["p99_latency_ms"],
            "train_wall_time_s": train_cost_s,
            "trainable_params": r["trainable_params"],
        })
    return out


def pareto_frontier(points: list[dict], *, maximize: str, minimize: str) -> list[dict]:
    """Keeps only the non-dominated points on (maximize, minimize).

    A point is dominated if another point is at least as good on both axes and
    strictly better on one -- dominated configurations are never the right
    deploy choice regardless of how the two axes get weighted, so dropping
    them is exactly what makes this a decision aid instead of a bigger table.
    """
    pts = sorted(points, key=lambda p: (-p[maximize], p[minimize]))
    frontier = []
    best_min_so_far = float("inf")
    for p in pts:
        if p[minimize] < best_min_so_far:
            frontier.append(p)
            best_min_so_far = p[minimize]
    return frontier
