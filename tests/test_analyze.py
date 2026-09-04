import sys
sys.path.insert(0, "src")

from adapt.analyze import pareto_frontier, rank_sweep_table


def test_pareto_drops_dominated_points():
    points = [
        {"name": "A", "accuracy": 0.80, "latency": 100},
        {"name": "B", "accuracy": 0.85, "latency": 90},   # dominates A on both
        {"name": "C", "accuracy": 0.90, "latency": 150},  # better acc, worse latency: on frontier
        {"name": "D", "accuracy": 0.70, "latency": 200},  # dominated by everything
    ]
    frontier = pareto_frontier(points, maximize="accuracy", minimize="latency")
    names = {p["name"] for p in frontier}
    assert "A" not in names, "A is strictly dominated by B and must be dropped"
    assert "D" not in names, "D is strictly dominated and must be dropped"
    assert "B" in names and "C" in names


def test_rank_sweep_table_computes_param_ratio():
    rows = [
        {"method": "lora", "lora_rank": 8, "accuracy": 0.70, "trainable_params": 1_000_000},
        {"method": "lora", "lora_rank": 8, "accuracy": 0.71, "trainable_params": 1_000_000},
        {"method": "lora", "lora_rank": 64, "accuracy": 0.72, "trainable_params": 8_000_000},
        {"method": "zero_shot", "lora_rank": None, "accuracy": 0.50, "trainable_params": 0},
    ]
    table = rank_sweep_table(rows)
    ranks = {row["rank"] for row in table}
    assert ranks == {8, 64}, "zero_shot rows must not leak into the rank table"

    r8 = next(row for row in table if row["rank"] == 8)
    assert r8["n_seeds"] == 2
    assert abs(r8["mean_accuracy"] - 0.705) < 1e-9
    assert abs(r8["param_ratio_vs_r64"] - 0.125) < 1e-9


def test_rank_significance_matrix_returns_significance_and_magnitude():
    """Regression test: this function used to compute the actual delta
    interval internally and then discard it, returning only a bool -- which
    is useless for a report (knowing r16 beats r8 doesn't say by how much).
    Must return both."""
    import numpy as np
    from adapt.analyze import rank_significance_matrix

    rng = np.random.default_rng(0)
    n = 300
    r4 = rng.random(n) < 0.40
    r8 = rng.random(n) < 0.48  # a real, meaningful gap over r4
    r16 = rng.random(n) < 0.49  # barely different from r8

    import os
    os.makedirs("results/per_item", exist_ok=True)
    for run_id, arr in [("lora_r4_seed0", r4), ("lora_r8_seed0", r8), ("lora_r16_seed0", r16)]:
        np.save(f"results/per_item/{run_id}.npy", arr)

    rows = [
        {"method": "lora", "lora_rank": 4, "run_id": "lora_r4_seed0"},
        {"method": "lora", "lora_rank": 8, "run_id": "lora_r8_seed0"},
        {"method": "lora", "lora_rank": 16, "run_id": "lora_r16_seed0"},
    ]
    result = rank_significance_matrix(rows)

    assert set(result.keys()) == {"r4_vs_r8", "r8_vs_r16"}
    for name, (sig, interval) in result.items():
        assert isinstance(sig, bool)
        assert hasattr(interval, "low") and hasattr(interval, "high"), \
            f"{name}: must return the actual Interval, not just a bool"
