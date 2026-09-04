"""Statistical machinery for comparing model configurations.

The reason this module exists at all: fine-tuning comparisons are almost always
reported as two bare numbers -- "base model 71%, fine-tuned 78%" -- with no
indication of whether that gap would survive re-running the experiment. On an
evaluation set of a few hundred items, a 7-point gap can easily be noise, and a
2-point gap almost always is.

Three ideas do the work here:

1. **Pairing.** Every configuration is scored on the *same* items, so the
   comparison can look at per-item differences instead of two independent
   averages. Pairing removes item difficulty from the variance, and it is free
   -- it costs nothing but running the same eval set through each config.

2. **Bootstrapping.** Rather than assuming a distribution for the metric (which
   is hopeless for things like ROUGE or pass@k), resample the data and watch
   how much the statistic moves.

3. **Multiplicity.** Sweeping LoRA ranks and data sizes means dozens of
   comparisons. At the 5% level, one comparison in twenty looks significant by
   chance alone, so an uncorrected sweep manufactures findings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Interval",
    "bootstrap_ci",
    "paired_delta_ci",
    "mcnemar_exact",
    "holm_bonferroni",
    "min_paired_samples",
]


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    point: float
    low: float
    high: float
    level: float = 0.95

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely on one side of zero.

        For a difference between two configurations this is the honest version
        of "is it significant" -- and unlike a p-value it also shows how large
        the effect might plausibly be.
        """
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:
        return f"{self.point:+.4f} [{self.low:+.4f}, {self.high:+.4f}]"


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int | None = 0,
) -> Interval:
    """Percentile bootstrap confidence interval for a mean.

    Resamples items with replacement `n_boot` times and reads the interval off
    the quantiles of the resampled means.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1-D array")

    rng = _rng(seed)
    n = values.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)

    tail = (1.0 - level) / 2.0
    return Interval(
        point=float(values.mean()),
        low=float(np.quantile(means, tail)),
        high=float(np.quantile(means, 1.0 - tail)),
        level=level,
    )


def paired_delta_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int | None = 0,
) -> Interval:
    """Confidence interval for the mean difference between two configs.

    `a` and `b` must be per-item scores for the SAME items in the SAME order --
    that alignment is the whole point. Resampling happens over items, and both
    configs move together on each resample, so any variance from some items
    simply being harder cancels out.

    Getting this wrong -- bootstrapping the two sets independently -- inflates
    the interval and hides real differences. It is the most common statistical
    error in model comparison write-ups.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired scores must align: got {a.shape} and {b.shape}")

    return bootstrap_ci(a - b, n_boot=n_boot, level=level, seed=seed)


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test for two systems scored right/wrong on the same items.

    Only the disagreements carry information. Items both systems get right, or
    both get wrong, say nothing about which is better -- so the test conditions
    on the discordant pairs and asks whether they split evenly.

    The exact binomial form is used rather than the chi-square approximation
    because discordant counts are often small, and the approximation is
    unreliable exactly there.

    Returns (a_only, b_only, p_value).
    """
    a_correct = np.asarray(a_correct).astype(bool)
    b_correct = np.asarray(b_correct).astype(bool)
    if a_correct.shape != b_correct.shape:
        raise ValueError("paired outcomes must align")

    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    n = a_only + b_only

    if n == 0:
        return a_only, b_only, 1.0

    # Two-sided exact binomial: probability of a split at least this lopsided
    # under a fair coin.
    from math import comb

    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return a_only, b_only, float(min(1.0, 2.0 * tail))


def holm_bonferroni(p_values: dict[str, float], *, alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down correction across a family of comparisons.

    A rank sweep crossed with data sizes produces dozens of p-values. Testing
    each at 0.05 means roughly one false positive per twenty comparisons, which
    is how sweeps manufacture findings that do not replicate.

    Holm is used rather than plain Bonferroni because it is uniformly more
    powerful while controlling the same family-wise error rate -- there is no
    reason to take the weaker option.

    Returns a mapping from comparison name to whether it survives.
    """
    if not p_values:
        return {}

    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    survives: dict[str, bool] = {}
    rejected_so_far = True

    for i, (name, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        # Step-down: once one test fails, every larger p-value fails too.
        if rejected_so_far and p <= threshold:
            survives[name] = True
        else:
            rejected_so_far = False
            survives[name] = False

    return survives


def min_paired_samples(
    effect: float,
    per_item_sd: float,
    *,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """How many evaluation items are needed to detect `effect` reliably.

    Worth computing *before* spending GPU hours, because the usual failure is
    running an expensive sweep on a 200-item eval set that could never have
    resolved the differences being looked for. If this returns 4,000 and the
    eval set has 300 items, the honest conclusion is that the experiment cannot
    answer the question -- and saying so is better than reporting noise.

    Uses the normal approximation for a paired two-sided test, which is fine at
    the sample sizes this regime implies.
    """
    if effect <= 0 or per_item_sd <= 0:
        raise ValueError("effect and per_item_sd must be positive")

    from statistics import NormalDist

    nd = NormalDist()
    z_alpha = nd.inv_cdf(1.0 - alpha / 2.0)
    z_beta = nd.inv_cdf(power)

    n = ((z_alpha + z_beta) * per_item_sd / effect) ** 2
    return int(np.ceil(n))
