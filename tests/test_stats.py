"""Tests for the comparison statistics.

These check statistical *behaviour*, not just that functions return numbers.
A confidence interval implementation that is subtly wrong still returns
plausible-looking intervals -- the only way to catch it is to simulate many
experiments and check the interval covers the truth as often as it claims to.
"""

import numpy as np
import pytest

from adapt.stats import (
    bootstrap_ci,
    holm_bonferroni,
    mcnemar_exact,
    min_paired_samples,
    paired_delta_ci,
)


def test_bootstrap_ci_is_calibrated():
    """A 95% interval should contain the true mean about 95% of the time.

    This is the test that actually validates the implementation. Simulate 400
    independent experiments, build an interval for each, and count coverage.
    Anything far from 0.95 means the interval is mis-calibrated -- too narrow
    and it manufactures significance, too wide and it hides real effects.
    """
    rng = np.random.default_rng(11)
    true_mean = 0.62
    covered = 0
    trials = 400

    for t in range(trials):
        sample = rng.normal(true_mean, 0.25, size=300)
        ci = bootstrap_ci(sample, n_boot=1500, seed=t)
        if ci.low <= true_mean <= ci.high:
            covered += 1

    coverage = covered / trials
    # Binomial noise on 400 trials is about +/- 2%, so allow a modest band.
    assert 0.91 <= coverage <= 0.98, f"coverage {coverage:.3f} is off nominal 0.95"


def test_paired_delta_detects_real_difference():
    """A genuine improvement should produce an interval clear of zero."""
    rng = np.random.default_rng(3)
    # Items vary a lot in difficulty; config B is uniformly 6 points better.
    difficulty = rng.normal(0.5, 0.3, size=500)
    a = difficulty
    b = difficulty + 0.06

    ci = paired_delta_ci(b, a, seed=1)
    assert ci.excludes_zero
    assert 0.04 < ci.point < 0.08


def test_paired_delta_rejects_noise():
    """Identical configs differing only by noise must NOT look significant."""
    rng = np.random.default_rng(5)
    difficulty = rng.normal(0.5, 0.3, size=500)
    a = difficulty + rng.normal(0, 0.05, size=500)
    b = difficulty + rng.normal(0, 0.05, size=500)

    ci = paired_delta_ci(a, b, seed=2)
    assert not ci.excludes_zero, f"found a difference that isn't there: {ci}"


def test_pairing_beats_unpaired_analysis():
    """Pairing should give a tighter interval than treating samples as independent.

    This is the concrete reason the eval set is shared across configurations:
    item difficulty cancels, so the same data resolves smaller effects.
    """
    rng = np.random.default_rng(7)
    difficulty = rng.normal(0.5, 0.4, size=400)
    a = difficulty
    b = difficulty + 0.03

    paired = paired_delta_ci(b, a, seed=4)
    paired_width = paired.high - paired.low

    # Unpaired: bootstrap each arm separately and add the variances.
    ci_a = bootstrap_ci(a, seed=4)
    ci_b = bootstrap_ci(b, seed=5)
    unpaired_width = (ci_a.high - ci_a.low) + (ci_b.high - ci_b.low)

    assert paired_width < unpaired_width / 2


def test_mcnemar_only_counts_disagreements():
    """Items both systems agree on carry no information."""
    # 90 agreements, then 10 items where A wins and 0 where B does.
    a = np.array([True] * 90 + [True] * 10)
    b = np.array([True] * 90 + [False] * 10)

    a_only, b_only, p = mcnemar_exact(a, b)
    assert (a_only, b_only) == (10, 0)
    assert p < 0.01

    # Perfect agreement -> nothing to test.
    _, _, p_same = mcnemar_exact(a, a)
    assert p_same == 1.0


def test_mcnemar_even_split_is_not_significant():
    a = np.array([True] * 8 + [False] * 8)
    b = np.array([False] * 8 + [True] * 8)
    a_only, b_only, p = mcnemar_exact(a, b)
    assert a_only == b_only == 8
    assert p > 0.9


def test_holm_is_stricter_than_uncorrected():
    """A borderline p-value in a large family should not survive."""
    family = {f"cmp_{i}": 0.04 for i in range(20)}
    survived = holm_bonferroni(family, alpha=0.05)
    assert not any(survived.values()), "20 borderline results all 'significant' is the bug"

    # A single overwhelming result still survives alongside weak ones.
    mixed = {"strong": 0.0001, **{f"weak_{i}": 0.4 for i in range(10)}}
    out = holm_bonferroni(mixed, alpha=0.05)
    assert out["strong"] is True
    assert not any(v for k, v in out.items() if k != "strong")


def test_holm_step_down_is_monotone():
    """Once a test fails, no larger p-value may pass."""
    family = {"a": 0.001, "b": 0.03, "c": 0.031, "d": 0.9}
    out = holm_bonferroni(family, alpha=0.05)
    order = ["a", "b", "c", "d"]
    seen_failure = False
    for name in order:
        if not out[name]:
            seen_failure = True
        elif seen_failure:
            pytest.fail(f"{name} passed after an earlier failure")


def test_min_samples_scales_inversely_with_squared_effect():
    """Halving the effect you want to detect should roughly quadruple the n."""
    big = min_paired_samples(effect=0.04, per_item_sd=0.3)
    small = min_paired_samples(effect=0.02, per_item_sd=0.3)
    assert 3.6 < small / big < 4.4


def test_min_samples_is_sobering():
    """Detecting a 1-point difference needs far more than a typical eval set."""
    n = min_paired_samples(effect=0.01, per_item_sd=0.35)
    assert n > 5000, "this is the number that should stop you over-claiming"


def test_alignment_is_enforced():
    with pytest.raises(ValueError):
        paired_delta_ci(np.zeros(10), np.zeros(11))
    with pytest.raises(ValueError):
        mcnemar_exact(np.zeros(5, bool), np.zeros(6, bool))
