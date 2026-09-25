import numpy as np
import pandas as pd

from moexlab.risk.sizing import kelly_ruin_probability, kelly_scale, required_scale


def test_kelly_scale_matches_mu_over_sigma2():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 20000)
    s = kelly_scale(r)
    assert abs(s - r.mean() / r.var()) / (r.mean() / r.var()) < 0.1


def test_no_edge_means_zero_kelly_and_infeasible_target():
    r = np.random.default_rng(1).normal(-0.0005, 0.01, 5000)
    assert kelly_scale(r) == 0.0
    assert required_scale(r, 0.05) is None


def test_target_above_kelly_growth_is_infeasible():
    # SR годовой ≈ 0.8: максимальный рост ≈ SR²/2 ≈ 32%/год << 10%/мес
    r = np.random.default_rng(2).normal(0.0005, 0.01, 20000)
    assert required_scale(r, 0.10) is None
    assert required_scale(r, 0.005) is not None


def test_kelly_drawdown_probability_formula():
    assert kelly_ruin_probability(1.0, 0.5) == 0.5          # полный Келли: P(DD 50%) = 50%
    assert abs(kelly_ruin_probability(0.5, 0.5) - 0.125) < 1e-12
