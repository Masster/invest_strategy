import numpy as np

from moexlab.statistics.inference import (benjamini_hochberg, bootstrap_mean_pvalue, deflated_sharpe_ratio,
                                          hansen_spa, pbo_cscv, whites_reality_check)


def test_bh_controls_fdr_on_uniform_pvalues():
    rng = np.random.default_rng(0)
    rej = [benjamini_hochberg(rng.random(100), 0.1).sum() for _ in range(200)]
    assert np.mean(np.array(rej) > 0) < 0.15


def test_bh_finds_strong_signals():
    p = np.r_[np.full(10, 1e-6), np.random.default_rng(1).random(90)]
    assert benjamini_hochberg(p, 0.1)[:10].all()


def test_spa_and_rc_reject_true_edge_and_accept_null():
    rng = np.random.default_rng(2)
    T, N = 1000, 30
    null = rng.standard_normal((T, N)) * 0.01
    edge = null.copy()
    edge[:, 0] += 0.003   # SR ≈ 0.3 в день
    assert hansen_spa(null, 500, 10, 1)["pvalue"] > 0.05
    assert hansen_spa(edge, 500, 10, 1)["pvalue"] < 0.05
    assert whites_reality_check(edge, 500, 10, 1)["pvalue"] < 0.05


def test_pbo_is_high_for_pure_noise_and_low_for_true_edge():
    rng = np.random.default_rng(3)
    noise = rng.standard_normal((1200, 40)) * 0.01
    assert pbo_cscv(noise, 10)["pbo"] > 0.3
    edge = noise.copy()
    edge[:, 5] += 0.004
    assert pbo_cscv(edge, 10)["pbo"] < 0.1


def test_dsr_penalizes_many_trials():
    rng = np.random.default_rng(4)
    M = rng.standard_normal((500, 200)) * 0.01
    sr = M.mean(0) / M.std(0, ddof=1)
    best = int(sr.argmax())
    assert deflated_sharpe_ratio(M[:, best], 200, sr)["dsr"] < 0.9


def test_bootstrap_pvalue():
    rng = np.random.default_rng(5)
    assert bootstrap_mean_pvalue(rng.standard_normal(500) + 0.3, 500) < 0.01
    assert bootstrap_mean_pvalue(rng.standard_normal(500) - 0.3, 500) > 0.9
