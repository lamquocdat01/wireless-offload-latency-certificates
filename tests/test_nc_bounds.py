"""Unit tests for code/nc_bounds.py, code/service_curve.py and code/replay.py (topic #19, P1).
Run from the topic folder:  python -m pytest -q tests
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import nc_bounds as nb  # noqa: E402
from service_curve import cert_exceedances, certified_T, cp_upper  # noqa: E402

TAU = 1000.0 / 30.0


# ------------------------------------------------------------------ Thm 1
def test_alpha_values():
    w = np.arange(0, 9)
    # K = 4, a = 0.25, b = 1.5: min{w, ceil(w/4) + floor(1.5 + 0.25 (w - 1))}
    exp = [0, 1, 2, 3, 3, 4, 4, 5, 5]
    assert nb.alpha_tb(w, 4, 0.25, 1.5).tolist() == exp


def test_alpha_is_upper_bound_of_policy():
    rng = np.random.default_rng(0)
    for K, a, b in [(2, 0.3, 1.6), (5, 0.1, 4.0), (8, 0.5, 2.0), (3, 0.2, 1.0)]:
        o = nb.token_bucket_opens(rng.random(800) < 0.6, K, a, b, phase=1)
        cs = np.r_[0, np.cumsum(o)]
        for w in range(1, 300):
            assert np.max(cs[w:] - cs[:-w]) <= nb.alpha_tb(w, K, a, b)


@pytest.mark.parametrize("K,a,b", [(2, 0.2, 1.4), (4, 0.3, 1.6), (6, 0.5, 2.0), (3, 0.1, 4.0)])
def test_thm1_prime_tightness(K, a, b):
    W = 600
    gap = nb.alpha_tb(np.arange(W + 1), K, a, b) - nb.alpha_star_tb(W, K, a, b)
    assert gap.min() >= 0 and gap.max() <= 1


def test_thm1_prime_counterexample_linear_gap():
    W = 3000
    gap = nb.alpha_tb(np.arange(W + 1), 2, 0.4, 1.5) - nb.alpha_star_tb(W, 2, 0.4, 1.5)
    assert gap[W] >= 70          # 1 + a <= b < 1 + 2a is NOT enough


def test_alpha_bit_staircase():
    assert nb.alpha_bit(0.0, 8.0, TAU, 4, 0.1, 1.2) == 0.0
    assert nb.alpha_bit(TAU / 2, 8.0, TAU, 4, 0.1, 1.2) == 8.0
    assert nb.alpha_bit(TAU, 8.0, TAU, 4, 0.1, 1.2) == 8.0 * nb.alpha_tb(1, 4, 0.1, 1.2)


# ------------------------------------------------------------------ Prop 3 / Thm 2
def test_prop3_identity_and_c_le_1():
    al = nb.alpha_tb(np.arange(4001), 8, 0.1, 2.0).astype(float)
    for c in (0.3, 0.9, 1.0, 1.4, 2.5):
        if (1 / 8 + 0.1) * c >= 1:
            continue
        assert nb.q_term(al, c) == pytest.approx(c * (1 + nb.sigma(al, 1 / c)), abs=1e-9)
        if c <= 1:
            assert nb.q_term(al, c) == pytest.approx(c)


def test_q_tb_matches_grid():
    for K, a, b, c in [(4, 0.1, 1.2, 1.5), (16, 0.2, 4.0, 2.0), (2, 0.05, 1.1, 1.8)]:
        al = nb.alpha_tb(np.arange(20001), K, a, b).astype(float)
        assert nb.q_tb(K, a, b, c) == pytest.approx(nb.q_term(al, c), abs=1e-9)


def test_q_from_sequences_equals_envelope():
    rng = np.random.default_rng(1)
    seqs = [(rng.random(300) < 0.4).astype(np.int8) for _ in range(5)]
    al = nb.window_counts(seqs, 300).astype(float)
    for c in (0.7, 1.3, 2.2):
        assert nb.q_from_sequences(seqs, c) == pytest.approx(nb.q_term(al, c), abs=1e-9)


def test_bound_max_le_sum():
    for K, Q, c in [(4, 0.6, 0.6), (8, 30.0, 1.5), (2, 6.0, 1.5)]:
        assert nb.e2e_bound_ms(K, Q, c, 7.2, 0, TAU) <= nb.e2e_bound_sum_ms(K, Q, 7.2, 0, TAU) + 1e-9


def test_k_max():
    # Delta = 200, T = 7.2, L = 20, c = 1.5: floor((172.8)/33.33 - 1.5) + 1 = floor(3.684) + 1 = 4
    assert nb.k_max(200, 7.2, 20, 1.5, TAU) == 4
    assert nb.k_max(10, 7.2, 20, 1.5, TAU) == 0
    assert nb.k_dev(200, 20, TAU) == 6


# ------------------------------------------------------------------ Thm 4 / Alg 1
def test_bmax_closed_sufficient():
    for D, T, L, c, K, a in [(200, 7.2, 20, 1.5, 4, 0.1), (500, 10, 0, 2.17, 16, 0.2), (1000, 50, 20, 1.07, 8, 0.3)]:
        bc = nb.b_max_closed(D, T, L, c, TAU, K, a)
        be = nb.b_max_exact(D, T, L, c, TAU, K, a)
        assert bc <= be + 1e-6
        if bc >= 0:
            assert nb.sigma_tb(K, a, bc, 1 / c) <= nb.burst_budget(D, T, L, c, TAU) + 1e-9


def test_bmax_c_le_1_infinite():
    assert nb.b_max_closed(200, 7.2, 20, 0.9, TAU, 4, 0.1) == math.inf
    assert nb.b_max_exact(200, 7.2, 20, 1.0, TAU, 4, 0.1) == math.inf


def test_alg1_output_feasible_and_minimal():
    p = nb.alg1(300, 8.2, 20, 1.5, TAU, 0.1, b_rule="min")
    assert p.admitted and nb.feasible(p.K, p.a, p.b, 300, 8.2, 20, 1.5, TAU)
    assert not nb.feasible(p.K + 1, p.a, p.b, 300, 8.2, 20, 1.5, TAU)
    assert p.rho == pytest.approx(1 / p.K + p.a)


def test_alg1_rejects_unstable():
    p = nb.alg1(1000, 8.2, 20, 3.0, TAU, 0.4)
    assert not p.admitted and p.reason.startswith("unstable")


# ------------------------------------------------------------------ Ext 2.2
def test_general_beta_reduces_to_thm2():
    K, a, b, c, T = 8, 0.1, 2.0, 1.4, 7.0
    s = 1.0
    R = s / (c * TAU)
    binv = lambda y: np.where(y > 0, T + y / R, 0.0)
    W = 3000
    al = nb.alpha_tb(np.arange(W + 1), K, a, b).astype(float)
    br2 = float(np.max(binv(s * (al + 1)) - np.arange(W + 1) * TAU))
    gen = max((K - 1) * TAU + float(binv(np.array([s]))[0]), br2)
    assert gen == pytest.approx(nb.e2e_bound_ms(K, nb.q_term(al, c), c, T, 0.0, TAU), abs=1e-9)


# ------------------------------------------------------------------ Prop 5
def test_cp_upper_known_value():
    # n = 1e6, m = 1000 exceedances, delta = 0.05 one-sided -> ~1.054e-3 (P0 report)
    assert cp_upper(1000, 10**6, 0.05) == pytest.approx(1.0536e-3, rel=2e-3)


def test_cert_exceedances_definition():
    from scipy import stats
    n, eps, d = 100000, 1e-3, 0.025
    m = cert_exceedances(n, eps, d)
    assert stats.binom.cdf(m, n, eps) <= d < stats.binom.cdf(m + 1, n, eps)
    assert cert_exceedances(100, 1e-3, 0.025) == -1


def test_certified_T_is_order_statistic():
    x = np.arange(1, 100001, dtype=float)
    c = certified_T(x, 1e-3, 0.025)
    assert c["T_ms"] == x[len(x) - c["m_star"] - 1]
    assert c["T_ms"] > np.quantile(x, 1 - 1e-3)       # conservative vs the empirical quantile
