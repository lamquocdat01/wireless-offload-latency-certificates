"""Unit tests for code/fec.py (Thm 6, Lemma 6.1, Prop 7). Run: python -m pytest -q tests"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import fec  # noqa: E402


def test_p_frame_indep_is_binomial_tail():
    assert fec.p_frame_indep(100, 3, 2e-3) == pytest.approx(stats.binom.sf(3, 103, 2e-3))
    assert fec.p_frame_indep(10, 0, 0.1) == pytest.approx(1 - 0.9 ** 10)


def test_r_star_minimal():
    for N, p in ((35, 2.3e-3), (100, 2.3e-3), (300, 1.4e-3)):
        r = fec.r_star_indep(N, p, 1e-3)
        assert r >= 0 and fec.p_frame_indep(N, r, p) <= 1e-3
        assert r == 0 or fec.p_frame_indep(N, r - 1, p) > 1e-3


def test_monotone_in_r():
    v = [fec.p_frame_indep(100, r, 5e-3) for r in range(20)]
    assert all(x >= y for x, y in zip(v, v[1:]))
    g = fec.GE(1e-4, 0.05, 0.4)
    w = [fec.p_frame_ge(g, 100, r) for r in range(12)]
    assert all(x >= y - 1e-15 for x, y in zip(w, w[1:]))


def test_ge_memoryless_equals_binomial():
    p = 1.3e-3
    g = fec.GE(0.3, 0.7, p / 0.3)        # p_gb + p_bg = 1 -> i.i.d. with rate p
    for N, r in ((35, 1), (100, 3), (300, 5)):
        assert fec.p_frame_ge(g, N, r) == pytest.approx(fec.p_frame_indep(N, r, p), rel=1e-9)


def test_ge_scaled_identity_and_stationary():
    g = fec.GE(2e-4, 0.1, 0.5)
    g1 = g.scaled(1.0)
    assert g1.p_gb == pytest.approx(g.p_gb) and g1.p_bg == pytest.approx(g.p_bg)
    g2 = g.scaled(2.0)
    P2 = g.P @ g.P
    assert g2.P == pytest.approx(P2)
    assert g.scaled(0.3).pi == pytest.approx(g.pi)


def test_ge_dp_vs_mc_small():
    rng = np.random.default_rng(0)
    g = fec.GE(1e-2, 0.2, 0.6)
    dp = fec.p_frame_ge(g, 20, 2)
    mc = fec.simulate_ge_frames(g, 20, 2, 200_000, rng)
    assert abs(dp - mc) < 5 * np.sqrt(dp * (1 - dp) / 200_000)


def test_loglik_bernoulli_case():
    rng = np.random.default_rng(1)
    x = (rng.random(20000) < 0.01).astype(np.int8)
    z, tail = fec._runs(x)
    p = x.mean()
    g = fec.GE(0.5, 0.5, 2 * p)          # memoryless: rate p
    assert fec.ge_loglik(g, z, tail) == pytest.approx(fec.bernoulli_loglik(x), rel=1e-9)


def test_empirical_block_rate():
    bad = np.zeros(1000, np.int8); bad[[5, 6, 250]] = 1
    rate, n = fec.empirical_block_rate(bad, 100, 1)
    assert n == 10 and rate == pytest.approx(0.1)
