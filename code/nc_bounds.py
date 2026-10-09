"""Network-calculus bounds for the gated-camera traffic class (topic #19, P1). Pure numpy.

Units: discrete frame slots (tau_f = 1/F). A frame occupies the uplink transmitter for c slots
(c = s F / R: frame size x frame rate / link rate, the *link load factor* of one frame). Radio
latency <= T [ms], server latency <= L_srv [ms]. Results in theory/THEORY_19.md:

  Thm 1   alpha(w) = min{w, ceil(w/K) + floor(b + a(w-1))}                 (token-bucket(K,a,b))
  Thm 1'  alpha*(w) >= alpha(w) - 1     if a <= 1/2 and b >= 1 + 2a        (tightness)
  Thm 2   L_e2e <= T + L_srv + tau_f max{K - 1 + c, Q(alpha, c)},  Q = max_w [(alpha(w)+1)c - w]
  Ext 2.2 L_e2e <= L_srv + max{(K-1) tau_f + beta^-1(s), h(alpha_bit + s, beta)}   (general beta)
  Prop 3  Q(alpha, c) = c (1 + sigma_alpha(1/c)),  sigma_alpha(r) = max_w [alpha(w) - r w]
  Thm 4   feasible set F, closed-form bucket cap b_max, Alg 1 (min-rate admission/dimensioning)

Nothing here reads data; see replay.py / service_curve.py for the trace side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPS_NUM = 1e-12


# ----------------------------------------------------------------------------- Thm 1

def alpha_tb(w, K: int, a: float, b: float) -> np.ndarray:
    """Thm 1 arrival curve of token-bucket(K, a, b), in frames, for integer windows w >= 0."""
    w = np.asarray(w, dtype=np.int64)
    out = np.minimum(w, np.ceil(w / K) + np.floor(b + a * (w - 1) + EPS_NUM))
    return np.where(w <= 0, 0, out).astype(np.int64)


def alpha_bit(t_ms, s_bits: float, tau_f_ms: float, K: int, a: float, b: float) -> np.ndarray:
    """Thm 1, bit / continuous-time form: alpha_bit(t) = s * alpha(ceil(t / tau_f)), alpha_bit(0) = 0.
    (A half-open interval of length t contains at most ceil(t / tau_f) slot instants.)"""
    t = np.asarray(t_ms, dtype=float)
    w = np.ceil(t / tau_f_ms - EPS_NUM).astype(np.int64)
    return s_bits * alpha_tb(np.maximum(w, 0), K, a, b)


def token_bucket_opens(fire: np.ndarray, K: int, a: float, b: float, phase: int = 0,
                       c0: float | None = None) -> np.ndarray:
    """Token-bucket(K,a,b) policy (identical to policy.token_bucket): refresh slots t = phase mod K
    always open; the gate opens a non-refresh slot iff it fires and the token level after refill
    (c_t = min(b, c_{t-1} + a)) is >= 1, then spends one token. c_{-1} = b unless c0 is given.
    The test c >= 1 uses a 1e-9 tolerance (exact-arithmetic semantics: 0.2 + 8 x 0.1 counts as 1);
    policy.py (P0) tests c >= 1.0 in raw floating point, which can delay an open by one slot."""
    n = len(fire)
    o = np.zeros(n, dtype=np.int8)
    o[phase % K::K] = 1
    c = float(b if c0 is None else c0)
    for t in range(n):
        c = min(b, c + a)
        if not o[t] and fire[t] and c >= 1.0 - 1e-9:
            o[t] = 1
            c -= 1.0
    return o


def alpha_star_tb(W: int, K: int, a: float, b: float) -> np.ndarray:
    """Exact window supremum alpha*(w), w = 0..W, of token-bucket(K,a,b): gate always fires, bucket
    full before the window, every refresh phase (verified against an exhaustive DP in
    verify_thm1.py). Returns int array of length W+1 with alpha*(0) = 0."""
    best = np.zeros(W + 1, dtype=np.int64)
    for phase in range(K):
        o = token_bucket_opens(np.ones(W, dtype=bool), K, a, b, phase)
        best = np.maximum(best, np.r_[0, np.cumsum(o)])
    return best


def window_counts(opens_by_seq: list[np.ndarray], W: int) -> np.ndarray:
    """Empirical arrival curve: max over sequences/positions of #opens in w slots, w = 0..W."""
    best = np.zeros(W + 1, dtype=np.int64)
    for o in opens_by_seq:
        cs = np.r_[0, np.cumsum(o, dtype=np.int64)]
        n = len(o)
        for w in range(1, W + 1):
            m = int(np.max(cs[w:] - cs[:-w])) if w <= n else int(cs[-1])
            if m > best[w]:
                best[w] = m
    return best


# ----------------------------------------------------------------------------- Prop 3 / Thm 2

def sigma(alpha_vals: np.ndarray, r: float) -> float:
    """sigma_alpha(r) = max_{w>=0} [alpha(w) - r w] for alpha given on w = 0..len-1 (alpha(0) = 0).
    The caller must make the grid long enough (see sigma_tb for the closed-form curve)."""
    w = np.arange(len(alpha_vals))
    return float(max(0.0, np.max(alpha_vals - r * w)))


def q_term(alpha_vals: np.ndarray, c: float) -> float:
    """Q(alpha, c) = max_{w>=0} [(alpha(w) + 1) c - w] in slots."""
    w = np.arange(len(alpha_vals))
    return float(np.max((alpha_vals + 1.0) * c - w))


def q_from_sequences(opens_by_seq: list[np.ndarray], c: float) -> float:
    """Q(alpha_hat, c) for the empirical envelope of the sequences, without a window grid:
    Q = c + max_seq [max sub-array sum of (c o_t - 1), empty allowed] (Kadane). Same as P0 q_exact."""
    best = 0.0
    for o in opens_by_seq:
        x = c * o.astype(float) - 1.0
        cs = np.r_[0.0, np.cumsum(x)]
        best = max(best, float(np.max(cs - np.minimum.accumulate(cs))))
    return best + c


def rho_tb(K: int, a: float) -> float:
    """Long-run (worst-case) activation rate of token-bucket(K,a,b): lim alpha(w)/w."""
    return min(1.0, 1.0 / K + a)


def sigma_tb(K: int, a: float, b: float, r: float, w_cap: int = 20_000_000) -> float:
    """sigma_alpha(r) of the Thm-1 curve. inf if r < rho (unstable), 0 if r >= 1.
    Exact up to the window cap: the search doubles the horizon W until the affine envelope
    alpha(w) - r w <= 1 + b - rho - (r - rho) w (w >= 1) cannot beat the running max; if the cap
    is hit the envelope value is returned (a valid upper bound, never an under-estimate)."""
    if r >= 1.0:
        return 0.0
    rho = 1.0 / K + a
    if r < rho - EPS_NUM:
        return math.inf
    env0 = 1.0 + b - rho                      # alpha(w) - rho w <= env0 for all w >= 1
    W = max(64, 4 * K)
    while True:
        best = sigma(alpha_tb(np.arange(W + 1), K, a, b).astype(float), r)
        tail = env0 - (r - rho) * (W + 1)
        if tail <= best:
            return best
        if W >= w_cap:
            return max(best, tail)
        W = min(2 * W, w_cap)


def q_tb(K: int, a: float, b: float, c: float) -> float:
    """Prop 3: Q(alpha, c) = c (1 + sigma_alpha(1/c)) for the Thm-1 curve."""
    return c * (1.0 + sigma_tb(K, a, b, 1.0 / c))


def e2e_bound_ms(K_dev: float, Q: float, c: float, T_ms: float, L_srv_ms: float, tau_f_ms: float
                 ) -> float:
    """Thm 2 (max form)."""
    return T_ms + L_srv_ms + tau_f_ms * max(K_dev - 1 + c, Q)


def e2e_bound_sum_ms(K_dev: float, Q: float, T_ms: float, L_srv_ms: float, tau_f_ms: float
                     ) -> float:
    """Naive sum form (device certificate + horizontal deviation), kept for comparison only."""
    return (K_dev - 1) * tau_f_ms + T_ms + Q * tau_f_ms + L_srv_ms


def k_max(delta_ms: float, T_ms: float, L_srv_ms: float, c: float, tau_f_ms: float) -> int:
    """Cor 2.1, branch 1: K - 1 + c <= D / tau_f with D = Delta - T - L_srv. 0 if no K >= 1 works."""
    D = delta_ms - T_ms - L_srv_ms
    return max(0, int(math.floor(D / tau_f_ms - c + EPS_NUM)) + 1)


def k_dev(delta_ms: float, L_srv_ms: float, tau_f_ms: float) -> int:
    """On-device reference (no link): (K - 1) tau_f + L_srv <= Delta."""
    return k_max(delta_ms, 0.0, L_srv_ms, 0.0, tau_f_ms)


# ----------------------------------------------------------------------------- Ext 2.2 (general beta)

def h_shift(alpha_bit_fn, beta_inv_fn, s_bits: float, t_grid_ms: np.ndarray) -> float:
    """h(alpha_bit + s, beta) = sup_{x>=0} [beta^-1(alpha_bit(x) + s) - x], evaluated on t_grid_ms
    (alpha_bit is a right-open staircase, so the sup is attained at x = k tau_f + 0; the caller
    passes those points)."""
    x = np.asarray(t_grid_ms, float)
    return float(np.max(beta_inv_fn(alpha_bit_fn(x) + s_bits) - x))


def e2e_bound_general_ms(K: int, tau_f_ms: float, s_bits: float, beta_inv_fn, alpha_bit_fn,
                         L_srv_ms: float, t_grid_ms: np.ndarray) -> float:
    """Ext 2.2: L <= L_srv + max{(K-1) tau_f + beta^-1(s), h(alpha_bit + s, beta)}."""
    return L_srv_ms + max((K - 1) * tau_f_ms + float(beta_inv_fn(np.array([s_bits]))[0]),
                          h_shift(alpha_bit_fn, beta_inv_fn, s_bits, t_grid_ms))


# ----------------------------------------------------------------------------- Thm 4 / Alg 1

def b_min_tight(a: float) -> float:
    """Smallest bucket depth of the tight regime of Thm 1' (b >= 1 + 2a, a <= 1/2)."""
    return 1.0 + 2.0 * a


def burst_budget(delta_ms: float, T_ms: float, L_srv_ms: float, c: float, tau_f_ms: float) -> float:
    """S = (Delta - T - L_srv)/(c tau_f) - 1: the largest admissible sigma_alpha(1/c) (Cor 2.1 + Prop 3)."""
    return (delta_ms - T_ms - L_srv_ms) / (c * tau_f_ms) - 1.0


def b_max_closed(delta_ms: float, T_ms: float, L_srv_ms: float, c: float, tau_f_ms: float,
                 K: int, a: float) -> float:
    """Thm 4 closed form (sufficient): for 1 < c < 1/rho,
        b_max = (1 - rho) (D/tau_f - 2c + 1) / (c - 1),     rho = 1/K + a, D = Delta - T - L_srv.
    +inf if c <= 1 (burst constraint vacuous, Prop 3); -inf if rho c >= 1 (unstable)."""
    rho = 1.0 / K + a
    if c <= 1.0:
        return math.inf
    if rho * c >= 1.0:
        return -math.inf
    D = (delta_ms - T_ms - L_srv_ms) / tau_f_ms
    return (1.0 - rho) * (D - 2.0 * c + 1.0) / (c - 1.0)


def b_max_exact(delta_ms: float, T_ms: float, L_srv_ms: float, c: float, tau_f_ms: float,
                K: int, a: float, b_hi: float = 1e4, tol: float = 1e-6) -> float:
    """Largest b with sigma_{alpha(K,a,b)}(1/c) <= S, exact Thm-1 curve (bisection; sigma is
    nondecreasing in b). +inf if c <= 1; -inf if infeasible even for b -> 0+ ... (returns the cap
    of the closed set; the gate needs b >= 1 to ever open)."""
    if c <= 1.0:
        return math.inf
    if (1.0 / K + a) * c >= 1.0:
        return -math.inf
    S = burst_budget(delta_ms, T_ms, L_srv_ms, c, tau_f_ms)
    ok = lambda b: sigma_tb(K, a, b, 1.0 / c) <= S + 1e-9
    lo = 0.0
    if not ok(lo):
        return -math.inf
    hi = max(1.0, b_max_closed(delta_ms, T_ms, L_srv_ms, c, tau_f_ms, K, a) + 4.0)
    while ok(hi) and hi < b_hi:
        hi *= 2.0
    if ok(hi):
        return math.inf
    # sigma is a step function of b (jumps where floor(b + a(w-1)) changes); bisection on the
    # boundary of {b : ok(b)} (a closed-left interval [0, b*) or [0, b*])
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class Plan:
    admitted: bool
    K: int
    a: float
    b: float
    rho: float
    K_max: int
    b_max: float
    reason: str


def alg1(delta_ms: float, T_ms: float, L_srv_ms: float, c: float, tau_f_ms: float, a_q: float,
         b_rule: str = "max", exact: bool = True) -> Plan:
    """Alg 1 (admission + dimensioning of one gated camera class).

    Input: SLA (Delta, eps) -> T = T_hat_eps (Prop 5), server latency L_srv, link load factor c,
    gate budget a_q required by the application (a_q in [0, 1/2]).
    Output: (K*, a*, b*) in F minimising the worst-case activation rate rho = 1/K + a, or reject.
      K* = K_max(Delta, T, c);  a* = a_q;  b* = b_max(Delta, T, c; K*, a_q) (b_rule='max': the
      most gate freedom the certificate allows) or b_min = 1 + 2a_q (b_rule='min').
    Optimal over F intersected with {a >= a_q} (Thm 4(iii)); F uses the exact sigma if exact=True,
    otherwise the closed-form b_max (a subset of F, still sufficient)."""
    Km = k_max(delta_ms, T_ms, L_srv_ms, c, tau_f_ms)
    if a_q > 0.5:
        return Plan(False, Km, a_q, math.nan, math.nan, Km, math.nan, "a_q > 1/2 outside Thm 1'")
    if Km < 1:
        return Plan(False, Km, a_q, math.nan, math.nan, Km, math.nan,
                    "deadline: Delta < T + L_srv + c tau_f (K_max < 1)")
    rho = rho_tb(Km, a_q)
    if rho * c >= 1.0:
        return Plan(False, Km, a_q, math.nan, rho, Km, -math.inf, "unstable: rho c >= 1")
    bm = (b_max_exact if exact else b_max_closed)(delta_ms, T_ms, L_srv_ms, c, tau_f_ms, Km, a_q)
    bmin = b_min_tight(a_q)
    if bm < bmin:
        return Plan(False, Km, a_q, math.nan, rho, Km, bm, "burst: b_max < 1 + 2a")
    b = bmin if b_rule == "min" else (bm if math.isfinite(bm) else math.inf)
    return Plan(True, Km, a_q, b, rho, Km, bm, "ok")


def feasible(K: int, a: float, b: float, delta_ms: float, T_ms: float, L_srv_ms: float, c: float,
             tau_f_ms: float) -> bool:
    """(K, a, b) in F: K <= K_max, sigma(1/c) <= S, b >= 1 + 2a, a <= 1/2, rho c < 1."""
    if a > 0.5 or b < b_min_tight(a) - EPS_NUM:
        return False
    if K < 1 or K > k_max(delta_ms, T_ms, L_srv_ms, c, tau_f_ms):
        return False
    if (1.0 / K + a) * c >= 1.0:
        return False
    return sigma_tb(K, a, b, 1.0 / c) <= burst_budget(delta_ms, T_ms, L_srv_ms, c, tau_f_ms) + 1e-9
