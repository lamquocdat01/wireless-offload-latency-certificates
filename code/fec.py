"""Frame-level certificate with packet-level erasure coding (topic #19, Thm 6 / Prop 7). Pure numpy/scipy.

A frame = N data packets + r repair packets of an MDS erasure code (any N of the N + r packets
recover the frame). A packet is BAD if it is lost or its one-way delay exceeds the certified T_hat.

  Thm 6   independent bad indicators with P(bad) <= p:  P_frame <= P(Bin(N + r, p) > r)
          r*(N, p, eps_frame) = min{ r >= 0 : P(Bin(N + r, p) > r) <= eps_frame }
          c_eff = c (N + r*) / N
  Prop 7  Gilbert-Elliott (GE) burst model: states G (never bad), B (bad w.p. h); transitions
          p_gb = P(G->B), p_bg = P(B->G). Fitted by maximum likelihood on the bad-indicator sequence of
          a session (run-compressed forward algorithm). P_frame under GE computed exactly by a
          (state, count) dynamic programme over the M = N + r packets of a frame, started in the
          stationary law. Time scaling: per-packet transition matrix P^(delta/iv) for a packet spacing
          delta [ms] when the trace spacing is iv [ms].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


# ------------------------------------------------------------------------------- Thm 6

def p_frame_indep(N: int, r: int, p: float) -> float:
    """P(Bin(N + r, p) > r)."""
    return float(stats.binom.sf(r, N + r, p))


def r_star_indep(N: int, p: float, eps_frame: float, r_max: int | None = None) -> int:
    """Smallest r with P(Bin(N + r, p) > r) <= eps_frame (P is nonincreasing in r). -1 if none <= r_max."""
    r_max = N if r_max is None else r_max
    for r in range(0, r_max + 1):
        if p_frame_indep(N, r, p) <= eps_frame:
            return r
    return -1


def union_bound(N: int, r: int, p: float) -> float:
    """Pessimistic reference without coding gain: (N + r) p (any bad packet breaks the frame)."""
    return min(1.0, (N + r) * p)


def c_eff(c: float, N: int, r: int) -> float:
    return c * (N + r) / N


# ------------------------------------------------------------------------------- Prop 7: GE

@dataclass
class GE:
    p_gb: float
    p_bg: float
    h: float           # P(bad | B)
    loglik: float = math.nan

    @property
    def P(self) -> np.ndarray:
        return np.array([[1 - self.p_gb, self.p_gb], [self.p_bg, 1 - self.p_bg]])

    @property
    def pi(self) -> np.ndarray:
        s = self.p_gb + self.p_bg
        return np.array([self.p_bg / s, self.p_gb / s])

    @property
    def mean_bad(self) -> float:
        return float(self.pi[1] * self.h)

    def scaled(self, k: float) -> "GE":
        """Per-step transition for a step of k trace intervals (k may be < 1): P^k (2-state chain)."""
        s = self.p_gb + self.p_bg
        lam = (1 - s) ** k                       # second eigenvalue of P^k
        return GE(self.pi[1] * (1 - lam), self.pi[0] * (1 - lam), self.h, self.loglik)


def _runs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode a 0/1 sequence as (zero-run lengths before each 1, trailing zeros): x = 0^{z0} 1 0^{z1} 1 ..."""
    ones = np.flatnonzero(x)
    z = np.diff(np.r_[-1, ones]) - 1
    tail = len(x) - 1 - ones[-1] if len(ones) else len(x)
    return z.astype(np.int64), int(tail)


def _mat_pow_log(M: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """M^n = exp(logscale) * R with max|R| = 1 (repeated squaring with renormalisation)."""
    R, ls = np.eye(2), 0.0
    A, la = M.copy(), 0.0
    while n:
        if n & 1:
            R = R @ A; ls += la
            m = np.abs(R).max(); R /= m; ls += math.log(m)
        n >>= 1
        if n:
            A = A @ A; la *= 2
            m = np.abs(A).max(); A /= m; la += math.log(m)
    return R, ls


def ge_loglik(params: GE, z: np.ndarray, tail: int) -> float:
    """Exact log-likelihood, vectorised. M1 = P diag(0, h) has rank one (its image is e_B), so after
    every bad packet the normalised forward vector is e_B and the likelihood factorises over the
    runs of good packets: ll = sum_i log(u_i' M0^{z_i} m1) + log(e_B' M0^{tail} 1), u_0 = pi,
    u_i = e_B (i >= 1), M0 = P diag(1, 1 - h), m1 = P[:, 1] h. M0^z via the 2x2 eigen-decomposition
    in log scale (real eigenvalues: M0 is entrywise nonnegative with positive off-diagonals)."""
    P = params.P
    M0 = P * np.array([1.0, 1 - params.h])[None, :]
    m1 = P[:, 1] * params.h
    lam, V = np.linalg.eig(M0)
    lam, V = lam.real, V.real
    o = np.argsort(-np.abs(lam)); lam, V = lam[o], V[:, o]
    Vi = np.linalg.inv(V)
    rho = lam[1] / lam[0]

    def logq(u, zz, w):
        zz = np.asarray(zz, dtype=np.int64)
        a, b = (u @ V), (Vi @ w)
        sgn = np.where(zz % 2 == 1, -1.0, 1.0) if rho < 0 else 1.0     # rho^z for integer z, rho < 0
        val = a[0] * b[0] + a[1] * b[1] * np.power(abs(rho), zz.astype(float)) * sgn
        return zz * math.log(lam[0]) + np.log(np.maximum(val, 1e-300))
    if len(z) == 0:
        return float(logq(params.pi, [tail], np.ones(2))[0])
    eB = np.array([0.0, 1.0])
    ll = float(logq(params.pi, z[:1], m1)[0]) + float(np.sum(logq(eB, z[1:], m1)))
    ll += float(logq(eB, [tail], np.ones(2))[0])
    return ll


def fit_ge(x: np.ndarray) -> GE:
    """MLE of (p_gb, p_bg, h) on a 0/1 sequence (1 = bad). The first state follows the stationary law."""
    z, tail = _runs(x.astype(np.int8))
    m = x.mean()
    sig = lambda u: 1 / (1 + np.exp(-u))

    def nll(u):
        g = GE(sig(u[0]), sig(u[1]), sig(u[2]))
        return -ge_loglik(g, z, tail)
    best = None
    for h0 in (0.05, 0.3, 0.9):
        for pbg0 in (0.01, 0.3):
            pgb0 = m * pbg0 / max(h0 - m, 1e-6)
            u0 = np.log(np.array([pgb0, pbg0, h0]) / (1 - np.array([pgb0, pbg0, h0])))
            r = optimize.minimize(nll, u0, method="Nelder-Mead",
                                  options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 3000})
            if best is None or r.fun < best.fun:
                best = r
    g = GE(sig(best.x[0]), sig(best.x[1]), sig(best.x[2]))
    g.loglik = -best.fun
    return g


def bernoulli_loglik(x: np.ndarray) -> float:
    m, n = int(x.sum()), len(x)
    p = m / n
    return m * math.log(p) + (n - m) * math.log(1 - p) if 0 < m < n else 0.0


def p_frame_ge(g: GE, N: int, r: int) -> float:
    """P(#bad > r among M = N + r consecutive packets), GE chain started in the stationary law."""
    M = N + r
    P = g.P
    # f[s, k]: prob of being in state s with k bad so far (k capped at r + 1)
    f = np.zeros((2, r + 2))
    emit = np.array([[1.0, 0.0], [1 - g.h, g.h]])          # [state, bad?]
    pi = g.pi
    for s in (0, 1):
        f[s, 0] += pi[s] * emit[s, 0]
        f[s, min(1, r + 1)] += pi[s] * emit[s, 1]
    for _ in range(M - 1):
        nf = np.zeros_like(f)
        for s in (0, 1):
            for s2 in (0, 1):
                w = f[s] * P[s, s2]
                nf[s2] += w * emit[s2, 0]
                nf[s2, 1:] += w[:-1] * emit[s2, 1]
                nf[s2, r + 1] += w[r + 1] * emit[s2, 1]
        f = nf
    return float(f[:, r + 1].sum())


def r_star_ge(g: GE, N: int, eps_frame: float, r_max: int | None = None) -> int:
    r_max = N if r_max is None else r_max
    for r in range(0, r_max + 1):
        if p_frame_ge(g, N, r) <= eps_frame:
            return r
    return -1


def simulate_ge_frames(g: GE, N: int, r: int, n_frames: int, rng: np.random.Generator,
                       chunk: int = 200_000) -> float:
    """MC estimate of P_frame under GE (frames independent, each started stationary)."""
    M, bad_frames, done = N + r, 0, 0
    while done < n_frames:
        k = min(chunk, n_frames - done)
        s = (rng.random(k) < g.pi[1]).astype(np.int8)
        cnt = np.zeros(k, dtype=np.int32)
        for t in range(M):
            if t:
                u = rng.random(k)
                s = np.where(s == 0, (u < g.p_gb), (u >= g.p_bg)).astype(np.int8)
            cnt += (s == 1) & (rng.random(k) < g.h)
        bad_frames += int(np.sum(cnt > r)); done += k
    return bad_frames / n_frames


def empirical_block_rate(bad: np.ndarray, M: int, r: int, step: int | None = None) -> tuple[float, int]:
    """Model-free: fraction of blocks of M consecutive trace packets with > r bad (non-overlapping
    blocks if step is None, else sliding with the given step)."""
    step = M if step is None else step
    cs = np.r_[0, np.cumsum(bad.astype(np.int64))]
    starts = np.arange(0, len(bad) - M + 1, step)
    cnt = cs[starts + M] - cs[starts]
    return float(np.mean(cnt > r)), int(len(starts))
