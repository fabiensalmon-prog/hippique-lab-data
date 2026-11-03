"""Logit conditionnel (Benter étage 1 et 2), Harville corrigé, rapport placé attendu."""
import math
import numpy as np
from scipy.optimize import minimize

from .features import FEATURES


class Design:
    """Courses -> matrice concaténée + index de segments (une course = un segment)."""

    def __init__(self, races, means=None, stds=None):
        rows, starts, y, self.meta = [], [], [], []
        for r in races:
            if r["winner_idx"] is None:
                continue
            starts.append(len(rows))
            y.append(len(rows) + r["winner_idx"])
            for x in r["rows"]:
                rows.append([np.nan if x["x"].get(k) is None else float(x["x"][k]) for k in FEATURES])
            self.meta.append(r)
        X = np.array(rows, dtype=float) if rows else np.zeros((0, len(FEATURES)))
        if means is None:
            means = np.nanmean(X, axis=0) if len(X) else np.zeros(len(FEATURES))
            stds = np.nanstd(X, axis=0) if len(X) else np.ones(len(FEATURES))
            means = np.nan_to_num(means); stds = np.where(np.nan_to_num(stds) < 1e-9, 1.0, np.nan_to_num(stds))
        self.means, self.stds = means, stds
        X = (X - means) / stds
        self.X = np.nan_to_num(np.clip(X, -5, 5))
        self.starts = np.array(starts, dtype=int)
        self.y = np.array(y, dtype=int)
        self.seg = np.repeat(np.arange(len(starts)), np.diff(np.append(self.starts, len(self.X)))) if len(starts) else np.zeros(0, int)


def seg_logsumexp(s, starts, seg):
    m = np.maximum.reduceat(s, starts)
    e = np.exp(s - m[seg])
    return m + np.log(np.add.reduceat(e, starts))


def fit_clogit(X, starts, seg, y, l2=1.0, w0=None):
    k = X.shape[1]

    def f(w):
        s = X @ w
        lse = seg_logsumexp(s, starts, seg)
        p = np.exp(s - lse[seg])
        nll = -(s[y].sum() - lse.sum()) + 0.5 * l2 * w @ w
        g = -(X[y].sum(0) - p @ X) + l2 * w
        return nll, g
    res = minimize(f, np.zeros(k) if w0 is None else w0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    return res.x


def probs(X, starts, seg, w):
    s = X @ w
    return np.exp(s - seg_logsumexp(s, starts, seg)[seg])


def split_probs(p, starts):
    return np.split(p, starts[1:])


# ---------- second étage ----------
def fit_stage2(pf_list, pm_list, winners):
    """pf_list/pm_list : listes d'arrays par course ; winners : index du gagnant. -> (alpha, beta)."""
    Z, starts, y = [], [], []
    for pf, pm, w in zip(pf_list, pm_list, winners):
        starts.append(len(Z)); y.append(len(Z) + w)
        Z.extend(np.column_stack([np.log(np.clip(pf, 1e-6, 1)), np.log(np.clip(pm, 1e-6, 1))]).tolist())
    Z = np.array(Z); starts = np.array(starts); y = np.array(y)
    seg = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, len(Z))))
    ab = fit_clogit(Z, starts, seg, y, l2=1e-3, w0=np.array([0.3, 1.0]))
    return float(ab[0]), float(ab[1])


def combine(pf, pm, alpha, beta):
    z = alpha * np.log(np.clip(pf, 1e-6, 1)) + beta * np.log(np.clip(pm, 1e-6, 1))
    z = np.exp(z - z.max())
    return z / z.sum()


def loglik(p_list, winners):
    return float(sum(math.log(max(1e-9, p[w])) for p, w in zip(p_list, winners)))


# ---------- Harville corrigé (Benter) ----------
def place_probs(p, k, gamma=0.81, delta=0.65):
    """Probabilité de finir dans les k premiers (k = 2 ou 3), Harville avec puissances gamma (2e) et delta (3e). O(n²)."""
    p = np.asarray(p, float); n = len(p)
    p2 = p ** gamma; p3 = p ** delta
    S2, S3 = p2.sum(), p3.sum()
    second = np.zeros(n); third = np.zeros(n)
    for i in range(n):
        if p[i] <= 0:
            continue
        s2 = S2 - p2[i]
        if s2 <= 0:
            continue
        q2 = p2 / s2; q2[i] = 0.0
        second += p[i] * q2
        if k >= 3:
            s3 = S3 - p3[i] - p3                      # dénominateur pour chaque 2e possible j
            w = np.where((s3 > 0) & (q2 > 0), p[i] * q2 / np.where(s3 > 0, s3, 1.0), 0.0)
            W = w.sum()
            t = p3 * (W - w); t[i] = 0.0
            third += t
    return p + second + (third if k >= 3 else 0)


def fit_harville(p_list, orders, grid=None):
    """Maximise la vraisemblance des 2e et 3e places observées. orders : index des 3 premiers."""
    grid = grid or [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best = (0.81, 0.65, -1e18)
    for g in grid:
        for dl in grid:
            ll = 0.0
            for p, o in zip(p_list, orders):
                if len(o) < 3:
                    continue
                a, b, c = o[:3]
                r2 = p ** g; r2[a] = 0; ll += math.log(max(1e-9, r2[b] / r2.sum()))
                r3 = p ** dl; r3[a] = 0; r3[b] = 0; ll += math.log(max(1e-9, r3[c] / r3.sum()))
            if ll > best[2]:
                best = (g, dl, ll)
    return best[0], best[1]


# ---------- rapport placé attendu ----------
def fit_sp_dividend(samples):
    """samples : (odds_ref, n_partants, dividende placé) des chevaux placés -> coefficients log-linéaires."""
    samples = [s for s in samples if s[2] and s[2] > 0 and s[0] and s[0] > 1 and s[1] and s[1] > 1]
    if len(samples) < 50:
        return [0.15, 0.45, 0.0]
    A = np.array([[1.0, math.log(o), math.log(n)] for o, n, _ in samples])
    b = np.array([math.log(dv) for _, _, dv in samples])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef.tolist()


def sp_expected(coef, odds, n):
    return max(1.05, math.exp(coef[0] + coef[1] * math.log(odds) + coef[2] * math.log(n)))
