"""Test de sous-ensembles de variables nouvelles + recalibrage du dividende 2sur4."""
import sys, json, math, pickle
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from hl.model import fit_clogit, fit_pair_dividend, pair_expected

D = pickle.load(open(sys.argv[2], "rb"))
df, races, BASE = D["runners"], D["races"], D["base"]
R = pd.DataFrame(races)
df["date"] = R["date"].values[df.race.values]; df["group"] = R["group"].values[df.race.values]
ok = df.groupby("race")["odds_final"].transform(lambda s: s.notna().all() and (s > 1).all())
df = df[ok & (df.groupby("race")["win"].transform("sum") == 1)].copy().sort_values(["race", "num"]).reset_index(drop=True)
df["pm"] = (1 / df.odds_final) / df.groupby("race")["odds_final"].transform(lambda s: (1 / s).sum())


def design(f, cols, mu=None, sd=None):
    X = f[cols].to_numpy(dtype=float)
    if mu is None:
        mu = np.nan_to_num(np.nanmean(X, 0)); sd = np.nanstd(X, 0); sd = np.where(np.nan_to_num(sd) < 1e-9, 1, np.nan_to_num(sd))
    X = np.nan_to_num(np.clip((X - mu) / sd, -5, 5))
    r = f.race.to_numpy(); st = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    seg = np.repeat(np.arange(len(st)), np.diff(np.r_[st, len(r)]))
    return X, st, seg, np.flatnonzero(f.win.to_numpy() == 1), mu, sd


def ll(s, st, seg, y):
    m = np.maximum.reduceat(s, st); return float((s[y] - (m + np.log(np.add.reduceat(np.exp(s - m[seg]), st)))).sum())


BASEF = ["b_" + k for k in BASE]
SUBS = {"actuel": [], "entourage": ["n_drv_disc_ae", "n_trn_disc_ae", "n_trn_hot_ae", "n_trn_track_ae", "n_drv_track_ae", "n_trn_drv_ae"],
        "entourage+etalon": ["n_drv_disc_ae", "n_trn_disc_ae", "n_trn_hot_ae", "n_trn_track_ae", "n_drv_track_ae", "n_trn_drv_ae", "n_sire_ae", "n_draw_ae"]}
out = {}
# deux découpages : stage1 <2024 / stage2 2024 / test 2025-26, et stage1 <2023 / stage2 2023 / test 2024 (robustesse)
for cut in (("2024-01-01", "2025-01-01", "2099"), ("2023-01-01", "2024-01-01", "2025-01-01")):
    S1 = df[df.date < cut[0]]; S2 = df[(df.date >= cut[0]) & (df.date < cut[1])]; TE = df[(df.date >= cut[1]) & (df.date < cut[2])]
    for g in ("TROT", "PLAT", "OBSTACLE"):
        s1, s2, te = S1[S1.group == g], S2[S2.group == g], TE[TE.group == g]
        lp = lambda f: np.log(np.clip(f.pm.to_numpy(), 1e-6, 1))
        _, st, sg, y, *_ = design(te, BASEF[:1]); ll0 = -float(np.log(np.diff(np.r_[st, len(te)])).sum()); llm = ll(lp(te), st, sg, y)
        for name, extra in SUBS.items():
            cols = BASEF + extra
            X1, a1, g1, y1, mu, sd = design(s1, cols); w = fit_clogit(X1, a1, g1, y1, l2=2.0)
            X2, a2, g2, y2, *_ = design(s2, cols, mu, sd); Xt, at, gt, yt, *_ = design(te, cols, mu, sd)
            ab = fit_clogit(np.column_stack([X2 @ w, lp(s2)]), a2, g2, y2, l2=1e-3, w0=np.array([0.3, 1.0]))
            llc = ll(np.column_stack([Xt @ w, lp(te)]) @ ab, at, gt, yt)
            out.setdefault(f"test>={cut[1]}", {}).setdefault(g, {})[name] = dict(
                delta_rho=round((llm - llc) / ll0, 5), alpha=round(ab[0], 3), courses=int(len(at)))
        print(cut, g, out[f"test>={cut[1]}"][g], flush=True)

# 2sur4 : recalibrage du dividende
odds = {(r, str(n)): o for r, n, o in zip(df.race, df.num.astype(str), df.odds_ref)}
nr = df.groupby("race").size().to_dict()
rows = []
for ri in nr:
    for combo, dv in (races[ri]["div_2s4"] or {}).items():
        a, _, b = combo.partition("-"); oa, ob = odds.get((ri, a)), odds.get((ri, b))
        if oa and ob and oa > 1 and ob > 1:
            rows.append((races[ri]["date"], oa, ob, nr[ri], dv))
S = pd.DataFrame(rows, columns=["date", "oa", "ob", "n", "div"])
A = S[S.date < "2025-01-01"]; B = S[S.date >= "2025-01-01"].copy()
c_old = fit_pair_dividend([tuple(x) for x in A[["oa", "ob", "n", "div"]].to_numpy()][-8000:])
c_all = fit_pair_dividend([tuple(x) for x in A[["oa", "ob", "n", "div"]].to_numpy()])


def feats(f):
    lo = np.log(f.oa * f.ob); lmin = np.log(np.minimum(f.oa, f.ob))
    return np.column_stack([np.ones(len(f)), lo, np.log(f.n), lo ** 2, lmin])


Xa = feats(A); ya = np.log(A["div"].to_numpy())
c_new, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
smear = float(np.mean(np.exp(ya - Xa @ c_new)))          # estimateur de Duan (moyenne sans hypothèse log-normale)
B["old"] = [pair_expected(c_old, a, b, n) for a, b, n in zip(B.oa, B.ob, B.n)]
B["all"] = [pair_expected(c_all, a, b, n) for a, b, n in zip(B.oa, B.ob, B.n)]
B["new"] = np.exp(feats(B) @ c_new) * smear
B["bk"] = pd.cut(B.old, [0, 2, 3, 4, 6, 10, 20, 1e9]).astype(str)
cal = {}
for k, g in B.groupby("bk"):
    cal[k] = dict(n=int(len(g)), reel_moyen=round(float(g["div"].mean()), 2), predit_actuel=round(float(g.old.mean()), 2),
                  predit_8000_vs_tout=round(float(g["all"].mean()), 2), predit_nouveau=round(float(g.new.mean()), 2))
err = lambda c: float(np.mean(np.abs(np.log(B[c]) - np.log(B["div"]))))
out["s4"] = dict(coef_nouveau=c_new.tolist(), smear=smear, calibration=cal,
                 erreur_log_moyenne=dict(actuel=round(err("old"), 4), nouveau=round(err("new"), 4)))
print(json.dumps(out["s4"], indent=0))
json.dump(out, open(sys.argv[3], "w"), indent=1)
