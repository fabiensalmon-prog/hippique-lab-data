"""Récurrences face au marché + valeur ajoutée des variables candidates. python3 analyse.py <repo> <full.pkl> <out.json>"""
import sys, json, math, pickle
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from hl.model import fit_clogit

D = pickle.load(open(sys.argv[2], "rb"))
df, races, BASE, NEW = D["runners"], D["races"], D["base"], D["new"]
for _c in df.columns:
    if isinstance(df[_c].dtype, pd.CategoricalDtype): df[_c] = df[_c].astype(object)
R = pd.DataFrame(races)
df["date"] = R["date"].values[df["race"].values]
df["group"] = R["group"].values[df["race"].values]
df["going"] = R["going"].values[df["race"].values]
df["wet"] = R["wet"].values[df["race"].values]
df["nr"] = R["n"].values[df["race"].values]
# marché final normalisé et marché du matin
ok = df.groupby("race")["odds_final"].transform(lambda s: s.notna().all() and (s > 1).all())
okr = df.groupby("race")["odds_ref"].transform(lambda s: s.notna().all() and (s > 1).all())
df = df[ok & okr & (df.groupby("race")["win"].transform("sum") == 1)].copy()
df["pm"] = (1 / df.odds_final) / df.groupby("race")["odds_final"].transform(lambda s: (1 / s).sum())
df["pr"] = (1 / df.odds_ref) / df.groupby("race")["odds_ref"].transform(lambda s: (1 / s).sum())
divsg = [r["div_sg"] for r in races]; divsp = [r["div_sp"] for r in races]
df["ret_sg"] = [divsg[r].get(str(n), 0.0) if w else 0.0 for r, n, w in zip(df.race, df.num, df.win)]
df["has_sp"] = [bool(divsp[r]) for r in df.race]
df["ret_sp"] = [divsp[r].get(str(n), 0.0) for r, n in zip(df.race, df.num)]
df["period"] = np.where(df.date < "2025-01-01", "A", "B")
print("partants", len(df), "courses", df.race.nunique(), df.date.min(), df.date.max(), flush=True)
out = {"n_runners": int(len(df)), "n_races": int(df.race.nunique()), "range": [df.date.min(), df.date.max()],
       "split": "A = jusqu'au 31/12/2024 (découverte), B = 2025-2026 (validation)"}


def tab(mask_or_key, frame=None, min_n=300):
    f = df if frame is None else frame
    g = f.groupby([mask_or_key, "period"]) if isinstance(mask_or_key, str) else f.groupby([mask_or_key, f.period])
    a = g.agg(n=("win", "size"), w=("win", "sum"), e=("pm", "sum"), er=("pr", "sum"), ret=("ret_sg", "sum"),
              nsp=("has_sp", "sum"), rsp=("ret_sp", "sum")).reset_index()
    a["ae"] = a.w / a.e; a["ae_ref"] = a.w / a.er; a["roi_sg"] = a.ret / a.n - 1; a["roi_sp"] = a.rsp / a.nsp.clip(lower=1) - 1
    a = a[a.n >= min_n]
    res = {}
    for _, r in a.iterrows():
        res.setdefault(str(r.iloc[0]), {})[r.period] = dict(n=int(r.n), wins=int(r.w), ae=round(r.ae, 3), ae_ref=round(r.ae_ref, 3),
                                                             roi_sg=round(r.roi_sg, 3), roi_sp=round(r.roi_sp, 3))
    return res


T = {}
df["odds_b"] = pd.cut(df.odds_ref, [1, 2, 3, 4, 6, 8, 11, 16, 25, 40, 1e9], labels=["1-2", "2-3", "3-4", "4-6", "6-8", "8-11", "11-16", "16-25", "25-40", "40+"]).astype(str)
for g in ("TROT", "PLAT", "OBSTACLE"):
    T[f"cote_matin_{g}"] = tab("odds_b", df[df.group == g])
df["fav"] = df.groupby("race")["odds_ref"].rank(method="first") == 1
T["favori_par_terrain"] = tab("going", df[df.fav & df.going.notna()], 200)
T["favori_pluie"] = tab("wet", df[df.fav])
df["nr_b"] = pd.cut(df.nr, [0, 7, 10, 13, 16, 30]).astype(str)
T["favori_nb_partants"] = tab("nr_b", df[df.fav])
T["sexe"] = tab("sexe")
T["age_trot"] = tab("age", df[df.group == "TROT"])
T["age_plat"] = tab("age", df[df.group == "PLAT"])
T["ferrure_trot"] = tab("deferre", df[df.group == "TROT"])
T["deferre_1ere_fois"] = tab("b_first_deferre", df[df.group == "TROT"])
T["oeilleres"] = tab("oeil")
T["changement_driver"] = tab("b_driver_change")
T["supplemente"] = tab("n_supplement")
T["jument_pleine"] = tab("n_jument_pleine", df[df.group == "TROT"], 50)
T["avis_entraineur_pos"] = tab("b_avis_pos")
T["avis_entraineur_neg"] = tab("b_avis_neg")
T["retour_repos_90j"] = tab("b_long_rest")
T["inedit"] = tab("b_inedit")
for k in ("n_trn_hot_ae", "n_drv_hot_ae", "n_sire_going_ae", "n_draw_ae", "b_trn_ae", "b_drv_ae", "n_trn_drv_ae", "n_horse_going_apt"):
    s = df[k].dropna()
    if len(s) < 1000:
        continue
    q = pd.Series(pd.qcut(s.rank(method="first"), 5, labels=["Q1 bas", "Q2", "Q3", "Q4", "Q5 haut"]).astype(str), index=s.index).reindex(df.index)
    T["quintile_" + k] = tab(q)
out["tables"] = T


# persistance : un entraîneur / driver / étalon « au-dessus du marché » en A le reste-t-il en B ?
def persist(key, frame=None, min_n=150):
    f = df if frame is None else frame
    a = f.groupby([key, "period"]).agg(n=("win", "size"), w=("win", "sum"), e=("pm", "sum"), ret=("ret_sg", "sum")).unstack()
    a = a.dropna()
    a = a[(a[("n", "A")] >= min_n) & (a[("n", "B")] >= min_n // 2)]
    aeA = (a[("w", "A")] + 5) / (a[("e", "A")] + 5); aeB = (a[("w", "B")] + 5) / (a[("e", "B")] + 5)
    top = aeA >= aeA.quantile(0.8); bot = aeA <= aeA.quantile(0.2)
    def grp(m):
        return dict(n_entites=int(m.sum()), partants_B=int(a[("n", "B")][m].sum()),
                    ae_A=round(float(a[("w", "A")][m].sum() / a[("e", "A")][m].sum()), 3),
                    ae_B=round(float(a[("w", "B")][m].sum() / a[("e", "B")][m].sum()), 3),
                    roi_sg_B=round(float(a[("ret", "B")][m].sum() / a[("n", "B")][m].sum() - 1), 3))
    return dict(entites=int(len(a)), corr_ae=round(float(np.corrcoef(np.log(aeA), np.log(aeB))[0, 1]), 3),
                top20=grp(top), bottom20=grp(bot))


P = {"entraineur": persist("trainer"), "driver_jockey": persist("driver"), "etalon": persist("sire")}
df["sire_going"] = df.sire.fillna("") + "|" + df.going.fillna("")
P["etalon_x_terrain_plat_obstacle"] = persist("sire_going", df[(df.group != "TROT") & df.going.notna() & df.sire.notna()], 60)
df["trn_hippo"] = df.trainer.fillna("") + "|" + R["hippo"].values[df.race.values]
P["entraineur_x_hippodrome"] = persist("trn_hippo", min_n=60)
df["trn_drv"] = df.trainer.fillna("") + "|" + df.driver.fillna("")
P["entraineur_x_driver"] = persist("trn_drv", min_n=60)
out["persistance"] = P
print("tables ok", flush=True)

# ---------- 2sur4 : dividende estimé (formule actuelle) vs réel ----------
from hl.model import fit_pair_dividend, pair_expected
odds_map = {(r, str(n)): o for r, n, o in zip(df.race, df.num.astype(str), df.odds_ref)}
nmap = df.groupby("race").size().to_dict()
S4 = []
for ri in nmap:
    rc = races[ri]
    for combo, dv in (rc["div_2s4"] or {}).items():
        a, _, b = combo.partition("-")
        oa, ob = odds_map.get((ri, a)), odds_map.get((ri, b))
        if oa and ob:
            S4.append((rc["date"], oa, ob, nmap[ri], dv))
S4 = pd.DataFrame(S4, columns=["date", "oa", "ob", "n", "div"])
coef = fit_pair_dividend([tuple(x) for x in S4[S4.date < "2025-01-01"][["oa", "ob", "n", "div"]].to_numpy()][-8000:])
B4 = S4[S4.date >= "2025-01-01"].copy()
B4["pred"] = [pair_expected(coef, a, b, n) for a, b, n in zip(B4.oa, B4.ob, B4.n)]
B4["pb"] = pd.cut(B4.pred, [0, 2, 3, 4, 6, 10, 20, 1e9]).astype(str)
B4["ratio"] = B4["div"] / B4.pred
out["deux_sur_quatre_dividende"] = dict(
    coef=coef, n_combinaisons_B=int(len(B4)),
    par_dividende_predit={k: dict(n=int(len(g)), predit_median=round(float(g.pred.median()), 2), reel_median=round(float(g["div"].median()), 2),
                                  reel_moyen=round(float(g["div"].mean()), 2), ratio_moyen=round(float(g.ratio.mean()), 3))
                          for k, g in B4.groupby("pb")})
# 2sur4 sur les deux favoris du matin : ROI réel
fav2 = df[df.has_sp].sort_values(["race", "odds_ref"]).groupby("race").head(2)
pairs = fav2.groupby("race").num.apply(lambda s: "-".join(str(v) for v in sorted(int(x) for x in s)))
ret = []
for ri, k in pairs.items():
    d4 = races[ri]["div_2s4"]
    if d4:
        ret.append((races[ri]["date"] >= "2025-01-01", d4.get(k, 0.0)))
ret = pd.DataFrame(ret, columns=["B", "r"])
out["deux_sur_quatre_2_favoris"] = {("B" if b else "A"): dict(n=int(len(g)), taux_gain=round(float((g.r > 0).mean()), 3),
                                                              roi=round(float(g.r.mean() - 1), 3)) for b, g in ret.groupby("B")}
print("2s4 ok", json.dumps(out["deux_sur_quatre_dividende"]["par_dividende_predit"]), flush=True)

# ---------- valeur ajoutée face au marché FINAL ----------
df = df.sort_values(["race", "num"]).reset_index(drop=True)


def design(frame, cols, mu=None, sd=None):
    X = frame[cols].to_numpy(dtype=float)
    if mu is None:
        mu = np.nanmean(X, 0); sd = np.nanstd(X, 0); mu = np.nan_to_num(mu); sd = np.where(np.nan_to_num(sd) < 1e-9, 1, np.nan_to_num(sd))
    X = np.nan_to_num(np.clip((X - mu) / sd, -5, 5))
    r = frame.race.to_numpy()
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    seg = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(r)]))
    y = np.flatnonzero(frame.win.to_numpy() == 1)
    return X, starts, seg, y, mu, sd


def ll(score, starts, seg, y):
    m = np.maximum.reduceat(score, starts)
    lse = m + np.log(np.add.reduceat(np.exp(score - m[seg]), starts))
    return float((score[y] - lse).sum())


S1 = df[df.date < "2024-01-01"]; S2 = df[(df.date >= "2024-01-01") & (df.date < "2025-01-01")]; TE = df[df.date >= "2025-01-01"]
NEWF = ["n_" + k for k in NEW if k not in ("fav_ref", "n_partants")]
BASEF = ["b_" + k for k in BASE]
VA = {}
for g in ("TROT", "PLAT", "OBSTACLE"):
    s1, s2, te = S1[S1.group == g], S2[S2.group == g], TE[TE.group == g]
    res = {"courses_test": int(te.race.nunique())}
    lp = lambda f: np.log(np.clip(f.pm.to_numpy(), 1e-6, 1))
    _, st_, sg_, y_, *_ = design(te, BASEF[:1])
    ll0 = -float(np.log(np.diff(np.r_[st_, len(te)])).sum())
    llm = ll(lp(te), st_, sg_, y_)
    res["rho_marche_final"] = round(1 - llm / ll0, 5)
    sc = {}
    for name, cols in (("actuel", BASEF), ("actuel+nouvelles", BASEF + NEWF)):
        X1, a1, g1, y1, mu, sd = design(s1, cols)
        w = fit_clogit(X1, a1, g1, y1, l2=2.0)
        X2, a2, g2, y2, *_ = design(s2, cols, mu, sd); Xt, at, gt, yt, *_ = design(te, cols, mu, sd)
        Z2 = np.column_stack([X2 @ w, lp(s2)]); Zt = np.column_stack([Xt @ w, lp(te)])
        ab = fit_clogit(Z2, a2, g2, y2, l2=1e-3, w0=np.array([0.3, 1.0]))
        llc = ll(Zt @ ab, at, gt, yt)
        res[name] = dict(rho_fond=round(1 - ll(Xt @ w, at, gt, yt) / ll0, 5), rho_combine=round(1 - llc / ll0, 5),
                         delta_rho_vs_final=round((1 - llc / ll0) - (1 - llm / ll0), 5), alpha=round(ab[0], 3), beta=round(ab[1], 3))
        sc[name] = (Xt @ w, X2 @ w)
        if name == "actuel+nouvelles":
            res["poids_nouvelles"] = sorted([(c[2:], round(float(v), 3)) for c, v in zip(cols, w) if c in NEWF], key=lambda t: -abs(t[1]))
    # une variable à la fois, au-dessus de actuel + marché
    one = {}
    for c in NEWF:
        X2, a2, g2, y2, mu, sd = design(s2, [c]); Xt, at, gt, yt, *_ = design(te, [c], mu, sd)
        Z2 = np.column_stack([sc["actuel"][1], lp(s2), X2[:, 0]]); Zt = np.column_stack([sc["actuel"][0], lp(te), Xt[:, 0]])
        ab = fit_clogit(Z2, a2, g2, y2, l2=1e-3, w0=np.array([0.3, 1.0, 0.0]))
        Zb = Z2[:, :2]; ab0 = fit_clogit(Zb, a2, g2, y2, l2=1e-3, w0=np.array([0.3, 1.0]))
        one[c[2:]] = dict(gain_ll_par_1000_courses=round((ll(Zt @ ab, at, gt, yt) - ll(Zt[:, :2] @ ab0, at, gt, yt)) / len(at) * 1000, 2),
                          coef=round(float(ab[2]), 3))
    res["une_par_une"] = dict(sorted(one.items(), key=lambda t: -t[1]["gain_ll_par_1000_courses"]))
    VA[g] = res
    print(g, json.dumps({k: v for k, v in res.items() if k not in ("une_par_une", "poids_nouvelles")}), flush=True)
out["valeur_ajoutee"] = VA
json.dump(out, open(sys.argv[3], "w"), indent=1, ensure_ascii=False, default=str)
print("ok")
