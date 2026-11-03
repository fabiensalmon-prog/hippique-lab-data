"""Entraînement V1 + backtest walk-forward mensuel.

python3 -m hl.train            -> models/latest.json, reports/backtest.json
Chaque mois de test n'est prédit qu'avec un modèle entraîné sur les mois précédents (étage 1),
et l'étage 2 n'est estimé que sur des prédictions hors échantillon des mois précédents.
Décisions sur le rapport de référence (matin), règlement sur les dividendes officiels.
"""
import json, math, os, sys, time
from collections import defaultdict
import numpy as np

from .store import iter_races
from .features import FeatureState, FEATURES
from .model import Design, fit_clogit, probs, split_probs, fit_stage2, combine, fit_harville, fit_sp_dividend, loglik
from .decide import analyse, size, n_places, RULES

ROOT = os.path.join(os.path.dirname(__file__), "..")
MIN_TRAIN_MONTHS = 2
L2 = 2.0


def build_dataset():
    st = FeatureState()
    out = []
    for race in iter_races():
        rows = st.features(race)
        st.update(race)
        if len(rows) < 4:
            continue
        nums = [r["num"] for r in rows]
        w = race["arrivee"][0] if race["arrivee"] else None
        out.append(dict(id=race["id"], date=race["date"], month=race["date"][:7], group=race["group"], rows=rows,
                        n=len(rows), winner_idx=nums.index(w) if w in nums else None,
                        order=[nums.index(a) for a in race["arrivee"][:3] if a in nums],
                        div_sg=race["div_sg"], div_sp=race["div_sp"], dead_heat=race["dead_heat"],
                        hippodrome=race["hippodrome"], heure=race["heure"]))
    return out, st


def fit_group(races):
    D = Design(races)
    if len(D.starts) < 50:
        return None
    w = fit_clogit(D.X, D.starts, D.seg, D.y, l2=L2)
    return dict(w=w.tolist(), means=D.means.tolist(), stds=D.stds.tolist(), n_races=int(len(D.starts)))


def predict_group(model, races):
    D = Design(races, np.array(model["means"]), np.array(model["stds"]))
    if not len(D.starts):
        return {}
    p = probs(D.X, D.starts, D.seg, np.array(model["w"]))
    return {r["id"]: pr for r, pr in zip(D.meta, split_probs(p, D.starts))}


def predict_any(model, race):
    """Prédiction d'une course sans résultat (live)."""
    tmp = dict(race, winner_idx=0)
    return predict_group(model, [tmp]).get(race["id"])


def mkt(r):
    o = [x["odds_ref"] for x in r["rows"]]
    if not all(v and v > 1 for v in o):
        return None
    inv = np.array([1 / v for v in o]); return inv / inv.sum()


def main():
    t0 = time.time()
    data, _ = build_dataset()
    months = sorted({r["month"] for r in data})
    print(f"{len(data)} courses, {len(months)} mois : {months[:1]} -> {months[-1:]}", flush=True)
    oos = {}                      # id -> P fondamentale hors échantillon
    report = dict(months=[], built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), n_races=len(data))
    bank = dict(capital=500.0, peak=500.0, mdd=0.0)
    flat = defaultdict(lambda: dict(n=0, staked=0.0, ret=0.0, wins=0))
    bets_log = []
    for mi, m in enumerate(months):
        train = [r for r in data if r["month"] < m]
        test = [r for r in data if r["month"] == m]
        if mi < MIN_TRAIN_MONTHS:
            continue
        for g in ("TROT", "PLAT"):
            mdl = fit_group([r for r in train if r["group"] == g])
            if mdl:
                oos.update(predict_group(mdl, [r for r in test if r["group"] == g]))
        # étage 2 + Harville + placé sur les mois précédents (hors échantillon)
        prev = [r for r in train if r["id"] in oos and r["winner_idx"] is not None and mkt(r) is not None]
        if len(prev) < 200:
            continue
        alpha, beta = fit_stage2([oos[r["id"]] for r in prev], [mkt(r) for r in prev], [r["winner_idx"] for r in prev])
        pc_prev = [combine(oos[r["id"]], mkt(r), alpha, beta) for r in prev]
        gamma, delta = fit_harville(pc_prev[-3000:], [r["order"] for r in prev[-3000:]])
        sp_samples = [(r["rows"][i]["odds_ref"], r["n"], r["div_sp"][str(r["rows"][i]["num"])])
                      for r in train[-6000:] for i in range(r["n"])
                      if str(r["rows"][i]["num"]) in r["div_sp"] and r["rows"][i]["odds_ref"]]
        params = dict(alpha=alpha, beta=beta, gamma=gamma, delta=delta, sp_coef=fit_sp_dividend(sp_samples))
        # évaluation du mois
        ev = [r for r in test if r["id"] in oos and r["winner_idx"] is not None and mkt(r) is not None]
        ll_f = loglik([oos[r["id"]] for r in ev], [r["winner_idx"] for r in ev])
        ll_m = loglik([mkt(r) for r in ev], [r["winner_idx"] for r in ev])
        ll_c = loglik([combine(oos[r["id"]], mkt(r), alpha, beta) for r in ev], [r["winner_idx"] for r in ev])
        ll_0 = sum(-math.log(r["n"]) for r in ev)
        rho = lambda ll: 1 - ll / ll_0 if ll_0 else None
        # paris simulés, journée par journée
        mb = dict(n=0, staked=0.0, pnl=0.0)
        for day in sorted({r["date"] for r in ev}):
            cap0 = bank["capital"]; spent = 0.0; day_pnl = 0.0
            for r in sorted([x for x in ev if x["date"] == day], key=lambda x: x["heure"] or 0):
                a = analyse(r["rows"], oos[r["id"]], params)
                for b in size(a["cands"], cap0, spent):
                    spent += b["stake"]
                    num = str(b["num"])
                    if b["market"] == "SIMPLE_GAGNANT":
                        div = r["div_sg"].get(num)
                        if div is not None and b["min_odds"] and div < b["min_odds"]:
                            spent -= b["stake"]; continue          # value disparue : ticket annulé
                    else:
                        div = r["div_sp"].get(num)
                    gross = b["stake"] * div if div else 0.0
                    day_pnl += gross - b["stake"]
                    fk = b["market"]; flat[fk]["n"] += 1; flat[fk]["staked"] += 1.0; flat[fk]["ret"] += (div or 0.0)
                    flat[fk]["wins"] += bool(div)
                    mb["n"] += 1; mb["staked"] += b["stake"]; mb["pnl"] += gross - b["stake"]
                    bets_log.append(dict(date=day, race=r["id"], market=fk, num=b["num"], odds=b["odds"], p=round(b["p"], 4),
                                         stake=b["stake"], div=div, pnl=round(gross - b["stake"], 2)))
            bank["capital"] += day_pnl
            bank["peak"] = max(bank["peak"], bank["capital"])
            bank["mdd"] = max(bank["mdd"], (bank["peak"] - bank["capital"]) / bank["peak"])
        report["months"].append(dict(month=m, races=len(ev), alpha=round(alpha, 3), beta=round(beta, 3),
                                     gamma=gamma, delta=delta, rho_fund=rho(ll_f), rho_mkt=rho(ll_m), rho_comb=rho(ll_c),
                                     delta_rho=(rho(ll_c) - rho(ll_m)) if ll_0 else None,
                                     bets=mb["n"], staked=round(mb["staked"], 2), pnl=round(mb["pnl"], 2),
                                     capital=round(bank["capital"], 2)))
        print(f"{m}: {len(ev)} courses  rho fond {rho(ll_f):.4f} marché {rho(ll_m):.4f} combiné {rho(ll_c):.4f}  "
              f"alpha {alpha:.2f} beta {beta:.2f}  paris {mb['n']} P&L {mb['pnl']:+.2f}  capital {bank['capital']:.2f}", flush=True)
    # modèle final pour le live : tout l'historique
    final = {g: fit_group([r for r in data if r["group"] == g]) for g in ("TROT", "PLAT")}
    prev = [r for r in data if r["id"] in oos and r["winner_idx"] is not None and mkt(r) is not None]
    if len(prev) >= 200:
        alpha, beta = fit_stage2([oos[r["id"]] for r in prev], [mkt(r) for r in prev], [r["winner_idx"] for r in prev])
        pcs = [combine(oos[r["id"]], mkt(r), alpha, beta) for r in prev[-3000:]]
        gamma, delta = fit_harville(pcs, [r["order"] for r in prev[-3000:]])
    else:
        alpha, beta, gamma, delta = 0.0, 1.0, 0.81, 0.65       # pas assez de recul : le marché seul décide
    sp_samples = [(r["rows"][i]["odds_ref"], r["n"], r["div_sp"][str(r["rows"][i]["num"])])
                  for r in data[-8000:] for i in range(r["n"])
                  if str(r["rows"][i]["num"]) in r["div_sp"] and r["rows"][i]["odds_ref"]]
    mths = report["months"]
    tot_races = sum(x["races"] for x in mths)
    agg = lambda k: (sum(x[k] * x["races"] for x in mths if x[k] is not None) / tot_races) if tot_races else None
    gate = dict(min_races=1000, min_delta_rho=0.005,
                n_races_oos=tot_races, delta_rho_oos=agg("delta_rho"))
    gate["passed"] = bool(gate["delta_rho_oos"] is not None and tot_races >= gate["min_races"]
                          and gate["delta_rho_oos"] >= gate["min_delta_rho"])
    latest = dict(version="HL-V1." + time.strftime("%Y%m%d", time.gmtime()), features=FEATURES, groups=final,
                  stage2=dict(alpha=alpha if gate["passed"] else 0.0, alpha_fitted=alpha, beta=beta, gamma=gamma, delta=delta,
                              sp_coef=fit_sp_dividend(sp_samples)),
                  gate=gate, data_range=[data[0]["date"], data[-1]["date"]] if data else None, rules=RULES,
                  trained_at=report["built_at"])
    # coefficients lisibles (importance)
    for g, mdl in final.items():
        if mdl:
            latest.setdefault("top_features", {})[g] = sorted(
                [(f, round(w, 3)) for f, w in zip(FEATURES, mdl["w"])], key=lambda t: -abs(t[1]))[:15]
    report.update(summary=dict(
        races_oos=tot_races, rho_fund=agg("rho_fund"), rho_mkt=agg("rho_mkt"), rho_comb=agg("rho_comb"),
        delta_rho=agg("delta_rho"), final_capital=round(bank["capital"], 2), max_drawdown=round(bank["mdd"], 4),
        flat={k: dict(v, roi=(v["ret"] - v["staked"]) / v["staked"] if v["staked"] else None) for k, v in flat.items()},
        bets=len(bets_log), gate=gate, seconds=round(time.time() - t0)))
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    json.dump(latest, open(os.path.join(ROOT, "models", "latest.json"), "w"), indent=1)
    json.dump(dict(report, bets_sample=bets_log[-300:]), open(os.path.join(ROOT, "reports", "backtest.json"), "w"), indent=1)
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
