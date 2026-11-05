"""Entraînement V1.1 + backtest walk-forward mensuel.

python3 -m hl.train            -> models/latest.json, reports/backtest.json
- Étage 1 (logit conditionnel trot / plat / obstacle) : chaque mois de test est prédit par un modèle entraîné
  uniquement sur les mois précédents.
- Étage 2, Harville corrigé, dividendes attendus (placé, couplé, 2sur4) : estimés sur les mois précédents seulement.
- Décisions sur la cote marché du matin (rapport de référence), règlement sur les dividendes officiels.
- Portes : le modèle n'entre dans les décisions que s'il améliore le marché (gain de ρ²) ; chaque marché n'est
  misé que si son ROI hors échantillon est positif sur au moins 300 paris.
"""
import json, math, os, time
from collections import defaultdict
import numpy as np

from .store import iter_races
from .features import FeatureState, FEATURES
from .model import (Design, fit_clogit, probs, split_probs, fit_stage2, combine, fit_harville, fit_sp_dividend,
                    fit_pair_dividend, loglik)
from .decide import analyse, size, RULES, MARKETS

ROOT = os.path.join(os.path.dirname(__file__), "..")
GROUPS = ("TROT", "PLAT", "OBSTACLE")
MIN_TRAIN_MONTHS = int(os.environ.get("HL_MIN_TRAIN_MONTHS", "2"))
# V1.2 : l'historique couvre plusieurs saisons ; le backtest commence en TEST_FROM (entraînement sur tout ce qui précède)
# et l'étage 1 est réestimé tous les REFIT_MONTHS mois (départ à chaud), l'étage 2 chaque mois.
TEST_FROM = os.environ.get("HL_TEST_FROM", "2023-01")
REFIT_MONTHS = int(os.environ.get("HL_REFIT_MONTHS", "3"))
L2 = 2.0
GATE = dict(min_races=1000, min_delta_rho=0.005, market_min_bets=300)


def build_dataset(compact=False):
    """compact=True : variables stockées en tableau float32 (ordre FEATURES) pour tenir en mémoire sur 6+ saisons."""
    st = FeatureState()
    out = []
    for race in iter_races():
        rows = st.features(race)
        st.update(race)
        if len(rows) < 4:
            continue
        if compact:
            for x in rows:
                x["x"] = np.array([np.nan if x["x"].get(k) is None else x["x"][k] for k in FEATURES], dtype=np.float32)
        nums = [r["num"] for r in rows]
        w = race["arrivee"][0] if race["arrivee"] else None
        out.append(dict(id=race["id"], date=race["date"], month=race["date"][:7], group=race["group"], rows=rows,
                        n=len(rows), winner_idx=nums.index(w) if w in nums else None,
                        order=[nums.index(a) for a in race["arrivee"][:3] if a in nums], arrivee=race["arrivee"],
                        div_sg=race["div_sg"], div_sp=race["div_sp"], div_cg=race["div_cg"], div_2s4=race["div_2s4"],
                        dead_heat=race["dead_heat"], hippodrome=race["hippodrome"], heure=race["heure"], mkt_src=race.get("mkt_src")))
    return out, st


def fit_group(races, w0=None):
    D = Design(races)
    if len(D.starts) < 50:
        return None
    w = fit_clogit(D.X, D.starts, D.seg, D.y, l2=L2, w0=w0)
    return dict(w=w.tolist(), means=D.means.tolist(), stds=D.stds.tolist(), n_races=int(len(D.starts)), features=list(FEATURES))


def predict_group(model, races):
    feats = model.get("features") or FEATURES[:len(model["w"])]     # modèles antérieurs à V1.2 : préfixe de FEATURES
    D = Design(races, np.array(model["means"]), np.array(model["stds"]), features=feats)
    if not len(D.starts):
        return {}
    p = probs(D.X, D.starts, D.seg, np.array(model["w"]))
    return {r["id"]: pr for r, pr in zip(D.meta, split_probs(p, D.starts))}


def predict_any(model, race):
    """Prédiction d'une course sans résultat (live)."""
    return predict_group(model, [dict(race, winner_idx=0)]).get(race["id"])


def mkt(r):
    o = [x["odds_ref"] for x in r["rows"]]
    if not all(v and v > 1 for v in o):
        return None
    inv = np.array([1 / v for v in o]); return inv / inv.sum()


def mkt_final(r):
    o = [x["odds_final"] for x in r["rows"]]
    if not all(v and v > 1 for v in o):
        return None
    inv = np.array([1 / v for v in o]); return inv / inv.sum()


def odds_of(r, num):
    return next((x["odds_ref"] for x in r["rows"] if str(x["num"]) == str(num)), None)


def dividend_params(train):
    """Régressions des dividendes attendus sur les courses d'entraînement récentes."""
    sp, cg, s4 = [], [], []
    for r in train[-8000:]:
        for x in r["rows"]:
            k = str(x["num"])
            if k in r["div_sp"] and x["odds_ref"]:
                sp.append((x["odds_ref"], r["n"], r["div_sp"][k]))
        for combo, dv in r["div_cg"].items():
            a, _, b = combo.partition("-")
            oa, ob = odds_of(r, a), odds_of(r, b)
            if oa and ob: cg.append((oa, ob, r["n"], dv))
        for combo, dv in r["div_2s4"].items():
            a, _, b = combo.partition("-")
            oa, ob = odds_of(r, a), odds_of(r, b)
            if oa and ob: s4.append((oa, ob, r["n"], dv))
    return dict(sp_coef=fit_sp_dividend(sp), cg_coef=fit_pair_dividend(cg), s4_coef=fit_pair_dividend(s4))


def settle(r, b):
    """Dividende pour 1 € du ticket simulé (0 si perdu, None si non réglable)."""
    k = str(b["num"])
    if b["market"] == "SIMPLE_GAGNANT":
        return r["div_sg"].get(k, 0.0)
    if b["market"] == "SIMPLE_PLACE":
        return r["div_sp"].get(k, 0.0) if r["div_sp"] else None
    if b["market"] == "COUPLE_GAGNANT":
        return r["div_cg"].get(k, 0.0) if r["div_cg"] else None
    return r["div_2s4"].get(k, 0.0) if r["div_2s4"] else None


def main():
    t0 = time.time()
    data, _ = build_dataset(compact=True)
    months = sorted({r["month"] for r in data})
    print(f"{len(data)} courses, {len(months)} mois : {months[:1]} -> {months[-1:]}", flush=True)
    oos = {}
    report = dict(months=[], built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), n_races=len(data))
    bank = dict(capital=500.0, peak=500.0, mdd=0.0)
    flat = defaultdict(lambda: dict(n=0, staked=0.0, ret=0.0, wins=0))
    by_group = defaultdict(lambda: dict(races=0, ll_f=0.0, ll_m=0.0, ll_c=0.0, ll_0=0.0))
    bets_log = []
    top1 = defaultdict(lambda: dict(n=0, p1=0, p2=0, p3=0, p4=0, ret_win=0.0, n_place=0, ret_place=0.0))
    stage1, last_fit = {}, None
    for mi, m in enumerate(months):
        if mi < MIN_TRAIN_MONTHS or m < TEST_FROM:
            continue
        train = [r for r in data if r["month"] < m]
        test = [r for r in data if r["month"] == m]
        if last_fit is None or mi - last_fit >= REFIT_MONTHS:
            for g in GROUPS:
                w0 = np.array(stage1[g]["w"]) if stage1.get(g) else None
                stage1[g] = fit_group([r for r in train if r["group"] == g], w0=w0)
            last_fit = mi
        for g in GROUPS:
            if stage1.get(g):
                oos.update(predict_group(stage1[g], [r for r in test if r["group"] == g]))
        prev = [r for r in train if r["id"] in oos and r["winner_idx"] is not None and mkt(r) is not None]
        if len(prev) < 200:
            continue
        alpha, beta = fit_stage2([oos[r["id"]] for r in prev], [mkt(r) for r in prev], [r["winner_idx"] for r in prev])
        pc_prev = [combine(oos[r["id"]], mkt(r), alpha, beta) for r in prev[-3000:]]
        gamma, delta = fit_harville(pc_prev, [r["order"] for r in prev[-3000:]])
        params = dict(alpha=alpha, beta=beta, gamma=gamma, delta=delta, **dividend_params(train))
        ev = [r for r in test if r["id"] in oos and r["winner_idx"] is not None and mkt(r) is not None]
        W = [r["winner_idx"] for r in ev]
        ll_f = loglik([oos[r["id"]] for r in ev], W)
        ll_m = loglik([mkt(r) for r in ev], W)
        ll_c = loglik([combine(oos[r["id"]], mkt(r), alpha, beta) for r in ev], W)
        ll_0 = sum(-math.log(r["n"]) for r in ev)
        for r in ev:
            G = by_group[r["group"]]; w = r["winner_idx"]
            G["races"] += 1; G["ll_0"] += -math.log(r["n"])
            G["ll_f"] += math.log(max(1e-9, oos[r["id"]][w])); G["ll_m"] += math.log(max(1e-9, mkt(r)[w]))
            G["ll_c"] += math.log(max(1e-9, combine(oos[r["id"]], mkt(r), alpha, beta)[w]))
        for r in ev:            # notre n°1 (probabilité combinée) : où finit-il, gagnant ou placé ?
            pc = combine(oos[r["id"]], mkt(r), alpha, beta)
            for key, idx in (("modele", int(np.argmax(pc))), ("favori_marche", int(np.argmax(mkt(r))))):
                num = r["rows"][idx]["num"]; arr = r["arrivee"]
                pos = arr.index(num) + 1 if num in arr else None
                T = top1[key]; T["n"] += 1; T["p1" if pos == 1 else "p2" if pos == 2 else "p3" if pos == 3 else "p4"] += 1
                T["ret_win"] += r["div_sg"].get(str(num), 0.0) if pos == 1 else 0.0
                if r["div_sp"]:
                    T["n_place"] += 1; T["ret_place"] += r["div_sp"].get(str(num), 0.0)
        # face au marché FINAL (celui qui fixe les rapports) : la vraie référence à battre
        prevf = [r for r in prev if mkt_final(r) is not None]
        evf = [r for r in ev if mkt_final(r) is not None]
        if len(prevf) >= 200 and evf:
            af, bf = fit_stage2([oos[r["id"]] for r in prevf], [mkt_final(r) for r in prevf], [r["winner_idx"] for r in prevf])
            llf0 = sum(-math.log(r["n"]) for r in evf)
            llfm = loglik([mkt_final(r) for r in evf], [r["winner_idx"] for r in evf])
            llfc = loglik([combine(oos[r["id"]], mkt_final(r), af, bf) for r in evf], [r["winner_idx"] for r in evf])
            final_rec = dict(races_final=len(evf), rho_mkt_final=1 - llfm / llf0, rho_comb_final=1 - llfc / llf0,
                             delta_rho_final=(1 - llfc / llf0) - (1 - llfm / llf0))
        else:
            final_rec = dict(races_final=0, rho_mkt_final=None, rho_comb_final=None, delta_rho_final=None)
        rho = lambda ll: 1 - ll / ll_0 if ll_0 else None
        mb = defaultdict(lambda: dict(n=0, pnl=0.0))
        for day in sorted({r["date"] for r in ev}):
            cap0 = bank["capital"]; spent = 0.0; day_pnl = 0.0
            for r in sorted([x for x in ev if x["date"] == day], key=lambda x: x["heure"] or 0):
                a = analyse(r["rows"], oos[r["id"]], params, race_key=r["id"], all_markets=True)
                for b in size(a["cands"], cap0, spent):
                    div = settle(r, b)
                    if div is None:
                        continue
                    spent += b["stake"]
                    pnl = b["stake"] * div - b["stake"]
                    day_pnl += pnl
                    F = flat[b["market"]]; F["n"] += 1; F["staked"] += 1.0; F["ret"] += div; F["wins"] += div > 0
                    if b["market"] == "SIMPLE_GAGNANT" and r["div_sp"]:      # le même cheval, joué placé
                        alt = r["div_sp"].get(str(b["num"]), 0.0)
                        A = flat["GAGNANT_JOUE_PLACE"]; A["n"] += 1; A["staked"] += 1.0; A["ret"] += alt; A["wins"] += alt > 0
                    mb[b["market"]]["n"] += 1; mb[b["market"]]["pnl"] += pnl
                    bets_log.append(dict(date=day, race=r["id"], market=b["market"], num=b["num"], p=round(b["p"], 4),
                                         odds=round(b["odds"], 2), stake=b["stake"], div=div, pnl=round(pnl, 2)))
            bank["capital"] += day_pnl
            bank["peak"] = max(bank["peak"], bank["capital"])
            bank["mdd"] = max(bank["mdd"], (bank["peak"] - bank["capital"]) / bank["peak"])
        rec = dict(month=m, races=len(ev), alpha=round(alpha, 3), beta=round(beta, 3), gamma=gamma, delta=delta,
                   rho_fund=rho(ll_f), rho_mkt=rho(ll_m), rho_comb=rho(ll_c),
                   delta_rho=(rho(ll_c) - rho(ll_m)) if ll_0 else None,
                   bets=sum(v["n"] for v in mb.values()), pnl=round(sum(v["pnl"] for v in mb.values()), 2),
                   by_market={k: dict(n=v["n"], pnl=round(v["pnl"], 2)) for k, v in mb.items()},
                   capital=round(bank["capital"], 2), **final_rec)
        report["months"].append(rec)
        print(f"{m}: {len(ev)} courses  ρ² fond {rho(ll_f):.4f} marché {rho(ll_m):.4f} combiné {rho(ll_c):.4f}  "
              f"α {alpha:.2f} β {beta:.2f}  paris {rec['bets']} P&L {rec['pnl']:+.2f}  capital {bank['capital']:.2f}", flush=True)
    # ---------- modèle final pour le live ----------
    final = {g: fit_group([r for r in data if r["group"] == g], w0=np.array(stage1[g]["w"]) if stage1.get(g) else None) for g in GROUPS}
    prev = [r for r in data if r["id"] in oos and r["winner_idx"] is not None and mkt(r) is not None]
    if len(prev) >= 200:
        alpha, beta = fit_stage2([oos[r["id"]] for r in prev], [mkt(r) for r in prev], [r["winner_idx"] for r in prev])
        pcs = [combine(oos[r["id"]], mkt(r), alpha, beta) for r in prev[-3000:]]
        gamma, delta = fit_harville(pcs, [r["order"] for r in prev[-3000:]])
    else:
        alpha, beta, gamma, delta = 0.0, 1.0, 0.81, 0.65
    mths = report["months"]
    tot = sum(x["races"] for x in mths)
    agg = lambda k: (sum(x[k] * x["races"] for x in mths if x[k] is not None) / tot) if tot else None
    totf = sum(x.get("races_final") or 0 for x in mths)
    dfin = (sum((x.get("delta_rho_final") or 0) * (x.get("races_final") or 0) for x in mths) / totf) if totf else None
    gate = dict(min_races=GATE["min_races"], min_delta_rho=GATE["min_delta_rho"], n_races_oos=tot, delta_rho_oos=agg("delta_rho"),
                n_races_final=totf, delta_rho_final=dfin, reference="marché final")
    # la porte se juge face au marché FINAL : battre la cote du matin ne suffit pas au pari mutuel
    gate["passed"] = bool(dfin is not None and totf >= GATE["min_races"] and dfin >= GATE["min_delta_rho"])
    flat_out = {k: dict(v, roi=(v["ret"] - v["staked"]) / v["staked"] if v["staked"] else None) for k, v in flat.items()}
    markets = {mk: bool(gate["passed"] and flat_out.get(mk, {}).get("n", 0) >= GATE["market_min_bets"]
                        and (flat_out.get(mk, {}).get("roi") or -1) > 0) for mk in MARKETS}
    gate["markets"] = markets
    latest = dict(version="HL-V1.2." + time.strftime("%Y%m%d", time.gmtime()), features=FEATURES, groups=final,
                  stage2=dict(alpha=alpha if gate["passed"] else 0.0, alpha_fitted=alpha, beta=beta, gamma=gamma, delta=delta,
                              markets=markets, **dividend_params(data)),
                  gate=gate, data_range=[data[0]["date"], data[-1]["date"]] if data else None, rules=RULES,
                  trained_at=report["built_at"])
    for g, mdl in final.items():
        if mdl:
            latest.setdefault("top_features", {})[g] = sorted(
                [(f, round(w, 3)) for f, w in zip(FEATURES, mdl["w"])], key=lambda t: -abs(t[1]))[:15]
    groups_out = {g: dict(races=v["races"], rho_fund=1 - v["ll_f"] / v["ll_0"], rho_mkt=1 - v["ll_m"] / v["ll_0"],
                          rho_comb=1 - v["ll_c"] / v["ll_0"]) for g, v in by_group.items() if v["ll_0"]}
    report.update(summary=dict(
        races_oos=tot, rho_fund=agg("rho_fund"), rho_mkt=agg("rho_mkt"), rho_comb=agg("rho_comb"), delta_rho=agg("delta_rho"),
        final_capital=round(bank["capital"], 2), max_drawdown=round(bank["mdd"], 4), flat=flat_out, by_group=groups_out,
        top1={k: dict(v, roi_win=(v["ret_win"] - v["n"]) / v["n"] if v["n"] else None,
                      roi_place=(v["ret_place"] - v["n_place"]) / v["n_place"] if v["n_place"] else None) for k, v in top1.items()},
        bets=len(bets_log), gate=gate, seconds=round(time.time() - t0)))
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    json.dump(latest, open(os.path.join(ROOT, "models", "latest.json"), "w"), indent=1)
    json.dump(dict(report, bets_sample=bets_log[-300:]), open(os.path.join(ROOT, "reports", "backtest.json"), "w"), indent=1)
    print(json.dumps(report["summary"], indent=1, default=str))


if __name__ == "__main__":
    main()
