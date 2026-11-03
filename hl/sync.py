"""Synchronisation run GitHub -> base du dashboard (exécuté par la tâche Claude).

python3 -m hl.sync <morning|midday> <date> <jour N> <dbx_dir> <out_dir>
  dbx_dir : export ArtifactData (collections meta, days, races, bets) + versions.json {"col/doc": version}
Règles : un ticket n'est jamais modifié ; une course déjà ANALYSEE n'est pas réécrite au run de midi ;
mises = fractions de Kelly × capital d'ouverture, plafonds 2 % par ticket et 6 % par jour (mises déjà prises comprises).
"""
import glob, json, os, sys
from .decide import size, RULES

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_dir(dbx, col):
    out = {}
    for p in glob.glob(os.path.join(dbx, col, "*.json")):
        out[os.path.basename(p)[:-5]] = json.load(open(p, encoding="utf-8"))
    return out


def pour_contre(p, market):
    pour, contre = [], []
    if (p.get("form") or 0) >= 0.55: pour.append(f"Forme récente solide (indice {p['form']:.2f})")
    if p.get("p_mkt") is not None and (p.get("p_fund") or 0) > p["p_mkt"]: pour.append(f"Le modèle fondamental le voit à {p['p_fund']*100:.0f} % contre {p['p_mkt']*100:.0f} % pour le marché")
    if market == "SIMPLE_PLACE" and p.get("p_place"): pour.append(f"Probabilité d'être placé {p['p_place']*100:.0f} % (Harville corrigé)")
    if (p.get("incident_risk") or 0) >= 0.2: contre.append(f"Risque de faute : {int(p['incident_risk']*100)} % de disqualifications récentes")
    if p.get("confidence") == "FAIBLE": contre.append("Peu d'historique dans notre base sur ce cheval")
    if market == "SIMPLE_PLACE": contre.append("Rapport placé estimé, le vrai dividende n'est connu qu'à l'arrivée")
    risk = "Rapport final sous le minimum (value absorbée)" if market == "SIMPLE_GAGNANT" else "Dividende placé plus faible que prévu"
    return pour or ["Avantage combiné modèle + marché positif"], contre, risk


def main(phase, d, day_n, dbx, out_dir):
    run = json.load(open(os.path.join(ROOT, "live", d, f"{phase}.json"), encoding="utf-8"))
    vers = json.load(open(os.path.join(dbx, "versions.json"))) if os.path.exists(os.path.join(dbx, "versions.json")) else {}
    meta = load_dir(dbx, "meta").get("state", {})
    days = load_dir(dbx, "days")
    races_db = {k: v for k, v in load_dir(dbx, "races").items() if v.get("date") == d}
    bets_db = {k: v for k, v in load_dir(dbx, "bets").items() if v.get("date") == d}
    prev_days = sorted(k for k in days if k < d)
    if d in days:
        capital = days[d]["opening_capital"]
    elif prev_days and days[prev_days[-1]].get("closing_capital") is not None:
        capital = days[prev_days[-1]]["closing_capital"]
    else:
        capital = meta.get("current_capital", 500.0)
    os.makedirs(out_dir, exist_ok=True)
    writes = []

    def put(col, did, data):
        p = os.path.join(out_dir, f"{col}__{did}.json")
        json.dump(data, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        w = dict(op="set", collection=col, doc_id=did, file_path=os.path.abspath(p))
        if vers.get(f"{col}/{did}"): w["if_version"] = vers[f"{col}/{did}"]
        writes.append(w)

    # une course = même journée + même R/C, quel que soit le nom d'hippodrome (compatibilité J1 / V0)
    by_rc = {f"{v['date']}_{k.split('_')[-1]}": k for k, v in races_db.items()}
    for r in run["races"]:
        old = by_rc.get(f"{d}_R{r['r']}C{r['c']}")
        if old:
            r["doc_id"] = old; r["race_id"] = races_db[old].get("race_id", r["race_id"])
    already = sum(b.get("stake", 0) for b in bets_db.values())
    cands, to_write = [], []
    for r in sorted(run["races"], key=lambda x: x["time"]):
        did = r["doc_id"]
        if phase == "midday" and races_db.get(did, {}).get("status") == "ANALYSEE":
            continue                                      # décision du matin conservée
        if phase == "midday" and r["status"] == "NON_ANALYSEE" and did in races_db:
            continue                                      # ne pas écraser une prédiction du matin par « hors délai »
        to_write.append(r)
        for c in r.get("cands", []):
            cands.append((r, c))
    sized = size([c for _, c in cands], capital, already)
    by_key = {(c["num"], c["market"], c["kelly_f"]): c for c in sized}
    new_bets = []
    for r, c in cands:
        s = by_key.get((c["num"], c["market"], c["kelly_f"]))
        if not s:
            continue
        TAG = dict(SIMPLE_GAGNANT=("WIN", "Simple gagnant"), SIMPLE_PLACE=("PLACE", "Simple placé"),
                   COUPLE_GAGNANT=("CG", "Couplé gagnant"), DEUX_SUR_QUATRE=("2S4", "2sur4"))
        tag, label = TAG[c["market"]]
        p = next((p for p in r["preds"] if p["num"] == c["num"]), None)
        if p is None:     # couple : on agrège les deux chevaux
            nums = [int(t) for t in str(c["num"]).split("-")]
            ps = [q for q in r["preds"] if q["num"] in nums]
            p = dict(p_fund=c["p"], p_mkt=None, edge=None, ev=c["ev_low"], confidence=min((q.get("confidence") or "FAIBLE") for q in ps) if ps else "FAIBLE",
                     form=min((q.get("form") or 0) for q in ps) if ps else None, incident_risk=max((q.get("incident_risk") or 0) for q in ps) if ps else None)
        bid = f"{d.replace('-', '')}_{r['race_id']}_N{c['num']}_{tag}"
        if bid in bets_db:
            continue
        pour, contre, risk = pour_contre(p, c["market"])
        if c["market"] in ("COUPLE_GAGNANT", "DEUX_SUR_QUATRE"):
            pour = [f"Probabilité du couple {c['p']*100:.1f} % (arrivées simulées, Harville corrigé)", f"Dividende attendu {c['div_exp']:.1f} € pour 1 €"]
            contre = ["Dividende estimé sur l'historique : le vrai rapport n'est connu qu'à l'arrivée"] + contre
            risk = "Un seul des deux chevaux dans l'arrivée"
        new_bets.append(dict(bet_id=bid, date=d, race_id=r["race_id"], meeting=r["meeting"], rc=f"R{r['r']}C{r['c']}",
                             time=r["time"], num=c["num"], horse=c["horse"], market=label, market_code=c["market"],
                             odds=c["odds"], odds_time=r.get("odds_time"), p_hl=p.get("p_fund"), p_mkt=p.get("p_mkt"),
                             p_dec=c["p"], p_low=c["p_low"], edge=p.get("edge"), ev=p.get("ev"), ev_low=c["ev_low"],
                             sp_div_exp=c.get("sp_div_exp") or c.get("div_exp"), confidence=p.get("confidence"), stake=s["stake"],
                             min_odds=c.get("min_odds") or 0, status="OPEN", pour=pour, contre=contre, risk=risk,
                             experts=f"Modèle {run['model']}", created_at=run["computed_at"], settlement=None))
    chosen = {(b["race_id"], b["num"]) for b in new_bets if b["market_code"] in ("SIMPLE_GAGNANT", "SIMPLE_PLACE")}
    for r in to_write:
        preds = r["preds"]
        for p in preds:
            p["selected"] = (r["race_id"], p["num"]) in chosen
        dec = r["decision"]
        if dec == "BET" and not any(b["race_id"] == r["race_id"] for b in new_bets):
            dec = "PASS"; r["reason"] += " — mise sous 1 € ou plafond du jour atteint"
        put("races", r["doc_id"], dict(date=d, race_id=r["race_id"], meeting=r["meeting"], r=r["r"], c=r["c"], time=r["time"],
            disc=r["disc"], dist=r["dist"], cond=r["cond"], n_runners=r["n_runners"], status=r["status"], decision=dec,
            reason=r["reason"], odds_time=r.get("odds_time"), odds_src=r.get("odds_src"), flags=[], shadow=r.get("shadow"),
            terrain=r.get("terrain"), captured_at=run["captured_at"], computed_at=run["computed_at"], model=run["model"],
            phase=phase, preds=preds, result=None))
    for b in new_bets:
        put("bets", b["bet_id"], b)
    # journée
    st = {k: v.get("status") for k, v in races_db.items()}
    dec = {k: v.get("decision") for k, v in races_db.items()}
    for r in to_write:
        st[r["doc_id"]] = r["status"]; dec[r["doc_id"]] = r["decision"]
    for r in run["races"]:
        st.setdefault(r["doc_id"], r["status"])
    cov = dict(expected=len(st), analyzed=sum(v == "ANALYSEE" for v in st.values()),
               partial=sum(v == "PARTIELLE" for v in st.values()), not_analyzed=sum(v == "NON_ANALYSEE" for v in st.values()))
    all_bets = list(bets_db.values()) + new_bets
    prev = days.get(d, {})
    put("days", d, dict(prev, date=d, day=day_n, opening_capital=capital, closing_capital=prev.get("closing_capital"),
        pnl=prev.get("pnl"), status_morning="COMPLETE" if cov["analyzed"] == cov["expected"] else "INCOMPLETE",
        status_close=prev.get("status_close", "PLANNED"), coverage=cov, n_bet=len(all_bets),
        n_pass=sum(1 for k, v in st.items() if v == "ANALYSEE" and dec.get(k) != "BET" and not any(b["race_id"] in k for b in all_bets)),
        exposure=round(sum(b["stake"] for b in all_bets), 2), n_settled=prev.get("n_settled", 0),
        morning_at=prev.get("morning_at") or run["computed_at"], midday_at=run["computed_at"] if phase == "midday" else prev.get("midday_at"),
        model=run["model"], kelly_fraction=RULES["kelly"], max_bet_pct=RULES["max_bet_pct"], max_day_pct=RULES["max_day_pct"],
        morning_note=(f"Modèle {run['model']} · " + ("validé" if run["gate"]["passed"] else "non validé : le marché décide seul (α = 0)")
                      + f" · {cov['partial']} partielle(s), {cov['not_analyzed']} hors délai")))
    put("meta", "state", dict(meta, day=day_n, current_date=d, current_capital=capital, model=run["model"],
        params=dict(run["rules"], **{k: run["params"][k] for k in ("alpha", "beta", "gamma", "delta")},
                    alpha_model=run["params"]["alpha"], w_experts=0, min_edge=run["rules"]["min_edge"], odds_haircut=run["rules"]["haircut"]),
        gate=run["gate"], updated_at=run["computed_at"], last_run=phase))
    bt = os.path.join(ROOT, "reports", "backtest.json")
    mp = os.path.join(ROOT, "models", "latest.json")
    if os.path.exists(bt) and os.path.exists(mp):
        rep, mdl = json.load(open(bt)), json.load(open(mp))
        put("reports", "backtest", dict(built_at=rep.get("built_at"), n_races=rep.get("n_races"), summary=rep.get("summary"),
            months=rep.get("months"), model=mdl.get("version"), gate=mdl.get("gate"), stage2=mdl.get("stage2"),
            top_features=mdl.get("top_features"), data_range=mdl.get("data_range")))
    json.dump(writes, open(os.path.join(out_dir, "_writes.json"), "w"))
    print(f"{phase} {d}: {len(to_write)} courses écrites, {len(new_bets)} tickets, exposition jour {sum(b['stake'] for b in all_bets):.2f} € "
          f"sur capital {capital:.2f} €, couverture {cov['analyzed']}/{cov['expected']} · {len(writes)} écritures")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
