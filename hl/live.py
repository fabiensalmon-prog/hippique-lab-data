"""Run quotidien (GitHub Actions).

python3 -m hl.live morning|midday|closing [AAAA-MM-JJ]
  morning/midday -> live/<date>/<phase>.json  : prédictions de tous les partants + candidats de pari (fractions de Kelly)
  closing        -> data/<année>/<date>.json.gz (historique) + live/<date>/results.json (format closing.py)
Les mises en euros sont calculées à la synchronisation avec le dashboard (capital = source de vérité du dashboard).
"""
import gzip, json, os, sys, time
from datetime import date, datetime, timezone, timedelta
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "collect"))
from pmu_history import collect_day, write_day          # noqa: E402
from .store import normalize, iter_races                 # noqa: E402
from .features import FeatureState                       # noqa: E402
from .train import predict_any                           # noqa: E402
from .decide import analyse, RULES                       # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PARIS = timezone(timedelta(hours=1))


def meeting_id(h):
    return (h or "INCONNU").upper().replace(" ", "")


def race_doc_id(race):
    return f"{race['date']}_{meeting_id(race['hippodrome'])}_R{race['r']}C{race['c']}"


def run_phase(phase, d):
    now_ms = int(os.environ.get("HL_NOW_MS") or time.time() * 1000)
    obj = json.load(gzip.open(os.environ["HL_TEST_DAY"], "rt")) if os.environ.get("HL_TEST_DAY") else collect_day(date.fromisoformat(d))
    os.makedirs(os.path.join(ROOT, "live", d), exist_ok=True)
    with gzip.open(os.path.join(ROOT, "live", d, f"raw_{phase}.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    mp = os.path.join(ROOT, "models", "latest.json")
    model = json.load(open(mp)) if os.path.exists(mp) else dict(
        version="HL-V1.0-sans-modele", groups={}, gate=dict(passed=False, n_races_oos=0, delta_rho_oos=None),
        stage2=dict(alpha=0.0, beta=1.0, gamma=0.81, delta=0.65, sp_coef=[0.15, 0.45, 0.0]))
    st = FeatureState()
    for r in iter_races(until=(date.fromisoformat(d) - timedelta(days=1)).isoformat()):
        st.update(r)
    params = model["stage2"]
    out = dict(date=d, phase=phase, model=model["version"], gate=model["gate"], params=params, rules=RULES,
               computed_at=datetime.now(PARIS).isoformat(timespec="seconds"),
               captured_at=obj["collected_at"], races=[])
    races = normalize(obj)
    for race in races:
        rows = st.features(race)
        started = race["heure"] and race["heure"] <= now_ms + 5 * 60 * 1000
        heure = datetime.fromtimestamp(race["heure"] / 1000, PARIS).strftime("%H:%M") if race["heure"] else ""
        doc = dict(race_id=f"{meeting_id(race['hippodrome'])}_R{race['r']}C{race['c']}", doc_id=race_doc_id(race),
                   meeting=race["hippodrome"], r=race["r"], c=race["c"], time=heure, disc=race["discipline"],
                   group=race["group"], dist=race["distance"], cond=race["categorie"], n_runners=len(rows),
                   terrain=race["terrain"], pen=race["pen"], libelle=race["libelle"])
        if started or len(rows) < 2:
            doc.update(status="NON_ANALYSEE", decision=None, reason="Départ avant le run (hors délai)", preds=[], cands=[])
            out["races"].append(doc); continue
        mdl = model["groups"].get(race["group"])
        pf = predict_any(mdl, dict(id=race["id"], rows=rows)) if mdl else None
        if pf is None:      # pas encore de modèle entraîné : heuristique musique du V0 (sans cotes)
            sc = np.array([3.0 * (r["x"].get("mus_form") if r["x"].get("mus_form") is not None else 0.25)
                           + 0.8 * (r["x"].get("mus_cons") or 0.0) - (r["x"].get("mus_dq") or 0.0) for r in rows])
            e = np.exp(2.6 * (sc - sc.max())); pf = e / e.sum()
        a = analyse(rows, pf, params)
        if not a["market_ok"]:
            doc.update(status="PARTIELLE", decision=None, reason="Prédiction journalisée, rapports probables PMU incomplets")
        else:
            doc.update(status="ANALYSEE", decision="BET" if a["cands"] else "PASS")
            if a["cands"]:
                doc["reason"] = " · ".join(f"{c['market'].replace('SIMPLE_', '')} N°{c['num']} EV basse {c['ev_low']*100:+.0f} %" for c in a["cands"])
            else:
                top = max(a["preds"], key=lambda p: p["ev_low"])
                doc["reason"] = (f"Meilleure EV basse N°{top['num']} {top['ev_low']*100:+.0f} % (edge {top['edge']*100:+.1f} pts) : sous le seuil"
                                 + ("" if model["gate"]["passed"] else " · modèle non validé : le marché décide seul"))
        sel = {c["num"] for c in a["cands"]}
        for p in a["preds"]:
            p["selected"] = p["num"] in sel
            x = next(r["x"] for r in rows if r["num"] == p["num"])
            p.update(jockey=next(r["driver"] for r in rows if r["num"] == p["num"]),
                     form=x.get("mus_form"), incident_risk=x.get("mus_dq"), n_runs=x.get("mus_n"),
                     confidence="MOYENNE" if x.get("has_hist") else "FAIBLE",
                     p_hl=p["p_fund"], p_exp=None, raw_signal=bool(p.get("edge_raw", 0) >= 0.05))
        doc.update(preds=a["preds"], cands=a["cands"], shadow=a.get("shadow"), odds_time=obj["collected_at"][11:16] + " UTC",
                   odds_src="PMU rapport de référence")
        out["races"].append(doc)
    json.dump(out, open(os.path.join(ROOT, "live", d, f"{phase}.json"), "w"), ensure_ascii=False, indent=0)
    n = len(out["races"]); an = sum(r["status"] == "ANALYSEE" for r in out["races"])
    print(f"{phase} {d}: {an}/{n} analysées, {sum(len(r['cands']) for r in out['races'])} candidats, modèle {model['version']}, gate {model['gate']['passed']}")


def run_closing(d):
    if os.environ.get("HL_TEST_DAY"):
        obj = json.load(gzip.open(os.environ["HL_TEST_DAY"], "rt"))
    else:
        obj = collect_day(date.fromisoformat(d))
        write_day(date.fromisoformat(d), obj)              # l'historique s'enrichit chaque soir
    res = {}
    for race in normalize(obj):
        if not race["arrivee"]:
            continue
        res[race_doc_id(race)] = dict(
            arrivee=race["arrivee"], rapport_gagnant=race["div_sg"].get(str(race["arrivee"][0])),
            rapport_place={k: v for k, v in race["div_sp"].items()},
            non_partants=[u["num"] for u in race["runners"] if u["np"]],
            final_odds={str(u["num"]): u["odds_final"] for u in race["runners"] if u["odds_final"]},
            dead_heat=race["dead_heat"], source="PMU rapports définitifs")
    os.makedirs(os.path.join(ROOT, "live", d), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "live", d, "results.json"), "w"), ensure_ascii=False)
    print(f"closing {d}: {len(res)} arrivées")


if __name__ == "__main__":
    phase = sys.argv[1]
    d = sys.argv[2] if len(sys.argv) > 2 else datetime.now(PARIS).date().isoformat()
    run_closing(d) if phase == "closing" else run_phase(phase, d)
