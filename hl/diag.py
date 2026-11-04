"""Diagnostic : où se trouve (ou pas) l'avantage du modèle, sur les prédictions hors échantillon du walk-forward.

python3 -m hl.diag  -> reports/diag.json
Pour chaque partant d'une course jamais vue : P modèle combiné, P marché (rapport de référence du matin),
cote finale, gagnant, dividendes gagnant et placé. On mesure le ROI de règles simples à mise fixe.
"""
import json, math, os, time
from collections import defaultdict
import numpy as np

from .train import build_dataset, fit_group, predict_group, mkt, GROUPS, MIN_TRAIN_MONTHS
from .model import fit_stage2, combine, place_probs, fit_harville
from .decide import n_places

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    t0 = time.time()
    data, _ = build_dataset()
    months = sorted({r["month"] for r in data})
    oos, rows = {}, []
    for mi, m in enumerate(months):
        train = [r for r in data if r["month"] < m]
        test = [r for r in data if r["month"] == m]
        if mi < MIN_TRAIN_MONTHS:
            continue
        for g in GROUPS:
            mdl = fit_group([r for r in train if r["group"] == g])
            if mdl:
                oos.update(predict_group(mdl, [r for r in test if r["group"] == g]))
        prev = [r for r in train if r["id"] in oos and r["winner_idx"] is not None and mkt(r) is not None]
        if len(prev) < 200:
            continue
        a, b = fit_stage2([oos[r["id"]] for r in prev], [mkt(r) for r in prev], [r["winner_idx"] for r in prev])
        gm, dl = fit_harville([combine(oos[r["id"]], mkt(r), a, b) for r in prev[-3000:]], [r["order"] for r in prev[-3000:]])
        for r in test:
            if r["id"] not in oos or r["winner_idx"] is None or mkt(r) is None or not r["div_sg"]:
                continue
            pm = mkt(r); pc = combine(oos[r["id"]], pm, a, b); k = n_places(r["n"])
            pp = place_probs(pc, k, gm, dl) if k else np.zeros(r["n"])
            ppm = place_probs(pm, k, gm, dl) if k else np.zeros(r["n"])
            for i, x in enumerate(r["rows"]):
                num = str(x["num"])
                rows.append(dict(src=r.get("mkt_src"), month=r["month"], g=r["group"], n=r["n"], pc=float(pc[i]), pm=float(pm[i]), pf=float(oos[r["id"]][i]),
                                 pp=float(pp[i]), ppm=float(ppm[i]), o_ref=x["odds_ref"], o_fin=x["odds_final"],
                                 win=int(i == r["winner_idx"]), dsg=r["div_sg"].get(num, 0.0),
                                 dsp=r["div_sp"].get(num, 0.0) if r["div_sp"] else None, k=k))
        print(m, len(rows), flush=True)

    def roi(sel, key):
        s = [x for x in sel if x[key] is not None]
        return dict(n=len(s), roi=round((sum(x[key] for x in s) - len(s)) / len(s), 4) if s else None,
                    hit=round(sum(x[key] > 0 for x in s) / len(s), 3) if s else None)

    out = dict(built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), runners=len(rows))
    # 1. biais favoris / outsiders du marché : ROI gagnant par tranche de cote finale
    buckets = [(1, 2), (2, 3), (3, 5), (5, 8), (8, 13), (13, 21), (21, 40), (40, 1000)]
    out["roi_by_odds"] = [dict(cote=f"{lo}-{hi}", **roi([x for x in rows if x["o_fin"] and lo <= x["o_fin"] < hi], "dsg")) for lo, hi in buckets]
    out["roi_place_by_odds"] = [dict(cote=f"{lo}-{hi}", **roi([x for x in rows if x["o_fin"] and lo <= x["o_fin"] < hi and x["k"]], "dsp")) for lo, hi in buckets]
    # 2. calibration : P combinée vs fréquence réelle
    cal = []
    for lo, hi in [(0, .02), (.02, .05), (.05, .1), (.1, .15), (.15, .2), (.2, .3), (.3, .45), (.45, 1)]:
        s = [x for x in rows if lo <= x["pc"] < hi]
        if s:
            cal.append(dict(tranche=f"{lo:.0%}-{hi:.0%}", n=len(s), p_moy=round(sum(x["pc"] for x in s) / len(s), 4),
                            freq=round(sum(x["win"] for x in s) / len(s), 4), p_marche=round(sum(x["pm"] for x in s) / len(s), 4)))
    out["calibration"] = cal
    # 3. ROI selon l'écart modèle / marché (ratio pc/pm), gagnant et placé
    edge = []
    for lo, hi in [(0, .8), (.8, .95), (.95, 1.05), (1.05, 1.15), (1.15, 1.3), (1.3, 1.6), (1.6, 100)]:
        s = [x for x in rows if lo <= x["pc"] / x["pm"] < hi]
        edge.append(dict(ratio=f"{lo}-{hi}", gagnant=roi(s, "dsg"), place=roi([x for x in s if x["k"]], "dsp")))
    out["roi_by_edge"] = edge
    # 4. même chose en plaçant : ratio P placé modèle / P placé marché
    edge_p = []
    for lo, hi in [(0, .9), (.9, 1.0), (1.0, 1.05), (1.05, 1.1), (1.1, 1.2), (1.2, 100)]:
        s = [x for x in rows if x["k"] and x["ppm"] > 0 and lo <= x["pp"] / x["ppm"] < hi]
        edge_p.append(dict(ratio=f"{lo}-{hi}", place=roi(s, "dsp")))
    out["roi_place_by_edge"] = edge_p
    # 5. cote du matin vs cote finale : nos chevaux « value » voient-ils leur cote baisser ?
    s = [x for x in rows if x["o_ref"] and x["o_fin"] and x["pc"] / x["pm"] >= 1.15]
    out["clv_value"] = dict(n=len(s), cote_matin_moy=round(sum(x["o_ref"] for x in s) / len(s), 2) if s else None,
                            cote_finale_moy=round(sum(x["o_fin"] for x in s) / len(s), 2) if s else None,
                            part_qui_baisse=round(sum(x["o_fin"] < x["o_ref"] for x in s) / len(s), 3) if s else None)
    # 6. par discipline, les meilleurs filtres
    out["by_group"] = {g: dict(gagnant_ratio_1_3=roi([x for x in rows if x["g"] == g and x["pc"] / x["pm"] >= 1.3], "dsg"),
                               place_ratio_1_1=roi([x for x in rows if x["g"] == g and x["k"] and x["ppm"] and x["pp"] / x["ppm"] >= 1.1], "dsp"))
                       for g in GROUPS}
    # 7. robustesse : la règle « trot, ratio ≥ 1,3 » par année et selon la source de la cote marché
    rule = lambda x: x["g"] == "TROT" and x["pc"] / x["pm"] >= 1.3
    out["rule_trot_13"] = dict(
        par_annee={y: roi([x for x in rows if rule(x) and x["month"].startswith(y)], "dsg") for y in sorted({x["month"][:4] for x in rows})},
        par_semestre={h: roi([x for x in rows if rule(x) and x["month"] >= h[:7] and x["month"] <= h[8:]], "dsg")
                      for h in ("2025-01_2025-06", "2025-07_2025-12", "2026-01_2026-06", "2026-07_2026-12")},
        cote_matin=roi([x for x in rows if rule(x) and x["src"] == "REFERENCE"], "dsg"),
        cote_directe=roi([x for x in rows if rule(x) and x["src"] == "DIRECT"], "dsg"),
        par_tranche_cote=[dict(cote=f"{lo}-{hi}", **roi([x for x in rows if rule(x) and x["o_ref"] and lo <= x["o_ref"] < hi], "dsg"))
                          for lo, hi in [(1, 4), (4, 8), (8, 15), (15, 30), (30, 1000)]],
        ratio_1_5=roi([x for x in rows if x["g"] == "TROT" and x["pc"] / x["pm"] >= 1.5], "dsg"),
        ratio_1_2=roi([x for x in rows if x["g"] == "TROT" and x["pc"] / x["pm"] >= 1.2], "dsg"))
    out["part_cote_directe"] = round(sum(x["src"] == "DIRECT" for x in rows) / len(rows), 3)
    out["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "reports", "diag.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
