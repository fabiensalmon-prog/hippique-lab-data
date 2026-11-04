"""Expérience : le gradient boosting (LightGBM, objectif de classement) bat-il le logit face au MARCHÉ FINAL ?

python3 -m hl.exp_gbm -> reports/exp_gbm.json
Découpage simple dans le temps : entraînement jusqu'au 31/12/2025, test sur 2026 (jamais vu).
Mesure : ρ² du marché final, du logit + marché, du GBM + marché, et des deux modèles + marché.
"""
import json, math, os
import numpy as np
import lightgbm as lgb

from .train import build_dataset, fit_group, predict_group, GROUPS
from .model import Design, fit_stage2, combine
from .features import FEATURES

ROOT = os.path.join(os.path.dirname(__file__), "..")
CUT = "2026-01"


def mfinal(r):
    of = [x["odds_final"] for x in r["rows"]]
    return (np.array([1 / v for v in of]) / sum(1 / v for v in of)) if all(v and v > 1 for v in of) else None


def gbm_fit(races, valid):
    D = Design(races); V = Design(valid, D.means, D.stds)
    def lab(Dz):
        y = np.zeros(len(Dz.X)); y[Dz.y] = 1; return y
    grp = np.diff(np.append(D.starts, len(D.X))); vgrp = np.diff(np.append(V.starts, len(V.X)))
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=31,
                       min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(D.X, lab(D), group=grp, eval_set=[(V.X, lab(V))], eval_group=[vgrp],
          eval_at=[1], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m, D.means, D.stds


def gbm_probs(m, means, stds, races):
    D = Design(races, means, stds)
    s = m.predict(D.X)
    out = {}
    for r, a, b in zip(D.meta, D.starts, np.append(D.starts[1:], len(s))):
        z = s[a:b]; e = np.exp(z - z.max()); out[r["id"]] = e / e.sum()
    return out


def main():
    data, _ = build_dataset()
    data = [r for r in data if r["winner_idx"] is not None and mfinal(r) is not None]
    tr = [r for r in data if r["month"] < CUT]; te = [r for r in data if r["month"] >= CUT]
    # calibration de l'étage 2 sur la fin de la période d'entraînement (hors échantillon pour les modèles)
    inner = [r for r in tr if r["month"] < "2025-09"]; cal = [r for r in tr if r["month"] >= "2025-09"]
    res = {}
    P_log_cal, P_log_te, P_gbm_cal, P_gbm_te = {}, {}, {}, {}
    for g in GROUPS:
        a_in = [r for r in inner if r["group"] == g]; a_cal = [r for r in cal if r["group"] == g]
        a_tr = [r for r in tr if r["group"] == g]; a_te = [r for r in te if r["group"] == g]
        if len(a_in) < 200:
            continue
        P_log_cal.update(predict_group(fit_group(a_in), a_cal)); P_log_te.update(predict_group(fit_group(a_tr), a_te))
        m, mu, sd = gbm_fit(a_in, a_cal[: len(a_cal) // 2])
        P_gbm_cal.update(gbm_probs(m, mu, sd, a_cal))
        m2, mu2, sd2 = gbm_fit([r for r in a_tr if r["month"] < "2025-11"], [r for r in a_tr if r["month"] >= "2025-11"])
        P_gbm_te.update(gbm_probs(m2, mu2, sd2, a_te))
        imp = sorted(zip(FEATURES, m2.feature_importances_), key=lambda t: -t[1])[:12]
        res.setdefault("importance", {})[g] = [(f, int(v)) for f, v in imp]
    calr = [r for r in cal if r["id"] in P_log_cal and r["id"] in P_gbm_cal]
    ter = [r for r in te if r["id"] in P_log_te and r["id"] in P_gbm_te]
    W = lambda rs: [r["winner_idx"] for r in rs]
    a1, b1 = fit_stage2([P_log_cal[r["id"]] for r in calr], [mfinal(r) for r in calr], W(calr))
    a2, b2 = fit_stage2([P_gbm_cal[r["id"]] for r in calr], [mfinal(r) for r in calr], W(calr))
    # les deux modèles ensemble : moyenne géométrique puis étage 2
    mix = lambda p, q: (lambda z: z / z.sum())(np.sqrt(np.clip(p, 1e-9, 1) * np.clip(q, 1e-9, 1)))
    a3, b3 = fit_stage2([mix(P_log_cal[r["id"]], P_gbm_cal[r["id"]]) for r in calr], [mfinal(r) for r in calr], W(calr))
    ll0 = sum(-math.log(r["n"]) for r in ter)
    rho = lambda ps: round(1 - sum(math.log(max(1e-9, p[r["winner_idx"]])) for p, r in zip(ps, ter)) / ll0, 4)
    res.update(test_courses=len(ter), periode_test=f"{ter[0]['date']} -> {ter[-1]['date']}",
               marche_final=rho([mfinal(r) for r in ter]),
               logit_seul=rho([P_log_te[r["id"]] for r in ter]), gbm_seul=rho([P_gbm_te[r["id"]] for r in ter]),
               logit_plus_marche=rho([combine(P_log_te[r["id"]], mfinal(r), a1, b1) for r in ter]),
               gbm_plus_marche=rho([combine(P_gbm_te[r["id"]], mfinal(r), a2, b2) for r in ter]),
               deux_plus_marche=rho([combine(mix(P_log_te[r["id"]], P_gbm_te[r["id"]]), mfinal(r), a3, b3) for r in ter]),
               poids=dict(logit=[round(a1, 3), round(b1, 3)], gbm=[round(a2, 3), round(b2, 3)], deux=[round(a3, 3), round(b3, 3)]))
    for k in ("logit_plus_marche", "gbm_plus_marche", "deux_plus_marche"):
        res["gain_" + k] = round(res[k] - res["marche_final"], 4)
    json.dump(res, open(os.path.join(ROOT, "reports", "exp_gbm.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
