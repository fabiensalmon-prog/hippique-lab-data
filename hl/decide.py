"""Moteur de décision V1 : simple gagnant + simple placé ; exotiques en observation. Partagé par le backtest et le live."""
import math
import numpy as np

from .model import combine, place_probs, sp_expected

RULES = dict(
    u_dec=0.10,          # incertitude relative sur P décision
    haircut=0.90,        # pari mutuel : décision sur 90 % du rapport probable
    min_edge=0.015, min_ev_low=0.03, min_p=0.05, max_odds=25.0,
    sp_min_ev_low=0.03, sp_min_p=0.25,
    kelly=0.25, max_bet_pct=0.02, max_day_pct=0.06, min_stake=1.0,
)


def n_places(n):
    return 3 if n >= 8 else (2 if n >= 4 else 0)


def market_probs(odds):
    inv = np.array([1.0 / o for o in odds]); return inv / inv.sum()


def analyse(rows, pf, params):
    """rows : partants (avec odds_ref) ; pf : P fondamentale. -> dict prédictions + candidats de pari (fractions de Kelly)."""
    n = len(rows)
    odds = [r.get("odds_ref") for r in rows]
    ok_market = all(o and o > 1.0 for o in odds)
    out = dict(market_ok=ok_market, preds=[], cands=[])
    if not ok_market:
        for r, p in zip(rows, pf):
            out["preds"].append(dict(num=r["num"], horse=r["nom"], p_fund=float(p)))
        return out
    pm = market_probs(odds)
    pd = combine(pf, pm, params["alpha"], params["beta"])
    k = n_places(n)
    pp = place_probs(pd, k, params["gamma"], params["delta"]) if k else np.zeros(n)
    R = RULES
    for i, r in enumerate(rows):
        o = odds[i]
        pl = pd[i] * (1 - R["u_dec"])
        ev = pd[i] * o - 1
        ev_low = pl * o * R["haircut"] - 1
        sp_div = sp_expected(params["sp_coef"], o, n) if k else None
        ppl = pp[i] * (1 - R["u_dec"]) if k else None
        sp_ev_low = (ppl * sp_div * R["haircut"] - 1) if k else None
        pr = dict(num=r["num"], horse=r["nom"], p_fund=float(pf[i]), p_mkt=float(pm[i]), p_dec=float(pd[i]),
                  p_place=float(pp[i]) if k else None, odds=o, edge=float(pd[i] - pm[i]), ev=float(ev),
                  ev_low=float(ev_low), sp_div_exp=sp_div, sp_ev_low=sp_ev_low, edge_raw=float(pf[i] - pm[i]))
        out["preds"].append(pr)
        if pr["edge"] >= R["min_edge"] and ev_low >= R["min_ev_low"] and pd[i] >= R["min_p"] and o <= R["max_odds"]:
            b = o * R["haircut"] - 1
            f = max(0.0, (b * pl - (1 - pl)) / b)
            out["cands"].append(dict(num=r["num"], horse=r["nom"], market="SIMPLE_GAGNANT", odds=o, p=float(pd[i]),
                                     p_low=float(pl), ev_low=float(ev_low), kelly_f=R["kelly"] * f,
                                     min_odds=math.ceil(10.0 / pl) / 10.0))
        if k and sp_ev_low >= R["sp_min_ev_low"] and pp[i] >= R["sp_min_p"]:
            b = sp_div * R["haircut"] - 1
            if b > 0:
                f = max(0.0, (b * ppl - (1 - ppl)) / b)
                out["cands"].append(dict(num=r["num"], horse=r["nom"], market="SIMPLE_PLACE", odds=o, p=float(pp[i]),
                                         p_low=float(ppl), ev_low=float(sp_ev_low), kelly_f=R["kelly"] * f,
                                         sp_div_exp=sp_div, min_odds=None))
    if params.get("alpha", 0) <= 0:
        out["cands"] = []            # modèle non validé : observation seulement, aucun ticket
    best = {}
    for c in out["cands"]:
        if c["market"] not in best or c["ev_low"] > best[c["market"]]["ev_low"]:
            best[c["market"]] = c
    out["cands"] = list(best.values())
    # exotiques en observation : meilleur couplé gagnant (Harville corrigé) enregistré, jamais misé
    if n >= 4:
        top = list(np.argsort(-pd)[:4])
        best = None
        for a in top:
            for b_ in top:
                if a == b_:
                    continue
                pab = pd[a] * (pd[b_] ** params["gamma"]) / (np.delete(pd ** params["gamma"], a).sum())
                pba = pd[b_] * (pd[a] ** params["gamma"]) / (np.delete(pd ** params["gamma"], b_).sum())
                if best is None or pab + pba > best[2]:
                    best = (int(rows[a]["num"]), int(rows[b_]["num"]), float(pab + pba))
        out["shadow"] = dict(couple_gagnant=sorted(best[:2]), p=best[2]) if best else None
    return out


def size(cands, capital, already=0.0):
    """Fractions de Kelly -> mises en euros avec plafonds par ticket et par jour. Une seule mise par course et par marché."""
    R = RULES
    stakes = [min(c["kelly_f"] * capital, R["max_bet_pct"] * capital) for c in cands]
    room = max(0.0, R["max_day_pct"] * capital - already)
    tot = sum(stakes)
    k = min(1.0, room / tot) if tot > 0 else 1.0
    out = []
    for c, s in zip(cands, stakes):
        st = math.floor(s * k * 2) / 2
        if st >= R["min_stake"]:
            out.append(dict(c, stake=st))
    return out
