"""Moteur de décision V1.1 : simple gagnant, simple placé, couplé gagnant, 2sur4. Partagé par le backtest et le live.

Chaque marché n'est misé en live que si params["markets"][marché] est vrai : le backtest l'active seulement
si son ROI hors échantillon est positif sur assez de paris (porte par marché, voir train.py).
"""
import math, zlib
import numpy as np

from .model import combine, place_probs, sp_expected, simulate_top, pair_probs, pair_expected

RULES = dict(
    u_dec=0.10,          # incertitude relative sur la probabilité de décision
    haircut=0.90,        # pari mutuel : décision sur 90 % du rapport attendu
    min_edge=0.015, min_ev_low=0.03, min_p=0.05, max_odds=25.0,
    sp_min_ev_low=0.03, sp_min_p=0.25,
    cg_min_ev_low=0.10, cg_min_p=0.04, s4_min_ev_low=0.10, s4_min_p=0.15, s4_min_runners=10,
    kelly=0.25, max_bet_pct=0.02, max_day_pct=0.06, min_stake=1.0, s4_min_stake=3.0,
)
MARKETS = ("SIMPLE_GAGNANT", "SIMPLE_PLACE", "COUPLE_GAGNANT", "DEUX_SUR_QUATRE")


def n_places(n):
    return 3 if n >= 8 else (2 if n >= 4 else 0)


def market_probs(odds):
    inv = np.array([1.0 / o for o in odds]); return inv / inv.sum()


def _kelly(p_low, div):
    b = div * RULES["haircut"] - 1
    return RULES["kelly"] * max(0.0, (b * p_low - (1 - p_low)) / b) if b > 0 else 0.0


def analyse(rows, pf, params, race_key="", all_markets=False):
    """rows : partants (odds_ref = cote marché) ; pf : P fondamentale.
    -> dict(preds, cands, shadow). all_markets=True (backtest) évalue tous les marchés quelle que soit leur porte."""
    n = len(rows)
    odds = [r.get("odds_ref") for r in rows]
    ok_market = all(o and o > 1.0 for o in odds)
    out = dict(market_ok=ok_market, preds=[], cands=[], shadow=None)
    if not ok_market:
        for r, p in zip(rows, pf):
            out["preds"].append(dict(num=r["num"], horse=r["nom"], p_fund=float(p)))
        return out
    R = RULES
    pm = market_probs(odds)
    pd = combine(pf, pm, params["alpha"], params["beta"])
    k = n_places(n)
    pp = place_probs(pd, k, params["gamma"], params["delta"]) if k else np.zeros(n)
    cands = []
    for i, r in enumerate(rows):
        o = odds[i]
        pl = pd[i] * (1 - R["u_dec"])
        ev_low = pl * o * R["haircut"] - 1
        sp_div = sp_expected(params["sp_coef"], o, n) if k else None
        ppl = pp[i] * (1 - R["u_dec"]) if k else None
        sp_ev_low = (ppl * sp_div * R["haircut"] - 1) if k else None
        out["preds"].append(dict(num=r["num"], horse=r["nom"], p_fund=float(pf[i]), p_mkt=float(pm[i]), p_dec=float(pd[i]),
                                 p_place=float(pp[i]) if k else None, odds=o, edge=float(pd[i] - pm[i]),
                                 ev=float(pd[i] * o - 1), ev_low=float(ev_low), sp_div_exp=sp_div, sp_ev_low=sp_ev_low,
                                 edge_raw=float(pf[i] - pm[i])))
        if pd[i] - pm[i] >= R["min_edge"] and ev_low >= R["min_ev_low"] and pd[i] >= R["min_p"] and o <= R["max_odds"]:
            cands.append(dict(num=r["num"], horse=r["nom"], market="SIMPLE_GAGNANT", odds=o, p=float(pd[i]), p_low=float(pl),
                              ev_low=float(ev_low), kelly_f=_kelly(pl, o), min_odds=math.ceil(10.0 / pl) / 10.0))
        if k and sp_ev_low >= R["sp_min_ev_low"] and pp[i] >= R["sp_min_p"]:
            cands.append(dict(num=r["num"], horse=r["nom"], market="SIMPLE_PLACE", odds=o, p=float(pp[i]), p_low=float(ppl),
                              ev_low=float(sp_ev_low), kelly_f=_kelly(ppl, sp_div), sp_div_exp=sp_div, min_odds=None))
    # couples : arrivées simulées par Harville corrigé
    if n >= 4 and (params.get("cg_coef") or params.get("s4_coef")):
        top = simulate_top(pd, params["gamma"], params["delta"], sims=3000, seed=zlib.crc32(race_key.encode()) & 0xffffffff)
        cg, s4 = pair_probs(top, n)
        cand_idx = list(np.argsort(-pd)[:6])
        shadow = {}
        for mk, mat, coef, mn_ev, mn_p in (("COUPLE_GAGNANT", cg, params.get("cg_coef"), R["cg_min_ev_low"], R["cg_min_p"]),
                                           ("DEUX_SUR_QUATRE", s4, params.get("s4_coef"), R["s4_min_ev_low"], R["s4_min_p"])):
            if not coef or (mk == "DEUX_SUR_QUATRE" and n < R["s4_min_runners"]):
                continue
            best = None
            for a in cand_idx:
                for b in cand_idx:
                    if b <= a:
                        continue
                    p = float(mat[min(a, b), max(a, b)])
                    if p <= 0:
                        continue
                    div = pair_expected(coef, odds[a], odds[b], n)
                    plo = p * (1 - R["u_dec"])
                    evl = plo * div * R["haircut"] - 1
                    combo = "-".join(str(v) for v in sorted((rows[a]["num"], rows[b]["num"])))
                    item = dict(combo=combo, p=p, div_exp=div, ev_low=evl,
                                horse=f"{rows[a]['nom']} + {rows[b]['nom']}")
                    if best is None or evl > best["ev_low"]:
                        best = item
                    if evl >= mn_ev and p >= mn_p:
                        cands.append(dict(num=combo, horse=item["horse"], market=mk, odds=div, p=p, p_low=plo, ev_low=evl,
                                          kelly_f=_kelly(plo, div), div_exp=div, min_odds=None))
            if best:
                shadow[mk] = best
        out["shadow"] = shadow or None
    enabled = params.get("markets") or {}
    if not all_markets:
        if params.get("alpha", 0) <= 0:
            cands = []                       # modèle non validé : observation seulement
        cands = [c for c in cands if enabled.get(c["market"])]
    best = {}
    for c in cands:                          # au plus un ticket par course et par marché
        if c["market"] not in best or c["ev_low"] > best[c["market"]]["ev_low"]:
            best[c["market"]] = c
    out["cands"] = list(best.values())
    return out


def size(cands, capital, already=0.0):
    """Fractions de Kelly -> mises en euros, plafonds 2 % par ticket et 6 % par jour (mises déjà prises comprises)."""
    R = RULES
    stakes = [min(c["kelly_f"] * capital, R["max_bet_pct"] * capital) for c in cands]
    room = max(0.0, R["max_day_pct"] * capital - already)
    tot = sum(stakes)
    k = min(1.0, room / tot) if tot > 0 else 1.0
    out = []
    for c, s in zip(cands, stakes):
        st = math.floor(s * k * 2) / 2
        mn = R["s4_min_stake"] if c["market"] == "DEUX_SUR_QUATRE" else R["min_stake"]
        if st >= mn:
            out.append(dict(c, stake=st))
    return out
