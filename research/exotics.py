"""Erreurs de prix des paris combinés : ROI réel de chaque combinaison de rangs de cote.
python3 exotics.py <dépôt> <sortie.json>
Pour chaque course où le pari est offert, on « joue » toutes les combinaisons de rangs (rang 1 = favori) et on
règle au rapport officiel (0 si la combinaison n'est pas dans les rapports). Découverte A = 2020-2024, validation B = 2025-26.
Deux scénarios : rangs selon la cote du matin (référence) et selon la cote finale (pari tardif)."""
import sys, os, json, gzip, glob, itertools
from collections import defaultdict
REPO = sys.argv[1]
TYPES = {"COUPLE_GAGNANT": 2, "COUPLE_PLACE": 2, "DEUX_SUR_QUATRE": 2, "TRIO": 3}
MAXR = 8          # rangs 1..8 détaillés, 9 à 12 regroupés en « 9+ » (au-delà : non joués)


def divs(rap, typ):
    out = {}
    for t in rap or []:
        if t.get("typePari") != typ:
            continue
        for x in t.get("rapports", []):
            d = x.get("dividendePourUnEuro")
            if d is None:
                continue
            try:
                k = "-".join(str(v) for v in sorted(int(s) for s in str(x.get("combinaison")).split("-")))
            except ValueError:
                continue
            out[k] = max(out.get(k, 0.0), d / 100.0)
    return out


def rate(p, key):
    x = p.get(key) or {}
    return float(x["rapport"]) if x.get("rapport") else None


# acc[(scénario, type, période, combo_de_rangs)] = [mises, retours, gagnants]
acc = defaultdict(lambda: [0, 0.0, 0])
nr_acc = defaultdict(lambda: [0, 0.0, 0])     # même chose, ventilé par taille de champ pour les meilleures cases
files = sorted(glob.glob(os.path.join(REPO, "data", "*", "*.json.gz")))
for i, f in enumerate(files):
    day = json.load(gzip.open(f, "rt", encoding="utf-8"))
    per = "A" if day["date"] < "2025-01-01" else "B"
    for x in day.get("races", []):
        parts = (x.get("participants") or {}).get("participants") or []
        live = [p for p in parts if p.get("statut") != "NON_PARTANT" and p.get("incident") != "NON_PARTANT"]
        n = len(live)
        if n < 6:
            continue
        rap = x.get("rapports") if isinstance(x.get("rapports"), list) else []
        for scen, key in (("matin", "dernierRapportReference"), ("final", "dernierRapportDirect")):
            od = [(rate(p, key), p.get("numPmu")) for p in live]
            if not all(o and o > 1 for o, _ in od):
                continue
            ranked = [num for _, num in sorted(od, key=lambda t: (t[0], t[1]))]
            rk = {num: min(r + 1, MAXR + 1) for r, num in enumerate(ranked)}
            for typ, k in TYPES.items():
                dv = divs(rap, typ)
                if not dv:
                    continue
                for combo in itertools.combinations(ranked[:12], k):
                    ck = "-".join(str(v) for v in sorted(combo))
                    rkey = "-".join(str(rk[c]) if rk[c] <= MAXR else f"{MAXR + 1}+" for c in combo)
                    r = dv.get(ck, 0.0)
                    a = acc[(scen, typ, per, rkey)]; a[0] += 1; a[1] += r; a[2] += r > 0
                    nb = "6-9" if n <= 9 else "10-13" if n <= 13 else "14+"
                    b = nr_acc[(scen, typ, per, rkey, nb)]; b[0] += 1; b[1] += r; b[2] += r > 0
    if i % 200 == 0:
        print(i, len(files), day["date"], flush=True)

res = {}
for (scen, typ, per, rkey), (n, ret, w) in acc.items():
    res.setdefault(scen, {}).setdefault(typ, {}).setdefault(rkey, {})[per] = dict(n=n, roi=round(ret / n - 1, 4), hit=round(w / n, 4))
res_nb = {}
for (scen, typ, per, rkey, nb), (n, ret, w) in nr_acc.items():
    res_nb.setdefault(scen, {}).setdefault(typ, {}).setdefault(rkey, {}).setdefault(nb, {})[per] = dict(n=n, roi=round(ret / n - 1, 4))
json.dump(dict(par_rangs=res, par_rangs_et_champ=res_nb), open(sys.argv[2], "w"), indent=0)
print("ok")
