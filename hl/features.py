"""Variables Hippique Lab V1.

Principe anti-fuite : FeatureState.features(race) n'utilise que l'état construit avec les courses
ANTÉRIEURES et les champs pré-course ; FeatureState.update(race) intègre ensuite le résultat.
"""
import math, re, unicodedata
from collections import defaultdict

PTS = {'1': 1.0, '2': .72, '3': .56, '4': .45, '5': .36, '6': .29, '7': .23, '8': .19, '9': .15, '0': .08}
HALF_LIFE = 180.0
LEX = {
    "k_malchance": ["gene", "enferme", "n'a pu s'exprimer", "sans espace", "bloque", "bouscule", "pas de passage",
                    "n'a jamais pu", "pas trouve d'ouverture", "contrarie", "mal parti", "pris dans"],
    "k_exterieur": ["a l'exterieur", "sans abri", "troisieme epaisseur", "deuxieme epaisseur", "en dehors",
                    "dos au vent", "au large"],
    "k_faute": ["fautif", "desuni", "faute", "a rompu", "galop", "s'est mis a la faute"],
    "k_finit": ["termine bien", "fini vite", "bonne fin", "en progres", "progresse", "bien fini", "rapide",
                "a bien termine", "s'est bien comporte", "honorable", "a conclu"],
    "k_neg": ["decevant", "jamais dans le coup", "sans reaction", "s'est arrete", "a cede", "lache", "n'a rien montre",
              "rien fait", "sans ressort"],
}


def _norm(s):
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def comment_flags(text):
    t = _norm(text)
    return {k: float(any(w in t for w in ws)) for k, ws in LEX.items()}


def parse_musique(s, group, disc):
    runs, season = [], 1.0
    for m in re.finditer(r'\((\d\d)\)|([0-9DATRG])([apmshc])', s or ''):
        if m.group(1):
            season *= 0.85
            continue
        place, kind = m.group(2), m.group(3)
        if group == "PLAT":
            wd = 1.0 if kind == 'p' else 0.5
        elif group == "OBSTACLE":
            wd = 1.0 if kind in 'hsc' else 0.5
        else:
            wd = 1.0 if (kind == 'a') == (disc == "ATTELE") else 0.6
        runs.append((PTS.get(place, 0.0), wd * season, place in 'DATRG'))
    runs = runs[:8]
    if not runs:
        return dict(mus_form=None, mus_cons=None, mus_dq=None, mus_n=0.0)
    ws = [0.78 ** i * w for i, (_, w, _) in enumerate(runs)]
    return dict(mus_form=sum(w * p for w, (p, _, _) in zip(ws, runs)) / sum(ws),
                mus_cons=sum(p >= .56 for p, _, _ in runs) / len(runs),
                mus_dq=sum(d for _, _, d in runs[:6]) / min(len(runs), 6), mus_n=float(len(runs)))


def _days(a, b):
    from datetime import date
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


class Decayed:
    """Compteurs à décroissance exponentielle (demi-vie 180 jours) : courses, victoires, victoires attendues par le marché."""
    __slots__ = ("n", "w", "e", "d", "hl")

    def __init__(self, hl=HALF_LIFE):
        self.n = self.w = self.e = 0.0
        self.d = None
        self.hl = hl

    def _decay(self, d):
        if self.d:
            f = 0.5 ** (_days(self.d, d) / self.hl)
            self.n *= f; self.w *= f; self.e *= f
        self.d = d

    def add(self, d, won, pexp):
        self._decay(d); self.n += 1; self.w += won; self.e += pexp

    def stats(self, d, prior_n=20.0, prior_wr=0.1):
        n, w, e = self.n, self.w, self.e
        if self.d:
            f = 0.5 ** (_days(self.d, d) / self.hl); n, w, e = n * f, w * f, e * f
        wr = (w + prior_wr * prior_n) / (n + prior_n)
        ae = (w + 5.0) / (e + 5.0)        # victoires réelles / attendues par le marché, lissé
        return wr, math.log(ae), math.log1p(n)


FEATURES = [
    # musique et fiche PMU
    "mus_form", "mus_cons", "mus_dq", "mus_n", "log_gains", "gains_per_race", "win_rate", "place_rate", "log_courses",
    "age", "log_gains_annee", "log_prix", "class_jump",
    # historique propre au cheval
    "has_hist", "log_days", "long_rest", "hist_beaten", "hist_win5", "hist_place5", "red_resid", "red_best", "elo",
    "dist_apt", "track_apt", "spec_apt", "terrain_apt", "dq_rate_hist", "first_deferre",
    "k_malchance", "k_exterieur", "k_faute", "k_finit", "k_neg",
    # entourage
    "drv_wr", "drv_ae", "drv_n", "trn_wr", "trn_ae", "trn_n", "combo_n", "combo_wr", "driver_change",
    # situation du jour
    "corde", "recul", "deferre4", "deferre_any", "oeil_on", "avis_pos", "avis_neg", "inedit", "poids", "valeur",
    # confrontations directes et écarts à l'arrivée
    "h2h_score", "h2h_n", "margin3",
    # vitesses normalisées au trot (réduction kilométrique corrigée de la piste, de la distance, du départ et du jour)
    "spd_last", "spd_avg3", "spd_best5", "spd_dist", "spd_n",
    # V1.2 (étude 2020-2026) : entourage par discipline / hippodrome / forme récente, duo entraîneur-driver, étalon, corde
    "drv_disc_ae", "trn_disc_ae", "trn_hot_ae", "trn_track_ae", "drv_track_ae", "trn_drv_ae", "sire_ae", "draw_ae",
]


def draw_key(race, u):
    """Biais de corde / place derrière l'autostart, par hippodrome, discipline et distance."""
    if race["group"] == "TROT" and not race["autostart"]:
        return None
    c = u["corde"] if race["group"] != "TROT" else u["num"]
    try:
        c = int(c)
    except (TypeError, ValueError):
        return None
    if c <= 0:
        return None
    b = 1 if c <= 2 else 2 if c <= 4 else 3 if c <= 7 else 4 if c <= 10 else 5
    return (race["hippodrome"], race["discipline"], int((race["distance"] or 0) / 200), b)


class FeatureState:
    def __init__(self):
        self.horse = {}                         # horse -> dict(hist=[...], elo, last_deferre)
        self.drv = defaultdict(Decayed)
        self.trn = defaultdict(Decayed)
        self.combo = defaultdict(lambda: [0.0, 0.0])
        self.par = {}                           # (hippodrome, spécialité, autostart, distance arrondie) -> [réduction de référence, n]
        self.variant = {}                       # (date, hippodrome, spécialité) -> [somme des écarts du jour, n]
        self.h2h = {}                           # (cheval a, cheval b) trié -> [a devant b, b devant a]
        self.ctx = defaultdict(lambda: Decayed(365.0))   # V1.2 : (type, clé...) -> victoires / attendues par le marché
        self.hot = defaultdict(lambda: Decayed(21.0))    # V1.2 : forme des 3 dernières semaines

    # ---------- variables pré-course ----------
    def features(self, race):
        d, grp = race["date"], race["group"]
        rows = []
        for u in race["runners"]:
            if u["np"]:
                continue
            f = dict(parse_musique(u["musique"], grp, race["discipline"]))
            nc = u["n_courses"] or 0
            f.update(log_gains=math.log1p(u["gains"]), gains_per_race=u["gains"] / max(nc, 1) / 1000.0,
                     win_rate=(u["n_vict"] + 0.5) / (nc + 5.0), place_rate=(u["n_places"] + 1.5) / (nc + 5.0),
                     log_courses=math.log1p(nc), age=u["age"], log_gains_annee=math.log1p(u["gains_annee"]),
                     log_prix=math.log1p(race["prix"]))
            h = self.horse.get(u["horse"])
            hist = h["hist"] if h else []
            if hist:
                last = hist[-1]
                days = _days(last["date"], d)
                recent = hist[-5:]
                f.update(has_hist=1.0, log_days=math.log1p(days), long_rest=float(days > 90),
                         hist_beaten=sum(x["beaten"] for x in recent) / len(recent),
                         hist_win5=sum(x["place"] == 1 for x in recent) / len(recent),
                         hist_place5=sum(1 <= x["place"] <= 3 for x in recent) / len(recent),
                         elo=h["elo"], dq_rate_hist=sum(x["dq"] for x in hist[-8:]) / len(hist[-8:]),
                         class_jump=math.log1p(race["prix"]) - sum(math.log1p(x["prix"]) for x in recent) / len(recent))
                rr = [x["red_resid"] for x in hist[-4:] if x["red_resid"] is not None]
                f["red_resid"] = -sum(rr[-3:]) / len(rr[-3:]) if rr else None   # signe : plus haut = plus rapide
                f["red_best"] = -min(rr) if rr else None
                def apt(cond):
                    sel = [x["beaten"] for x in hist if cond(x)]
                    allb = sum(x["beaten"] for x in hist) / len(hist)
                    return (allb - sum(sel) / len(sel)) * min(1.0, len(sel) / 3.0) if sel else None
                f["dist_apt"] = apt(lambda x: abs((x["distance"] or 0) - (race["distance"] or 0)) <= 200)
                f["track_apt"] = apt(lambda x: x["hippodrome"] == race["hippodrome"])
                f["spec_apt"] = apt(lambda x: x["discipline"] == race["discipline"])
                f["terrain_apt"] = (apt(lambda x: x["pen"] is not None and abs(x["pen"] - race["pen"]) <= 0.6)
                                    if grp in ("PLAT", "OBSTACLE") and race["pen"] is not None else None)
                f.update(comment_flags(last["comment"]))
                f["first_deferre"] = float(bool(u["deferre"]) and not last["deferre"]) if grp == "TROT" else 0.0
            else:
                f.update(has_hist=0.0, long_rest=0.0, first_deferre=0.0, **{k: 0.0 for k in LEX})
            drv = self.drv[u["driver"]].stats(d) if u["driver"] else (None, None, None)
            trn = self.trn[u["trainer"]].stats(d) if u["trainer"] else (None, None, None)
            cb = self.combo.get((u["horse"], u["driver"]), [0.0, 0.0])
            f.update(drv_wr=drv[0], drv_ae=drv[1], drv_n=drv[2], trn_wr=trn[0], trn_ae=trn[1], trn_n=trn[2],
                     combo_n=math.log1p(cb[0]), combo_wr=(cb[1] + 0.2) / (cb[0] + 2.0),
                     driver_change=float(u["driver_change"]),
                     corde=u["corde"] if (grp == "PLAT" or race["autostart"]) else None,
                     recul=float((u["recul"] or 0) > 0) if grp == "TROT" else None,
                     deferre4=float(u["deferre"] == "DEFERRE_ANTERIEURS_POSTERIEURS"),
                     deferre_any=float("DEFERRE" in u["deferre"]),
                     oeil_on=float(u["oeilleres"] not in ("", "SANS_OEILLERES")),
                     avis_pos=float(u["avis"] == "POSITIF"), avis_neg=float(u["avis"] == "NEGATIF"),
                     inedit=float(u["inedit"]), poids=(u["poids"] / 10.0 if u["poids"] else None) if grp != "TROT" else None,
                     valeur=u["valeur"] if grp != "TROT" else None)
            # confrontations directes avec les partants du jour
            diff = meet = 0.0
            for v in race["runners"]:
                if v["np"] or v["horse"] == u["horse"]:
                    continue
                a, b = sorted((u["horse"], v["horse"]))
                rec = self.h2h.get((a, b))
                if rec:
                    mine, theirs = (rec[0], rec[1]) if a == u["horse"] else (rec[1], rec[0])
                    diff += mine - theirs; meet += mine + theirs
            f["h2h_score"] = diff / (meet + 2.0)
            f["h2h_n"] = math.log1p(meet)
            figs = []
            for x in (hist or []):
                if x.get("sp_raw") is not None:
                    v = self.variant.get(x["var_key"])
                    vm = v[0] / v[1] if v and v[1] else 0.0
                    figs.append((x["distance"], max(-5.0, min(5.0, -(x["sp_raw"] - vm)))))
            if grp == "TROT" and figs:
                fv = [y for _, y in figs]
                near = [y for dd, y in figs if dd and race["distance"] and abs(dd - race["distance"]) <= 200]
                f.update(spd_last=fv[-1], spd_avg3=sum(fv[-3:]) / len(fv[-3:]), spd_best5=max(fv[-5:]),
                         spd_dist=sum(near[-3:]) / len(near[-3:]) if near else None, spd_n=math.log1p(len(fv)))
            else:
                f.update(spd_last=None, spd_avg3=None, spd_best5=None, spd_dist=None, spd_n=0.0)
            mg = [x["lengths"] for x in hist[-3:] if x.get("lengths") is not None] if hist else []
            f["margin3"] = -sum(mg) / len(mg) if mg else None
            # V1.2 : log(victoires / victoires attendues par le marché), lissé, informations antérieures uniquement
            tr, dr, sire = u["trainer"], u["driver"], u.get("sire")
            ae = lambda store, key: store[key].stats(d)[1] if key in store else 0.0
            dk = draw_key(race, u)
            f.update(drv_disc_ae=ae(self.ctx, ("DC", dr, race["discipline"])) if dr else None,
                     trn_disc_ae=ae(self.ctx, ("TC", tr, race["discipline"])) if tr else None,
                     trn_hot_ae=ae(self.hot, tr) if tr else None,
                     trn_track_ae=ae(self.ctx, ("TT", tr, race["hippodrome"])) if tr else None,
                     drv_track_ae=ae(self.ctx, ("DT", dr, race["hippodrome"])) if dr else None,
                     trn_drv_ae=ae(self.ctx, ("TD", tr, dr)) if tr and dr else None,
                     sire_ae=ae(self.ctx, ("S", grp, sire)) if sire else None,
                     draw_ae=ae(self.ctx, ("DR",) + dk) if dk else None)
            rows.append(dict(num=u["num"], nom=u["nom"], horse=u["horse"], driver=u["driver"], x=f,
                             odds_ref=u["odds_mkt"], odds_final=u["odds_final"], place=u["place"]))
        return rows

    # ---------- intégration du résultat ----------
    def update(self, race):
        d = race["date"]
        runners = [u for u in race["runners"] if not u["np"]]
        n = len(runners)
        if n < 2:
            return
        inv = {u["num"]: 1.0 / u["odds_final"] for u in runners if u["odds_final"]}
        s = sum(inv.values()) or 1.0
        reds = sorted(u["red_km"] for u in runners if u["red_km"])
        med = reds[len(reds) // 2] if reds else None
        winner = race["arrivee"][0] if race["arrivee"] else None
        par_val = var_key = None
        if race["group"] == "TROT" and med and race["distance"]:
            pk = (race["hippodrome"], race["discipline"], race["autostart"], int(round(race["distance"] / 100.0)) * 100)
            var_key = (d, race["hippodrome"], race["discipline"])
            pv = self.par.get(pk)
            if pv and pv[1] >= 3:
                par_val = pv[0]
                vv = self.variant.setdefault(var_key, [0.0, 0])
                vv[0] += (med - par_val) / 1000.0; vv[1] += 1
            if pv:
                pv[0] += 0.08 * (med - pv[0]); pv[1] += 1
            else:
                self.par[pk] = [float(med), 1]
        for u in runners:
            pl = u["place"] if isinstance(u["place"], int) and u["place"] > 0 else None
            dq = "DISQUALIFIE" in u["incident"] or u["incident"] in ("ARRETE", "TOMBE", "DEROBE")
            h = self.horse.setdefault(u["horse"], dict(hist=[], elo=1500.0))
            h["hist"].append(dict(date=d, place=pl or 99, beaten=((pl - 1) / (n - 1)) if pl else 1.0, dq=float(dq),
                                  distance=race["distance"], hippodrome=race["hippodrome"], discipline=race["discipline"],
                                  prix=race["prix"], pen=race["pen"], deferre=u["deferre"], comment=u["comment"],
                                  red_resid=((u["red_km"] - med) / 1000.0) if (u["red_km"] and med and not dq) else None,
                                  lengths=min(u.get("lengths_behind"), 20.0) if u.get("lengths_behind") is not None else None,
                                  sp_raw=((u["red_km"] - par_val) / 1000.0) if (par_val and u["red_km"] and not dq) else None,
                                  var_key=var_key,
                                  meteo=race.get("meteo")))
            h["hist"] = h["hist"][-20:]
            won = float(u["num"] == winner)
            pe = inv.get(u["num"], 0.0) / s
            if u["driver"]:
                self.drv[u["driver"]].add(d, won, pe)
            if u["trainer"]:
                self.trn[u["trainer"]].add(d, won, pe)
                self.hot[u["trainer"]].add(d, won, pe)
            tr, dr, sire = u["trainer"], u["driver"], u.get("sire")
            keys = []
            if dr: keys += [("DC", dr, race["discipline"]), ("DT", dr, race["hippodrome"])]
            if tr: keys += [("TC", tr, race["discipline"]), ("TT", tr, race["hippodrome"])]
            if tr and dr: keys.append(("TD", tr, dr))
            if sire: keys.append(("S", race["group"], sire))
            dk = draw_key(race, u)
            if dk: keys.append(("DR",) + dk)
            for k in keys:
                self.ctx[k].add(d, won, pe)
            cb = self.combo.setdefault((u["horse"], u["driver"]), [0.0, 0.0]); cb[0] += 1; cb[1] += won
        # confrontations directes : qui a fini devant qui (non classés derniers)
        rank = {u["horse"]: (u["place"] if isinstance(u["place"], int) and u["place"] > 0 else 99) for u in runners}
        hs = list(rank)
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                ra, rb = rank[hs[i]], rank[hs[j]]
                if ra == rb:
                    continue
                a, b = sorted((hs[i], hs[j]))
                rec = self.h2h.setdefault((a, b), [0, 0])
                first = hs[i] if ra < rb else hs[j]
                rec[0 if first == a else 1] += 1
        # Elo multi-joueurs : chaque paire, non classés derniers ex aequo
        order = {u["horse"]: (u["place"] if isinstance(u["place"], int) and u["place"] > 0 else 99) for u in runners}
        elo = {k: self.horse[k]["elo"] for k in order}
        K = 32.0 / (n - 1)
        delta = defaultdict(float)
        ks = list(order)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = ks[i], ks[j]
                if order[a] == order[b]:
                    continue
                ea = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400.0))
                sa = 1.0 if order[a] < order[b] else 0.0
                delta[a] += K * (sa - ea); delta[b] -= K * (sa - ea)
        for k, v in delta.items():
            self.horse[k]["elo"] += v
