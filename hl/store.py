"""Lecture des journées PMU collectées -> courses normalisées, dans l'ordre chronologique."""
import gzip, json, glob, os, re

# écart au cheval précédent (plat / obstacle) en longueurs
_ECART = {"NEZ": 0.05, "COURTE_TETE": 0.1, "TETE": 0.2, "COURTE_ENCOLURE": 0.25, "ENCOLURE": 0.3, "DEMI_LONGUEUR": 0.5,
          "TROIS_QUARTS_LONGUEUR": 0.75, "LONGUEUR": 1.0, "DISTANCE": 15.0, "LOIN": 15.0, "DEAD_HEAT": 0.0, "TRES_LOIN": 20.0}


def ecart_longueurs(d):
    if not d:
        return None
    ident = (d.get("identifiant") or "").upper()
    if ident in _ECART:
        return _ECART[ident]
    lib = (d.get("libelleCourt") or "").replace(",", ".").replace("L", "").strip()
    try:
        if "/" in lib:
            a, b = lib.split()[-1].split("/") if " " in lib else lib.split("/")
            whole = float(lib.split()[0]) if " " in lib else 0.0
            return whole + float(a) / float(b)
        return float(lib)
    except (ValueError, IndexError):
        return None
from datetime import date
import unicodedata


def _norm_name(s):
    s = unicodedata.normalize("NFD", (s or "").upper())
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn").strip() or None

ROOT = os.environ.get("HL_DATA") or os.path.join(os.path.dirname(__file__), "..", "data")
DISC = {"ATTELE": "TROT", "MONTE": "TROT", "PLAT": "PLAT", "HAIE": "OBSTACLE", "STEEPLECHASE": "OBSTACLE", "CROSS": "OBSTACLE"}


def load_day(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def day_files(root=ROOT, until=None):
    fs = sorted(glob.glob(os.path.join(root, "*", "*.json.gz")))
    if until:
        fs = [f for f in fs if os.path.basename(f)[:10] <= until]
    return fs


def _div(rapports, typ):
    """{combinaison: dividende pour 1 €} pour un type de pari."""
    out = {}
    for t in rapports or []:
        if t.get("typePari") != typ:
            continue
        for x in t.get("rapports", []):
            d = x.get("dividendePourUnEuro")
            if d is not None:
                k = str(x.get("combinaison"))
                if "-" in k:   # combinaisons non ordonnées : clé triée « 1-7 »
                    try: k = "-".join(str(v) for v in sorted(int(t) for t in k.split("-")))
                    except ValueError: pass
                out[k] = d / 100.0
    return out


def _gains(p):
    g = p.get("gainsParticipant") or {}
    return (g.get("gainsCarriere") or 0) / 100.0, (g.get("gainsAnneeEnCours") or 0) / 100.0


def _rap(p, key):
    x = p.get(key) or {}
    r = x.get("rapport")
    return float(r) if r else None


def normalize(dayobj):
    """-> liste de courses : dict(meta, runners[]) ; seulement trot attelé/monté et plat, France."""
    out = []
    for x in dayobj.get("races", []):
        c = x.get("course") or {}
        grp = DISC.get(c.get("discipline"))
        if not grp:
            continue
        parts = (x.get("participants") or {}).get("participants") or []
        if not parts:
            continue
        arr = c.get("ordreArrivee") or []
        rap = x.get("rapports") if isinstance(x.get("rapports"), list) else []
        pen = c.get("penetrometre") or {}
        try:
            penv = float(str(pen.get("valeurMesure", "")).replace(",", "."))
        except ValueError:
            penv = None
        cat = c.get("categorieParticularite") or ""
        race = dict(
            id=f"{dayobj['date']}_R{x['r']}C{x['c']}", date=dayobj["date"], r=x["r"], c=x["c"],
            hippodrome=x.get("hippodrome") or (c.get("hippodrome") or {}).get("libelleCourt"),
            libelle=c.get("libelle"), heure=c.get("heureDepart"), group=grp, discipline=c.get("discipline"),
            distance=c.get("distance"), corde=c.get("corde"), prix=(c.get("montantPrix") or 0),
            autostart="AUTOSTART" in cat, handicap="HANDICAP" in cat, reclamer="RECLAMER" in cat,
            categorie=cat, piste=c.get("typePiste"), pen=penv, terrain=pen.get("intitule"),
            conditions=c.get("conditions") or "", n_declared=c.get("nombreDeclaresPartants"),
            meteo=dict(temp=(x.get("meteo") or {}).get("temperature"), ciel=(x.get("meteo") or {}).get("nebulositeLibelleCourt"),
                       vent=(x.get("meteo") or {}).get("forceVent")) if x.get("meteo") else None,
            arrivee=[a[0] if isinstance(a, list) and a else a for a in arr],
            dead_heat=any(isinstance(a, list) and len(a) > 1 for a in arr),
            div_sg=_div(rap, "SIMPLE_GAGNANT"), div_sp=_div(rap, "SIMPLE_PLACE"),
            div_cg=_div(rap, "COUPLE_GAGNANT"), div_2s4=_div(rap, "DEUX_SUR_QUATRE"),
            runners=[])
        for p in parts:
            np_ = p.get("statut") == "NON_PARTANT" or p.get("incident") == "NON_PARTANT"
            gc, gy = _gains(p)
            com = (p.get("commentaireApresCourse") or {}).get("texte") or ""
            race["runners"].append(dict(
                num=p.get("numPmu"), nom=p.get("nom"), horse=p.get("idCheval") or p.get("nom"),
                age=p.get("age"), sexe=p.get("sexe"), driver=p.get("driver"), trainer=p.get("entraineur"),
                musique=p.get("musique") or "", np=np_,
                gains=gc, gains_annee=gy, n_courses=p.get("nombreCourses") or 0, n_vict=p.get("nombreVictoires") or 0,
                n_places=p.get("nombrePlaces") or 0, valeur=p.get("handicapValeur"), poids=p.get("handicapPoids"),
                corde=p.get("placeCorde"), recul=p.get("handicapDistance"), deferre=p.get("deferre") or "",
                oeilleres=p.get("oeilleres") or "", driver_change=bool(p.get("driverChange")),
                inedit=bool(p.get("indicateurInedit")), avis=p.get("avisEntraineur") or "",
                sire=_norm_name(p.get("nomPere")),
                odds_ref=_rap(p, "dernierRapportReference"), odds_final=_rap(p, "dernierRapportDirect"),
                # --- informations APRÈS course : jamais utilisées comme variables de cette course ---
                place=p.get("ordreArrivee"), incident=p.get("incident") or "", red_km=p.get("reductionKilometrique"),
                temps=p.get("tempsObtenu"), comment=com, ecart=ecart_longueurs(p.get("distanceChevalPrecedent"))))
        fin = sorted([u for u in race["runners"] if isinstance(u["place"], int) and u["place"] > 0], key=lambda u: u["place"])
        cum = 0.0
        for u in race["runners"]:
            u["lengths_behind"] = None
        for k, u in enumerate(fin):
            if k > 0:
                e = u["ecart"]
                cum = cum + e if e is not None else None if cum is None else cum + 1.0
            u["lengths_behind"] = cum if k > 0 else 0.0
        live = [u for u in race["runners"] if not u["np"]]
        if live and all(u["odds_ref"] for u in live):
            race["mkt_src"] = "REFERENCE"
            for u in race["runners"]: u["odds_mkt"] = u["odds_ref"]
        else:
            # Pas de rapport de référence complet : la course n'a pas de marché utilisable avant le départ.
            # (Utiliser la cote finale ici faisait fuir l'information d'après-course dans le backtest.)
            race["mkt_src"] = "AUCUN"
            for u in race["runners"]: u["odds_mkt"] = None
        out.append(race)
    out.sort(key=lambda r: (r["date"], r["heure"] or 0, r["r"], r["c"]))
    return out


def iter_races(root=ROOT, until=None):
    for f in day_files(root, until):
        for r in normalize(load_day(f)):
            yield r
