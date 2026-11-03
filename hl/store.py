"""Lecture des journées PMU collectées -> courses normalisées, dans l'ordre chronologique."""
import gzip, json, glob, os, re
from datetime import date

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
DISC = {"ATTELE": "TROT", "MONTE": "TROT", "PLAT": "PLAT"}   # obstacle exclu du pilote


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
                out[str(x.get("combinaison"))] = d / 100.0
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
                odds_ref=_rap(p, "dernierRapportReference"), odds_final=_rap(p, "dernierRapportDirect"),
                # --- informations APRÈS course : jamais utilisées comme variables de cette course ---
                place=p.get("ordreArrivee"), incident=p.get("incident") or "", red_km=p.get("reductionKilometrique"),
                temps=p.get("tempsObtenu"), comment=com))
        live = [u for u in race["runners"] if not u["np"]]
        if live and all(u["odds_ref"] for u in live):
            race["mkt_src"] = "REFERENCE"
            for u in race["runners"]: u["odds_mkt"] = u["odds_ref"]
        else:
            race["mkt_src"] = "DIRECT"
            for u in race["runners"]: u["odds_mkt"] = u["odds_final"]
        out.append(race)
    out.sort(key=lambda r: (r["date"], r["heure"] or 0, r["r"], r["c"]))
    return out


def iter_races(root=ROOT, until=None):
    for f in day_files(root, until):
        for r in normalize(load_day(f)):
            yield r
