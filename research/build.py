"""Table plate point-in-time : variables actuelles (hl.features) + variables candidates, toutes calculées
avec l'information antérieure à la course. Usage : python3 build.py <repo> <out.pkl>"""
import sys, os, math, gzip, json, pickle, time, unicodedata
from collections import defaultdict
REPO = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, REPO)
os.environ["HL_DATA"] = os.path.join(REPO, "data")
from hl.store import day_files, load_day, normalize
from hl.features import FeatureState, FEATURES, _days


def norm(s):
    s = unicodedata.normalize("NFD", (s or "").upper())
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn").strip()


class AE:
    """victoires / victoires attendues par le marché final, décroissance exponentielle."""
    __slots__ = ("n", "w", "e", "d", "hl")

    def __init__(self, hl):
        self.n = self.w = self.e = 0.0; self.d = None; self.hl = hl

    def add(self, d, won, pe):
        if self.d:
            f = 0.5 ** (_days(self.d, d) / self.hl); self.n *= f; self.w *= f; self.e *= f
        self.d = d; self.n += 1; self.w += won; self.e += pe

    def get(self, d, k=5.0):
        if not self.d:
            return 0.0, 0.0
        f = 0.5 ** (_days(self.d, d) / self.hl)
        return math.log((self.w * f + k) / (self.e * f + k)), math.log1p(self.n * f)


GOING = {"BON": "BON", "BON LEGER": "SEC", "LEGER": "SEC", "SEC": "SEC", "TRES SEC": "SEC", "BON SOUPLE": "BS",
         "SOUPLE": "SOUPLE", "TRES SOUPLE": "LOURD", "COLLANT": "LOURD", "LOURD": "LOURD", "TRES LOURD": "LOURD",
         "PSF STANDARD": "PSF", "PSF RAPIDE": "PSF", "PSF LENTE": "PSF", "STANDARD": "PSF"}


def going(race):
    t = norm(race.get("terrain"))
    if not t:
        return None
    if t.startswith("PSF") or "FIBRE" in t:
        return "PSF"
    return GOING.get(t, t[:10])


def wet(meteo_code):
    t = norm(meteo_code)
    return any(w in t for w in ("PLUIE", "AVERSE", "ORAGE", "BRUINE", "NEIGE"))


def dist_b(race):
    d = race["distance"] or 0
    if race["group"] == "TROT":
        return "T<2200" if d < 2200 else "T<2800" if d < 2800 else "T+"
    return "P<1500" if d < 1500 else "P<2000" if d < 2000 else "P<2500" if d < 2500 else "P+"


NEW = ["sire_ae", "sire_n", "sire_going_ae", "sire_dist_ae", "sire_wet_ae", "dam_sire_ae",
       "trn_hot_ae", "trn_hot_n", "drv_hot_ae", "drv_hot_n", "trn_track_ae", "drv_track_ae", "trn_drv_ae", "trn_drv_n",
       "trn_disc_ae", "drv_disc_ae", "trn_return_ae", "owner_ae", "draw_ae", "horse_wet_apt", "horse_going_apt",
       "supplement", "jument_pleine", "n_partants", "wt_change", "fav_ref"]


class NewState:
    def __init__(self):
        self.t = defaultdict(lambda: AE(365.0))
        self.hot = defaultdict(lambda: AE(21.0))
        self.hh = {}           # cheval -> list (going, wet, beaten)
        self.lastw = {}        # cheval -> dernier poids

    def feats(self, race, u, ex, base_x, n):
        d, g = race["date"], race["group"]
        sire, dam = ex.get("pere"), ex.get("pere_mere")
        gb, db, w = going(race), dist_b(race), race["_wet"]
        f = {}
        a, k = self.t[("S", g, sire)].get(d) if sire else (0.0, 0.0); f["sire_ae"], f["sire_n"] = a, k
        f["sire_going_ae"] = (self.t[("SG", g, sire, gb)].get(d)[0] - a) if sire and gb else None
        f["sire_dist_ae"] = (self.t[("SD", g, sire, db)].get(d)[0] - a) if sire else None
        f["sire_wet_ae"] = (self.t[("SW", g, sire, w)].get(d)[0] - a) if sire else None
        f["dam_sire_ae"] = self.t[("DS", g, dam)].get(d)[0] if dam else None
        tr, dr = u["trainer"], u["driver"]
        f["trn_hot_ae"], f["trn_hot_n"] = self.hot[("T", tr)].get(d, 3.0) if tr else (None, None)
        f["drv_hot_ae"], f["drv_hot_n"] = self.hot[("D", dr)].get(d, 3.0) if dr else (None, None)
        f["trn_track_ae"] = self.t[("TT", tr, race["hippodrome"])].get(d)[0] if tr else None
        f["drv_track_ae"] = self.t[("DT", dr, race["hippodrome"])].get(d)[0] if dr else None
        f["trn_drv_ae"], f["trn_drv_n"] = self.t[("TD", tr, dr)].get(d) if tr and dr else (None, None)
        f["trn_disc_ae"] = self.t[("TC", tr, race["discipline"])].get(d)[0] if tr else None
        f["drv_disc_ae"] = self.t[("DC", dr, race["discipline"])].get(d)[0] if dr else None
        lr = base_x.get("long_rest") or 0.0
        f["trn_return_ae"] = self.t[("TR", tr)].get(d)[0] if (tr and lr) else 0.0
        f["owner_ae"] = self.t[("O", ex.get("owner"))].get(d)[0] if ex.get("owner") else None
        f["draw_ae"] = self.t[self._drawkey(race, u)].get(d)[0] if self._drawkey(race, u) else None
        hh = self.hh.get(u["horse"], [])
        allb = sum(b for _, _, b in hh) / len(hh) if hh else None
        sel = [b for _, ww, b in hh if ww == w]
        f["horse_wet_apt"] = ((allb - sum(sel) / len(sel)) * min(1, len(sel) / 3)) if (sel and w) else (0.0 if hh else None)
        sel = [b for gg, _, b in hh if gg == gb] if gb else []
        f["horse_going_apt"] = ((allb - sum(sel) / len(sel)) * min(1, len(sel) / 3)) if sel else None
        f["supplement"] = float(bool(ex.get("supplement")))
        f["jument_pleine"] = float(bool(ex.get("jp")))
        f["n_partants"] = float(n)
        lw = self.lastw.get(u["horse"])
        f["wt_change"] = (u["poids"] - lw) / 10.0 if (g != "TROT" and u["poids"] and lw) else None
        f["fav_ref"] = None
        return f

    @staticmethod
    def _drawkey(race, u):
        if race["group"] == "TROT" and not race["autostart"]:
            return None
        c = u["corde"] if race["group"] != "TROT" else u["num"]
        if not c:
            return None
        try:
            c = int(c)
        except (TypeError, ValueError):
            return None
        b = 1 if c <= 2 else 2 if c <= 4 else 3 if c <= 7 else 4 if c <= 10 else 5
        return ("DR", race["hippodrome"], race["discipline"], int((race["distance"] or 0) / 200), b)

    def update(self, race, exmap):
        d, g = race["date"], race["group"]
        runners = [u for u in race["runners"] if not u["np"]]
        n = len(runners)
        inv = {u["num"]: 1.0 / u["odds_final"] for u in runners if u["odds_final"]}
        s = sum(inv.values())
        if n < 2 or not s:
            return
        win = race["arrivee"][0] if race["arrivee"] else None
        gb, db, w = going(race), dist_b(race), race["_wet"]
        for u in runners:
            ex = exmap.get(u["num"], {})
            won = float(u["num"] == win); pe = inv.get(u["num"], 0.0) / s
            sire, dam, tr, dr = ex.get("pere"), ex.get("pere_mere"), u["trainer"], u["driver"]
            keys = []
            if sire:
                keys += [("S", g, sire), ("SD", g, sire, db), ("SW", g, sire, w)]
                if gb: keys.append(("SG", g, sire, gb))
            if dam: keys.append(("DS", g, dam))
            if tr:
                keys += [("TT", tr, race["hippodrome"]), ("TC", tr, race["discipline"])]
                self.hot[("T", tr)].add(d, won, pe)
                h = self._last.get(u["horse"])
                if h and _days(h, d) > 90: keys.append(("TR", tr))
            if dr:
                keys += [("DT", dr, race["hippodrome"]), ("DC", dr, race["discipline"])]
                self.hot[("D", dr)].add(d, won, pe)
            if tr and dr: keys.append(("TD", tr, dr))
            if ex.get("owner"): keys.append(("O", ex["owner"]))
            dk = self._drawkey(race, u)
            if dk: keys.append(dk)
            for k in keys:
                self.t[k].add(d, won, pe)
            pl = u["place"] if isinstance(u["place"], int) and u["place"] > 0 else None
            b = ((pl - 1) / (n - 1)) if pl else 1.0
            self.hh.setdefault(u["horse"], []).append((gb, w, b)); self.hh[u["horse"]] = self.hh[u["horse"]][-15:]
            if u["poids"]: self.lastw[u["horse"]] = u["poids"]
            self._last[u["horse"]] = d

    _last = {}


def extras(dayobj):
    out = {}
    for x in dayobj.get("races", []):
        parts = (x.get("participants") or {}).get("participants") or []
        m = x.get("meteo") or {}
        rid = f"{dayobj['date']}_R{x['r']}C{x['c']}"
        out[rid] = (dict(wet=wet(m.get("nebulositeLibelleCourt")), temp=m.get("temperature"), vent=m.get("forceVent"),
                         ciel=m.get("nebulositeLibelleCourt")),
                    {p.get("numPmu"): dict(pere=norm(p.get("nomPere")) or None, pere_mere=norm(p.get("nomPereMere")) or None,
                                           owner=norm(p.get("proprietaire")) or None, supplement=p.get("supplement"),
                                           jp=p.get("jumentPleine"), sexe=p.get("sexe")) for p in parts})
    return out


def main():
    st, ns = FeatureState(), NewState()
    from array import array
    NUM = set(["odds_ref", "odds_final"] + ["b_" + k for k in FEATURES] + ["n_" + k for k in NEW])
    INT = {"race", "place", "win"}
    class Cols(dict):
        def __missing__(self, k):
            v = array("f") if k in NUM else array("i") if k in INT else []
            self[k] = v; return v
    cols = Cols()
    nan = float("nan")
    races_meta = []
    t0 = time.time()
    files = day_files(os.path.join(REPO, "data"))
    for i, f in enumerate(files):
        day = load_day(f)
        exr = extras(day)
        for race in normalize(day):
            rm, exmap = exr.get(race["id"], ({}, {}))
            race["_wet"] = bool(rm.get("wet"))
            rows = st.features(race)
            n = len(rows)
            if n >= 4:
                live = {u["num"]: u for u in race["runners"] if not u["np"]}
                win = race["arrivee"][0] if race["arrivee"] else None
                ri = len(races_meta)
                races_meta.append(dict(id=race["id"], date=race["date"], group=race["group"], disc=race["discipline"],
                                       hippo=race["hippodrome"], going=going(race), wet=race["_wet"], temp=rm.get("temp"),
                                       ciel=rm.get("ciel"), n=n, dist=race["distance"], prix=race["prix"],
                                       div_sg=race["div_sg"], div_sp=race["div_sp"], div_cg=race["div_cg"],
                                       div_2s4=race["div_2s4"], arrivee=race["arrivee"][:4]))
                for r in rows:
                    u = live[r["num"]]; ex = exmap.get(r["num"], {})
                    nf = ns.feats(race, u, ex, r["x"], n)
                    cols["race"].append(ri); cols["num"].append(r["num"])
                    cols["odds_ref"].append(r["odds_ref"] or nan); cols["odds_final"].append(r["odds_final"] or nan)
                    cols["place"].append(r["place"] if isinstance(r["place"], int) else 0)
                    cols["win"].append(int(r["num"] == win))
                    cols["trainer"].append(u["trainer"]); cols["driver"].append(u["driver"])
                    cols["sire"].append(ex.get("pere")); cols["sexe"].append(ex.get("sexe")); cols["age"].append(u["age"])
                    cols["deferre"].append(u["deferre"]); cols["oeil"].append(u["oeilleres"]); cols["corde_raw"].append(u["corde"])
                    for k in FEATURES:
                        v = r["x"].get(k); cols["b_" + k].append(nan if v is None else float(v))
                    for k in NEW:
                        v = nf.get(k); cols["n_" + k].append(nan if v is None else float(v))
            st.update(race); ns.update(race, exmap)
        if i % 100 == 0:
            print(f"{i}/{len(files)} {os.path.basename(f)} runners={len(cols['race'])} {time.time()-t0:.0f}s", flush=True)
    import pandas as pd
    import numpy as np
    data = {}
    for k in list(cols):
        v = cols.pop(k)
        data[k] = np.frombuffer(v, dtype=np.float32 if k in NUM else np.int32).copy() if isinstance(v, array) else pd.Categorical(v)
        del v
    df = pd.DataFrame(data)
    pickle.dump(dict(runners=df, races=races_meta, base=FEATURES, new=NEW), open(OUT, "wb"), protocol=4)
    print("done", len(df), len(races_meta), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
