"""Hippique Lab — collecte de l'historique PMU (tourne dans GitHub Actions).

Modes :
  python3 collect/pmu_history.py day 2026-11-02          # une journée (test)
  python3 collect/pmu_history.py backfill 30             # remonte 30 jours de plus dans le passé
Sortie : data/<AAAA>/<AAAA-MM-JJ>.json.gz  (programme + partants + rapports définitifs, courses françaises)
         data/progress.json                (plus ancienne journée collectée, objectif)
Bibliothèque standard uniquement. Une requête toutes les ~0,7 s.
"""
import gzip, json, os, sys, time, urllib.request, urllib.error
from datetime import date, timedelta

BASE = "https://online.turfinfo.api.pmu.fr/rest/client/1/programme"
UA = {"User-Agent": "HippiqueLab-research/0.1 (usage personnel, collecte lente)", "Accept": "application/json"}
PAUSE = 0.7
ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
TARGET_OLDEST = date(2023, 1, 1)   # objectif : environ 3 saisons

def get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                body = r.read()
                return json.loads(body.decode("utf-8")) if body else None
        except urllib.error.HTTPError as e:
            if e.code in (204, 404): return None
            time.sleep(3 * (i + 1))
        except Exception:
            time.sleep(3 * (i + 1))
    return None

def collect_day(d):
    ds = d.strftime("%d%m%Y")
    prog = get(f"{BASE}/{ds}"); time.sleep(PAUSE)
    out = dict(date=d.isoformat(), collected_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), ok=bool(prog), races=[])
    if not prog: return out
    reunions = prog.get("programme", {}).get("reunions", [])
    for reu in reunions:
        pays = (reu.get("pays") or {}).get("code")
        if pays not in ("FRA", None): continue
        r = reu.get("numOfficiel")
        for c in reu.get("courses", []):
            k = c.get("numOrdre")
            parts = get(f"{BASE}/{ds}/R{r}/C{k}/participants"); time.sleep(PAUSE)
            rap = get(f"{BASE}/{ds}/R{r}/C{k}/rapports-definitifs"); time.sleep(PAUSE)
            out["races"].append(dict(r=r, c=k, hippodrome=(reu.get("hippodrome") or {}).get("libelleCourt"),
                                     course=c, participants=parts, rapports=rap))
    return out

def write_day(d, obj):
    folder = os.path.join(ROOT, str(d.year)); os.makedirs(folder, exist_ok=True)
    with gzip.open(os.path.join(folder, f"{d.isoformat()}.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))

def progress():
    p = os.path.join(ROOT, "progress.json")
    return (json.load(open(p)) if os.path.exists(p) else {}), p

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "day":
        d = date.fromisoformat(sys.argv[2]); obj = collect_day(d); write_day(d, obj)
        n = len(obj["races"]); np = sum(1 for x in obj["races"] if x["participants"]); nr = sum(1 for x in obj["races"] if x["rapports"])
        print(f"{d}: programme={'ok' if obj['ok'] else 'ABSENT'} courses_FR={n} partants={np} rapports={nr}")
        if n:
            x = next((x for x in obj["races"] if x["participants"]), None)
            if x:
                p0 = (x["participants"].get("participants") or [{}])[0]
                print("champs partant:", sorted(p0.keys()))
                print("champs course:", sorted(x["course"].keys()))
            y = next((x for x in obj["races"] if x["rapports"]), None)
            if y: print("rapports (types):", [t.get("typePari") for t in y["rapports"]][:15] if isinstance(y["rapports"], list) else type(y["rapports"]).__name__)
    elif mode == "backfill":
        ndays = int(sys.argv[2]); st, path = progress()
        start = date.fromisoformat(st["oldest_done"]) - timedelta(days=1) if st.get("oldest_done") else date.today() - timedelta(days=1)
        d, done = start, 0
        while done < ndays and d >= TARGET_OLDEST:
            obj = collect_day(d); write_day(d, obj); done += 1
            st.update(oldest_done=d.isoformat(), newest_done=st.get("newest_done") or d.isoformat(), target=TARGET_OLDEST.isoformat())
            json.dump(st, open(path, "w"), indent=1)
            print(f"{d}: {len(obj['races'])} courses", flush=True)
            d -= timedelta(days=1)
        print("termine" if d < TARGET_OLDEST else f"reste jusqu'au {TARGET_OLDEST}")
