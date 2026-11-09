"""Sonde : quelles adresses PMU donnent les cotes en direct et les rapports probables avant le départ ?"""
import json, time, urllib.request, urllib.error
from datetime import datetime, timezone, timedelta
UA = {"User-Agent": "HippiqueLab-research/0.1", "Accept": "application/json"}
def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
            b = r.read(); return r.status, (json.loads(b) if b else None)
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        return str(e)[:60], None
now = datetime.now(timezone.utc); ds = (now + timedelta(hours=1)).strftime("%d%m%Y")
st, prog = get(f"https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{ds}")
nxt = []
for reu in prog["programme"]["reunions"]:
    for c in reu["courses"]:
        hd = c.get("heureDepart", 0) / 1000
        if hd > now.timestamp():
            nxt.append((hd, reu["numOfficiel"], c["numOrdre"], (reu.get("pays") or {}).get("code"), c.get("statut")))
nxt.sort(); print("prochaines:", [(datetime.fromtimestamp(h, timezone.utc).strftime("%H:%M"), r, k, p, s) for h, r, k, p, s in nxt[:5]])
h, r, k, p, s = next(x for x in nxt if x[3] == "FRA")
for base in ("https://online.turfinfo.api.pmu.fr/rest/client/1/programme", "https://online.turfinfo.api.pmu.fr/rest/client/7/programme",
             "https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"):
    for suf in ("participants", "combinaisons", "combinaisons?specialisation=INTERNET", "rapports-probables", "rapports/COUPLE_GAGNANT",
                "rapports/SIMPLE_GAGNANT", "cotes", "participants?specialisation=INTERNET"):
        url = f"{base}/{ds}/R{r}/C{k}/{suf}"; st, js = get(url)
        snip = ""
        if js is not None:
            if isinstance(js, dict):
                snip = str(list(js.keys()))[:200]
                if "participants" in js:
                    p0 = js["participants"][0]; snip += " | " + json.dumps({kk: p0.get(kk) for kk in ("numPmu", "dernierRapportDirect", "dernierRapportReference")}, ensure_ascii=False)[:400]
                else:
                    snip += " | " + json.dumps(js, ensure_ascii=False)[:500]
            else:
                snip = json.dumps(js, ensure_ascii=False)[:500]
        print(st, url.split("/programme/")[0][-9:], suf, snip)
        time.sleep(0.4)
