import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from hl.model import fit_pair_dividend, pair_expected
D = pickle.load(open(sys.argv[2], "rb")); df, races = D["runners"], D["races"]
df = df[["race", "num", "odds_ref"]].copy(); df["num"] = df.num.astype(str)
odds = {(r, n): o for r, n, o in zip(df.race, df.num, df.odds_ref)}
nr = df.groupby("race").size().to_dict()
for typ in ("div_2s4", "div_cg"):
    rows = []
    for ri in nr:
        for combo, dv in (races[ri][typ] or {}).items():
            a, _, b = combo.partition("-"); oa, ob = odds.get((ri, a)), odds.get((ri, b))
            if oa and ob and oa > 1 and ob > 1: rows.append((races[ri]["date"], oa, ob, nr[ri], dv))
    S = pd.DataFrame(rows, columns=["date", "oa", "ob", "n", "div"])
    A = S[S.date < "2025-01-01"]; B = S[S.date >= "2025-01-01"].copy()
    c = fit_pair_dividend([tuple(x) for x in A[["oa", "ob", "n", "div"]].to_numpy()][-8000:])
    B["new"] = [pair_expected(c, a, b, n) for a, b, n in zip(B.oa, B.ob, B.n)]
    B["old"] = [pair_expected(c[:4], a, b, n) for a, b, n in zip(B.oa, B.ob, B.n)]
    B["bk"] = pd.cut(B.old, [0, 2, 3, 4, 6, 10, 20, 50, 1e9]).astype(str)
    print(typ, B.groupby("bk").agg(n=("div", "size"), reel=("div", "mean"), ancien=("old", "mean"), nouveau=("new", "mean")).round(2).to_string())
