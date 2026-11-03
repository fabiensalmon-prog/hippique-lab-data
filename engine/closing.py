"""Hippique Lab — clôture : règlement des tickets + métriques modèle vs marché.

Entrées :
  db_dir   : export de la base (ArtifactData list ... out_dir) contenant races/, bets/, days/, meta/, metrics/
  results  : JSON {race_id: {"arrivee":[n1,n2,...], "rapport_gagnant": 4.2 (pour 1 €), "non_partants":[..],
                              "final_odds": {num: cote finale}  (optionnel, pour CLV et rapport minimal), "source": "..."}}
             race_id = id de document races (ex. "2026-11-03_VINCENNES_R1C1"). Course absente => PENDING.
  date     : journée à clôturer
Sortie : out_dir/_writes.json (batch ArtifactData) + documents JSON.
Règles : aucun résultat estimé ; ticket sans résultat vérifiable => PENDING ; non-partant => CANCELLED (remboursé).
"""
import json, sys, os, glob, math

VERS = {}
def load(dir_, col):
    """Lit l'export ArtifactData (out_dir). Les versions viennent de <db_dir>/versions.json {"col/doc_id": version}."""
    if not VERS and os.path.exists(os.path.join(dir_, 'versions.json')):
        VERS.update(json.load(open(os.path.join(dir_, 'versions.json'))))
    out = {}
    for p in glob.glob(os.path.join(dir_, col, '*.json')):
        did = os.path.basename(p)[:-5]
        out[did] = (json.load(open(p)), VERS.get(f"{col}/{did}"))
    return out

def main(db_dir, results_path, date, out_dir, now_iso):
    res = json.load(open(results_path))
    # correspondance tolérante : même journée + même numéro de réunion et de course (R1C3), quel que soit le nom d'hippodrome
    res_rc = {f"{k[:10]}_{k.split('_')[-1]}": v for k, v in res.items()}
    def lookup(doc_id):
        return res.get(doc_id) or res_rc.get(f"{doc_id[:10]}_{doc_id.split('_')[-1]}")
    races = load(db_dir, 'races'); bets = load(db_dir, 'bets'); days = load(db_dir, 'days')
    meta = load(db_dir, 'meta').get('state'); metrics = load(db_dir, 'metrics')
    os.makedirs(out_dir, exist_ok=True); writes = []
    def put(col, did, data, ver, op='set'):
        p = os.path.join(out_dir, f"{col}__{did}.json"); json.dump(data, open(p, 'w'), ensure_ascii=False)
        w = dict(op=op, collection=col, doc_id=did, file_path=os.path.abspath(p))
        if ver: w['if_version'] = ver
        writes.append(w)
    # 1. résultats dans les courses du jour
    for rid, (r, ver) in races.items():
        x = lookup(rid)
        if r.get('date') != date or not x: continue
        put('races', rid, dict(r, result=dict(arrivee=x['arrivee'], rapport_gagnant=x.get('rapport_gagnant'),
            non_partants=x.get('non_partants', []), source=x.get('source'), settled_at=now_iso)), ver)
        races[rid] = (dict(r, result=x), ver)
    # 2. règlement des tickets
    day_pnl, n_settled, pend = 0.0, 0, 0
    for bid, (b, ver) in bets.items():
        if b.get('date') != date or (b.get('settlement') or {}).get('status') in ('WIN', 'LOSS', 'CANCELLED'): continue
        rid = f"{date}_{b['race_id']}"; x = lookup(rid)
        if not x:
            st = dict(status='PENDING', note='Résultat non vérifiable', settled_at=now_iso); pend += 1
        else:
            fo = (x.get('final_odds') or {}).get(str(b['num'])) or (x.get('final_odds') or {}).get(b['num'])
            if b['num'] in x.get('non_partants', []):
                st = dict(status='CANCELLED', note='Non-partant, mise remboursée', gross_return=b['stake'], net_pnl=0.0)
            elif b.get('market_code') == 'SIMPLE_PLACE':
                rp = x.get('rapport_place') or {}
                dv = rp.get(str(b['num']))
                if not rp: st = dict(status='PENDING', note='Rapports placé non trouvés'); pend += 1
                elif dv: st = dict(status='WIN', final_odds=dv, gross_return=round(b['stake']*dv, 2), net_pnl=round(b['stake']*dv - b['stake'], 2))
                else: st = dict(status='LOSS', final_odds=fo, gross_return=0.0, net_pnl=-b['stake'])
            elif x['arrivee'] and x['arrivee'][0] == b['num']:
                rap = x.get('rapport_gagnant')
                if rap is None: st = dict(status='PENDING', note='Rapport officiel non trouvé'); pend += 1
                elif rap < b['min_odds']: st = dict(status='CANCELLED', final_odds=rap, note=f"Rapport {rap} < minimal {b['min_odds']} : value disparue", gross_return=b['stake'], net_pnl=0.0)
                else: st = dict(status='WIN', final_odds=rap, gross_return=round(b['stake']*rap, 2), net_pnl=round(b['stake']*rap - b['stake'], 2))
            else:
                if fo is not None and fo < b['min_odds']:
                    st = dict(status='CANCELLED', final_odds=fo, note=f"Cote finale {fo} < minimale {b['min_odds']}", gross_return=b['stake'], net_pnl=0.0)
                else:
                    st = dict(status='LOSS', final_odds=fo, gross_return=0.0, net_pnl=-b['stake'])
            if fo: st['clv'] = round(b['odds']/fo - 1, 4)
            st.update(settled_at=now_iso, source=x.get('source'))
            if st['status'] in ('WIN', 'LOSS'): day_pnl += st['net_pnl']; n_settled += 1
        put('bets', bid, dict(b, settlement=st), ver)
    # 3. métriques cumulées sur toutes les courses réglées
    eps = 1e-6; n_r = n_run = 0; B = dict(hl=0, mkt=0); L = dict(hl=0, mkt=0, dec=0); RK = dict(hl=0, mkt=0); nm = nd = 0
    bins = [dict(lo=i/20, hi=(i+1)/20, n=0, sp=0.0, w=0) for i in range(20)]
    dis = dict(n=0, sh=0.0, sm=0.0, w=0); raw = dict(n=0, wins=0, ret=0.0)
    for rid, (r, _) in races.items():
        x = r.get('result')
        if not x or not x.get('arrivee') or not r.get('preds'): continue
        win = x['arrivee'][0]; nps = set(x.get('non_partants', []))
        ps = [p for p in r['preds'] if p['num'] not in nps]
        if not any(p['num'] == win for p in ps): continue
        s_hl = sum(p['p_hl'] for p in ps)
        n_r += 1; n_run += len(ps)
        B['hl'] += sum((p['p_hl']/s_hl - (p['num'] == win))**2 for p in ps)
        L['hl'] += -math.log(max(eps, next(p['p_hl'] for p in ps if p['num'] == win)/s_hl))
        RK['hl'] += 1 + sorted([-p['p_hl'] for p in ps]).index(-next(p['p_hl'] for p in ps if p['num'] == win))
        for p in ps:
            q = p['p_hl']/s_hl; b_ = bins[min(19, int(q*20))]; b_['n'] += 1; b_['sp'] += q; b_['w'] += (p['num'] == win)
        if all(p.get('p_mkt') is not None for p in ps):
            nm += 1; s_m = sum(p['p_mkt'] for p in ps); s_d = sum(p['p_dec'] for p in ps)
            B['mkt'] += sum((p['p_mkt']/s_m - (p['num'] == win))**2 for p in ps)
            L['mkt'] += -math.log(max(eps, next(p['p_mkt'] for p in ps if p['num'] == win)/s_m))
            L['dec'] += -math.log(max(eps, next(p['p_dec'] for p in ps if p['num'] == win)/s_d))
            RK['mkt'] += 1 + sorted([-p['p_mkt'] for p in ps]).index(-next(p['p_mkt'] for p in ps if p['num'] == win))
            for p in ps:
                if p['p_hl'] - p['p_mkt'] >= 0.10: dis['n'] += 1; dis['sh'] += p['p_hl']; dis['sm'] += p['p_mkt']; dis['w'] += (p['num'] == win)
                if p.get('raw_signal') and not p.get('selected'):
                    raw['n'] += 1
                    if p['num'] == win: raw['wins'] += 1; raw['ret'] += (x.get('rapport_gagnant') or p['odds'])
    clvs = [b['settlement']['clv'] for b, _ in bets.values() if (b.get('settlement') or {}).get('clv') is not None]
    for bid, (b, _) in bets.items():
        for w in writes:
            if w['collection'] == 'bets' and w['doc_id'] == bid:
                s = json.load(open(w['file_path']))['settlement']
                if s.get('clv') is not None: clvs.append(s['clv'])
    m = dict(updated_at=now_iso, n_races=n_r, n_runners=n_run, n_races_market=nm,
             brier_hl=B['hl']/n_r if n_r else None, brier_mkt=B['mkt']/nm if nm else None,
             logloss_hl=L['hl']/n_r if n_r else None, logloss_mkt=L['mkt']/nm if nm else None, logloss_dec=L['dec']/nm if nm else None,
             rank_hl=RK['hl']/n_r if n_r else None, rank_mkt=RK['mkt']/nm if nm else None,
             calibration=[dict(p=b['sp']/b['n'], freq=b['w']/b['n'], n=b['n']) for b in bins if b['n']],
             disagree=[dict(label='P HL ≥ P marché + 10 pts', n=dis['n'], p_hl=dis['sh']/dis['n'], p_mkt=dis['sm']/dis['n'], freq=dis['w']/dis['n'])] if dis['n'] else None,
             raw_signals=dict(n=raw['n'], wins=raw['wins'], roi=(raw['ret']-raw['n'])/raw['n'] if raw['n'] else None),
             clv=sum(clvs)/len(clvs) if clvs else None)
    put('metrics', 'cumul', m, metrics.get('cumul', (None, None))[1])
    # 4. journée + capital du lendemain
    d, dver = days[date]
    closing = round(d['opening_capital'] + day_pnl, 2)
    put('days', date, dict(d, pnl=round(day_pnl, 2), closing_capital=closing, n_settled=n_settled, n_pending=pend,
        status_close='DONE' if pend == 0 else 'PENDING', close_at=now_iso), dver)
    mm, mver = meta
    put('meta', 'state', dict(mm, current_capital=closing, updated_at=now_iso, last_run='closing',
        peak_capital=max(mm.get('peak_capital', 500.0), closing)), mver)
    json.dump(writes, open(os.path.join(out_dir, '_writes.json'), 'w'))
    print(f"P&L {day_pnl:+.2f} € · réglés {n_settled} · en attente {pend} · courses notées {n_r}")
    print(f"CAPITAL DE DÉPART DU PROCHAIN RUN = {closing:.2f} €")

if __name__ == '__main__':
    main(*sys.argv[1:6])
