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
    races = load(db_dir, 'races'); bets = load(db_dir, 'bets'); labbets = load(db_dir, 'labbets'); days = load(db_dir, 'days')
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
            non_partants=x.get('non_partants', []), rapport_place=x.get('rapport_place'), rapport_cg=x.get('rapport_cg'), rapport_2s4=x.get('rapport_2s4'), source=x.get('source'), settled_at=now_iso)), ver)
        races[rid] = (dict(r, result=x), ver)
    # 2. règlement des tickets
    day_pnl, n_settled, pend = 0.0, 0, 0
    for col, bets_c in (('bets', bets), ('labbets', labbets)):
     for bid, (b, ver) in bets_c.items():
         if b.get('date') != date or (b.get('settlement') or {}).get('status') in ('WIN', 'LOSS', 'CANCELLED'): continue
         rid = f"{date}_{b['race_id']}"; x = lookup(rid)
         if not x:
             st = dict(status='PENDING', note='Course pas encore courue ou résultat officiel pas encore publié', settled_at=now_iso); pend += 1
         else:
             fo = (x.get('final_odds') or {}).get(str(b['num'])) or (x.get('final_odds') or {}).get(b['num'])
             if b['num'] in x.get('non_partants', []):
                 st = dict(status='CANCELLED', note='Non-partant, mise remboursée', gross_return=b['stake'], net_pnl=0.0)
             elif b.get('market_code') in ('COUPLE_GAGNANT', 'DEUX_SUR_QUATRE'):
                 mp = x.get('rapport_cg' if b['market_code'] == 'COUPLE_GAGNANT' else 'rapport_2s4') or {}
                 parts = [int(t) for t in str(b['num']).split('-')]
                 dv = mp.get(str(b['num']))
                 if any(q in x.get('non_partants', []) for q in parts):
                     st = dict(status='CANCELLED', note='Un cheval du couple non-partant, mise remboursée', gross_return=b['stake'], net_pnl=0.0)
                 elif not mp: st = dict(status='PENDING', note='Rapports du pari non trouvés'); pend += 1
                 elif dv: st = dict(status='WIN', final_odds=dv, gross_return=round(b['stake']*dv, 2), net_pnl=round(b['stake']*dv - b['stake'], 2))
                 else: st = dict(status='LOSS', gross_return=0.0, net_pnl=-b['stake'])
             elif b.get('market_code') == 'SIMPLE_PLACE':
                 rp = x.get('rapport_place') or {}
                 dv = rp.get(str(b['num']))
                 if not rp: st = dict(status='PENDING', note='Rapports placé non trouvés'); pend += 1
                 elif dv: st = dict(status='WIN', final_odds=dv, gross_return=round(b['stake']*dv, 2), net_pnl=round(b['stake']*dv - b['stake'], 2))
                 else: st = dict(status='LOSS', final_odds=fo, gross_return=0.0, net_pnl=-b['stake'])
             elif x['arrivee'] and x['arrivee'][0] == b['num']:
                 rap = x.get('rapport_gagnant')
                 if rap is None: st = dict(status='PENDING', note='Rapport officiel non trouvé'); pend += 1
                 else: st = dict(status='WIN', final_odds=rap, gross_return=round(b['stake']*rap, 2), net_pnl=round(b['stake']*rap - b['stake'], 2))
             else:
                 st = dict(status='LOSS', final_odds=fo, gross_return=0.0, net_pnl=-b['stake'])
             if fo: st['clv'] = round(b['odds']/fo - 1, 4)
             if fo and b.get('min_odds') and fo < b['min_odds'] and st['status'] in ('WIN', 'LOSS'):
                 st['note'] = f"Cote finale {fo} sous la cote minimale {b['min_odds']} : ticket maintenu (un pari validé ne s'annule pas)"
             st.update(settled_at=now_iso, source=x.get('source'))
             if st['status'] in ('WIN', 'LOSS'): day_pnl += st['net_pnl']; n_settled += 1
         put(col, bid, dict(b, settlement=st), ver)
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
    # analyse « gagnant ou placé ? » : où finissent nos choix, et ce qu'aurait rapporté chaque façon de les jouer
    def acc(): return dict(n=0, p1=0, p2=0, p3=0, p4=0, ret_win=0.0, n_place=0, ret_place=0.0)
    alt = dict(top1=acc(), tickets=acc())
    def add(A, num, x):
        arr = x.get('arrivee') or []
        pos = arr.index(num) + 1 if num in arr else None
        A['n'] += 1
        A['p1' if pos == 1 else 'p2' if pos == 2 else 'p3' if pos == 3 else 'p4'] += 1
        A['ret_win'] += (x.get('rapport_gagnant') or 0.0) if pos == 1 else 0.0
        rp = x.get('rapport_place')
        if rp:
            A['n_place'] += 1; A['ret_place'] += rp.get(str(num), 0.0)
    race_by_id = {}
    for rid, (r, _) in races.items():
        x = r.get('result')
        if not x or not x.get('arrivee') or not r.get('preds'): continue
        race_by_id[rid] = (r, x)
        ps = [p for p in r['preds'] if p['num'] not in set(x.get('non_partants', []))]
        if ps:
            top = max(ps, key=lambda p: p.get('p_dec') if p.get('p_dec') is not None else p['p_hl'])
            add(alt['top1'], top['num'], x)
    for bid, (b, _) in bets.items():
        if b.get('market_code', 'SIMPLE_GAGNANT') != 'SIMPLE_GAGNANT': continue
        rx = race_by_id.get(f"{b['date']}_{b['race_id']}")
        if rx: add(alt['tickets'], b['num'], rx[1])
    for A in alt.values():
        A['roi_win'] = (A['ret_win'] - A['n']) / A['n'] if A['n'] else None
        A['roi_place'] = (A['ret_place'] - A['n_place']) / A['n_place'] if A['n_place'] else None
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
             clv=sum(clvs)/len(clvs) if clvs else None, alt=alt)
    put('metrics', 'cumul', m, metrics.get('cumul', (None, None))[1])
    # 4. journée + capital du lendemain
    d, dver = days[date]
    # P&L du jour = tous les tickets réglés de la journée (déjà réglés lors d'un run précédent + réglés maintenant) : idempotent
    final_st = {}
    for bid, (b, _) in bets.items():
        if b.get('date') == date: final_st[bid] = b.get('settlement') or {}
    for w in writes:
        if w['collection'] == 'bets':
            final_st[w['doc_id']] = json.load(open(w['file_path'])).get('settlement') or {}
    day_pnl = sum(s.get('net_pnl') or 0 for s in final_st.values() if s.get('status') in ('WIN', 'LOSS'))
    n_settled = sum(1 for s in final_st.values() if s.get('status') in ('WIN', 'LOSS', 'CANCELLED'))
    pend = sum(1 for s in final_st.values() if s.get('status') not in ('WIN', 'LOSS', 'CANCELLED'))
    closing = round(d['opening_capital'] + day_pnl, 2)
    lab_st = {bid: (b.get('settlement') or {}) for bid, (b, _) in labbets.items() if b.get('date') == date}
    for w in writes:
        if w['collection'] == 'labbets':
            lab_st[w['doc_id']] = json.load(open(w['file_path'])).get('settlement') or {}
    lab_pnl = sum(x.get('net_pnl') or 0 for x in lab_st.values() if x.get('status') in ('WIN', 'LOSS'))
    lab_pend = sum(1 for x in lab_st.values() if x.get('status') not in ('WIN', 'LOSS', 'CANCELLED'))
    lab_open = d.get('lab_opening_capital') if d.get('lab_opening_capital') is not None else 500.0
    lab_close = round(lab_open + lab_pnl, 2)
    put('days', date, dict(d, pnl=round(day_pnl, 2), closing_capital=closing, n_settled=n_settled, n_pending=pend,
        lab_pnl=round(lab_pnl, 2), lab_closing_capital=lab_close, lab_n_pending=lab_pend,
        status_close='DONE' if pend == 0 else 'PENDING', close_at=now_iso), dver)
    mm, mver = meta
    put('meta', 'state', dict(mm, current_capital=closing if pend == 0 else mm.get('current_capital', d['opening_capital']),
        lab_capital=lab_close if lab_pend == 0 else mm.get('lab_capital', lab_open), updated_at=now_iso, last_run='closing' if pend == 0 else 'resultats',
        peak_capital=max(mm.get('peak_capital', 500.0), closing)), mver)
    json.dump(writes, open(os.path.join(out_dir, '_writes.json'), 'w'))
    print(f"P&L {day_pnl:+.2f} € · réglés {n_settled} · en attente {pend} · courses notées {n_r} · labo P&L {lab_pnl:+.2f} € ({len(lab_st)} tickets, {lab_pend} en attente)")
    print(f"CAPITAL DE DÉPART DU PROCHAIN RUN = {closing:.2f} €" + ("" if pend == 0 else f" (provisoire : {pend} ticket(s) encore à régler)"))

if __name__ == '__main__':
    main(*sys.argv[1:6])
