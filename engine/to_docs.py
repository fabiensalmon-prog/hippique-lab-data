"""Convertit un run du moteur en documents pour la base du dashboard (ids stables => idempotent)."""
import json, sys, os

def pour_contre(pred, race):
    f = pred['features']; pour, contre = [], []
    if f['form'] >= 0.55: pour.append(f"Forme récente solide (indice {f['form']:.2f})")
    if f['consistency'] >= 0.5: pour.append(f"Régulier : {int(f['consistency']*100)} % de places dans les 3 sur ses dernières sorties")
    if f['entourage']: pour.append(f"Driver/jockey de premier plan ({pred['jockey']})")
    if pred.get('p_exp') and pred['p_exp'] >= 0.15: pour.append(f"Bien vu des pronostiqueurs (P experts {pred['p_exp']*100:.0f} %)")
    if f['incident_risk'] >= 0.2: contre.append(f"Risque de faute : {int(f['incident_risk']*100)} % de disqualifications/arrêts récents")
    if f['n_runs'] < 5: contre.append(f"Historique court ({f['n_runs']} courses lisibles)")
    if pred['p_mkt'] < pred['p_hl'] / 2: contre.append("Fort désaccord avec le marché : possible information manquante dans le modèle (classe, engagement)")
    contre.append("Modèle V0 non calibré : edge à confirmer par le journal")
    risk = ("Faute / disqualification au trot" if race['disc'] in ('TA','TM') and f['incident_risk'] >= 0.15
            else "Rapport final sous le minimum (value absorbée par le marché)")
    return pour or ["Signal combiné modèle + marché positif"], contre, risk

def main(run_path, out_dir, day_number, versions_path=None, status_close=None):
    run = json.load(open(run_path)); date = run['date']
    VERS = json.load(open(versions_path)) if versions_path and os.path.exists(versions_path) else {}
    os.makedirs(out_dir, exist_ok=True)
    writes = []
    def put(col, did, data):
        p = os.path.join(out_dir, f"{col}__{did}.json"); json.dump(data, open(p, 'w'), ensure_ascii=False)
        w = dict(op='set', collection=col, doc_id=did, file_path=os.path.abspath(p))
        if VERS.get(f"{col}/{did}"): w['if_version'] = VERS[f"{col}/{did}"]
        writes.append(w)
    races = run['races']
    an = [r for r in races if r['status'] == 'ANALYSEE']
    cov = dict(expected=len(races), analyzed=len(an),
               partial=sum(r['status'] == 'PARTIELLE' for r in races), not_analyzed=sum(r['status'] == 'NON_ANALYSEE' for r in races))
    bets = run['bets']
    exposure = round(sum(b['stake'] for b in bets), 2)
    sel = {(b['race_id'], b['num']) for b in bets}
    for r in races:
        preds = []
        for p in r['preds']:
            q = {k: p.get(k) for k in ('num','horse','jockey','p_fund','p_exp','p_hl','p_mkt','p_dec','odds','edge','edge_raw','ev','ev_low','confidence','raw_signal')}
            q['selected'] = (r['race_id'], p['num']) in sel
            q['form'] = p['features']['form']; q['incident_risk'] = p['features']['incident_risk']; q['n_runs'] = p['features']['n_runs']
            preds.append(q)
        if r['decision'] == 'BET' and not any(q['selected'] for q in preds):
            r['decision'] = 'PASS'
        put('races', f"{date}_{r['race_id']}", dict(date=date, race_id=r['race_id'], meeting=r['meeting'], r=r['r'], c=r['c'], time=r['time'],
            disc=r['disc'], dist=r['dist'], cond=r['cond'], n_runners=r['n_runners'], status=r['status'], decision=r['decision'],
            reason=r['reason'], odds_time=r['odds_time'], odds_src=r['odds_src'], overround=r['overround'], flags=r['flags'],
            captured_at=run['capture_partants'], computed_at=run['computed_at'], model=run['model'], preds=preds, result=None))
    for b in bets:
        race = next(r for r in races if r['race_id'] == b['race_id'])
        pred = next(p for p in race['preds'] if p['num'] == b['num'])
        pour, contre, risk = pour_contre(pred, race)
        ex = []
        # experts : rang dans chaque source
        put('bets', b['bet_id'], dict(b, date=date, pour=pour, contre=contre, risk=risk,
            experts=f"P experts {pred['p_exp']*100:.0f} %" if pred.get('p_exp') else "Pas de signal experts",
            created_at=run['computed_at'], settlement=None))
    n_bet = len(bets); n_pass = sum(1 for r in races if r['status'] == 'ANALYSEE' and r['decision'] == 'PASS')
    status = 'COMPLETE' if cov['analyzed'] == cov['expected'] else 'INCOMPLETE'
    put('days', date, dict(date=date, day=day_number, opening_capital=run['capital'], closing_capital=None, pnl=None,
        status_morning=status, status_close=status_close or 'PLANNED', coverage=cov, n_bet=n_bet, n_pass=n_pass, exposure=exposure,
        n_settled=0, morning_at=run['computed_at'], model=run['model'],
        morning_note=f"{cov['partial']} course(s) partielles (marché non capturé ou donnée incohérente), {cov['not_analyzed']} partie(s) avant le run",
        kelly_fraction=run['params']['kelly'], max_bet_pct=run['params']['max_bet_pct'], max_day_pct=run['params']['max_day_pct']))
    prev = json.load(open(os.path.join(os.path.dirname(versions_path), 'meta', 'state.json'))) if versions_path and os.path.exists(os.path.join(os.path.dirname(versions_path), 'meta', 'state.json')) else {}
    put('meta', 'state', dict(prev, day=day_number, current_date=date, current_capital=run['capital'], initial_capital=500.0, pilot_start='2026-11-03',
        model=run['model'], params=run['params'], updated_at=run['computed_at'], last_run='morning'))
    json.dump(writes, open(os.path.join(out_dir, '_writes.json'), 'w'), indent=0)
    print(len(writes), 'docs', cov, 'bets', n_bet, 'exposure', exposure)

if __name__ == '__main__':
    # usage : to_docs.py run.json out_dir day_number [db_export_dir/versions.json]
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None)
