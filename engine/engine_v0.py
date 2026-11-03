"""Hippique Lab — moteur V0 (heuristique transparente, NON calibrée).

Chaîne : musique -> sous-ratings -> force -> P_fondamentale (softmax sur le peloton)
         -> + signal experts (pooling log-linéaire) -> P_HL
         -> seulement ensuite : marché (rapports probables PMU normalisés) -> edge / EV / incertitude -> BET/PASS
Le marché n'entre JAMAIS dans P_fondamentale ni dans P_HL.
"""
import re, math, json, sys, importlib.util
from datetime import datetime, timezone, timedelta

PARIS = timezone(timedelta(hours=1))
MODEL_VERSION = "HL-V0.1-heuristique"

# --- paramètres du moteur (versionnés) ---
P = dict(
    decay=0.78, season_decay=0.85, max_runs=8,
    beta=2.6,              # température softmax (force -> proba)
    w_experts=0.35,        # poids du signal experts dans le pooling (V0, experts non notés)
    u_base=0.30,           # incertitude relative de base (modèle non calibré)
    u_thin=0.15,           # incertitude additionnelle si < 4 courses connues
    alpha_model=0.25,      # poids du modèle dans la combinaison décisionnelle (Benter) — V0 : prior faible, à apprendre
    beta_market=1.0,
    u_dec=0.10,            # incertitude relative sur P_décision
    min_edge=0.015, min_ev_low=0.03, min_p=0.05, max_odds=25.0,
    odds_haircut=0.90,     # pari mutuel : la cote finale dérive, on décide sur 90 % du rapport probable
    kelly=0.25, max_bet_pct=0.02, max_day_pct=0.06,
)
PTS = {'1':1.0,'2':0.72,'3':0.56,'4':0.45,'5':0.36,'6':0.29,'7':0.23,'8':0.19,'9':0.15,'0':0.08,'D':0.0,'A':0.0,'T':0.0,'R':0.0,'G':0.0}
ELITE = {"ABRIVARD M.","RAFFIN E.","BAZIRE N.","THOMAIN D.","GELORMINI G.","OUVRIE F.","ROCHARD B.","LEBOURGEOIS Y.",
         "PLOQUIN PAU. PH.","BARRIER A.","ABRIVARD A.","LAMY R.","COLLETTE A.","MARTENS C.","NIVARD F.","VERVA PIE. Y.",
         "BARZALONA M.","GUYON M.","BOUDOT P. C.","PASQUIER S.","DEMURO C.","LECOEUVRE C.","HAMELIN A.","MADAMET A.",
         "BRIAND THE.","BEKAERT D.","LAMAZIERE L.","CALLIER P.","CHAUVE-LAFFAY BEN."}

def parse_musique(s, disc):
    """-> liste de (pts, weight_discipline, is_incident) du plus récent au plus ancien, avec saut de saison."""
    runs, season = [], 1.0
    for m in re.finditer(r'\((\d\d)\)|([0-9DATRG])([apmshc])', s or ''):
        if m.group(1):
            season *= P['season_decay']; continue
        place, kind = m.group(2), m.group(3)
        if disc == 'TA':   wd = 1.0 if kind == 'a' else 0.6
        elif disc == 'TM': wd = 1.0 if kind == 'm' else 0.6
        else:              wd = 1.0 if kind == 'p' else 0.5
        runs.append((PTS.get(place, 0.1), wd * season, place in 'DATRG'))
    return runs[:P['max_runs']]

def horse_features(runner, disc, field_val_mean):
    num, name, jock, mus, val = runner[:5]
    runs = parse_musique(mus, disc)
    n = len(runs)
    if n == 0:
        form, cons, inc = 0.25, 0.0, 0.15
    else:
        ws = [P['decay']**i * w for i, (_, w, _) in enumerate(runs)]
        form = sum(w*p for w, (p, _, _) in zip(ws, runs)) / sum(ws)
        cons = sum(1 for (p, _, _) in runs if p >= 0.56) / n
        inc = sum(1 for (_, _, d) in runs[:6] if d) / min(n, 6)
    ent = 0.25 if jock in ELITE else 0.0
    vt = 0.0
    if disc == 'PLAT' and val is not None and field_val_mean is not None:
        vt = 0.06 * (val - field_val_mean)      # course à conditions : la valeur compte
    # shrink vers la moyenne si historique court
    shrink = min(1.0, n / 4.0)
    strength = shrink * (3.0*form + 0.8*cons - (1.0*inc if disc in ('TA','TM') else 0)) + (1-shrink)*1.2 + ent + vt
    return dict(n_runs=n, form=round(form,3), consistency=round(cons,3), incident_risk=round(inc,3),
                entourage=ent, valeur_term=round(vt,3), strength=round(strength,4))

def softmax(xs, beta):
    m = max(xs); e = [math.exp(beta*(x-m)) for x in xs]; s = sum(e); return [v/s for v in e]

def expert_probs(nums, pronos):
    if not pronos: return None
    score = {n: 0.0 for n in nums}
    for src, lst in pronos.items():
        for k, n in enumerate(lst):
            if n in score: score[n] += 0.72**k
    k = len(pronos); xs = [score[n]/k for n in nums]
    return softmax(xs, 2.2)

def analyse_race(race, now_iso):
    disc = 'PLAT' if race['disc'].startswith('PLAT') else race['disc']
    runners = [r for r in race['runners'] if r[4] != "NP?"]
    if not runners:
        return dict(race_id=f"{race['meeting'].replace(' ','')}_R{race['r']}C{race['c']}", meeting=race['meeting'], r=race['r'], c=race['c'],
                    time=race['time'], disc=race['disc'], dist=race['dist'], cond=race['cond'], n_runners=0,
                    status='NON_ANALYSEE', decision=None, reason='Départ avant le run (hors délai)', odds_time=None, odds_src=None,
                    overround=None, flags=race['flags'], preds=[])
    vals = [r[4] for r in runners if isinstance(r[4], (int, float))]
    vmean = sum(vals)/len(vals) if (vals and race['disc'] == 'PLAT') else None
    feats = [horse_features(r, disc, vmean) for r in runners]
    nums = [r[0] for r in runners]
    p_fund = softmax([f['strength'] for f in feats], P['beta'])
    p_exp = expert_probs(nums, race['pronos'])
    if p_exp:
        w = P['w_experts']
        logs = [(1-w)*math.log(a) + w*math.log(b) for a, b in zip(p_fund, p_exp)]
        p_hl = softmax(logs, 1.0)
    else:
        p_hl = p_fund
    odds = race['odds']
    market_ok = bool(odds) and all(n in odds for n in nums)
    p_mkt = None
    if market_ok:
        inv = {n: 1/odds[n] for n in nums}; s = sum(inv.values())
        p_mkt = [inv[n]/s for n in nums]
        overround = s
    p_dec = None
    if market_ok:
        l = [P['alpha_model']*math.log(a) + P['beta_market']*math.log(b) for a, b in zip(p_hl, p_mkt)]
        p_dec = softmax(l, 1.0)
    preds = []
    for i, r in enumerate(runners):
        f = feats[i]
        u = P['u_base'] + (P['u_thin'] if f['n_runs'] < 4 else 0)
        conf = 'FAIBLE' if f['n_runs'] < 4 else ('MOYENNE' if u <= 0.30 else 'FAIBLE')
        d = dict(num=r[0], horse=r[1], jockey=r[2], musique=r[3], features=f,
                 p_fund=round(p_fund[i],4), p_exp=round(p_exp[i],4) if p_exp else None, p_hl=round(p_hl[i],4),
                 uncertainty=u, p_low=round(p_hl[i]*(1-u),4), confidence=conf)
        if market_ok:
            o = odds[r[0]]
            pdl = p_dec[i]*(1-P['u_dec'])
            d.update(odds=o, p_mkt=round(p_mkt[i],4), p_dec=round(p_dec[i],4), p_dec_low=round(pdl,4),
                     edge_raw=round(p_hl[i]-p_mkt[i],4), edge=round(p_dec[i]-p_mkt[i],4),
                     ev_raw=round(p_hl[i]*o-1,4), ev=round(p_dec[i]*o-1,4), ev_low=round(pdl*o*P['odds_haircut']-1,4),
                     raw_signal=(p_hl[i]-p_mkt[i] >= 0.05 and p_hl[i]*(1-u)*o*P['odds_haircut']-1 >= 0.10))
        preds.append(d)
    # statut
    started = race['time'] <= now_iso[11:16]
    if started or not runners:
        status, decision, reason = 'NON_ANALYSEE', None, 'Départ avant le run (hors délai)'
    elif any('incohérente' in fl for fl in race['flags']):
        status, decision, reason = 'PARTIELLE', None, race['flags'][0]
    elif not market_ok:
        status, decision, reason = 'PARTIELLE', None, 'Prédiction journalisée, marché non capturé' + (' (cotes incomplètes)' if odds else '')
    else:
        status = 'ANALYSEE'
        cands = [d for d in preds if d['edge'] >= P['min_edge'] and d['ev_low'] >= P['min_ev_low']
                 and d['p_dec'] >= P['min_p'] and d['odds'] <= P['max_odds'] and d['confidence'] != 'FAIBLE']
        if cands:
            best = max(cands, key=lambda d: d['ev_low'])
            decision, reason = 'BET', f"N°{best['num']} : edge {best['edge']*100:+.1f} pts, EV basse {best['ev_low']*100:+.0f} %"
            best['selected'] = True
        else:
            top = max(preds, key=lambda d: d['ev_low'])
            decision = 'PASS'
            reason = (f"Meilleure EV N°{top['num']} {top['ev']*100:+.0f} % mais non robuste "
                      f"(EV basse {top['ev_low']*100:+.0f} %, edge {top['edge']*100:+.1f} pts, conf. {top['confidence']})")
            nraw = sum(1 for d in preds if d['raw_signal'])
            if nraw: reason += f" · {nraw} signal(s) brut(s) HL écarté(s) par la combinaison marché" 
    return dict(race_id=f"{race['meeting'].replace(' ','')}_R{race['r']}C{race['c']}", meeting=race['meeting'], r=race['r'], c=race['c'],
                time=race['time'], disc=race['disc'], dist=race['dist'], cond=race['cond'], n_runners=len(runners),
                status=status, decision=decision, reason=reason, odds_time=race['odds_time'], odds_src=race['odds_src'],
                overround=round(overround,3) if market_ok else None, flags=race['flags'], preds=preds)

def size_bets(results, capital):
    bets = []
    for res in results:
        for d in res['preds']:
            if d.get('selected'):
                pl, o = d['p_dec_low'], d['odds']*P['odds_haircut']
                b = o - 1; f = max(0.0, (b*pl - (1-pl))/b)
                stake = min(P['kelly']*f*capital, P['max_bet_pct']*capital)
                bets.append(dict(race=res, pred=d, raw=stake))
    tot = sum(b['raw'] for b in bets); cap = P['max_day_pct']*capital
    k = min(1.0, cap/tot) if tot > 0 else 1.0
    out = []
    for b in bets:
        st = math.floor(b['raw']*k*2)/2
        if st < 1:
            b['race']['decision'] = 'PASS'; b['race']['reason'] += ' — mise Kelly < 1 €, PASS'; b['pred']['selected'] = False; continue
        d = b['pred']
        min_odds = math.ceil(1.0/(d['p_dec_low'])*10)/10
        out.append(dict(bet_id=f"{DATE.replace('-','')}_{b['race']['race_id']}_N{d['num']}_WIN", race_id=b['race']['race_id'],
                        meeting=b['race']['meeting'], rc=f"R{b['race']['r']}C{b['race']['c']}", time=b['race']['time'],
                        num=d['num'], horse=d['horse'], market='SIMPLE GAGNANT', odds=d['odds'], odds_time=b['race']['odds_time'],
                        p_hl=d['p_hl'], p_mkt=d['p_mkt'], p_dec=d['p_dec'], p_low=d['p_dec_low'], edge=d['edge'], ev=d['ev'], ev_low=d['ev_low'], edge_raw=d['edge_raw'],
                        confidence=d['confidence'], stake=st, min_odds=min_odds, status='OPEN'))
    return out

if __name__ == '__main__':
    path, capital, now_iso = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    spec = importlib.util.spec_from_file_location('raw', path); raw = importlib.util.module_from_spec(spec); spec.loader.exec_module(raw)
    DATE = raw.DATE
    results = [analyse_race(r, now_iso) for r in raw.RACES]
    bets = size_bets(results, capital)
    json.dump(dict(date=DATE, model=MODEL_VERSION, params=P, computed_at=now_iso, capture_partants=raw.CAPTURE_PARTANTS,
                   capital=capital, races=results, bets=bets), open(sys.argv[4], 'w'), ensure_ascii=False, indent=1)
    for r in results:
        top = sorted(r['preds'], key=lambda d: -d['p_hl'])[:3]
        print(f"{r['race_id']:<22}{r['time']} {r['status']:<12}{(r['decision'] or '-'):<5} {r['reason'][:90]}")
        print('    top:', ', '.join(f"{d['num']}:{d['p_hl']*100:.0f}%" + (f"/m{d['p_mkt']*100:.0f}%" if d.get('p_mkt') is not None else '') for d in top))
    print('BETS', json.dumps([(b['rc'], b['num'], b['horse'], b['odds'], b['stake'], b['min_odds']) for b in bets], ensure_ascii=False))
