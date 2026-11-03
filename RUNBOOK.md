# Hippique Lab — Runbook d'exécution (pilote 30 jours)

Dashboard : https://claude.ai/artifact/F5rA1u75w63t1pk6NnLm2X (base `db` de l'artifact, écrite avec ArtifactData).
Code moteur : docs du projet `claude/engine_v0.py`, `claude/to_docs.py`, `claude/closing.py`, exemple de capture `claude/raw_20261103.py`.
Pilote : J1 = 2026-11-03, capital initial 500 € fictifs. Jour N = nombre de jours depuis le 03/11 + 1.

## Collections de la base
- `meta/state` : day, current_date, current_capital, initial_capital, model, params, updated_at, last_run.
- `days/<YYYY-MM-DD>` : opening_capital, closing_capital, pnl, coverage{expected,analyzed,partial,not_analyzed}, n_bet, n_pass, exposure, status_morning, status_close.
- `races/<date>_<MEETING>_R<r>C<c>` : statut, décision, raison, preds[] (tous les partants : p_fund, p_exp, p_hl, p_mkt, p_dec, odds, edge, ev_low…), result.
- `bets/<YYYYMMDD>_<MEETING>_R<r>C<c>_N<num>_WIN` : ticket immuable + settlement.
- `metrics/cumul` : Brier, log loss, rang du gagnant (HL vs marché), calibration, désaccords, signaux bruts, CLV.

## Règles non négociables
1. Une prédiction n'existe que si elle est écrite AVANT le départ de la course (computed_at < heure de départ). Course déjà partie → NON_ANALYSEE.
2. PARTIEL ≠ PASS. Sans cote PMU vérifiée pour tous les partants → PARTIELLE (prédiction quand même journalisée).
3. Le run est COMPLET seulement si toutes les courses françaises PMU trot/plat du jour sont ANALYSEE. Les BET d'un run incomplet restent valides course par course ; la couverture est affichée.
4. Jamais de résultat estimé : sans arrivée + rapport officiels → PENDING.
5. Les tickets ne sont jamais modifiés après le départ, sauf le champ settlement à la clôture.
6. opening_capital(J+1) = closing_capital(J). Si la clôture J est PENDING, ouvrir J+1 sur le capital connu et le noter.
7. Toute écriture sur un document existant passe son `if_version` (versions lues via ArtifactData list).

## Sources (ordre de préférence)
- Programme + partants + musiques + pronostics : `https://www.canalturf.com/courses_liste_pronostics_demain.php` (page « demain » vue la veille, ou rechercher « pronostics PMU <date> canalturf »), puis chaque page course canalturf.
- Rapports probables PMU : geny.com pages `/course/<id>-<slug>/cotes` (trouver via WebSearch « <nom du prix> <date> cotes » ou la page programme geny `https://www.geny.com/en/programme/<YYYY-MM-DD>`), à défaut zone-turf programme de la réunion (colonne PMU).
- Résultats + rapports officiels : pages résultats canalturf / geny `arrivee-rapports` / zeturf course-du-jour.
- Ne contourner aucun site refusé ; noter la source dans `odds_src` / `source`.

## Run du matin (09:00)
1. Lire `meta/state` et `days` (ArtifactData list avec out_dir dans un dossier `dbx/`), noter les versions dans `dbx/versions.json` ({"meta/state": v, ...}).
2. Récupérer TOUTES les réunions françaises PMU trot/plat du jour (ignorer obstacle et étranger ; le noter). Pour chaque course : partants (num, cheval, driver/jockey, musique, valeur pour le plat), pronostics (canalturf, zeturf, presse), rapports probables PMU + heure.
3. Écrire `data/raw_<YYYYMMDD>.py` au format de `raw_20261103.py` (disc : TA / TM / PLAT / PLAT_H ; non-partant = 5e champ "NP?").
4. `python3 engine_v0.py data/raw_<date>.py <capital_ouverture> <now ISO+01:00> out/run_<date>.json`
5. `python3 to_docs.py out/run_<date>.json out/docs_<date> <jour N> dbx/versions.json` puis ArtifactData `batch` avec `_writes.json` (≤ 50 par batch).
6. Résumé : RUN J<N> COMPLET/INCOMPLET — x/y analysées, BET, PASS, exposition.

## Clôture (23:00)
1. Exporter `races` (where date == jour), `bets`, `days`, `meta`, `metrics` dans `dbx/` + `dbx/versions.json`.
2. Pour chaque course du jour ayant des prédictions : arrivée officielle, rapport simple gagnant PMU pour 1 €, non-partants, et si possible cote finale des chevaux joués. Écrire `results_<date>.json` {race_doc_id: {arrivee, rapport_gagnant, non_partants, final_odds, source}}.
3. `python3 closing.py dbx results_<date>.json <date> out/close_<date> <now ISO>` puis batch `_writes.json`.
4. Phrase finale obligatoire : `CAPITAL DE DÉPART DU PROCHAIN RUN = XXX €`.

## Évolution du modèle (ne pas faire pendant un run)
alpha_model (poids du modèle dans P décision) reste 0.25 tant que metrics/cumul n'a pas ≥ 300 courses. Il ne monte que si log loss HL < log loss marché hors échantillon.
