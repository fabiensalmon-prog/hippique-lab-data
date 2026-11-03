# Hippique Lab — Runbook V1

Dashboard : https://claude.ai/artifact/F5rA1u75w63t1pk6NnLm2X · Dépôt : fabiensalmon-prog/hippique-lab-data (privé)
Pilote : J1 = 2026-11-03 (moteur V0), J2 et suivants : moteur V1. Capital initial 500 € fictifs.

## Qui fait quoi
| Heure (Paris) | Où | Quoi |
|---|---|---|
| Toutes les heures | GitHub `collect.yml` | Remonte l'historique PMU, 30 jours par run, jusqu'au 01/11/2024 |
| 04:40 | GitHub `train.yml` | Variables, logit conditionnel trot/plat, backtest walk-forward, second étage, porte de validation → `models/latest.json`, `reports/backtest.json` |
| 08:50 | GitHub `daily.yml` morning | Programme + partants + rapports probables du jour → `live/<date>/morning.json` |
| 09:33 | Tâche Claude « Run matin » | `python3 -m hl.sync morning …` → dashboard (courses, prédictions, tickets en euros) |
| 13:20 | GitHub `daily.yml` midday | Rapports probables frais → `live/<date>/midday.json` |
| 14:03 | Tâche Claude « Run marché » | `hl.sync midday` : décide les courses restées partielles, sans toucher aux décisions du matin |
| 22:30 | GitHub `daily.yml` closing | Arrivées et rapports définitifs → `live/<date>/results.json` + journée ajoutée à l'historique |
| 23:07 | Tâche Claude « Clôture » | `engine/closing.py` : règlement des tickets (gagnant et placé), métriques, capital J+1 |

## Règles non négociables
1. Une prédiction n'existe que si elle est calculée avant le départ ; course partie → NON_ANALYSEE.
2. PARTIEL ≠ PASS : sans rapports probables PMU pour tous les partants → PARTIELLE.
3. Tickets immuables après création, sauf le champ settlement à la clôture.
4. Jamais de résultat estimé : sans rapport officiel → PENDING.
5. opening_capital(J+1) = closing_capital(J).
6. Porte de validation : le modèle ne participe aux décisions (α > 0) que si, hors échantillon, le gain de ρ² sur le marché est ≥ 0,005 sur au moins 1 000 courses. Sinon α = 0, le marché décide seul et aucun ticket n'est pris (observation).
7. Mises : ¼ Kelly sur la probabilité basse, 2 % du capital par ticket, 6 % par jour.

## Modèle V1
- Étage 1 : logit conditionnel séparé trot / plat, 54 variables (`hl/features.py`), toutes calculées avec l'information antérieure à la course.
- Étage 2 (Benter) : softmax(α·log P fondamentale + β·log P marché), estimé sur prédictions hors échantillon.
- Placé : Harville corrigé (γ, δ estimés), dividende placé attendu par régression sur l'historique.
- Exotiques : meilleur couplé gagnant enregistré en observation, jamais misé.
- Backtest : décision sur le rapport de référence du matin, règlement sur les dividendes officiels.
