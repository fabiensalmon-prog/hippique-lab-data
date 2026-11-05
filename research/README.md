# Étude des récurrences (05/11/2026)

Scripts de l'étude qui a conduit au modèle V1.2. Découverte = jusqu'au 31/12/2024, validation = 2025-2026.

- `build.py <dépôt> <sortie.pkl>` : table plate point-in-time (variables actuelles + candidates).
- `analyse.py <dépôt> <table.pkl> <sortie.json>` : récurrences face au marché final (A/E, ROI réels), persistance
  entraîneur / driver / étalon, valeur ajoutée de chaque variable → `reports/etude_recurrences.json`.
- `subset.py` : sous-ensembles retenus sur deux découpages + recalibrage du dividende 2sur4 → `reports/etude_variables.json`.
- `s4cal.py` : calibration du dividende attendu couplé / 2sur4 par tranche.
- `exotics.py` : ROI réel de chaque combinaison de rangs de cote (couplé gagnant / placé, 2sur4, trio), matin et final → `reports/etude_combines.json`.
