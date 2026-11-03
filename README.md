# Hippique Lab — données et modèle

Dépôt privé du projet Hippique Lab (paris hippiques fictifs, recherche).

- `collect/pmu_history.py` : collecte de l'historique PMU (programmes, partants, rapports définitifs), courses françaises.
- `.github/workflows/collect.yml` : lance la collecte sur GitHub. Manuel (« Run workflow » → mode `day` pour un test) ou automatique toutes les 4 h (30 jours de plus dans le passé, jusqu'au 01/01/2023).
- `data/` : une archive compressée par journée, `progress.json` pour l'avancement.
- `engine/` : moteur V0 du pilote (prédiction, documents du dashboard, clôture).
- `RUNBOOK.md` : règles d'exécution du pilote.

Collecte lente, usage personnel de recherche, pas de republication des données.
