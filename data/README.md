# Data

Nothing in this folder is committed. Recreate it with:

```bash
docker compose -f docker/docker-compose.yml exec jupyter python -m src.data --kaggle
```

| File | Source | Used for |
|---|---|---|
| `raw/winequality-red.csv`, `raw/winequality-white.csv` | [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (Cortez et al., 2009). Red + white Vinho Verde, 6,497 samples | ML model |
| `raw/winemag-data-130k-v2.csv` | [Kaggle Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews) (WineEnthusiast, ~130k reviews) | Sommelier catalog |
| `processed/wine_quality.csv` | red + white combined, duplicates removed, `wine_type` + `quality_tier` added | training |
| `processed/portugal_wines.csv` | Kaggle rows where `country == "Portugal"`, plus inferred `colour` | retrieval (RAG) |

**Kaggle manual fallback:** if the automatic download fails, download the zip from the Kaggle page,
unzip, and put `winemag-data-130k-v2.csv` in `data/raw/`. Then rerun the command above.

**Cite both datasets in the slides.** UCI: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis,
*Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems, 2009.
