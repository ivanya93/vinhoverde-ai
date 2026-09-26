# 🍷 VinhoVerde AI

**MLOps & LLMOps mini-project, EMBAAI, Porto Business School (2026)**

VinhoVerde AI is one app with two products, both run with the same MLOps discipline:

| | 🧪 Quality Lab (MLOps) | 🍷 Sommelier (LLMOps) |
|---|---|---|
| **User** | Winemaker, before bottling | Wine-shop customer |
| **Input** | 11 lab measurements + red/white | A question in plain language |
| **Output** | Quality tier: *standard / good / premium* + recommended action | Recommendations **only from our catalog of Portuguese wines** |
| **Data** | UCI Wine Quality: red + white Vinho Verde (6.5k samples) | Kaggle Wine Reviews, filtered to Portugal (~5k wines) |
| **What we version** | The model → MLflow Model Registry | The prompt → MLflow Prompt Registry |
| **How we compare** | Macro-F1 on a held-out test set | Evaluation Set: coverage, grounding, refusals |
| **How we ship** | Move `@champion` → `POST /reload` | Move `@champion` → `POST /prompt/reload` |

## Business value

- **Winery:** triage every batch in seconds from routine lab tests. That means fewer tasting-panel hours, and premium batches don't get sold as bulk wine (we track *premium recall*).
- **Wine shop:** a 24/7 sommelier that sells *what is actually on the shelf*, and politely refuses anything that isn't about wine.

## Quick start

```bash
cp docker/.env.example docker/.env   # add GEMINI_API_KEY (+ Kaggle token, optional)
make up          # MLflow :5001 · JupyterLab :8888 · App :8080
make data        # download UCI + Kaggle, build processed tables
make train       # 4 model configurations -> MLflow
make register    # best run -> @candidate -> gate -> @champion
make promote-prompt   # score 3 prompt modes, promote the winner (~6 min)
make reload      # app picks up both champions
```

Open **http://localhost:8080**. No `make`? Every target is a one-line `docker compose` command (see `Makefile`).

## Architecture

```
            ┌──────────── docker compose ────────────────────────────────┐
 browser ──►│ app :8080  FastAPI + UI                                    │
            │   /predict ──► models:/VinhoVerde-Quality-Classifier@champion
            │   /chat ─────► catalog search (TF-IDF) ─► prompts:/sommelier@champion ─► Gemini
            │                                                            │
            │ mlflow :5001  tracking · model registry · prompt registry · traces
            │                                                            │
            │ jupyter :8888  EDA + runs src/data → train → register → evaluate_prompts
            └────────────────────────────────────────────────────────────┘
 GitHub Actions: ci.yml (tests on every push) · retrain.yml (bonus: scheduled retrain + gate)
```

## Repository layout

```
├── api/app.py                 FastAPI: /predict /chat /reload /prompt/reload /health
├── ui/index.html              The "lovable" UI (Quality Lab + Sommelier chat)
├── src/
│   ├── config.py              Names, features, tiers, gate threshold
│   ├── data.py                Download + clean UCI and Kaggle
│   ├── features.py            Feature engineering (inside the model pipeline)
│   ├── train.py               4 configurations → MLflow runs
│   ├── register.py            Register → gate → promote
│   ├── catalog.py             Retrieval over Portuguese wines
│   ├── prompt_modes.py        3 prompt versions (single source of truth)
│   ├── evaluation_set.py      Fixed evaluation questions
│   ├── evaluate_prompts.py    Score → register prompts → gate → promote
│   └── llm_client.py          Provider seam (Gemini via OpenAI SDK)
├── notebooks/                 01_eda · 02_pipeline_walkthrough
├── tests/                     Offline tests on synthetic data
├── docker/                    Compose stack, Dockerfiles, pinned requirements
├── .github/workflows/         ci.yml · retrain.yml
├── docs/PRESENTATION.md       5-minute pitch plan
└── GUIDE.md                   Step-by-step plan to finish the project
```

## Known limitations

- **The prompt eval has a blind spot for false refusals.** `refusal_accuracy` only checks
  that off-topic questions get refused — it never checked whether an on-topic one gets
  wrongly refused. We hit this live: the `grounded` prompt (with a well-formed catalog
  context) refused *"A Douro red under 20 euros for steak?"* once, at `temperature=0.4`,
  even though replaying the exact same prompt 4/4 times produced correct, grounded
  answers. It was non-deterministic LLM flakiness, not a bug in the prompt or the
  retrieval — but the eval couldn't have caught it either way.
  `src/evaluate_prompts.py` now tracks a `false_refusal_rate` metric (checked over the
  on-topic cases) so this failure mode is at least visible per prompt mode, even though
  it isn't folded into `overall_score` yet.

## Data credits

- P. Cortez et al., *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems, 2009 (UCI Wine Quality).
- WineEnthusiast reviews, via Kaggle ([zynicide/wine-reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews)).
