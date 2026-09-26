# 5-minute pitch plan

**Rule:** one idea per slide, and show the system working rather than describing it. Aim for 4:45 so there's slack.

| # | Slide | Time | Speaker | Content |
|---|---|---|---|---|
| 1 | **The problem** | 0:30 | E | Vinho Verde producers test every batch in the lab anyway, but quality is still judged by a slow tasting panel. Shops sell wines customers can't navigate. One sentence per side. |
| 2 | **VinhoVerde AI** | 0:30 | E | Two products, one platform: Quality Lab (winemaker) and Sommelier (customer). Show the UI screenshot. The value: faster triage, no premium batch sold as bulk, a sommelier that sells what's in stock. |
| 3 | **Data and model** | 0:45 | A | UCI red + white (6.5k → ~5.3k after removing duplicates, and *why*). 3 tiers. 5 engineered features inside the pipeline. 4 configurations compared in MLflow (screenshot), winner by macro-F1 plus premium recall. |
| 4 | **MLOps loop** | 0:45 | B | Diagram: train → register → `@candidate` → **gate** → `@champion` → API. Registry screenshot with versions and aliases. CI green plus the retrain workflow summary. "Moving an alias *is* the deployment." |
| 5 | **Live demo** | 1:30 | D + C | See the script below. |
| 6 | **LLMOps and lessons** | 0:45 | C | Prompt-mode comparison table (basic vs grounded). Grounding went from ~0 to high once retrieval was added. Refusals gated at 100%. One lesson learned, one next step. |

## Demo script (1:30)

1. **Quality Lab** (30s): *Load red example* → Predict → read the tier and the action aloud. Point at the footer: "served by model version N from the registry."
2. **Promotion** (30s, optional if time is tight): in the terminal run `make register` (pre-trained new config) → `make reload` → predict again → the footer version changes. *Backup: screenshot of before and after.*
3. **Sommelier** (30s): click *"Ask the sommelier about this wine"*, then ask one chip question such as *"A Douro red under 20 euros for steak"*. Point at the wine cards: real catalog entries. Ask *"How do I file my taxes?"* and it refuses.

**If anything fails live:** switch to the backup video at once. Don't debug on stage.

## Screenshot checklist

- [ ] EDA: tier balance and drivers of quality
- [ ] MLflow: run comparison (4 configs)
- [ ] MLflow: model registry with `@champion` and `@candidate`
- [ ] Terminal: gate PASS, and one gate FAIL
- [ ] MLflow: prompt comparison (3 modes) and one trace
- [ ] GitHub Actions: CI green and the retrain summary
- [ ] UI: both tabs
- [ ] `/docs`: API page

## Likely questions (prepare 1–2 sentence answers)

- **Why macro-F1 and not accuracy?** Classes are imbalanced. Accuracy rewards always predicting the majority class.
- **Why tiers instead of predicting the 0–10 score?** A winery makes a 3-way decision (bulk, regular label or reserve). Tiers map to actions, and the extreme scores (3, 9) are too rare to learn.
- **How do you avoid training/serving skew?** Feature engineering lives inside the registered pipeline, and requirements are pinned identically for notebook and app.
- **What stops a bad model reaching production?** The gate: an absolute floor plus "not worse than the current champion" on the same test set.
- **How do you know the LLM isn't making wines up?** The grounding metric, plus a prompt that restricts it to retrieved catalog entries. We show a trace.
- **Why TF-IDF and not embeddings?** About 5k short documents, no extra service, and it's explainable. Embeddings would be the next step.
- **What about data drift?** New vintages mean new data. `retrain.yml` retrains weekly and the gate decides. Next step: monitor the input distributions.
- **Limitations?** Quality labels are subjective tasters' scores, and the UCI data only covers Vinho Verde. The Kaggle reviews come from one publication's critics.
