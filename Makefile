# Shortcuts. Run from the repository root.  `make help` lists them.
COMPOSE = docker compose -f docker/docker-compose.yml
RUN     = $(COMPOSE) exec jupyter python -m

.PHONY: help up down logs data train register eval promote-prompt reload test
help:            ## list commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'
up:              ## build and start MLflow, JupyterLab and the app
	$(COMPOSE) up -d --build
down:            ## stop everything (data and MLflow state are kept)
	$(COMPOSE) down
logs:            ## follow the app logs
	$(COMPOSE) logs -f app
data:            ## download UCI + Kaggle and build the processed tables
	$(RUN) src.data --kaggle
train:           ## train every configuration, log to MLflow
	$(RUN) src.train --mlflow-uri http://mlflow:5000
register:        ## register best run, run the gate, promote to @champion
	$(RUN) src.register --mlflow-uri http://mlflow:5000
eval:            ## score all prompt modes (no promotion), ~6 min
	$(RUN) src.evaluate_prompts
promote-prompt:  ## score prompt modes and promote the winner
	$(RUN) src.evaluate_prompts --promote
reload:          ## tell the app to pick up new @champion model + prompt
	curl -s -X POST localhost:8080/reload; echo; curl -s -X POST localhost:8080/prompt/reload; echo
test:            ## run the test suite inside the container
	$(COMPOSE) exec jupyter pytest -q
