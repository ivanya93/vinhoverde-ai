"""
VinhoVerde AI — one service, two products:

    /predict   Quality Lab: lab measurements -> quality tier, served from models:/...@champion
    /chat      Sommelier:   question -> retrieved Portuguese wines -> prompts:/sommelier@champion
    /          The UI (ui/index.html)

Both products follow the same rule: what is served is whatever holds the @champion alias in
MLflow. Promoting in the registry + calling /reload (or /prompt/reload) is the deployment.
"""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.genai
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src import catalog as catalog_mod
from src import config, llm_client, prompt_modes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vinhoverde")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_URI = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
PROMPT_URI = f"prompts:/{prompt_modes.PROMPT_NAME}@champion"
CHAT_EXPERIMENT = os.getenv("MLFLOW_CHAT_EXPERIMENT", "Sommelier-Chat")
UI_FILE = Path(__file__).resolve().parents[1] / "ui" / "index.html"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
app = FastAPI(title="VinhoVerde AI", version="1.0")

# ------------------------------------------------------------------------------------
# State, loaded lazily: the API starts before MLflow is ready and before anything is
# trained, so loading at import time would leave it broken until a restart.
# ------------------------------------------------------------------------------------
_state = {"model": None, "model_version": None, "model_error": None,
          "prompt": None, "prompt_version": None, "prompt_mode": None, "tracing": False}
_llm = llm_client.build_client()


def get_model(force: bool = False):
    if _state["model"] is not None and not force:
        return _state["model"]
    try:
        _state["model"] = mlflow.sklearn.load_model(MODEL_URI)
        mv = MlflowClient().get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
        _state["model_version"] = mv.version
        _state["model_error"] = None
    except Exception as exc:  # noqa: BLE001
        _state["model"], _state["model_error"] = None, str(exc)
    return _state["model"]


def get_prompt(force: bool = False) -> str:
    """The champion prompt template; falls back to the local default mode if none yet."""
    if _state["prompt"] is not None and not force:
        return _state["prompt"]
    try:
        p = mlflow.genai.load_prompt(PROMPT_URI)
        _state.update(prompt=p.template, prompt_version=p.version,
                      prompt_mode=(p.tags or {}).get("mode", "registered"))
    except Exception as exc:  # noqa: BLE001
        logger.info("No champion prompt yet (%s) — using local '%s'", exc, prompt_modes.DEFAULT_MODE)
        _state.update(prompt=prompt_modes.PROMPT_MODES[prompt_modes.DEFAULT_MODE]["template"],
                      prompt_version="local", prompt_mode=prompt_modes.DEFAULT_MODE)
    return _state["prompt"]


def ensure_tracing():
    if _state["tracing"]:
        return
    try:
        mlflow.set_experiment(CHAT_EXPERIMENT)
        mlflow.openai.autolog()
        _state["tracing"] = True
    except Exception as exc:  # noqa: BLE001 — no MLflow should cost observability, not the service
        logger.warning("Tracing off: %s", exc)


# ------------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------------
class Wine(BaseModel):
    fixed_acidity: float = Field(7.4, alias="fixed acidity")
    volatile_acidity: float = Field(0.7, alias="volatile acidity")
    citric_acid: float = Field(0.0, alias="citric acid")
    residual_sugar: float = Field(1.9, alias="residual sugar")
    chlorides: float = 0.076
    free_sulfur_dioxide: float = Field(11.0, alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(34.0, alias="total sulfur dioxide")
    density: float = 0.9978
    pH: float = 3.51
    sulphates: float = 0.56
    alcohol: float = 9.4
    wine_type: str = "red"

    model_config = {"populate_by_name": True}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump(by_alias=True)])[config.INPUT_COLUMNS]


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []


# ------------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(UI_FILE)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _state["model"] is not None,
            "model_uri": MODEL_URI, "model_version": _state["model_version"],
            "prompt_version": _state["prompt_version"], "prompt_mode": _state["prompt_mode"],
            "catalog_loaded": catalog_mod.load() is not None, "llm": llm_client.describe()}


@app.post("/predict")
def predict(wine: Wine):
    model = get_model()
    if model is None:
        raise HTTPException(503, f"No model at {MODEL_URI}. Train + register first. ({_state['model_error']})")
    X = wine.to_frame()
    tier = str(model.predict(X)[0])
    proba = dict(zip(model.classes_, model.predict_proba(X)[0].round(3).tolist()))
    return {"quality_tier": tier, "probabilities": proba, "action": config.TIER_ACTIONS[tier],
            "model_version": _state["model_version"], "input": wine.model_dump(by_alias=True)}


@app.post("/reload")
def reload_model():
    if get_model(force=True) is None:
        raise HTTPException(503, f"Could not load {MODEL_URI}: {_state['model_error']}")
    return {"reloaded": True, "model_version": _state["model_version"]}


@app.post("/chat")
def chat(req: ChatRequest):
    if _llm is None:
        raise HTTPException(503, "No GEMINI_API_KEY configured. See docker/.env.example.")
    ensure_tracing()

    catalog = catalog_mod.load()
    retrieved = catalog.search(req.message) if catalog else pd.DataFrame()
    context = catalog_mod.Catalog.to_context(retrieved) if catalog else "(catalog not loaded)"
    prompt = prompt_modes.render(get_prompt(), req.message, context)

    messages = [t.model_dump() for t in req.history[-6:]] + [{"role": "user", "content": prompt}]
    try:
        out = _llm.chat.completions.create(model=llm_client.get_model(), messages=messages,
                                           max_tokens=llm_client.get_max_tokens(), temperature=0.4)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"LLM call failed: {exc}") from exc

    wines = [] if retrieved.empty else retrieved[["title", "variety", "province", "points", "price"]] \
        .fillna("").to_dict(orient="records")
    return {"answer": out.choices[0].message.content, "wines": wines,
            "prompt_version": _state["prompt_version"], "prompt_mode": _state["prompt_mode"]}


@app.post("/prompt/reload")
def reload_prompt():
    get_prompt(force=True)
    return {"reloaded": True, "prompt_version": _state["prompt_version"], "mode": _state["prompt_mode"]}


@app.get("/examples")
def examples():
    """Two realistic presets for the UI (typical UCI values)."""
    return {
        "red": Wine().model_dump(by_alias=True),
        "white": Wine(**{"fixed acidity": 6.8, "volatile acidity": 0.26, "citric acid": 0.34,
                         "residual sugar": 5.2, "chlorides": 0.04, "free sulfur dioxide": 35,
                         "total sulfur dioxide": 138, "density": 0.9935, "pH": 3.2,
                         "sulphates": 0.49, "alcohol": 11.4, "wine_type": "white"}).model_dump(by_alias=True),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
