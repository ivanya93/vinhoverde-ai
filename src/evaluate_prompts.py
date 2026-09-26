"""
Score every Prompt Mode on the Evaluation Set, register each as a prompt version, and move
@champion to the winner if it passes the gate. Same shape as the model cycle:

    models   train -> register versions -> evaluate -> @champion -> API serves
    prompts  write -> register versions -> evaluate -> @champion -> Sommelier serves

    docker compose exec app python -m src.evaluate_prompts            # score only
    docker compose exec app python -m src.evaluate_prompts --promote  # and promote

3 modes x 10 cases = 30 calls. At the free tier's 5 requests/minute: ~6 minutes.
"""

import argparse
import logging
import os
import time

import mlflow
import mlflow.genai

from src import catalog as catalog_mod
from src import evaluation_set, llm_client
from src.prompt_modes import PROMPT_MODES, PROMPT_NAME, REFUSAL, render

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Sommelier-Prompt-Engineering"
MIN_REFUSAL_ACCURACY = 1.0  # answering off-topic questions is a hard fail
MIN_OVERALL_SCORE = 0.50


def looks_like_refusal(answer: str) -> bool:
    text = answer.lower()
    return REFUSAL.lower()[:40] in text or ("i can only help" in text) or ("only help with wine" in text)


def is_grounded(answer: str, retrieved) -> bool:
    """Did the answer name at least one producer from the wines we actually retrieved?"""
    text = answer.lower()
    wineries = [str(w).lower() for w in retrieved["winery"].dropna() if len(str(w)) > 3]
    return any(w in text for w in wineries)


class RateLimiter:
    """Stay under the provider's requests/minute; a 429 would otherwise score as a zero."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last = 0.0

    def wait(self):
        gap = time.time() - self.last
        if self.interval and gap < self.interval:
            time.sleep(self.interval - gap)
        self.last = time.time()


def score_mode(client, model, max_tokens, template, catalog, limiter, cases=None) -> dict:
    cases = cases or evaluation_set.all_cases()
    coverage, grounded, latencies = [], [], []
    refusals_ok = off_topic = failures = truncated = 0
    false_refusals = on_topic = 0

    for case in cases:
        retrieved = catalog.search(case["query"]) if catalog else None
        context = catalog_mod.Catalog.to_context(retrieved) if catalog else "(catalog not loaded)"
        prompt = render(template, case["query"], context)

        limiter.wait()
        start = time.time()
        try:
            out = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.4)
            answer = out.choices[0].message.content or ""
            truncated += out.choices[0].finish_reason == "length"
        except Exception as exc:  # noqa: BLE001 — a failed call counts against the mode
            logger.warning("    call failed: %s", exc)
            failures += 1
            answer = ""
        latencies.append(time.time() - start)

        if case["kind"] == "off_topic":
            off_topic += 1
            refusals_ok += looks_like_refusal(answer)
            continue

        # refusal_accuracy above only checks off-topic cases get refused; it says nothing
        # about an on-topic one wrongly getting refused. We found exactly that live: the
        # "grounded" champion refused "A Douro red under 20 euros for steak?" once, at
        # temperature=0.4, even though the catalog context and prompt were both fine and
        # 4/4 replays answered correctly. That kind of flake was invisible to this eval
        # until we started counting it here.
        on_topic += 1
        false_refusals += looks_like_refusal(answer)

        text = answer.lower()
        hits = sum(any(s in text for s in group) for group in case["must_mention"])
        coverage.append(hits / len(case["must_mention"]))
        if retrieved is not None:
            grounded.append(float(is_grounded(answer, retrieved)))

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    m = {
        "key_point_coverage": mean(coverage),
        "grounding_rate": mean(grounded),
        "refusal_accuracy": refusals_ok / off_topic if off_topic else 0.0,
        "false_refusal_rate": false_refusals / on_topic if on_topic else 0.0,
        "avg_latency_seconds": mean(latencies),
        "truncated_responses": truncated,
        "failed_calls": failures,
    }
    # Not folded into overall_score yet — surfaced for visibility first. Worth discussing:
    # should a mode that occasionally refuses good questions lose points even if its
    # average coverage/grounding look great?
    m["overall_score"] = (0.35 * m["key_point_coverage"] + 0.35 * m["grounding_rate"]
                          + 0.30 * m["refusal_accuracy"])
    return m


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sommelier Prompt Modes")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--modes", nargs="*", default=list(PROMPT_MODES))
    parser.add_argument("--rpm", type=int, default=int(os.getenv("LLM_REQUESTS_PER_MINUTE", 5)))
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    args = parser.parse_args()

    if not llm_client.is_configured():
        raise SystemExit("No GEMINI_API_KEY. Put it in docker/.env (see docker/.env.example).")
    catalog = catalog_mod.load()
    if catalog is None:
        logger.warning("No Portuguese catalog — run `python -m src.data --kaggle`. "
                       "Grounding will score 0 for every mode.")

    client, model, max_tokens = llm_client.build_client(), llm_client.get_model(), llm_client.get_max_tokens()
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.openai.autolog()  # every call becomes a trace next to the score it produced

    limiter = RateLimiter(args.rpm)
    cases = evaluation_set.all_cases()
    logger.info("Evaluating %d modes x %d cases with %s (~%.0f min)\n", len(args.modes), len(cases),
                model, len(args.modes) * len(cases) / max(args.rpm, 1))

    results, versions = [], {}
    with mlflow.start_run(run_name="prompt-mode-comparison"):
        mlflow.log_params({"model": model, "max_tokens": max_tokens, "cases": len(cases),
                           "rpm": args.rpm})
        for name in args.modes:
            mode = PROMPT_MODES[name]
            version = mlflow.genai.register_prompt(
                name=PROMPT_NAME, template=mode["template"],
                commit_message=f"{name}: {mode['description']}", tags={"mode": name})
            versions[name] = version.version

            with mlflow.start_run(run_name=name, nested=True):
                metrics = score_mode(client, model, max_tokens, mode["template"], catalog, limiter, cases)
                mlflow.log_params({"mode": name, "prompt_version": version.version})
                mlflow.log_metrics(metrics)
            results.append((name, metrics))
            logger.info("  %-20s overall %.3f | coverage %.2f | grounding %.2f | refusals %.2f | "
                        "false refusals %.2f",
                        name, metrics["overall_score"], metrics["key_point_coverage"],
                        metrics["grounding_rate"], metrics["refusal_accuracy"],
                        metrics["false_refusal_rate"])

        results.sort(key=lambda r: r[1]["overall_score"], reverse=True)
        winner, wm = results[0]
        failures = []
        if wm["refusal_accuracy"] < MIN_REFUSAL_ACCURACY:
            failures.append(f"refusal_accuracy {wm['refusal_accuracy']:.2f} < {MIN_REFUSAL_ACCURACY}")
        if wm["overall_score"] < MIN_OVERALL_SCORE:
            failures.append(f"overall_score {wm['overall_score']:.2f} < {MIN_OVERALL_SCORE}")
        mlflow.log_params({"best_mode": winner})
        mlflow.log_metric("gate_passed", 0 if failures else 1)

        if failures:
            logger.info("\nGate FAILED for %s: %s. Champion unchanged.", winner, "; ".join(failures))
            return
        logger.info("\nGate passed for %s.", winner)
        if not args.promote:
            logger.info("Run again with --promote to move @champion.")
            return
        mlflow.genai.set_prompt_alias(PROMPT_NAME, "champion", versions[winner])
        logger.info("@champion -> %s v%s. Reload: curl -X POST http://localhost:8080/prompt/reload",
                    PROMPT_NAME, versions[winner])


if __name__ == "__main__":
    main()
