from types import SimpleNamespace as NS

import pandas as pd
import pytest

from src import evaluate_prompts as ep
from src.catalog import Catalog
from src.data import build_portugal_catalog
from src.prompt_modes import PROMPT_MODES, REFUSAL, render
from src import config
from src.evaluation_set import OFF_TOPIC


def make_catalog(raw_dir):
    return Catalog(build_portugal_catalog(raw_dir / config.KAGGLE_FILE))


def test_filters_parse_colour_price_region():
    f = Catalog.parse_filters("A Douro red under 20 euros please, Portuguese style")
    assert f == {"colour": "red", "max_price": 20.0, "region": "douro"}
    assert "region" not in Catalog.parse_filters("any Portuguese wine?")  # 'port' must not match


def test_search_respects_filters(raw_dir):
    cat = make_catalog(raw_dir)
    res = cat.search("white wine for grilled sardines", k=3)
    assert (res["colour"] == "white").all()
    assert "Alvarinho" in res.iloc[0]["title"]
    assert (cat.search("red under 15")["price"] <= 15).all()


def test_templates_render_both_variables():
    for mode in PROMPT_MODES.values():
        out = render(mode["template"], "Q?", "CTX")
        assert "{{" not in out and "Q?" in out


def test_refusal_detection():
    assert ep.looks_like_refusal(REFUSAL)
    assert not ep.looks_like_refusal("Try an Alvarinho, crisp acidity.")


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        prompt = kw["messages"][-1]["content"]
        off = any(c["query"] in prompt for c in OFF_TOPIC)
        return NS(choices=[NS(message=NS(content=REFUSAL if off else self.text), finish_reason="stop")])


@pytest.mark.skipif("grounded" not in PROMPT_MODES,
                    reason="write the 'grounded' prompt in src/prompt_modes.py first")
def test_score_mode_rewards_grounded_answers(raw_dir):
    """
    the pytest, this checks the thing that actually
    justifies it: a grounded answer that names a real catalog wine should score higher
    than an ungrounded one that doesn't. If this fails, we'll read what score_mode is computing
    in src/evaluate_prompts.py — the metric might need a look, or our prompt might.
    """
    cat = make_catalog(raw_dir)
    limiter = ep.RateLimiter(0)
    good = ep.score_mode(FakeLLM("Casa Alta Alvarinho: crisp acidity"), "m", 100,
                         PROMPT_MODES["grounded"]["template"], cat, limiter)
    bad = ep.score_mode(FakeLLM("Any nice wine will do."), "m", 100,
                        PROMPT_MODES["basic"]["template"], cat, limiter)
    assert good["refusal_accuracy"] == 1.0
    assert good["overall_score"] > bad["overall_score"]
