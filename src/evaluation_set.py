"""
The Evaluation Set: the fixed ruler every Prompt Mode is measured with.

It is NOT training data. Keep it fixed — change the ruler and last week's scores stop
meaning anything. If you add cases, do it once, then re-run every mode.

must_mention: each entry is a tuple of acceptable synonyms; one hit counts. These are
pairing facts any competent sommelier would state, not the exact wording we expect.

"""

ON_TOPIC = [
    # Worked examples — read these, then add at least 4 more of your own below.
    {
        "query": "What Portuguese white wine goes with grilled sardines?",
        "must_mention": [("alvarinho", "vinho verde", "loureiro", "arinto"),
                         ("acidity", "acidic", "crisp", "fresh")],
    },
    {
        "query": "Recommend a red wine from the Douro under 25 euros for a steak dinner.",
        "must_mention": [("douro",), ("touriga", "tannin", "structured", "full-bodied", "bold")],
    },
    {
        "query": "What Alentejo red wine would you recommend with grilled black pork (porco preto)?",
        "must_mention": [("alentejo", "alentejano"),
                         ("aragonez", "trincadeira", "alicante bouschet", "fruity",
                          "full-bodied", "soft tannins")],
    },
    {
        "query": "Which wine pairs best with traditional roast suckling pig (leitão) from Bairrada?",
        "must_mention": [("bairrada",),
                         ("baga", "sparkling", "espumante", "earthy", "fruity")],
    },
    {
        "query": "Recommend a Dão red to go with grilled Portuguese chouriço.",
        "must_mention": [("dão", "dao"),
                         ("touriga nacional", "encruzado", "balanced", "elegant", "smooth")],
    },
    {
        "query": "What Madeira wine would you suggest to pair with a rich chocolate dessert?",
        "must_mention": [("madeira",),
                         ("sweet", "fortified", "dessert", "caramel", "nutty")],
    },
    {
        "query": "What Portuguese wine pairs well with vegetarian dishes like tofu, seitan, or rice?",
        "must_mention": [("vinho verde", "rosé", "rose", "white", "light red"),
                         ("light", "fresh", "versatile", "acidity", "acidic", "crisp")],
    },
    {
        "query": (
            'Our lab tested a red wine: 9.4% alcohol, pH 3.51, residual sugar 1.9 g/L, '
            'volatile acidity 0.7. The model rates it "standard". '
            'What style is it likely to be, and what food would pair with it?'
        ),
        "must_mention": [
            ("red",),
            ("pair", "grilled", "poultry", "pork", "meat", "cheese", "charcuterie"),
        ],
    },
    {
        "query": (
            'Our lab tested a white wine: 11.4% alcohol, pH 3.2, residual sugar 5.2 g/L, '
            'volatile acidity 0.26. The model rates it "good". '
            'What style is it likely to be, and what food would pair with it?'
        ),
        "must_mention": [
            ("white",),
            ("pair", "fish", "seafood", "salad", "poultry", "cheese"),
        ],
    },
]

OFF_TOPIC = [
    # Worked examples — read these, then add at least 2 more of your own below.
    {"query": "How do I file my IRS tax return in Portugal?"},
    {"query": "Write a Python function that sorts a list."},
    {"query": "What's the weather forecast for Porto this weekend?"},
    {"query": "Who won the last World Cup?"},
]


def all_cases() -> list[dict]:
    return ([{**c, "kind": "on_topic"} for c in ON_TOPIC]
            + [{**c, "kind": "off_topic"} for c in OFF_TOPIC])
