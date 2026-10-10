MODEL_PRICING = {
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
    "claude-3-5-haiku-latest": {
        "input": 0.80,
        "output": 4.00,
    },
    "claude-3-5-sonnet-latest": {
        "input": 3.00,
        "output": 15.00,
    },
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate the estimated request cost in USD."""

    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    if model not in MODEL_PRICING:
        raise ValueError(f"No pricing configured for model: {model}")

    prices = MODEL_PRICING[model]

    cost = (
        input_tokens * prices["input"]
        + output_tokens * prices["output"]
    ) / 1_000_000

    return round(cost, 6)