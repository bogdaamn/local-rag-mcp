PRICING = {
    "qwen3:0.6b": {"input": 0.20, "output": 0.20},  # Fireworks AI via OpenRouter, Sep 2026
}
DEFAULT_PRICING = {"input": 0.0, "output": 0.0}

_warned_models = set()


def estimate_cost(model, input_tokens, output_tokens):
    """$ = input_tokens/1e6 * price_in + output_tokens/1e6 * price_out.
    Unknown models default to $0.00 with a one-time warning rather than
    raising."""
    prices = PRICING.get(model)
    if prices is None:
        if model not in _warned_models:
            print(f"⚠️  No pricing data for model '{model}'; defaulting to $0.00")
            _warned_models.add(model)
        prices = DEFAULT_PRICING

    return (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]
