from telemetry import pricing


def test_estimate_cost_for_known_model():
    cost = pricing.estimate_cost("qwen3:0.6b", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 0.40  # $0.20/M in + $0.20/M out


def test_estimate_cost_zero_tokens_is_zero():
    assert pricing.estimate_cost("qwen3:0.6b", 0, 0) == 0.0


def test_estimate_cost_unknown_model_defaults_to_zero(monkeypatch, capsys):
    monkeypatch.setattr(pricing, "_warned_models", set())

    cost = pricing.estimate_cost("some-other-model", 1_000_000, 1_000_000)

    assert cost == 0.0
    assert "some-other-model" in capsys.readouterr().out


def test_estimate_cost_unknown_model_warns_only_once(monkeypatch, capsys):
    monkeypatch.setattr(pricing, "_warned_models", set())

    pricing.estimate_cost("some-other-model", 1, 1)
    pricing.estimate_cost("some-other-model", 1, 1)

    assert capsys.readouterr().out.count("some-other-model") == 1
