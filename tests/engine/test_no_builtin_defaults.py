"""yian has no built-in provider or model: each comes from a layer, or the run fails."""

import copy
import json

import pytest

from yian.compile.llm_cascade import resolve_llm_cascade
from yian.compose.loader import load_flow
from yian.llm_clients.auth import ApiKeyCredentials, AuthStore
from yian.llm_clients.connection import connection_snapshot, resolve_model_config
from yian.run import run_flow

_FIXES = ("llm_config.provider/model", "--provider/--model", "yian auth <provider>")


@pytest.fixture(autouse=True)
def no_environment_defaults(monkeypatch):
    monkeypatch.delenv("YIAN_DEFAULT_PROVIDER", raising=False)
    monkeypatch.delenv("YIAN_DEFAULT_MODEL", raising=False)


def _flow(llm_config=None):
    flow_layer = "" if llm_config is None else f"llm_config: {llm_config}\n"
    return load_flow(
        "id: f\nname: f\n" + flow_layer
        + "nodes:\n  write: {kind: agent, prompt: hi}\noutput: ${write.output}\n"
    )


def _resolve(loaded, layer=None):
    snapshot = connection_snapshot()
    return resolve_llm_cascade(
        copy.deepcopy(loaded.compiled),
        dict(layer or {}),
        defaults=snapshot.defaults,
        provider_models=snapshot.provider_models,
        stray_model=snapshot.stray_model,
    )


def _fake_model(monkeypatch):
    class _Model:
        def invoke(self, messages):
            class _Response:
                content = "ok"

            return _Response()

    monkeypatch.setattr("yian.llm_clients.model_from_config", lambda config: _Model())


def _forbid_model(monkeypatch):
    def forbidden(config):
        pytest.fail("no model may be built when the run cannot start")

    monkeypatch.setattr("yian.llm_clients.model_from_config", forbidden)


def test_no_model_anywhere_fails_before_any_node(monkeypatch, tmp_path):
    _forbid_model(monkeypatch)
    log = tmp_path / "r.jsonl"
    with pytest.raises(ValueError) as caught:
        run_flow(_flow(), {}, log=log)
    message = str(caught.value)
    assert "Agent 'write'" in message
    for fix in _FIXES:
        assert fix in message
    assert not log.exists()


def test_provider_without_model_names_that_provider(monkeypatch, tmp_path):
    _forbid_model(monkeypatch)
    with pytest.raises(ValueError) as caught:
        run_flow(_flow("{provider: anthropic}"), {}, log=tmp_path / "r.jsonl")
    message = str(caught.value)
    assert "Agent 'write'" in message
    assert "provider 'anthropic'" in message
    assert "yian auth anthropic" in message

    nested = load_flow(
        "id: f\nname: f\nllm_config: {provider: anthropic}\n"
        "defs:\n  sub:\n    nodes:\n      draft: {kind: agent, prompt: hi}\n"
        "    output: ${draft.output}\n"
        "nodes:\n  review: {kind: call, call: sub}\noutput: ${review.output}\n"
    )
    with pytest.raises(ValueError, match="Agent 'review/draft' has no model for provider 'anthropic'"):
        run_flow(nested, {})


@pytest.mark.parametrize(
    ("env", "layer", "flow_config", "expected"),
    [
        ({}, {}, None, ("anthropic", "saved-anth")),
        ({"YIAN_DEFAULT_PROVIDER": "openai"}, {}, None, ("openai", "saved-oai")),
        (
            {"YIAN_DEFAULT_PROVIDER": "openai", "YIAN_DEFAULT_MODEL": "env-m"},
            {},
            None,
            ("openai", "env-m"),
        ),
        (
            {"YIAN_DEFAULT_PROVIDER": "openai", "YIAN_DEFAULT_MODEL": "env-m"},
            {"provider": "anthropic"},
            None,
            ("anthropic", "saved-anth"),
        ),
        (
            {"YIAN_DEFAULT_PROVIDER": "openai", "YIAN_DEFAULT_MODEL": "env-m"},
            {"model": "host-m"},
            None,
            ("openai", "host-m"),
        ),
        (
            {"YIAN_DEFAULT_PROVIDER": "openai", "YIAN_DEFAULT_MODEL": "env-m"},
            {"model": "host-m"},
            "{provider: anthropic, model: flow-m}",
            ("anthropic", "flow-m"),
        ),
    ],
)
def test_each_layer_wins_over_lower_layers(monkeypatch, env, layer, flow_config, expected):
    store = AuthStore()
    store.save("openai", ApiKeyCredentials("k1"), "saved-oai")
    store.save("anthropic", ApiKeyCredentials("k2"), "saved-anth")
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    config = _resolve(_flow(flow_config), layer)["write"]

    assert (config["provider"], config["model"]) == expected


def test_run_records_the_layer_that_wins(monkeypatch, tmp_path):
    store = AuthStore()
    store.save("openai", ApiKeyCredentials("k1"), "saved-oai")
    store.save("anthropic", ApiKeyCredentials("k2"), "saved-anth")
    monkeypatch.setenv("YIAN_DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("YIAN_DEFAULT_MODEL", "env-m")
    _fake_model(monkeypatch)
    log = tmp_path / "r.jsonl"

    result = run_flow(_flow(), {}, log=log, llm_config={"provider": "anthropic"})

    assert result.status == "succeeded"
    started = json.loads(log.read_text().splitlines()[0])
    assert started["kind"] == "run_started"
    recorded = started["payload"]["effective_llm_by_static_path"]["write"]
    assert (recorded["provider"], recorded["model"]) == ("anthropic", "saved-anth")


def test_default_model_without_provider_is_named_only_when_needed(monkeypatch, tmp_path):
    monkeypatch.setenv("YIAN_DEFAULT_MODEL", "m")

    config = _resolve(_flow("{provider: anthropic, model: x}"))["write"]
    assert (config["provider"], config["model"]) == ("anthropic", "x")

    _forbid_model(monkeypatch)
    with pytest.raises(ValueError) as caught:
        run_flow(_flow(), {}, log=tmp_path / "r.jsonl")
    message = str(caught.value)
    assert "Agent 'write'" in message
    assert "YIAN_DEFAULT_MODEL is set but names no provider" in message


def test_python_api_resolve_model_config_without_model():
    with pytest.raises(ValueError) as caught:
        resolve_model_config({})
    message = str(caught.value)
    assert message.startswith("No model is set")
    for fix in _FIXES:
        assert fix in message
