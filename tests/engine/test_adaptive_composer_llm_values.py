"""`llm_config` values on an adaptive gate's composer, end to end: run start, call time, errors."""

import json

import pytest

from yian.compose import load_flow
from yian.run import run_flow

_SITE = "node 'ask' adaptive_questions.llm_config"
_COMPOSER = "ask__compose"

_COMPOSED = [{
    "question": "Which database?",
    "header": "DB",
    "options": [
        {"label": "Postgres", "description": "Relational."},
        {"label": "SQLite", "description": "Embedded."},
    ],
    "multi_select": False,
}]


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    """Record the config of each model built; each composes ``_COMPOSED``."""
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    built = []

    class Chat:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise AssertionError("structured path expected")

        def with_structured_output(self, schema):
            class Bound:
                def invoke(self, messages):
                    return schema.model_validate({"items": _COMPOSED})

            return Bound()

    def model_from_config(config):
        built.append(config)
        return Chat()

    monkeypatch.setattr("yian.llm_clients.model_from_config", model_from_config)
    return built


def _gate(llm_config, *, inputs="{proj: !str }", typedefs="", pick=""):
    """A flow whose ``ask`` gate composes its questions with ``llm_config``."""
    return f"""id: f
name: f
{typedefs}input: {inputs}
nodes:
{pick}  ask:
    kind: human_input
    input: {{proj: '${{input.proj}}'}}
    adaptive_questions:
      prompt: "Q ${{proj}}"
      llm_config: {llm_config}
output: ${{ask.output}}
"""


def _run(text, inputs, tmp_path):
    log = tmp_path / "run.jsonl"
    requests = []
    result = run_flow(
        load_flow(text), inputs, log=log,
        tools={"ask_user": lambda **request: requests.append(request) or {"DB": "Postgres"}},
    )
    records = [json.loads(line) for line in log.read_text().splitlines()]
    return result, records, requests


def _composer_steps(records):
    return [
        record["payload"]["step"] for record in records
        if record["kind"] == "node_step" and record["payload"]["node_id"] == _COMPOSER
    ]


def test_a_composer_model_from_an_input_resolves_at_run_start(tmp_path, fake_model):
    text = _gate("{provider: openai, model: '${input.m}'}", inputs="{proj: !str , m: !str }")
    result, records, requests = _run(text, {"proj": "x", "m": "gpt-input"}, tmp_path)
    assert result.status == "succeeded", result.failure
    assert result.output == {"DB": "Postgres"}
    assert requests[0]["questions"][0]["header"] == "DB"
    started = records[0]["payload"]
    entry = started["effective_llm_by_static_path"][_COMPOSER]
    assert (entry["provider"], entry["model"]) == ("openai", "gpt-input")
    assert "call_time_llm" not in started
    assert _composer_steps(records) == []
    [config] = fake_model
    assert (config["provider"], config["model"]) == ("openai", "gpt-input")


_PICK = "  pick: {kind: code, output: str, code: \"return 'gpt-picked'\"}\n"


def test_a_composer_model_from_an_upstream_node_resolves_at_call_time(tmp_path, fake_model):
    text = _gate("{provider: openai, model: '${pick.output}'}", pick=_PICK)
    result, records, requests = _run(text, {"proj": "x"}, tmp_path)
    assert result.status == "succeeded", result.failure
    assert requests[0]["questions"][0]["question"] == "Which database?"
    started = records[0]["payload"]
    assert _COMPOSER not in started["effective_llm_by_static_path"]
    assert _COMPOSER in started["call_time_llm"]["agents"]
    [step] = _composer_steps(records)
    recorded = step["effective_llm_config"]
    assert (recorded["provider"], recorded["model"]) == ("openai", "gpt-picked")
    [config] = fake_model
    assert (config["provider"], config["model"]) == ("openai", "gpt-picked")


_CFG = "typedefs:\n  Cfg: {provider: str, model: str}\n"


def test_a_composer_whole_record_from_an_input_resolves_at_run_start(tmp_path, fake_model):
    text = _gate("'${input.cfg}'", inputs="{proj: !str , cfg: !Cfg }", typedefs=_CFG)
    cfg = {"provider": "anthropic", "model": "claude-input"}
    result, records, _requests = _run(text, {"proj": "x", "cfg": cfg}, tmp_path)
    assert result.status == "succeeded", result.failure
    entry = records[0]["payload"]["effective_llm_by_static_path"][_COMPOSER]
    assert (entry["provider"], entry["model"]) == ("anthropic", "claude-input")
    [config] = fake_model
    assert (config["provider"], config["model"]) == ("anthropic", "claude-input")


def test_a_composer_whole_record_from_an_upstream_node_resolves_at_call_time(
    tmp_path, fake_model,
):
    pick = "  pick: {kind: code, output: Cfg, code: \"return {'provider': 'openai', " \
        "'model': 'whole'}\"}\n"
    text = _gate("!Cfg ${pick.output}", typedefs=_CFG, pick=pick)
    result, records, _requests = _run(text, {"proj": "x"}, tmp_path)
    assert result.status == "succeeded", result.failure
    [step] = _composer_steps(records)
    assert step["effective_llm_config"]["model"] == "whole"
    [config] = fake_model
    assert (config["provider"], config["model"]) == ("openai", "whole")


def test_a_composer_run_start_error_names_the_gate(tmp_path, fake_model):
    text = _gate("{provider: '${input.p}', model: m}", inputs="{proj: !str , p: !str }")
    with pytest.raises(ValueError) as caught:
        run_flow(load_flow(text), {"proj": "x", "p": "nope"}, log=tmp_path / "run.jsonl")
    message = str(caught.value)
    assert message.startswith(f"{_SITE}.provider: not a known provider")
    assert _COMPOSER not in message
    assert fake_model == []


def test_a_composer_call_time_error_names_the_gate(fake_model):
    pick = "  pick: {kind: code, output: str, code: \"return 'nope'\"}\n"
    text = _gate("{provider: '${pick.output}', model: m}", pick=pick)
    result = run_flow(load_flow(text), {"proj": "x"})
    assert result.status == "failed"
    assert result.failure.node_id == _COMPOSER
    assert result.failure.code == "contract.input"
    assert result.failure.message.startswith(f"{_SITE}.provider: not a known provider")
    assert fake_model == []


def test_a_composer_whole_record_error_names_the_gate_field(fake_model):
    typedefs = "typedefs:\n  Cfg: {provider: str, model: str, openai: object}\n"
    pick = "  pick: {kind: code, output: Cfg, code: \"return {'provider': 'openai', " \
        "'model': 'm', 'openai': {'temperature': 9.0}}\"}\n"
    text = _gate("!Cfg ${pick.output}", typedefs=typedefs, pick=pick)
    result = run_flow(load_flow(text), {"proj": "x"})
    assert result.status == "failed"
    assert result.failure.code == "contract.input"
    assert result.failure.message.startswith(f"{_SITE}.openai.temperature: ")
    assert fake_model == []


def test_a_composer_call_time_error_in_a_defs_call_names_the_gate_and_callsite(fake_model):
    text = """id: f
name: f
defs:
  child:
    input: {p: !str }
    nodes:
      ask:
        kind: human_input
        adaptive_questions:
          prompt: "Q"
          llm_config: {provider: '${input.p}', model: m}
    output: ${ask.output}
nodes:
  pick: {kind: code, output: str, code: "return 'nope'"}
  sub: {kind: call, call: child, input: {p: '${pick.output}'}}
output: ${sub.output}
"""
    result = run_flow(load_flow(text), {})
    assert result.status == "failed"
    assert result.failure.code == "contract.input"
    assert result.failure.message.startswith(f"{_SITE}.provider in 'sub': not a known provider")
    assert fake_model == []


def test_a_composer_value_that_breaks_its_tag_in_a_defs_call_names_the_field_and_callsite(
    fake_model,
):
    text = """id: f
name: f
llm_config: {provider: openai}
defs:
  child:
    input: {m: !Optional(str) }
    nodes:
      ask:
        kind: human_input
        adaptive_questions:
          prompt: "Q"
          llm_config: {model: !str '${input.m}'}
    output: ${ask.output}
nodes:
  pick: {kind: code, output: Optional(str), code: "return None"}
  sub: {kind: call, call: child, input: {m: '${pick.output}'}}
output: ${sub.output}
"""
    result = run_flow(load_flow(text), {})
    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.message.startswith(f"{_SITE}.model in 'sub': "), result.failure.message
    assert fake_model == []
