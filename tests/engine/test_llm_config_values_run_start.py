"""Run start checks the run's inputs before it resolves the llm cascade."""

import json

import pytest

from yian.compose import load_flow
from yian.nodes.contracts import Parameter
from yian.run import _run_start_inputs, run_flow
from yian.typesys.values import TypeCheckError

_FLOW = """id: f
name: f
input:
  topic: !str
  window: !int 30
nodes:
  agent:
    kind: agent
    input: {topic: "${input.topic}", window: "${input.window}"}
    prompt: "hi ${topic} ${window}"
output: ${agent.output}
"""

_GOOD_LLM = {"provider": "openai", "model": "gpt-5.5"}
_BAD_LLM = {"bogus": 1}
_BAD_LLM_ERROR = r"llm_config has unknown field\(s\) \['bogus'\]"


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    """Answer every model call with the prompt it was sent, and keep env keys out."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    calls = []

    class _Model:
        def invoke(self, messages):
            calls.append(messages)

            class _Reply:
                content = "ok"

            return _Reply()

    monkeypatch.setattr("yian.llm_clients.model_from_config", lambda cfg: _Model())
    return calls


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_an_unknown_input_is_reported_before_a_bad_llm_config(tmp_path):
    log = tmp_path / "run.jsonl"
    with pytest.raises(ValueError, match="unknown input 'nope'"):
        run_flow(load_flow(_FLOW), {"topic": "a", "nope": 1}, log=log, llm_config=_BAD_LLM)
    assert not log.exists()


def test_an_uncoercible_input_is_reported_before_a_bad_llm_config(tmp_path):
    log = tmp_path / "run.jsonl"
    with pytest.raises(TypeCheckError, match="input 'window': integer with 5000 digits"):
        run_flow(
            load_flow(_FLOW), {"topic": "a", "window": "1" * 5000},
            log=log, llm_config=_BAD_LLM,
        )
    assert not log.exists()


def test_a_bad_llm_config_alone_is_reported_as_before(tmp_path):
    log = tmp_path / "run.jsonl"
    with pytest.raises(ValueError, match=_BAD_LLM_ERROR):
        run_flow(load_flow(_FLOW), {"topic": "a"}, log=log, llm_config=_BAD_LLM)
    assert not log.exists()


def test_a_missing_required_input_stays_a_logged_start_failure(tmp_path, fake_model):
    log = tmp_path / "run.jsonl"
    result = run_flow(load_flow(_FLOW), {}, log=log, llm_config=_GOOD_LLM)
    assert result.status == "failed"
    records = _records(log)
    assert records[0]["kind"] == "run_started"
    failed = [r for r in records if r["kind"] == "node_failed"]
    assert [f["payload"]["failure"]["node_id"] for f in failed] == ["__start__"]
    failure = failed[0]["payload"]["failure"]
    assert failure["code"] == "binding.evaluate"
    assert "required input 'topic' is unbound" in failure["message"]
    assert fake_model == []


def test_a_missing_required_input_does_not_preempt_a_bad_llm_config(tmp_path):
    # A missing required input is START's logged failure, so the cascade's error,
    # raised before the log exists, is the one reported.
    log = tmp_path / "run.jsonl"
    with pytest.raises(ValueError, match=_BAD_LLM_ERROR):
        run_flow(load_flow(_FLOW), {}, log=log, llm_config=_BAD_LLM)
    assert not log.exists()


_GOLDEN_RUN_STARTED = {
    "kind": "run_started",
    "payload": {
        "effective_llm_by_static_path": {
            "agent": {
                "_connection": {
                    "credential_id": None,
                    "endpoint": "https://api.openai.com/v1",
                    "method": "api_key",
                    "source": "missing",
                    "store_path": None,
                },
                "model": "gpt-5.5",
                "provider": "openai",
            }
        },
        "flow_fingerprint": (
            "3d9b9bbd0808d27babce1bded77b3e90b642d179cc9678abcae0f4da66d4bfa7"
        ),
        "initial_pool": {
            "start_id": "__start__",
            "store": {"__start__": {"kind": "object", "value": {"topic": "ACME"}}},
            "system": {
                "now": "<now>",
                "run_id": {"kind": "str", "value": "golden"},
                "today": "<today>",
            },
        },
        "learnable": False,
        "memories": None,
        "memory_snapshots": None,
        "memory_stacks": None,
        "num_workers": 8,
        "start_wiring_override": {"topic": "${input.topic}"},
    },
    "run_id": "golden",
    "schema": 9,
    "seq": 0,
}


def test_a_run_with_a_default_input_records_and_returns_as_before(tmp_path, fake_model):
    log = tmp_path / "run.jsonl"
    result = run_flow(
        load_flow(_FLOW), {"topic": "ACME"}, log=log, run_id="golden", llm_config=_GOOD_LLM,
    )
    assert result.status == "succeeded"
    assert result.input == {"topic": "ACME"}
    assert result.output == "ok"
    # START still fills the default: the model sees window=30.
    assert "hi ACME 30" in str(fake_model[0])
    started = _records(log)[0]
    system = started["payload"]["initial_pool"]["system"]
    assert system["now"]["kind"] == "datetime" and system["today"]["kind"] == "date"
    system["now"], system["today"] = "<now>", "<today>"
    assert started == _GOLDEN_RUN_STARTED


def test_run_start_inputs_fill_valid_defaults_and_leave_bad_ones_to_start():
    params = [
        Parameter("topic", required=True),
        Parameter("window", has_default=True, default=30),
        Parameter("bad", has_default=True, default=float("nan")),
        Parameter("given", has_default=True, default="unused"),
    ]
    supplied = {"given": None}
    assert _run_start_inputs(params, supplied) == {"given": None, "window": 30}
    assert supplied == {"given": None}
