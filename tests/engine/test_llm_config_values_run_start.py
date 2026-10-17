"""Run start checks the run's inputs, then resolves and checks `llm_config` values reading them."""

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


def _effective(log):
    return _records(log)[0]["payload"]["effective_llm_by_static_path"]


def _models(log):
    return {path: (cfg["provider"], cfg["model"]) for path, cfg in _effective(log).items()}


_VALUES = """id: f
name: f
input:
INPUTS
nodes:
  chat:
    kind: agent
    prompt: Go.
    llm_config: LLM
output: ${chat.output}
"""


def _values(inputs: str, llm: str) -> str:
    return _VALUES.replace("INPUTS", inputs).replace("LLM", llm)


def test_provider_and_model_from_inputs_land_in_the_run_started_map(tmp_path, fake_model):
    text = _values("  p: !str\n  m: !str", "{provider: '${input.p}', model: '${input.m}'}")
    log = tmp_path / "run.jsonl"
    result = run_flow(load_flow(text), {"p": "anthropic", "m": "claude-x"}, log=log)
    assert result.status == "succeeded"
    assert _models(log) == {"chat": ("anthropic", "claude-x")}
    assert len(fake_model) == 1


def test_an_omitted_input_with_a_default_gives_its_default(tmp_path):
    text = _values("  m: !str gpt-from-default", "{model: '${input.m}'}")
    log = tmp_path / "run.jsonl"
    assert run_flow(load_flow(text), {}, log=log).status == "succeeded"
    assert _models(log) == {"chat": ("openai", "gpt-from-default")}


def test_an_omitted_optional_model_falls_through_to_the_saved_model(tmp_path):
    from yian.llm_clients.auth import ApiKeyCredentials, AuthStore

    AuthStore().save("anthropic", ApiKeyCredentials("saved-key"), "claude-saved")
    text = _values("  m: !Optional(str)", "{provider: anthropic, model: '${input.m}'}")
    log = tmp_path / "run.jsonl"
    assert run_flow(load_flow(text), {}, log=log).status == "succeeded"
    assert _models(log) == {"chat": ("anthropic", "claude-saved")}
    given = tmp_path / "given.jsonl"
    assert run_flow(load_flow(text), {"m": "claude-given"}, log=given).status == "succeeded"
    assert _models(given) == {"chat": ("anthropic", "claude-given")}


def test_a_missing_required_input_the_config_reads_fails_before_the_log(tmp_path, fake_model):
    text = _values("  m: !str", "{model: '${input.m}'}")
    log = tmp_path / "run.jsonl"
    with pytest.raises(ValueError, match=r"node 'chat' llm_config\.model: .*input 'm'"):
        run_flow(load_flow(text), {}, log=log)
    assert not log.exists()
    assert fake_model == []


def test_an_input_the_config_reads_is_checked_against_its_contract(tmp_path):
    text = _values("  t: !float", "{openai: {temperature: '${input.t}'}}")
    log = tmp_path / "run.jsonl"
    with pytest.raises(
        ValueError, match=r"node 'chat' llm_config\.openai\.temperature: input 't'"
    ):
        run_flow(load_flow(text), {"t": "warm"}, log=log)
    assert not log.exists()


_WHOLE_RECORD = """id: f
name: f
typedefs:
  Cfg:
    provider: str
    model: str
input:
  cfg: !Cfg
llm_config: ${input.cfg}
defs:
  child:
    nodes:
      inner: {kind: agent, prompt: Inner.}
    output: ${inner.output}
nodes:
  chat:
    kind: agent
    prompt: Go.
  c:
    kind: call
    call: child
    input: {}
    depends_on: [chat]
output: ${c.output}
"""


def test_a_flow_level_whole_record_from_an_input_reaches_every_agent(tmp_path):
    log = tmp_path / "run.jsonl"
    cfg = {"provider": "anthropic", "model": "claude-whole"}
    assert run_flow(load_flow(_WHOLE_RECORD), {"cfg": cfg}, log=log).status == "succeeded"
    assert _models(log) == {
        "chat": ("anthropic", "claude-whole"), "c/inner": ("anthropic", "claude-whole"),
    }


_LIB = """id: lib
name: lib
input: {m: !str }
llm_config: {model: '${input.m}'}
defs:
  inner:
    nodes:
      a: {kind: agent, prompt: Go.}
    output: ${a.output}
nodes:
  c: {kind: call, call: inner, input: {}}
output: ${c.output}
"""


def test_a_file_with_defs_loaded_as_the_root_resolves_its_input_value(tmp_path):
    log = tmp_path / "run.jsonl"
    flow = load_flow(_LIB, source_label="lib.yaml")
    assert run_flow(flow, {"m": "gpt-lib"}, log=log).status == "succeeded"
    assert _models(log) == {"c/a": ("openai", "gpt-lib")}


_PASS_DOWN = """id: f
name: f
input: {m: !str }
llm_config: {provider: anthropic, model: '${input.m}'}
defs:
  child:
    nodes:
      a: {kind: agent, prompt: Go.}
      b: {kind: agent, prompt: Go., llm_config: {model: own-model}}
    output: ${a.output}
nodes:
  top: {kind: agent, prompt: Top.}
  c:
    kind: call
    call: child
    input: {}
    depends_on: [top]
output: ${c.output}
"""


def test_a_root_flow_input_value_is_passed_down_to_child_agents(tmp_path):
    log = tmp_path / "run.jsonl"
    assert run_flow(load_flow(_PASS_DOWN), {"m": "claude-root"}, log=log).status == "succeeded"
    assert _models(log) == {
        "top": ("anthropic", "claude-root"),
        "c/a": ("anthropic", "claude-root"),
        "c/b": ("anthropic", "own-model"),
    }


def test_a_run_start_agent_binds_no_reserved_llm_config_input(tmp_path):
    text = _values("  m: !str", "{model: '${input.m}'}")
    flow = load_flow(text)
    assert "__llm_config" in flow.compiled.wiring["chat"]
    log = tmp_path / "run.jsonl"
    assert run_flow(flow, {"m": "gpt-learn"}, log=log, learnable=True).status == "succeeded"
    succeeded = [
        r for r in _records(log)
        if r["kind"] == "node_succeeded" and r["payload"]["node_id"] == "chat"
    ]
    capture = succeeded[0]["payload"]["forward_capture"]
    assert capture is not None
    assert "__llm_config" not in capture["bound_inputs"]
    assert "__llm_config" not in capture["input_names"]
    assert capture["node_data"]["effective_llm_config"]["model"] == "gpt-learn"


_MIXED = """id: f
name: f
input: {p: !str }
llm_config: {provider: '${input.p}', model: '${pick.output}', openai: {temperature: 0.5}}
defs:
  child:
    nodes:
      a: {kind: agent, prompt: Go., llm_config: {model: child-model}}
    output: ${a.output}
nodes:
  pick:
    kind: agent
    prompt: Pick.
    output: str
    llm_config: {model: pick-model}
  c:
    kind: call
    call: child
    input: {}
    depends_on: [pick]
output: ${c.output}
"""


def test_a_mixed_flow_layer_passes_its_literal_and_input_fields_and_drops_the_rest(tmp_path):
    # Every agent sets its own model, so the flow's run-time model reaches none of them.
    log = tmp_path / "run.jsonl"
    assert run_flow(load_flow(_MIXED), {"p": "openai"}, log=log).status == "succeeded"
    effective = _effective(log)
    assert _models(log) == {"pick": ("openai", "pick-model"), "c/a": ("openai", "child-model")}
    assert effective["c/a"]["openai"] == {"temperature": 0.5}
    assert effective["pick"]["openai"] == {"temperature": 0.5}


def _fails_before_any_node(text, inputs, match, tmp_path, fake_model):
    """Run ``text`` and expect run start to fail with ``match``, with no log and no node run."""
    log = tmp_path / "failed.jsonl"
    events = []
    with pytest.raises(ValueError, match=match) as caught:
        run_flow(load_flow(text), inputs, log=log, on_event=events.append)
    assert not log.exists()
    assert events == []
    assert fake_model == []
    return str(caught.value)


def test_a_provider_from_an_input_is_checked_against_its_connection(
    tmp_path, monkeypatch, fake_model
):
    import time

    from yian.llm_clients.auth import AuthStore, OAuthCredentials

    auth = tmp_path / "auth"
    auth.mkdir(mode=0o700)
    monkeypatch.setenv("YIAN_AUTH_FILE", str(auth / "auth.json"))
    AuthStore().save(
        "openai",
        OAuthCredentials("access-secret", "refresh-secret", time.time() + 3600, "account"),
        "gpt-5.5",
    )
    text = _values("  p: !str", "{provider: '${input.p}', openai: {temperature: 0.2}}")
    message = _fails_before_any_node(
        text, {"p": "openai"},
        r"^node 'chat' llm_config\.openai: `openai\.temperature` is not accepted by the "
        r"ChatGPT subscription endpoint",
        tmp_path, fake_model,
    )
    assert "secret" not in message


def test_a_whole_block_from_an_input_that_breaks_the_schema_fails_at_run_start(
    tmp_path, fake_model
):
    text = _values("  blk: !object", "{openai: '${input.blk}'}")
    _fails_before_any_node(
        text, {"blk": {"bogus": 1}},
        r"^node 'chat' llm_config\.openai: `openai\.bogus` is not a supported field",
        tmp_path, fake_model,
    )


def test_a_block_field_from_an_input_out_of_range_fails_at_its_field(tmp_path, fake_model):
    text = _values(
        "  b: !int", "{provider: anthropic, model: m, anthropic: {thinking: "
        "{type: enabled, budget_tokens: '${input.b}'}}}",
    )
    _fails_before_any_node(
        text, {"b": 5}, r"^node 'chat' llm_config\.anthropic: .*budget_tokens.*1024",
        tmp_path, fake_model,
    )


def test_an_unknown_provider_from_an_input_fails_naming_the_field(tmp_path, fake_model):
    text = _values("  p: !str", "{provider: '${input.p}'}")
    message = _fails_before_any_node(
        text, {"p": "bogus-provider"},
        r"^node 'chat' llm_config\.provider: not a known provider; use one of: openai, ",
        tmp_path, fake_model,
    )
    assert "bogus-provider" not in message


def test_a_literal_typed_provider_given_an_unknown_member_fails_at_run_start(
    tmp_path, fake_model
):
    text = _values("  p: !Some", "{provider: '${input.p}'}").replace(
        "input:", "typedefs:\n  Some: Literal(openai, nope)\ninput:"
    )
    assert run_flow(load_flow(text), {"p": "openai"}).status == "succeeded"
    fake_model.clear()
    _fails_before_any_node(
        text, {"p": "nope"}, r"^node 'chat' llm_config\.provider: not a known provider",
        tmp_path, fake_model,
    )


def test_a_flow_level_unknown_provider_from_an_input_fails_at_the_flow_field(
    tmp_path, fake_model
):
    text = _PASS_DOWN.replace("provider: anthropic", "provider: '${input.p}'").replace(
        "input: {m: !str }", "input: {m: !str , p: !str }"
    )
    _fails_before_any_node(
        text, {"m": "x", "p": "nope"}, r"^flow llm_config\.provider: not a known provider",
        tmp_path, fake_model,
    )


def test_an_unknown_field_in_a_whole_record_from_an_input_fails_at_run_start(
    tmp_path, fake_model
):
    text = _values("  cfg: !object", "'${input.cfg}'")
    _fails_before_any_node(
        text, {"cfg": {"model": "m", "bogus": 1}},
        r"^node 'chat' llm_config: llm_config has unknown field\(s\) \['bogus'\]",
        tmp_path, fake_model,
    )


_COMPOSER = """id: f
name: f
input: {proj: !str , m: !str , p: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: '${input.proj}'}
    adaptive_questions:
      prompt: "Q ${proj}"
      llm_config: {provider: '${input.p}', model: '${input.m}'}
output: ${ask.output}
"""


def test_a_composer_value_from_an_input_resolves_at_run_start(tmp_path, fake_model):
    from yian.compile.llm_cascade import resolve_run_llm

    flow = load_flow(_COMPOSER).compiled
    composer = next(
        node_id for node_id, node in flow.nodes.items()
        if getattr(node, "llm_config_label", None) is not None
    )
    effective = resolve_run_llm(
        flow, {}, defaults={}, run_inputs={"proj": "x", "m": "claude-c", "p": "anthropic"},
    ).effective
    assert effective[composer]["provider"] == "anthropic"
    assert effective[composer]["model"] == "claude-c"


def test_a_composer_value_error_names_the_gate(tmp_path, fake_model):
    site = r"node 'ask' adaptive_questions\.llm_config"
    _fails_before_any_node(
        _COMPOSER, {"proj": "x", "p": "openai"},
        rf"^{site}\.model: input 'm' is required and was not given", tmp_path, fake_model,
    )
    _fails_before_any_node(
        _COMPOSER, {"proj": "x", "m": "m", "p": "nope"},
        rf"^{site}\.provider: not a known provider", tmp_path, fake_model,
    )
