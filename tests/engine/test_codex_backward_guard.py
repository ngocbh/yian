"""Reject backward effects when accepted execution used the Codex provider."""

from copy import deepcopy

from langchain_core.messages import AIMessage
import pytest

import agent_composer.llm_clients as llm_clients
import agent_composer.run as run_module
from agent_composer.compose import load_flow
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import Node
from agent_composer.nodes.code import CodeNode
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.loop import LoopNode
from agent_composer.nodes.start import StartNode
from agent_composer.run import _replay_run_log, resume_flow, run_flow
from agent_composer.runtime.log import (
    BackwardStartedRecord,
    JsonlRunLog,
    MemoryRunLog,
    NodeFailedRecord,
    NodeSucceededRecord,
)
from agent_composer.runtime.state import commit_prepared, prepare_record
from tests.engine._memory_fakes import _Frame, _Memory, _Session
from tests.engine.test_agent_backward import _BackwardChat, _reply

_MIXED = """id: mixed-providers
name: mixed_providers
llm_config: {provider: ollama, model: offline-test}
input:
  text: str
nodes:
  native:
    kind: agent
    input: {text: '${input.text}'}
    prompt: ${text}
  api:
    kind: agent
    input: {text: '${input.text}'}
    prompt: ${text}
output:
  native: ${native.output}
  api: ${api.output}
"""


def _mixed_flow(*, provider="codex", memory=False):
    source = _MIXED
    if memory:
        source = source.replace(
            "name: mixed_providers", "name: mixed_providers\nlong_term_memory: test"
        )
    flow = load_flow(source)
    # Native selection is still private; these tests exercise recorded policy
    # through programmatic authored config and an offline model factory.
    flow.compiled.nodes["native"].own_llm_config = {
        "provider": provider,
        "model": "offline-test",
    }
    return flow


def _forward_model(monkeypatch, reply=None):
    configs = []

    def factory(config):
        configs.append(deepcopy(config))
        return _BackwardChat([AIMessage(content="answer") if reply is None else reply])

    monkeypatch.setattr(llm_clients, "model_from_config", factory)
    return configs


def _forbid_effects(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "rejected backward invoked a callback, opened memory, or appended a record"
        )

    monkeypatch.setattr(llm_clients, "model_from_config", forbidden)
    for node_type in (Node, AgentNode, StartNode, EndNode, LoopNode):
        monkeypatch.setattr(node_type, "backward", forbidden)
    monkeypatch.setattr(AgentNode, "forward", forbidden)
    monkeypatch.setattr(_Memory, "open_run", forbidden)
    monkeypatch.setattr(_Frame, "learn", forbidden)
    monkeypatch.setattr(_Session, "commit", forbidden)
    monkeypatch.setattr(MemoryRunLog, "append", forbidden)
    monkeypatch.setattr(JsonlRunLog, "append", forbidden)


def _assert_rejected_unchanged(result, feedback):
    forward = deepcopy(
        (result.input, result.status, result.output, result.failure, result.events)
    )
    previous, learning = result.backward_outcome, result.learning
    handle = result._backward_handle
    before = deepcopy(handle.records) if result.log is None else result.log.read_bytes()
    for _ in range(2):
        with pytest.raises(ValueError, match="Codex.*forward-only"):
            result.backward(feedback)
        assert (
            result.input,
            result.status,
            result.output,
            result.failure,
            result.events,
        ) == forward
        assert result.backward_outcome is previous and result.learning is learning
        assert not handle._unusable
        if result.log is None:
            assert handle.records == before
        else:
            assert result.log.read_bytes() == before
            with JsonlRunLog.resume(result.log):
                pass


@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize(
    "feedback",
    [None, {}, {"native": None, "api": None}, {"api": "Review API only."}, ""],
)
def test_mixed_codex_run_rejects_the_whole_pass_before_effects(
    tmp_path, monkeypatch, durable, memory, feedback
):
    flow = _mixed_flow(memory=memory)
    configs = _forward_model(monkeypatch)
    backend = _Memory()
    result = run_flow(
        flow,
        {"text": "hello"},
        learnable=True,
        log=tmp_path / "mixed.jsonl" if durable else None,
        memory={"test": backend} if memory else None,
    )
    assert result.status == "succeeded" and result.output == {
        "native": "answer",
        "api": "answer",
    }
    assert sorted(config["provider"] for config in configs) == ["codex", "ollama"]
    assert backend.learn_contexts == backend.commit_calls == []
    assert backend.close_count == int(memory)
    _forbid_effects(monkeypatch)
    _assert_rejected_unchanged(result, feedback)
    assert len(backend.open_snapshots) == int(memory)


def test_terminal_resume_remains_inspection_only_and_retains_the_guard(
    tmp_path, monkeypatch
):
    flow = _mixed_flow(provider="CoDeX", memory=True)
    _forward_model(monkeypatch)
    backend = _Memory()
    result = run_flow(
        flow,
        {"text": "hello"},
        log=tmp_path / "resume.jsonl",
        memory={"test": backend},
        learnable=True,
    )
    prefix = result.log.read_bytes()
    _forbid_effects(monkeypatch)
    inspected = resume_flow(flow, log=result.log)
    assert inspected.status == result.status and inspected.output == result.output
    assert (
        inspected.events == []
        and inspected.backward_outcome is inspected.learning is None
    )
    assert result.log.read_bytes() == prefix
    _assert_rejected_unchanged(inspected, None)
    assert len(backend.open_snapshots) == backend.close_count == 1


def test_rejection_preserves_an_observed_historical_backward_attempt(
    tmp_path, monkeypatch
):
    flow = _mixed_flow()
    _forward_model(monkeypatch)
    result = run_flow(
        flow, {"text": "hello"}, log=tmp_path / "attempt.jsonl", learnable=True
    )
    with JsonlRunLog.resume(result.log) as log:
        state = _replay_run_log(flow, log)
        commit_prepared(
            log, state, prepare_record(state, BackwardStartedRecord(1, None))
        )
    _forbid_effects(monkeypatch)
    inspected = resume_flow(flow, log=result.log)
    assert inspected.backward_outcome.failure.code == "backward.interrupted"
    _assert_rejected_unchanged(inspected, "Retry review.")


@pytest.mark.parametrize(
    "recorded_provider,live_provider", [("codex", "ollama"), ("ollama", "codex")]
)
def test_recorded_provider_controls_backward_after_live_config_mutation(
    monkeypatch, recorded_provider, live_provider
):
    flow = _mixed_flow(provider=recorded_provider)
    _forward_model(monkeypatch)
    result = run_flow(flow, {"text": "hello"}, learnable=True)
    flow.compiled.nodes["native"].own_llm_config["provider"] = live_provider
    flow.compiled.nodes["native"].llm_config = {
        "provider": live_provider,
        "model": "mutated",
    }
    result._backward_handle.program.nodes["native"].llm_config = {
        "provider": live_provider,
        "model": "mutated",
    }
    if recorded_provider == "codex":
        _forbid_effects(monkeypatch)
        _assert_rejected_unchanged(result, "Review.")
    else:
        configs = _forward_model(
            monkeypatch, _reply(input_feedback={"text": ["Refine."]})
        )
        outcome = result.backward({"native": "Review."})
        assert outcome.status == "succeeded"
        assert outcome.feedback.input_feedback == {"text": ("Refine.",)}
        assert configs == [{"provider": "ollama", "model": "offline-test"}]


def test_accepted_failed_codex_invocation_also_blocks_backward(monkeypatch):
    flow = _mixed_flow()
    _forward_model(monkeypatch, RuntimeError("forward provider failed"))
    result = run_flow(flow, {"text": "hello"}, learnable=True)
    assert result.status == "failed"
    failures = [
        envelope.record
        for envelope in result._backward_handle.records
        if type(envelope.record) is NodeFailedRecord
    ]
    assert len(failures) == 1 and failures[0].forward_capture.scope.static_path == (
        "native",
    )
    _forbid_effects(monkeypatch)
    _assert_rejected_unchanged(result, "Explain failure.")


def test_unexecuted_codex_branch_preserves_api_backward_and_memory_learning(
    monkeypatch,
):
    flow = load_flow("""id: selected-api
name: selected_api
llm_config: {provider: ollama, model: offline-test}
long_term_memory: test
input: {text: str}
nodes:
  gate:
    kind: case
    cases:
      - when: 1 == 1
        then: api
    else: native
  api:
    kind: agent
    input: {text: '${input.text}'}
    prompt: ${text}
  native:
    kind: agent
    prompt: unused
output: ${api.output | native.output}
""")
    flow.compiled.nodes["native"].own_llm_config = {
        "provider": "codex",
        "model": "offline-test",
    }
    configs = _forward_model(monkeypatch)
    backend = _Memory()
    result = run_flow(flow, {"text": "hello"}, memory={"test": backend}, learnable=True)
    assert result.status == "succeeded" and result.output == "answer"
    assert [config["provider"] for config in configs] == ["ollama"]
    configs = _forward_model(monkeypatch, _reply(input_feedback={"text": ["Refine."]}))
    outcome = result.backward("Review API.")
    assert outcome.status == "succeeded"
    assert outcome.feedback.input_feedback == {"text": ("Refine.",)}
    assert configs == [{"provider": "ollama", "model": "offline-test"}]
    assert len(backend.open_snapshots) == backend.close_count == 2
    assert len(backend.learn_contexts) == len(backend.commit_calls) == 1


def test_codex_looking_custom_capture_is_not_provider_authority(monkeypatch):
    flow = load_flow("""id: custom-capture
name: custom_capture
nodes:
  value:
    kind: code
    code: return 'answer'
output: ${value.output}
""")
    monkeypatch.setattr(
        CodeNode,
        "capture_backward_context",
        lambda self: {
            "effective_llm_config": {"provider": "codex", "model": "not-authority"}
        },
    )
    result = run_flow(flow, {}, learnable=True)
    assert result.status == "succeeded"
    assert result._backward_handle.records[0].record.effective_llm_by_static_path == {}
    assert result.backward("Review.").status == "succeeded"


def test_absent_recorded_execution_scope_preserves_existing_backward(monkeypatch):
    monkeypatch.setattr(run_module, "apply_execution_layout", lambda *_args: None)
    _forward_model(monkeypatch)
    result = run_flow(_mixed_flow(), {"text": "hello"}, learnable=True)
    assert result.status == "succeeded"
    records = result._backward_handle.records
    assert (
        records[0].record.effective_llm_by_static_path["native"]["provider"] == "codex"
    )
    captures = [
        item.record.forward_capture
        for item in records
        if type(item.record) is NodeSucceededRecord
    ]
    assert captures and all(capture.scope is None for capture in captures)
    assert result.backward().status == "succeeded"


def test_nested_pruned_codex_history_blocks_after_terminal_resume(
    tmp_path, monkeypatch
):
    flow = load_flow("""id: nested-codex
name: nested_codex
llm_config: {provider: ollama, model: offline-test}
defs:
  body:
    input: {n: int}
    nodes:
      gate:
        kind: case
        cases:
          - when: input.n == 0
            then: native
        else: local
      native:
        kind: agent
        prompt: Produce the next value.
        output: {n: int}
      local:
        kind: code
        code: "return {'n': 2}"
        output: {n: int}
      joined:
        kind: code
        input: {n: '${native.output.n | local.output.n}'}
        code: return inputs
        output: {n: int}
    output: ${joined.output}
  item:
    input: {n: int}
    nodes:
      loop:
        kind: loop
        call: body
        times: 2
        input: {n: '${input.n}'}
    output: ${loop.output}
  batch:
    nodes:
      many:
        kind: map
        call: item
        over: [0, 0]
        input: {n: '${item}'}
    output: ${many.output}
nodes:
  outer:
    kind: call
    call: batch
output: ${outer.output}
""")
    native = (
        flow.compiled.nodes["outer"]
        .child.nodes["many"]
        .child.nodes["loop"]
        .child.nodes["native"]
    )
    native.own_llm_config = {"provider": "codex", "model": "no-structured-sentinel"}
    configs = _forward_model(monkeypatch, AIMessage(content='{"n": 1}'))
    result = run_flow(flow, {}, log=tmp_path / "nested.jsonl", learnable=True)
    assert result.status == "succeeded", result.failure
    assert result.output == [{"n": 2}, {"n": 2}]
    assert [config["provider"] for config in configs] == ["codex", "codex"]
    with JsonlRunLog.resume(result.log) as log:
        state = _replay_run_log(flow, log)
    accepted = [
        item
        for item in state.trace.occurrences
        if item.record.forward_capture.scope.static_path
        == ("outer", "many", "loop", "native")
    ]
    assert len(accepted) == 2
    assert all(item.node_id not in state.flow.nodes for item in accepted)
    for item in accepted:
        scope = item.record.forward_capture.scope
        assert [position.kind for position in scope.occurrence_path] == [
            "call",
            "map_item",
            "loop_body",
        ]
        assert (
            state.effective_llm_by_static_path["/".join(scope.static_path)]["provider"]
            == "codex"
        )
    _forbid_effects(monkeypatch)
    inspected = resume_flow(flow, log=result.log)
    assert inspected.output == result.output and inspected.events == []
    _assert_rejected_unchanged(inspected, None)
