"""Invocation-local AGENT evidence from evaluation through durable acceptance."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import threading
import time

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
import pytest

import yian.llm_clients as llm_clients
import yian.tools as tools
from yian.compile.model import END_ID, START_ID, GrowthRejected
from yian.memory import (
    DeclSite,
    EvaluationEvidence,
    EvaluationTrace,
    MemoryContext,
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryScope,
    MemorySnapshotEnvelope,
    MemorySpec,
    MemoryStack,
    ModelTextInteraction,
    ModelValueInteraction,
    ToolCallInteraction,
    ToolResultInteraction,
)
from yian.nodes.agent import AgentNode
from yian.nodes.agent.context import DEFAULT_SYSTEM
from yian.nodes.base import Grow, Node, NodeKind, Output, Route
from yian.nodes.contracts import Parameter, ValueContract
from yian.runtime.eval_node import (
    EvaluatedOutcome,
    EvaluationFailure,
    eval_node,
)
from yian.runtime.log import (
    NodeFailedRecord,
    NodeSucceededRecord,
)
from yian.runtime.state import (
    commit_prepared,
    prepare_node_started,
    prepare_node_succeeded,
    prepare_record,
)
from yian.typesys import Type, ValueKind, VariablePool
from tests.engine._fakes import FuncNode
from tests.engine._graph_builder import _forward_capture, _engine, _graph


class _StringSubclass(str):
    """A valid provider string whose carrier type is not exact ``str``."""


def _scope(
    node_id: str,
    *,
    occurrence: tuple[MemoryOccurrenceSegment, ...] = (),
) -> MemoryScope:
    """Return one valid flow-plus-agent key scope for a test node."""
    return MemoryScope(("flow", node_id), ("outer", "local"), occurrence)


def _occurrence(index: int) -> tuple[MemoryOccurrenceSegment, ...]:
    return (MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ("flow",), index),)


def _learning_configuration(
    *nodes: Node,
) -> tuple[
    tuple[MemorySpec, ...],
    tuple[MemoryStack, ...],
    tuple[MemorySnapshotEnvelope, ...],
]:
    """Return one canonical root-owned key table, stacks and snapshots for the nodes' scopes."""
    scopes = [node.memory_scope for node in nodes if type(node.memory_scope) is MemoryScope]
    keys = sorted({key for scope in scopes for key in scope.memories})
    memories = tuple(
        MemorySpec(key, key, {}, (DeclSite((), ("root",), None),), False) for key in keys
    )
    by_locator = {scope.static_path: MemoryStack(scope.static_path, scope.memories) for scope in scopes}
    stacks = tuple(by_locator[locator] for locator in sorted(by_locator))
    return memories, stacks, tuple(MemorySnapshotEnvelope(key, {}) for key in keys)


@contextmanager
def _empty_context(_scope, _request, *, check_stopped):
    """Represent enabled memory whose policies recalled no visible sections."""
    yield MemoryContext()


class _PlainChat:
    """Return one visible free-text response and retain no provider state."""

    def __init__(self, text: str) -> None:
        self.text = text

    def invoke(self, _messages):
        return AIMessage(
            content=self.text,
            additional_kwargs={"reasoning": "private"},
            response_metadata={"usage": {"tokens": 99}},
        )


class _SubclassTextChat:
    """Return visible content as a valid string subclass."""

    def invoke(self, _messages):
        return SimpleNamespace(content=_StringSubclass("subclass answer"))


class _SurrogateTextChat:
    """Return provider text that has no valid canonical UTF-8 representation."""

    def invoke(self, _messages):
        return SimpleNamespace(content=chr(0xD800))


class _NativeStructuredChat:
    """Return one parsed value through LangChain's native structured seam."""

    def __init__(self, value: dict) -> None:
        self.value = value

    def invoke(self, _messages):
        raise AssertionError("native structured output must not invoke the raw model")

    def with_structured_output(self, schema):
        value = self.value

        class _Bound:
            def invoke(self, _messages):
                return schema.model_validate(value)

        return _Bound()


class _FallbackStructuredChat:
    """Return one invalid and then one valid visible fallback response."""

    def __init__(self) -> None:
        self.replies = ["not json", '{"value": 7}']

    def invoke(self, _messages):
        return AIMessage(content=self.replies.pop(0))


class _FallbackContentBlockChat:
    """Return JSON text beside provider-private reasoning and metadata blocks."""

    def invoke(self, _messages):
        return SimpleNamespace(
            content=[
                {"type": "reasoning", "reasoning": "SECRET_REASONING"},
                {"type": "text", "text": '{"value": 13}'},
                {"type": "metadata", "value": "SECRET_BLOCK_METADATA"},
            ],
            response_metadata={"private": "SECRET_PROVIDER_METADATA"},
        )


class _ToolChat:
    """Run one tool turn and then return one final visible answer."""

    def __init__(self) -> None:
        self.replies = [
            AIMessage(
                content="private plan",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {"topic": "ACME"},
                        "id": "provider-call-id",
                        "type": "tool_call",
                    }
                ],
                response_metadata={"usage": {"tokens": 5}},
            ),
            AIMessage(content="finished", response_metadata={"model": "secret"}),
        ]

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return self.replies.pop(0)


class _StructuredToolChat:
    """Run a tool discovery loop followed by native or fallback generation."""

    def __init__(self, *, native: bool) -> None:
        self.replies = [
            AIMessage(
                content="hidden tool-turn prose",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {"topic": "ACME"},
                        "id": "provider-id",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="visible discovery reply"),
        ]
        if not native:
            self.replies.append(AIMessage(content='{"value": 9}'))

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return self.replies.pop(0)

    def with_structured_output(self, schema):
        class _Bound:
            def invoke(self, _messages):
                return schema.model_validate({"value": 9})

        return _Bound()


class _SubclassToolChat:
    """Return one accepted tool name as a valid string subclass."""

    def __init__(self) -> None:
        self.replies = [
            SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "name": _StringSubclass("lookup"),
                        "args": {"topic": "ACME"},
                        "id": "provider-id",
                    }
                ],
            ),
            AIMessage(content="finished"),
        ]

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return self.replies.pop(0)


class _MutatingTraceNode(Node):
    """Mutate the live record after the evaluator snapshots its bound inputs."""

    kind = NodeKind.CODE
    needs_evaluation_trace = True

    def bind_reserved(self, _wiring, _pool, *, bind=None):
        return {"payload": {"items": [1]}}

    def forward(self, inputs, *, evaluation_trace: EvaluationTrace):
        inputs["payload"]["items"].append(2)
        return Output(inputs)


class _UncanonicalTraceNode(Node):
    """Expose an unsupported bound value before the node body may execute."""

    kind = NodeKind.CODE
    needs_evaluation_trace = True

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.ran = False

    def bind_reserved(self, _wiring, _pool, *, bind=None):
        return {"unordered": {1, 2}}

    def forward(self, _inputs, *, evaluation_trace: EvaluationTrace):
        self.ran = True
        return Output("unexpected")


class _EvidenceOutputNode(Node):
    """A kind-blind trace-capable test node returning one configured result."""

    kind = NodeKind.CODE
    needs_evaluation_trace = True

    def __init__(self, node_id: str, result) -> None:
        super().__init__(node_id)
        self.result = result

    def forward(self, _inputs, *, evaluation_trace: EvaluationTrace):
        return self.result


class _EvidenceRouteNode(_EvidenceOutputNode):
    kind = NodeKind.CASE


class _EvidenceGrowNode(_EvidenceOutputNode):
    kind = NodeKind.CALL
    is_spawner = True
    grow_depth_delta = 1


def _captured_agent(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chat,
    llm_config: dict | None = None,
) -> tuple[AgentNode, EvaluatedOutcome]:
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)
    node = AgentNode(
        "agent",
        prompt="Answer now",
        llm_config=llm_config or {},
    )
    node.memory_scope = _scope(node.id)
    evaluated = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )
    return node, evaluated


def test_evaluation_trace_protocol_is_runtime_checkable_and_write_only() -> None:
    assert isinstance(
        SimpleNamespace(record_prompt=lambda *_args, **_kwargs: None,
                        record_interaction=lambda _interaction: None),
        EvaluationTrace,
    )
    assert not hasattr(EvaluationTrace, "snapshot")
    assert not hasattr(EvaluationTrace, "evidence")


def test_scope_free_model_evidence_is_accepted_with_forward_capture(monkeypatch):
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: _PlainChat("done"))
    node = AgentNode("work")
    engine = _engine(
        _graph([node], [(START_ID, node.id), (node.id, END_ID)]), learnable=True,
    )
    list(engine.run())
    record = next(
        item.record for item in engine.state.trace.occurrences if item.node_id == node.id
    )
    assert record.evidence.scope is None
    assert record.evidence.interactions == (ModelTextInteraction("done"),)
    assert record.forward_capture.bound_inputs == {}
    assert record.forward_capture.scope.static_path == (node.id,)
    from yian.runtime.learning import project_outcome_episodes

    assert project_outcome_episodes(engine.state.run_id, record) == ()


@pytest.mark.parametrize("stage", ["recall", "factory", "model", "success"])
def test_actual_render_capture_survives_later_failures_without_fabricating_evidence(monkeypatch, stage):
    from yian.compose.helpers.types import read_variables
    from yian.memory import MemoryRecallError, MemorySection

    node = AgentNode("work", prompt="${instruction}: ${topic}", variables=read_variables({
        "instruction": "str = Summarize clearly",
    }, {}), llm_config={"provider": "test", "model": "chosen"})
    node.params = [Parameter("topic")]
    node.memory_scope = _scope(node.id)
    section = MemorySection(node.memory_scope.memories[0], "Relevant memory.")
    factory_calls, model_calls, recall_calls = [], [], []

    @contextmanager
    def recall(scope, request, *, check_stopped):
        recall_calls.append((scope, request.recall))
        if stage == "recall":
            raise MemoryRecallError("recall failed")
        yield MemoryContext((section,))

    class Chat:
        def invoke(self, messages):
            model_calls.append(messages)
            if stage == "model":
                raise ValueError("model failed")
            return AIMessage(content="answer", response_metadata={"private": "not retained"})

    def factory(config):
        factory_calls.append(dict(config))
        if stage == "factory":
            raise ValueError("factory failed")
        return Chat()

    monkeypatch.setattr(llm_clients, "model_from_config", factory)
    memories, stacks, snapshots = _learning_configuration(node)
    engine = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)],
                            wiring={"work": {"topic": "reefs"}}), learnable=True,
                     memories=memories, memory_stacks=stacks, memory_snapshots=snapshots, memory_context_opener=recall)
    list(engine.run())
    record = next(item.record for item in engine.state.trace.occurrences if item.node_id == "work")
    assert type(record) is (NodeSucceededRecord if stage == "success" else NodeFailedRecord)
    assert record.forward_capture.bound_inputs == {"topic": "reefs"}
    assert record.forward_capture.node_data == {
        "prompt_template": "${instruction}: ${topic}",
        "variables": {"instruction": "Summarize clearly"},
        "variable_domains": {"instruction": "str"},
        "rendered_prompt": "Summarize clearly: reefs",
        "effective_llm_config": {"provider": "test", "model": "chosen"},
    }
    visible_prompt = stage in ("model", "success")
    assert len(recall_calls) == 1
    assert len(factory_calls) == (stage != "recall")
    assert len(model_calls) == visible_prompt
    assert record.evidence.prompt == ("Summarize clearly: reefs" if visible_prompt else None)
    assert record.evidence.base_system_prompt == (DEFAULT_SYSTEM if visible_prompt else None)
    assert record.evidence.memory_sections == ((section,) if visible_prompt else ())
    assert record.evidence.interactions == ((ModelTextInteraction("answer"),) if stage == "success" else ())


def test_capture_gate_requires_exact_flag_and_trait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: _PlainChat("done"),
    )
    scoped = AgentNode("scoped")
    scoped.memory_scope = _scope(scoped.id)
    disabled = eval_node(
        scoped,
        None,
        VariablePool(),
        memory_context_opener=_empty_context,
    )
    assert disabled.evidence is None

    unscoped = AgentNode("unscoped")
    unscoped_evidence = eval_node(
        unscoped,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    ).evidence
    assert unscoped_evidence.scope is None
    assert unscoped_evidence.interactions == (ModelTextInteraction("done"),)

    ordinary = FuncNode("ordinary", lambda _inputs: "done")
    ordinary.memory_scope = _scope(ordinary.id)
    assert eval_node(
        ordinary,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    ).evidence is None

    with pytest.raises(TypeError, match="capture_evidence must be a bool"):
        eval_node(scoped, None, VariablePool(), capture_evidence=1)


def test_capture_snapshots_canonical_inputs_before_node_mutation() -> None:
    node = _MutatingTraceNode("mutate")
    node.memory_scope = _scope(node.id)

    evaluated = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert evaluated.outcome == Output({"payload": {"items": [1, 2]}})
    assert evaluated.evidence is not None
    assert evaluated.evidence.bound_inputs == {"payload": {"items": [1]}}
    detached = evaluated.evidence.bound_inputs
    assert detached is not None
    detached["payload"]["items"].append(9)
    assert evaluated.evidence.bound_inputs == {"payload": {"items": [1]}}


def test_input_snapshot_canonicalization_fault_propagates_before_node_run() -> None:
    node = _UncanonicalTraceNode("bad-input")
    node.memory_scope = _scope(node.id)

    with pytest.raises(ValueError, match="unordered"):
        eval_node(
            node,
            None,
            VariablePool(),
            capture_evidence=True,
            memory_context_opener=_empty_context,
        )

    assert node.ran is False


def test_binding_failure_captures_scope_with_absent_inputs_and_no_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: pytest.fail("binding failure must not create a model"),
    )
    node = AgentNode("agent")
    node.params = [Parameter(name="required", required=True)]
    node.memory_scope = _scope(node.id)
    flow = SimpleNamespace(wiring={node.id: {}})

    evaluated = eval_node(
        node,
        flow,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert isinstance(evaluated.outcome, EvaluationFailure)
    assert evaluated.evidence is not None
    assert evaluated.evidence.bound_inputs is None
    assert evaluated.evidence.base_system_prompt is None
    assert evaluated.evidence.interactions == ()


def test_oversized_recall_inputs_fail_identically_with_or_without_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recall context validation is semantic while evidence remains observational."""
    nested: dict = {}
    for _ in range(160):
        nested = {"nested": nested}
    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: pytest.fail("invalid recall context must precede the model"),
    )
    node = AgentNode("agent", prompt="answer")
    node.params = [Parameter(name="payload", required=True)]
    node.memory_scope = _scope(node.id)
    pool = VariablePool()
    pool.set("source", nested)
    flow = SimpleNamespace(wiring={node.id: {"payload": "${source.output}"}})

    capture_off = eval_node(node, flow, pool, memory_context_opener=_empty_context)
    capture_on = eval_node(
        node,
        flow,
        pool,
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert isinstance(capture_off.outcome, EvaluationFailure)
    assert capture_on.outcome == capture_off.outcome
    assert capture_on.outcome.failure.message == "memory recall context is invalid"
    assert capture_on.evidence is not None
    assert capture_on.evidence.bound_inputs is None
    assert capture_on.evidence.interactions == ()


def test_plain_agent_records_only_visible_prompt_and_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evaluated = _captured_agent(monkeypatch, chat=_PlainChat("answer"))

    assert evaluated.outcome == Output("answer")
    evidence = evaluated.evidence
    assert evidence is not None
    assert evidence.bound_inputs == {}
    assert (evidence.base_system_prompt, evidence.prompt) == (
        DEFAULT_SYSTEM,
        "Answer now",
    )
    assert evidence.memory_sections == ()
    assert evidence.interactions == (ModelTextInteraction("answer"),)
    assert "private" not in repr(evidence)
    assert "tokens" not in repr(evidence)


def test_visible_string_subclass_has_capture_off_on_behavioral_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: _SubclassTextChat(),
    )
    node = AgentNode("agent", prompt="answer")
    node.memory_scope = _scope(node.id)

    capture_off = eval_node(
        node,
        None,
        VariablePool(),
        memory_context_opener=_empty_context,
    )
    capture_on = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert capture_off.outcome == capture_on.outcome == Output("subclass answer")
    assert type(capture_on.outcome.value) is str
    assert capture_on.evidence is not None
    assert capture_on.evidence.interactions == (
        ModelTextInteraction("subclass answer"),
    )
    assert type(capture_on.evidence.interactions[0].text) is str


def test_invalid_unicode_text_is_ignored_by_observational_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed trace material cannot change the underlying AGENT outcome."""
    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: _SurrogateTextChat(),
    )
    node = AgentNode("agent", prompt="answer")
    node.memory_scope = _scope(node.id)

    capture_off = eval_node(
        node,
        None,
        VariablePool(),
        memory_context_opener=_empty_context,
    )
    capture_on = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    expected = Output(chr(0xD800))
    assert capture_off.outcome == capture_on.outcome == expected
    assert capture_on.evidence is not None
    assert capture_on.evidence.interactions == ()


def test_invalid_unicode_recall_prompt_fails_identically_with_or_without_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recall rejects an invalid prompt while tracing stays observational."""
    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: pytest.fail("invalid recall context must precede the model"),
    )
    node = AgentNode("agent", prompt=chr(0xD800))
    node.memory_scope = _scope(node.id)

    capture_off = eval_node(
        node,
        None,
        VariablePool(),
        memory_context_opener=_empty_context,
    )
    capture_on = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert isinstance(capture_off.outcome, EvaluationFailure)
    assert capture_on.outcome == capture_off.outcome
    assert capture_on.outcome.failure.message == "memory recall context is invalid"
    assert capture_on.evidence is not None
    assert capture_on.evidence.base_system_prompt is None
    assert capture_on.evidence.prompt is None
    assert capture_on.evidence.interactions == ()


def test_native_structured_agent_records_only_parsed_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: _NativeStructuredChat({"value": 7}),
    )
    node = AgentNode(
        "agent",
        prompt="number",
        llm_config={"provider": "anthropic", "model": "claude-sonnet-4-5"},
    )
    node.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))
    node.memory_scope = _scope(node.id)
    evaluated = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert evaluated.outcome == Output(7)
    assert evaluated.evidence is not None
    assert evaluated.evidence.interactions == (ModelValueInteraction(7),)


def test_fallback_structured_records_every_raw_retry_without_parsed_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _FallbackStructuredChat()
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)
    node = AgentNode(
        "agent",
        prompt="number",
        retries=1,
        llm_config={"provider": "vllm", "model": "no-structured-sentinel"},
    )
    node.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))
    node.memory_scope = _scope(node.id)

    evaluated = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert evaluated.outcome == Output(7)
    assert evaluated.evidence is not None
    assert evaluated.evidence.interactions == (
        ModelTextInteraction("not json"),
        ModelTextInteraction('{"value": 7}'),
    )


def test_fallback_structured_parses_only_visible_text_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: _FallbackContentBlockChat(),
    )
    node = AgentNode(
        "agent",
        prompt="number",
        retries=0,
        llm_config={"provider": "vllm", "model": "no-structured-sentinel"},
    )
    node.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))
    node.memory_scope = _scope(node.id)

    evaluated = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert evaluated.outcome == Output(13)
    assert evaluated.evidence is not None
    assert evaluated.evidence.interactions == (
        ModelTextInteraction('{"value": 13}'),
    )
    rendered = repr(evaluated.evidence)
    assert "SECRET_REASONING" not in rendered
    assert "SECRET_BLOCK_METADATA" not in rendered
    assert "SECRET_PROVIDER_METADATA" not in rendered


def test_invalid_unicode_fallback_trace_does_not_consume_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evidence rejection stays outside structured provider/parse retry semantics."""
    text = '{"value":"' + chr(0xD800) + '"}'

    class _SurrogateStructuredChat:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return SimpleNamespace(content=text)

    chat = _SurrogateStructuredChat()
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)
    node = AgentNode(
        "agent",
        prompt="text",
        retries=2,
        llm_config={"provider": "vllm", "model": "no-structured-sentinel"},
    )
    node.output_contract = ValueContract(
        Type(
            kind=ValueKind.OBJECT,
            fields={"value": Type.scalar(ValueKind.STRING)},
            required=frozenset({"value"}),
        )
    )
    node.memory_scope = _scope(node.id)

    evaluated = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert evaluated.outcome == Output({"value": chr(0xD800)})
    assert chat.calls == 1
    assert evaluated.evidence is not None
    assert evaluated.evidence.interactions == ()


def test_tool_loop_records_calls_results_and_final_text_in_visible_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def lookup(topic: str) -> str:
        """Return one deterministic public tool result."""
        return f"found:{topic}"

    monkeypatch.setitem(
        tools.TOOL_REGISTRY,
        "lookup",
        StructuredTool.from_function(lookup, name="lookup"),
    )
    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: _ToolChat(),
    )
    node = AgentNode(
        "agent",
        prompt="lookup",
        tools=["lookup"],
    )
    node.memory_scope = _scope(node.id)
    evaluated = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert evaluated.outcome == Output("finished")
    assert evaluated.evidence is not None
    assert evaluated.evidence.interactions == (
        ToolCallInteraction("lookup", {"topic": "ACME"}),
        ToolResultInteraction("lookup", "found:ACME"),
        ModelTextInteraction("finished"),
    )
    assert "provider-call-id" not in repr(evaluated.evidence)
    assert "private plan" not in repr(evaluated.evidence)


def test_tool_loop_records_valid_ungranted_call_before_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: _ToolChat(),
    )
    node = AgentNode("agent", prompt="lookup", tools=[])
    node.memory_scope = _scope(node.id)

    evaluated = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert isinstance(evaluated.outcome, EvaluationFailure)
    assert "not granted" in evaluated.outcome.failure.message
    assert evaluated.evidence is not None
    assert evaluated.evidence.interactions == (
        ToolCallInteraction("lookup", {"topic": "ACME"}),
    )


def test_recursive_ungranted_tool_arguments_do_not_change_failure_with_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments: dict = {}
    for _ in range(2_000):
        arguments = {"nested": arguments}

    class _RecursiveCallChat:
        def invoke(self, _messages):
            return SimpleNamespace(
                content="",
                tool_calls=[{"name": "lookup", "args": arguments, "id": "1"}],
            )

    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: _RecursiveCallChat(),
    )
    node = AgentNode("agent", prompt="lookup", tools=[])
    node.memory_scope = _scope(node.id)

    capture_off = eval_node(
        node,
        None,
        VariablePool(),
        memory_context_opener=_empty_context,
    )
    capture_on = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert isinstance(capture_off.outcome, EvaluationFailure)
    assert isinstance(capture_on.outcome, EvaluationFailure)
    assert capture_on.outcome.failure.code == capture_off.outcome.failure.code
    assert capture_on.outcome.failure.message == capture_off.outcome.failure.message
    assert capture_on.evidence is not None
    assert capture_on.evidence.interactions == ()


def test_tool_name_subclass_has_capture_off_on_behavioral_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def lookup(topic: str) -> str:
        """Record one exact tool invocation."""
        seen.append(topic)
        return f"found:{topic}"

    monkeypatch.setitem(
        tools.TOOL_REGISTRY,
        "lookup",
        StructuredTool.from_function(lookup, name="lookup"),
    )
    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: _SubclassToolChat(),
    )
    node = AgentNode(
        "agent",
        prompt="lookup",
        tools=["lookup"],
    )
    node.memory_scope = _scope(node.id)

    capture_off = eval_node(
        node,
        None,
        VariablePool(),
        memory_context_opener=_empty_context,
    )
    capture_on = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert capture_off.outcome == capture_on.outcome == Output("finished")
    assert seen == ["ACME", "ACME"]
    assert capture_on.evidence is not None
    assert capture_on.evidence.interactions == (
        ToolCallInteraction("lookup", {"topic": "ACME"}),
        ToolResultInteraction("lookup", "found:ACME"),
        ModelTextInteraction("finished"),
    )
    call, result, _ = capture_on.evidence.interactions
    assert type(call.name) is str
    assert type(result.name) is str


def test_tool_trace_uses_the_same_effective_arguments_as_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def noop() -> str:
        """Return one deterministic result without arguments."""
        return "ok"

    class _FalsyArgumentsChat:
        def __init__(self) -> None:
            self.replies = [
                SimpleNamespace(
                    content="",
                    tool_calls=[{"name": "noop", "args": [], "id": "1"}],
                ),
                AIMessage(content="finished"),
            ]

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            return self.replies.pop(0)

    monkeypatch.setitem(
        tools.TOOL_REGISTRY,
        "noop",
        StructuredTool.from_function(noop, name="noop"),
    )
    monkeypatch.setattr(
        llm_clients,
        "model_from_config",
        lambda _config: _FalsyArgumentsChat(),
    )
    node = AgentNode(
        "agent",
        prompt="run",
        tools=["noop"],
    )
    node.memory_scope = _scope(node.id)

    evaluated = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert evaluated.outcome == Output("finished")
    assert evaluated.evidence is not None
    assert evaluated.evidence.interactions == (
        ToolCallInteraction("noop", {}),
        ToolResultInteraction("noop", "ok"),
        ModelTextInteraction("finished"),
    )


def test_invalid_unicode_tool_result_is_ignored_by_observational_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unencodable visible tool result does not abort the ordinary tool loop."""
    def lookup() -> str:
        """Return one intentionally unencodable tool result."""
        return chr(0xD800)

    chat = _ToolChat()
    chat.replies[0].tool_calls[0]["args"] = {}
    monkeypatch.setitem(
        tools.TOOL_REGISTRY,
        "lookup",
        StructuredTool.from_function(lookup, name="lookup"),
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)
    node = AgentNode(
        "agent",
        prompt="lookup",
        tools=["lookup"],
    )
    node.memory_scope = _scope(node.id)

    evaluated = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert evaluated.outcome == Output("finished")
    assert evaluated.evidence is not None
    assert evaluated.evidence.interactions == (
        ToolCallInteraction("lookup", {}),
        ModelTextInteraction("finished"),
    )


@pytest.mark.parametrize("native", [True, False], ids=["native", "fallback"])
def test_tool_structured_output_retains_discovery_and_generation_order(
    monkeypatch: pytest.MonkeyPatch,
    native: bool,
) -> None:
    def lookup(topic: str) -> str:
        """Return one deterministic public tool result."""
        return f"found:{topic}"

    monkeypatch.setitem(
        tools.TOOL_REGISTRY,
        "lookup",
        StructuredTool.from_function(lookup, name="lookup"),
    )
    chat = _StructuredToolChat(native=native)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)
    node = AgentNode(
        "agent",
        prompt="lookup then score",
        tools=["lookup"],
        retries=0,
        llm_config=(
            {"provider": "anthropic", "model": "claude-sonnet-4-5"}
            if native
            else {"provider": "vllm", "model": "no-structured-sentinel"}
        ),
    )
    node.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))
    node.memory_scope = _scope(node.id)

    evaluated = eval_node(
        node,
        None,
        VariablePool(),
        capture_evidence=True,
        memory_context_opener=_empty_context,
    )

    assert evaluated.outcome == Output(9)
    assert evaluated.evidence is not None
    prefix = (
        ToolCallInteraction("lookup", {"topic": "ACME"}),
        ToolResultInteraction("lookup", "found:ACME"),
        ModelTextInteraction("visible discovery reply"),
    )
    expected_final = (
        ModelValueInteraction(9)
        if native
        else ModelTextInteraction('{"value": 9}')
    )
    assert evaluated.evidence.interactions == prefix + (expected_final,)


def _inflight_state(node: Node):
    """Prepare one static test node as the sole in-flight evaluation."""
    flow = _graph(
        [node],
        [(START_ID, node.id), (node.id, END_ID)],
    )
    memories, stacks, snapshots = _learning_configuration(node)
    engine = _engine(
        flow,
        memories=memories, memory_stacks=stacks,
        memory_snapshots=snapshots,
        learnable=True,
    )
    commit_prepared(
        engine.log,
        engine.state,
        prepare_node_started(engine.state, START_ID),
    )
    commit_prepared(
        engine.log,
        engine.state,
        prepare_node_succeeded(engine.state, START_ID, {},
                               forward_capture=_forward_capture(engine.state.flow.nodes[START_ID], {})),
    )
    commit_prepared(
        engine.log,
        engine.state,
        prepare_node_started(engine.state, node.id),
    )
    return engine.state


def test_state_validates_evidence_scope_and_trait_for_live_and_replay() -> None:
    node = _EvidenceOutputNode("work", Output("done"))
    node.memory_scope = _scope(node.id)
    evidence = EvaluationEvidence(node.memory_scope, {}, None, None)
    state = _inflight_state(node)

    prepared = prepare_node_succeeded(
        state,
        node.id,
        "done",
        evidence=evidence,
        forward_capture=_forward_capture(node, {}),
    )
    assert type(prepared.record) is NodeSucceededRecord
    assert prepared.record.evidence == evidence

    mismatched = EvaluationEvidence(
        _scope(node.id, occurrence=_occurrence(1)),
        {},
        None,
        None,
    )
    with pytest.raises(ValueError, match="evidence scope does not match"):
        prepare_node_succeeded(
            _inflight_state(node),
            node.id,
            "done",
            evidence=mismatched,
        )
    with pytest.raises(ValueError, match="evidence scope does not match"):
        prepare_record(
            _inflight_state(node),
            replace(prepared.record, evidence=mismatched),
        )

    ordinary = FuncNode("ordinary", lambda _inputs: "done")
    ordinary.memory_scope = _scope(ordinary.id)
    ordinary_evidence = EvaluationEvidence(ordinary.memory_scope, {}, None, None)
    with pytest.raises(ValueError, match="ineligible for evaluation evidence"):
        prepare_node_succeeded(
            _inflight_state(ordinary),
            ordinary.id,
            "done",
            evidence=ordinary_evidence,
        )


@pytest.mark.parametrize("arm", ["output", "route", "grow"])
def test_dispatcher_contract_failures_retain_the_same_evidence(arm: str) -> None:
    if arm == "output":
        node = _EvidenceOutputNode("work", Output("not-an-int"))
        node.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))
        nodes = [node]
        edges = [(START_ID, node.id), (node.id, END_ID)]
        grow_builder = lambda _spawner, _seed: None
    elif arm == "route":
        node = _EvidenceRouteNode("work", Route("absent"))
        target = FuncNode("target", lambda _inputs: "done")
        nodes = [node, target]
        edges = [
            (START_ID, node.id),
            (node.id, target.id, "present"),
            (target.id, END_ID),
        ]
        grow_builder = lambda _spawner, _seed: None
    else:
        node = _EvidenceGrowNode("work", Grow({"bad": True}))
        nodes = [node]
        edges = [(START_ID, node.id), (node.id, END_ID)]

        def grow_builder(_spawner, _seed):
            error = TypeError("rejected growth")
            raise GrowthRejected(error) from error

    node.memory_scope = _scope(node.id)
    memories, stacks, snapshots = _learning_configuration(node)
    engine = _engine(
        _graph(nodes, edges),
        grow_builder=grow_builder,
        memories=memories, memory_stacks=stacks,
        memory_snapshots=snapshots,
        learnable=True,
    )

    list(engine.run())

    failures = [
        envelope.record
        for envelope in engine.log
        if type(envelope.record) is NodeFailedRecord
    ]
    assert len(failures) == 1
    assert failures[0].evidence is not None
    assert failures[0].evidence.scope == node.memory_scope
    assert failures[0].evidence.bound_inputs == {}


def test_pooled_dispatch_discards_late_sibling_evidence_with_its_outcome() -> None:
    sibling_entered = threading.Event()

    class _LateSibling(_EvidenceOutputNode):
        def forward(self, _inputs, *, evaluation_trace: EvaluationTrace):
            sibling_entered.set()
            time.sleep(0.1)
            return Output("late")

    class _FailAfterSiblingStarts(_EvidenceOutputNode):
        def forward(self, _inputs, *, evaluation_trace: EvaluationTrace):
            assert sibling_entered.wait(1)
            return Output("not-an-int")

    failing = _FailAfterSiblingStarts("failing", Output(None))
    failing.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))
    sibling = _LateSibling("sibling", Output(None))
    failing.memory_scope = _scope(failing.id)
    sibling.memory_scope = _scope(sibling.id)
    flow = _graph(
        [failing, sibling],
        [
            (START_ID, failing.id),
            (START_ID, sibling.id),
            (failing.id, END_ID),
            (sibling.id, END_ID),
        ],
    )
    memories, stacks, snapshots = _learning_configuration(failing, sibling)
    engine = _engine(
        flow,
        num_workers=2,
        memories=memories, memory_stacks=stacks,
        memory_snapshots=snapshots,
        learnable=True,
    )

    list(engine.run())

    records = [envelope.record for envelope in engine.log]
    failure = next(record for record in records if type(record) is NodeFailedRecord)
    assert failure.failure.node_id == failing.id
    assert failure.evidence is not None
    assert not any(
        type(record) is NodeSucceededRecord and record.node_id == sibling.id
        for record in records
    )
