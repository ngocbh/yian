"""Raw node evaluation for the append-only dispatcher.

The runtime consumes one closed worker result. Streaming remains transient and is
forwarded only through the injected callback; host-control and callback failures stay
outside the node-failure sum.
"""

import threading
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest

from yian.events import StreamChunk
from yian.diagnostics import OriginRef
from yian.nodes.base import (
    Grow,
    Node,
    NodeKind,
    Output,
    Route,
    _HostStop,
)
from yian.nodes.contracts import Parameter
from yian.runtime.eval_node import (
    EvaluatedOutcome,
    EvaluationFailure,
    eval_node,
)
from yian.typesys.pool import VariablePool
from tests.engine._fakes import BranchNode, FailNode, FuncNode, StreamNode, derive_wiring


def _raw(node, *, emit_stream=None, stop_event=None):
    """Evaluate one fake against its flow-owned wiring."""
    flow = SimpleNamespace(wiring=derive_wiring({node.id: node}))
    return eval_node(
        node,
        flow,
        VariablePool(),
        emit_stream=emit_stream,
        stop_event=stop_event,
    ).outcome


class _Spawner(Node):
    """Minimal trait-declared spawner for the raw Grow result arm."""

    kind = NodeKind.CODE
    is_spawner = True

    def forward(self, inputs):
        return Grow(seed={"value": 1})


class _LeafGrow(Node):
    """Return Grow without the spawner trait to exercise the raw guard."""

    kind = NodeKind.CODE

    def forward(self, inputs):
        return Grow(seed={})


class _RaisesHostStop(Node):
    """Expose the private host-control exception without a node failure wrapper."""

    kind = NodeKind.CODE

    def forward(self, inputs):
        raise _HostStop("stop-now")


class _GeneratorRaisesHostStop(Node):
    """Yield once, then propagate one exact host-control signal while draining."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, expected: _HostStop) -> None:
        super().__init__(node_id)
        self.expected = expected

    def forward(self, inputs):
        yield StreamChunk(self.id, "output", "before-stop")
        raise self.expected


def test_public_evaluation_always_returns_frozen_slotted_wrapper() -> None:
    node = FuncNode("work", lambda _: "done")
    flow = SimpleNamespace(wiring=derive_wiring({node.id: node}))

    evaluated = eval_node(node, flow, VariablePool())

    assert type(evaluated) is EvaluatedOutcome
    assert evaluated == EvaluatedOutcome(Output("done"), None)
    assert [field.name for field in fields(EvaluatedOutcome)] == [
        "outcome",
        "evidence",
        "bindings",
        "backward_capture",
    ]
    assert not hasattr(evaluated, "__dict__")
    with pytest.raises(FrozenInstanceError):
        evaluated.evidence = None


def test_raw_evaluation_returns_closed_success_route_and_grow_values():
    output = _raw(FuncNode("work", lambda _: "done"))
    routed = _raw(BranchNode("branch", "chosen"))
    grown = _raw(_Spawner("spawn"))

    assert output == Output("done")
    assert routed == Route("chosen")
    assert grown == Grow(seed={"value": 1})


def test_render_capture_is_opt_in_single_write_and_detached_until_close():
    retained = []
    supplied = {"local": {"items": [1]}}

    class Capturing(Node):
        needs_render_capture = True

        def capture_backward_context(self):
            return {"static": "recorded"}

        def forward(self, inputs, *, capture_render=None):
            if capture_render is not None:
                retained.append(capture_render)
                capture_render(supplied)
                supplied["local"]["items"].append(2)
            return Output("done")

    node = Capturing("custom")
    assert eval_node(node, None, VariablePool()).backward_capture is None
    assert retained == []
    evaluated = eval_node(node, None, VariablePool(), capture_bindings=True)
    assert evaluated.outcome == Output("done")
    assert evaluated.backward_capture.node_data == {
        "static": "recorded", "local": {"items": [1]}, "effective_llm_config": None,
    }
    with pytest.raises(RuntimeError, match="closed"):
        retained[0]({"late": "write"})
    assert evaluated.backward_capture.node_data["local"] == {"items": [1]}


@pytest.mark.parametrize("publication,match", [
    ("duplicate", "already recorded"),
    ("reserved", "recorded runtime authority"),
    ("static", "replace static"),
    ("nonfinite", "non-finite"),
])
def test_invalid_render_capture_is_a_host_fault(publication, match):
    closed = []

    class Capturing(Node):
        needs_render_capture = True

        def capture_backward_context(self):
            return {"static": "recorded"}

        def forward(self, inputs, *, capture_render):
            closed.append(capture_render)
            if publication == "duplicate":
                capture_render({"first": 1})
                capture_render({"second": 2})
            elif publication == "reserved":
                capture_render({"effective_llm_config": {"model": "invented"}})
            elif publication == "static":
                capture_render({"static": "changed"})
            else:
                capture_render({"variables": {"nested": (float("inf"),)}})
            return Output("must not run")

    with pytest.raises((RuntimeError, ValueError), match=match):
        eval_node(Capturing("custom"), None, VariablePool(), capture_bindings=True)
    with pytest.raises(RuntimeError, match="closed"):
        closed[0]({})


def test_render_capture_preserves_original_host_exception_and_closes(monkeypatch):
    import yian.runtime.eval_node as evaluator

    expected = MemoryError("capture allocation failed")
    closed = []
    canonical = evaluator.canonical_seed

    def fail_dynamic(value, **kwargs):
        if kwargs.get("reject_nonfinite"):
            raise expected
        return canonical(value, **kwargs)

    class Capturing(Node):
        needs_render_capture = True

        def forward(self, inputs, *, capture_render):
            closed.append(capture_render)
            capture_render({})
            return Output("must not run")

    monkeypatch.setattr(evaluator, "canonical_seed", fail_dynamic)
    with pytest.raises(MemoryError) as caught:
        eval_node(Capturing("custom"), None, VariablePool(), capture_bindings=True)
    assert caught.value is expected
    with pytest.raises(RuntimeError, match="closed"):
        closed[0]({})


def test_scoped_render_capture_requires_recorded_model_map_entry(monkeypatch):
    import yian.nodes.agent.node as agent_module
    from yian.nodes.agent import AgentNode
    from yian.nodes.contracts import ExecutionScope

    node = AgentNode("a/runtime/id", prompt="hello", llm_config={"model": "ambient"})
    node.execution_scope = ExecutionScope(("authored", "agent"))
    calls = []
    monkeypatch.setitem(agent_module.MODES, "plain", lambda ctx: calls.append(ctx.prompt) or Output(ctx.prompt))
    with pytest.raises(ValueError, match="missing static path 'authored/agent'"):
        eval_node(node, None, VariablePool(), capture_bindings=True, effective_llm_by_static_path={})
    assert calls == []
    config = {"provider": "test", "model": "recorded", "temperature": 0.5}
    evaluated = eval_node(node, None, VariablePool(), capture_bindings=True,
                          effective_llm_by_static_path={"authored/agent": config})
    config["model"] = "later mutation"
    assert evaluated.backward_capture.node_data["effective_llm_config"]["model"] == "recorded"
    ordinary = eval_node(node, None, VariablePool(), effective_llm_by_static_path={})
    assert ordinary.outcome == evaluated.outcome
    assert ordinary.backward_capture is None


def test_raw_streaming_uses_only_the_injected_transient_callback():
    chunks = []

    result = _raw(StreamNode("agent", ["a", "b"]), emit_stream=chunks.append)

    assert result == Output({"output": "ab"})
    assert [chunk.chunk for chunk in chunks] == ["a", "b"]
    assert all(isinstance(chunk, StreamChunk) for chunk in chunks)


def test_raw_failure_is_frozen_slotted_and_preserves_diagnostics():
    node = FuncNode("work", lambda inputs: inputs)
    node.params = [Parameter(name="required", required=True)]
    node._wiring_src = {}

    result = _raw(node)

    assert isinstance(result, EvaluationFailure)
    assert result.failure.node_id == "work"
    assert result.failure.cause_type == "BindingError"
    assert result.failure.traceback is not None
    assert "BindingError" in result.failure.traceback
    assert [field.name for field in fields(EvaluationFailure)] == ["failure"]
    assert not any(
        hasattr(result, alias)
        for alias in ("node_id", "error", "error_type", "traceback")
    )
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.failure = result.failure


def test_raw_node_raise_and_invalid_grow_are_failures_not_observers():
    raised = _raw(FailNode("raise", "boom"))
    invalid_grow = _raw(_LeafGrow("leaf"))

    assert isinstance(raised, EvaluationFailure)
    assert raised.failure.code == "node.execute"
    assert (raised.failure.message, raised.failure.cause_type) == (
        "boom",
        "RuntimeError",
    )
    assert isinstance(invalid_grow, EvaluationFailure)
    assert invalid_grow.failure.code == "contract.node_result"
    assert "not a spawner" in invalid_grow.failure.message


class _RaisingStreamNode(Node):
    """Raise while a generator body is being advanced."""

    kind = NodeKind.CODE

    def forward(self, inputs):
        yield StreamChunk(self.id, "output", "first")
        raise ValueError("stream body failed")


class _InvalidStreamReturnNode(Node):
    """Return a value outside NodeResult from a generator body."""

    kind = NodeKind.CODE

    def forward(self, inputs):
        yield StreamChunk(self.id, "output", "before-invalid-return")
        return "invalid"


def test_stream_body_exception_and_invalid_return_use_distinct_stages() -> None:
    """Generator execution is node code; its terminal value is a result contract."""
    body = _RaisingStreamNode("body", origin=OriginRef("node:body"))
    invalid = _InvalidStreamReturnNode("invalid", origin=OriginRef("node:invalid"))
    chunks = []
    invalid_chunks = []

    body_failure = _raw(body, emit_stream=chunks.append)
    result_failure = _raw(invalid, emit_stream=invalid_chunks.append)

    assert [chunk.chunk for chunk in chunks] == ["first"]
    assert isinstance(body_failure, EvaluationFailure)
    assert body_failure.failure.code == "node.execute"
    assert body_failure.failure.trace == (body.origin,)
    assert body_failure.failure.cause_type == "ValueError"
    assert "ValueError: stream body failed" in (body_failure.failure.traceback or "")
    assert isinstance(result_failure, EvaluationFailure)
    assert [chunk.chunk for chunk in invalid_chunks] == ["before-invalid-return"]
    assert result_failure.failure.code == "contract.node_result"
    assert result_failure.failure.trace == (invalid.origin,)
    assert result_failure.failure.cause_type == "RuntimeError"
    assert "generator did not return a NodeResult" in result_failure.failure.message


def test_raw_stream_callback_failure_propagates_unchanged():
    expected = RuntimeError("callback failed")

    def reject(_chunk):
        raise expected

    with pytest.raises(RuntimeError) as caught:
        _raw(StreamNode("agent", ["a"]), emit_stream=reject)

    assert caught.value is expected


def test_raw_stream_callback_failure_closes_the_node_generator_before_returning():
    closed = []

    class ResourceStreamNode(Node):
        """Streaming fake whose generator records deterministic cleanup."""

        kind = NodeKind.CODE

        def forward(self, inputs):
            try:
                yield StreamChunk(self.id, "output", "first")
                yield StreamChunk(self.id, "output", "second")
                return Output("done")
            finally:
                closed.append(self.id)

    expected = RuntimeError("callback failed")

    def reject(_chunk):
        raise expected

    with pytest.raises(RuntimeError) as caught:
        _raw(ResourceStreamNode("resource"), emit_stream=reject)

    assert caught.value is expected
    assert closed == ["resource"]


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_raw_stream_cleanup_failure_does_not_replace_callback_failure():
    class CleanupFailsNode(Node):
        """Raise during generator close after yielding one callback-bound chunk."""

        kind = NodeKind.CODE

        def forward(self, inputs):
            try:
                yield StreamChunk(self.id, "output", "first")
            finally:
                raise ValueError("cleanup failed")

    expected = RuntimeError("callback failed")

    def reject(_chunk):
        raise expected

    with pytest.raises(RuntimeError) as caught:
        _raw(CleanupFailsNode("resource"), emit_stream=reject)

    assert caught.value is expected


def test_raw_host_stop_propagates_from_node_and_preset_event():
    with pytest.raises(_HostStop) as raised:
        _raw(_RaisesHostStop("work"))
    assert str(raised.value) == "stop-now"

    stop_event = threading.Event()
    stop_event.set()
    with pytest.raises(_HostStop) as preset:
        _raw(FuncNode("never", lambda _: "unexpected"), stop_event=stop_event)
    assert preset.value.primary is False


def test_raw_generator_host_stop_propagates_by_identity() -> None:
    expected = _HostStop("stream-stop")
    chunks = []

    with pytest.raises(_HostStop) as caught:
        _raw(_GeneratorRaisesHostStop("stream", expected), emit_stream=chunks.append)

    assert caught.value is expected
    assert [chunk.chunk for chunk in chunks] == ["before-stop"]
