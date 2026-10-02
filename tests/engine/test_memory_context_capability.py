"""Source-neutral, execution-scoped access to optional conversation turns."""

import threading
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from agent_composer.events import StreamChunk
from agent_composer.memory import ConversationInvocation, DeclSite, MemoryContext, MemoryScope, MemorySpec
from agent_composer.nodes.base import Grow, Node, NodeKind, Output, _HostStop
from agent_composer.nodes.contracts import Assertion
from agent_composer.runtime.eval_node import EvaluationFailure, eval_node
from agent_composer.typesys.pool import VariablePool
from tests.engine._fakes import FuncNode


class _Consumer(Node):
    kind = NodeKind.CODE
    needs_memory_context = True

    def __init__(self, body):
        super().__init__("work")
        self.memory_scope = MemoryScope(("work",), ("session",))
        self.body = body

    def forward(self, inputs, *, open_memory_context):
        return self.body(open_memory_context)


def _evaluate(node, opener, **kwargs):
    return eval_node(
        node, None, VariablePool(), memory_context_opener=opener,
        memory_invocation=ConversationInvocation("run", 3), **kwargs,
    ).outcome


def _open_session_runtime(memory):
    """Open a runtime whose one AGENT-owned key ``session`` is backed by ``memory``."""
    from agent_composer.runtime.memory_runtime import MemoryRuntime

    spec = MemorySpec("session", "session", {}, (DeclSite((), ("root",), "work"),), True)
    return MemoryRuntime.open(memories=(spec,), origins={"session": None}, memory={"session": memory})


def _opener(events, turn=None, *, stop=None, cleanup=None):
    @contextmanager
    def open_context(scope, request, *, check_stopped):
        events.append(("enter", scope, request))
        check_stopped()
        try:
            yield MemoryContext(transcript=turn)
        except BaseException as exc:
            events.append(("failed", exc, None if stop is None else stop.is_set()))
            raise
        else:
            events.append(("completed",))
        finally:
            if cleanup is not None:
                raise cleanup
    return open_context


def test_lazy_single_entry_binds_detached_identity_and_closes_after_execution():
    events, retained = [], []
    descriptor = {"format": {"version": 1}}
    turn = object()

    def body(begin):
        assert events == []
        retained.append(begin)
        assert begin("system", "prompt", descriptor).transcript is turn
        descriptor["format"]["version"] = 2
        events.append(("body",))
        return Output("done")

    node = _Consumer(body)
    assert _evaluate(node, _opener(events, turn)) == Output("done")
    assert [event[0] for event in events] == ["enter", "body", "completed"]
    assert events[0][2].invocation == ConversationInvocation("run", 3)
    assert events[0][2].recall.consumer_static_path == ("work",)
    assert events[0][2].compatibility == {"format": {"version": 1}}
    with pytest.raises(RuntimeError, match="closed"):
        retained[0]("system", "prompt", {})


def test_no_eager_turn_and_no_memory_remains_dormant():
    def forbidden(*args, **kwargs):
        pytest.fail("an unused capability must not open a turn")

    assert _evaluate(_Consumer(lambda begin: Output(7)), forbidden) == Output(7)
    with pytest.raises(RuntimeError, match="requires an injected memory context"):
        _evaluate(_Consumer(lambda begin: Output(begin("system", "prompt", {}))), None)
    node = _Consumer(lambda begin: Output(begin("system", "prompt", {})))
    node.memory_scope = None
    assert _evaluate(node, forbidden) == Output(MemoryContext())
    assert Node.needs_memory_context is False
    assert eval_node(FuncNode("ordinary", lambda _: 7), None, VariablePool(), memory_context_opener=forbidden).outcome == Output(7)


@pytest.mark.parametrize("result", [None, 7, Grow({})])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_invalid_node_result_closes_failed_before_failure_conversion(result, cleanup_fails):
    events = []
    def body(begin):
        begin("system", "prompt", {})
        return result
    cleanup = ValueError("result cleanup failed") if cleanup_fails else None
    outcome = _evaluate(_Consumer(body), _opener(events, cleanup=cleanup))
    assert isinstance(outcome, EvaluationFailure)
    assert outcome.failure.code == "contract.node_result"
    assert outcome.failure.cause_type == "RuntimeError"
    assert events[-1][0] == "failed"
    assert "_ResultContractError" not in outcome.failure.traceback
    assert f"RuntimeError: {outcome.failure.message}" in outcome.failure.traceback
    if cleanup_fails and hasattr(RuntimeError(), "add_note"):
        assert "result cleanup failed" in outcome.failure.traceback


def test_postassertion_failure_closes_failed():
    events = []
    node = _Consumer(lambda begin: (begin("system", "prompt", {}), Output(7))[1])
    node.post_asserts = [Assertion("false", "must fail")]
    outcome = _evaluate(node, _opener(events))
    assert outcome.failure.code == "assertion.failed"
    assert events[-1][0] == "failed"


def test_repeated_entry_is_host_fault_and_closes_first_turn():
    events = []
    def body(begin):
        begin("system", "prompt", {})
        begin("system", "prompt", {})
        return Output(7)
    with pytest.raises(RuntimeError, match="already"):
        _evaluate(_Consumer(body), _opener(events))
    assert [event[0] for event in events] == ["enter", "failed"]


@pytest.mark.parametrize("streaming", [False, True])
def test_opener_host_fault_keeps_identity_in_both_execution_shapes(streaming):
    error = RuntimeError("opening failed")
    def opener(*args, **kwargs):
        raise error
    def stream(begin):
        begin("system", "prompt", {})
        yield StreamChunk("work", "output", "never")
        return Output(7)
    body = stream if streaming else lambda begin: Output(begin("system", "prompt", {}))
    with pytest.raises(RuntimeError) as caught:
        _evaluate(_Consumer(body), opener)
    assert caught.value is error


@pytest.mark.parametrize("streaming", [False, True])
def test_primary_stop_is_set_before_turn_cleanup_and_keeps_identity(streaming):
    events, stop = [], threading.Event()
    error = _HostStop("requested")
    def body(begin):
        begin("system", "prompt", {})
        raise error
    def stream(begin):
        begin("system", "prompt", {})
        yield StreamChunk("work", "output", "before")
        raise error
    with pytest.raises(_HostStop) as caught:
        _evaluate(_Consumer(stream if streaming else body),
                  _opener(events, stop=stop, cleanup=RuntimeError("cleanup")), stop_event=stop)
    assert caught.value is error
    assert events[-1][0] == "failed"
    assert events[-1][2] is True
    if hasattr(error, "add_note"):
        assert any("cleanup" in note for note in getattr(error, "__notes__", []))


def test_stream_lifetime_includes_drain_and_callback_failure_cleanup():
    events = []
    error = RuntimeError("stream callback")
    def stream(begin):
        begin("system", "prompt", {})
        try:
            yield StreamChunk("work", "output", "one")
            pytest.fail("callback should abort iteration")
        finally:
            events.append(("generator closed",))
    def emit(chunk):
        assert [event[0] for event in events] == ["enter"]
        raise error
    with pytest.raises(RuntimeError) as caught:
        _evaluate(_Consumer(stream), _opener(events), emit_stream=emit)
    assert caught.value is error
    assert [event[0] for event in events] == ["enter", "generator closed", "failed"]


def test_callback_primary_stop_marks_signal_before_turn_unwind():
    events, stop = [], threading.Event()
    error = _HostStop("callback stop")
    def stream(begin):
        begin("system", "prompt", {})
        yield StreamChunk("work", "output", "one")
        return Output(7)
    def emit(chunk):
        raise error
    with pytest.raises(_HostStop) as caught:
        _evaluate(_Consumer(stream), _opener(events, stop=stop), emit_stream=emit, stop_event=stop)
    assert caught.value is error
    assert events[-1][2] is True


@pytest.mark.parametrize("host_stop", [False, True])
@pytest.mark.parametrize("real_backend", [False, True])
def test_callback_error_retains_conversation_cleanup_notes_without_throwing_into_generator(
    monkeypatch, host_stop, real_backend,
):
    events = []
    error = _HostStop("callback stop") if host_stop else RuntimeError("callback failure")
    cleanup = ValueError("conversation close failed")
    stop = threading.Event()
    def stream(begin):
        begin("system", "prompt", {})
        try:
            yield StreamChunk("work", "output", "one")
        except GeneratorExit:
            events.append(("generator exit",))
            raise
        except BaseException as unexpected:
            pytest.fail(f"callback exception entered user generator: {unexpected!r}")
        finally:
            events.append(("generator closed",))
    def emit(chunk):
        raise error
    node = _Consumer(stream)
    runtime = None
    opener = _opener(events, stop=stop, cleanup=cleanup)
    if real_backend:
        from agent_composer.memory import ConversationClosed
        from agent_composer.memory import session as backend

        memory = backend.SessionMemory()
        runtime = _open_session_runtime(memory)
        publish = backend._Turn._publish
        def fail_close(turn, checkpoint):
            if isinstance(checkpoint, ConversationClosed):
                raise cleanup
            return publish(turn, checkpoint)
        monkeypatch.setattr(backend._Turn, "_publish", fail_close)
        opener = runtime.open_context
    try:
        with pytest.raises(type(error)) as caught:
            _evaluate(node, opener, emit_stream=emit, stop_event=stop)
        assert caught.value is error
        assert ("generator exit",) in events
        assert ("generator closed",) in events
        assert stop.is_set() is host_stop
        if not real_backend:
            assert isinstance(events[-1][1], GeneratorExit)
        if hasattr(error, "add_note"):
            notes = getattr(error, "__notes__", [])
            assert sum("conversation close failed" in note for note in notes) == 1
    finally:
        if runtime is not None:
            runtime.close()


def test_broken_cleanup_note_hook_does_not_replace_primary_host_error():
    class StopWithoutNotes(_HostStop):
        @property
        def add_note(self):
            raise RuntimeError("note hook failed")
    error = StopWithoutNotes("original stop")
    def body(begin):
        begin("system", "prompt", {})
        raise error
    with pytest.raises(StopWithoutNotes) as caught:
        _evaluate(_Consumer(body), _opener([], cleanup=RuntimeError("cleanup failed")))
    assert caught.value is error


@pytest.mark.parametrize("num_workers", [0, 2])
def test_dispatch_binds_accepted_identity_before_worker_submission(num_workers, monkeypatch):
    from agent_composer.compile.model import START_ID, END_ID
    from agent_composer.compile.expand import build_grow
    from agent_composer.runtime.engine import FlowEngine
    from agent_composer.runtime.log import NodeStartedRecord
    from agent_composer.runtime.state import RunState
    from tests.engine._graph_builder import _engine, _graph

    events = []
    node = _Consumer(lambda begin: (begin("system", "prompt", {}), Output(7))[1])
    prepared = _engine(_graph([node], [(START_ID, "work"), ("work", END_ID)]), num_workers=num_workers)
    caller = threading.get_ident()
    reads = []

    get_attribute = RunState.__getattribute__
    def guarded_get_attribute(state, key):
        if key == "accepted_starts":
            assert threading.get_ident() == caller, "worker consulted mutable start authority"
            reads.append(key)
        return get_attribute(state, key)

    monkeypatch.setattr(RunState, "__getattribute__", guarded_get_attribute)
    engine = FlowEngine(prepared.state, prepared.log, grow_builder=build_grow,
                        num_workers=num_workers, memory_context_opener=_opener(events))
    list(engine.run())
    start = next(envelope for envelope in prepared.log
                 if isinstance(envelope.record, NodeStartedRecord) and envelope.record.node_id == "work")
    assert events[0][2].invocation == ConversationInvocation(start.run_id, start.seq)
    assert reads
    assert events[-1][0] == "completed"
    assert not hasattr(engine.state, "conversation_state")


def test_scope_and_invocation_are_frozen_before_node_execution():
    events = []
    invocation = ConversationInvocation("original", 7)
    def body(begin):
        node.memory_scope = MemoryScope(("replacement",), ("other",))
        object.__setattr__(invocation, "run_id", "mutated")
        begin("system", "prompt", {})
        return Output(7)
    node = _Consumer(body)
    eval_node(node, None, VariablePool(), memory_context_opener=_opener(events),
              memory_invocation=invocation)
    assert events[0][1].static_path == ("work",)
    assert events[0][2].invocation == ConversationInvocation("original", 7)


def test_malformed_scope_is_host_fault_even_for_custom_run_override():
    class CustomRun(_Consumer):
        def run(self, inputs, **caps):
            pytest.fail("invalid scope must be rejected before running")
    node = CustomRun(None)
    node.memory_scope = "invalid"
    with pytest.raises(TypeError, match="MemoryScope"):
        _evaluate(node, _opener([]))


@pytest.mark.parametrize("streaming", [False, True])
def test_generator_and_body_errors_close_failed_without_masking_node_failure(streaming):
    events = []
    def body(begin):
        begin("system", "prompt", {})
        raise ValueError("body rejected")
    def stream(begin):
        begin("system", "prompt", {})
        yield StreamChunk("work", "output", "one")
        raise ValueError("body rejected")
    result = _evaluate(_Consumer(stream if streaming else body), _opener(events))
    assert result.failure.code == "node.execute"
    assert result.failure.message == "body rejected"
    assert events[-1][0] == "failed"


@pytest.mark.parametrize("failure_stage", ["body", "postassertion", "result"])
def test_real_turn_rethrow_is_not_reported_as_secondary_cleanup_failure(failure_stage):
    from agent_composer.memory.session import SessionMemory

    error = ValueError("body rejected")
    def body(begin):
        begin("system", "prompt", {})
        if failure_stage == "body":
            raise error
        return None if failure_stage == "result" else Output(7)
    node = _Consumer(body)
    if failure_stage == "postassertion":
        node.post_asserts = [Assertion("false", "must fail")]
    memory = SessionMemory()
    runtime = _open_session_runtime(memory)
    try:
        result = _evaluate(node, runtime.open_context)
        assert isinstance(result, EvaluationFailure)
        assert "Memory context cleanup" not in result.failure.traceback
        assert not getattr(error, "__notes__", [])
        with memory.open_run() as opened:
            assert opened.snapshot["conversation"]["closed_invocations"][-1]["status"] == "failed"
    finally:
        runtime.close()


def test_real_backend_cleanup_note_survives_node_error_unwrapping(monkeypatch):
    from agent_composer.memory import ConversationClosed
    from agent_composer.memory import session as backend

    error = ValueError("body rejected")
    def body(begin):
        begin("system", "prompt", {})
        raise error
    node = _Consumer(body)
    memory = backend.SessionMemory()
    runtime = _open_session_runtime(memory)
    publish = backend._Turn._publish
    def fail_close(turn, checkpoint):
        if isinstance(checkpoint, ConversationClosed):
            raise RuntimeError("close storage failed")
        return publish(turn, checkpoint)
    monkeypatch.setattr(backend._Turn, "_publish", fail_close)
    try:
        outcome = _evaluate(node, runtime.open_context)
        assert outcome.failure.code == "node.execute"
        if hasattr(error, "add_note"):
            assert "close storage failed" in outcome.failure.traceback
            assert any("close storage failed" in note for note in getattr(error, "__notes__", []))
    finally:
        runtime.close()


def _fifo_scenario():
    """Hold a streaming execution open while the next same-leaf turn queues."""
    from agent_composer.memory.session import SessionMemory

    memory = SessionMemory()
    paused, release, second_entered = threading.Event(), threading.Event(), threading.Event()
    prior = []
    def first_body(begin):
        begin("system", "prompt", {})
        yield StreamChunk("work", "output", "one")
        return None  # Shape rejection must happen before the lease is released.
    def second_body(begin):
        turn = begin("system", "prompt", {})
        prior.extend(turn.transcript.view.closed_invocations)
        second_entered.set()
        return Output(7)
    first, second = _Consumer(first_body), _Consumer(second_body)
    runtime = _open_session_runtime(memory)
    def emit(chunk):
        paused.set()
        assert release.wait(3)
    def evaluate(node, run, emit_stream=None):
        return eval_node(node, None, VariablePool(), memory_context_opener=runtime.open_context,
                         memory_invocation=ConversationInvocation(run, 1), emit_stream=emit_stream)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            a = executor.submit(evaluate, first, "a", emit)
            try:
                assert paused.wait(3)
                b = executor.submit(evaluate, second, "b")
                with memory._condition:
                    assert memory._condition.wait_for(lambda: memory._queue, timeout=3)
                assert not second_entered.is_set()
            finally:
                release.set()
            assert a.result(3).outcome.failure.code == "contract.node_result"
            assert b.result(3).outcome == Output(7)
        assert len(prior) == 1
        assert prior[0].invocation == ConversationInvocation("a", 1)
        assert prior[0].status == "failed"
        memory.reset()
    finally:
        runtime.close()


def test_fifo_turn_covers_stream_drain_and_shape_check_in_bounded_child():
    child = subprocess.run(
        [sys.executable, "-c", "from tests.engine.test_memory_context_capability import _fifo_scenario; _fifo_scenario()"],
        capture_output=True, text=True, timeout=15,
    )
    assert child.returncode == 0, child.stdout + child.stderr


def test_missing_scoped_opener_remains_setup_fault_for_custom_run():
    class CustomRun(_Consumer):
        def run(self, inputs, **caps):
            pytest.fail("missing memory injection must be rejected before execution")

    with pytest.raises(RuntimeError, match="requires an injected memory context"):
        _evaluate(CustomRun(None), None)
