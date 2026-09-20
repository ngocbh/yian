"""Run-scoped setup and recall for injected memory implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any

import pytest

from agent_composer.diagnostics import OriginRef
from agent_composer.memory import (
    ConversationInvocation,
    ConversationRequest,
    LearnContext,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOperationError,
    MemoryOwner,
    MemoryRecallError,
    MemoryScope,
    MemorySetupError,
    MemorySnapshotEnvelope,
    RecallContext,
)
from agent_composer.compile.model import END_ID, START_ID
from agent_composer.events import RunSucceeded
from agent_composer.runtime.log import (
    MemoryRunLog,
)
from agent_composer.runtime.memory_runtime import (
    MemoryRuntime,
)
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_node_started,
    prepare_node_succeeded,
    prepare_run_started,
)
from agent_composer.typesys import VariablePool
from tests.engine._fakes import FuncNode
from tests.engine._graph_builder import _forward_capture, _graph


@dataclass
class _Frame:
    """Tiny policy frame whose recall behavior is supplied by the test."""

    value: object
    calls: list[RecallContext]
    learn_value: object = None
    learn_calls: list[LearnContext] = field(default_factory=list)

    def recall(self, context: RecallContext | None = None) -> str:
        assert context is not None
        self.calls.append(context)
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value  # type: ignore[return-value]

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        self.learn_calls.append(context)
        if isinstance(self.learn_value, BaseException):
            raise self.learn_value
        if callable(self.learn_value):
            return self.learn_value(context)
        if self.learn_value is None:
            return []
        return self.learn_value  # type: ignore[return-value]


class _Session:
    """One observable fake session with independently configured frames."""

    def __init__(
        self,
        name: str,
        events: list[tuple[Any, ...]],
        values: dict[MemoryFrameRef, object],
        *,
        learn_values: dict[MemoryFrameRef, object] | None = None,
        commit_value: object = None,
        snapshot: object | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.values = values
        self.learn_values = learn_values or {}
        self.commit_value = commit_value
        self._snapshot = {"policy": name} if snapshot is None else snapshot
        self.close_error = close_error
        self.calls: dict[MemoryFrameRef, list[RecallContext]] = {}
        self.frames: dict[MemoryFrameRef, _Frame] = {}

    @property
    def snapshot(self):
        self.events.append(("snapshot", self.name))
        return self._snapshot

    def open_frame(self, spec: MemoryFrameSpec) -> _Frame:
        self.events.append(("open_frame", self.name, spec.ref))
        calls: list[RecallContext] = []
        self.calls[spec.ref] = calls
        frame = _Frame(
            self.values.get(spec.ref, ""),
            calls,
            self.learn_values.get(spec.ref),
        )
        self.frames[spec.ref] = frame
        return frame

    def commit(
        self,
        commit_id: str,
        mutations: tuple[MemoryMutation, ...],
    ) -> None:
        self.events.append(("commit", self.name, commit_id, mutations))
        if isinstance(self.commit_value, BaseException):
            raise self.commit_value
        if callable(self.commit_value):
            return self.commit_value(commit_id, mutations)
        return self.commit_value  # type: ignore[return-value]

    def close(self) -> None:
        self.events.append(("close", self.name))
        if self.close_error is not None:
            raise self.close_error


class _Memory:
    """Tiny Memory-protocol implementation returning one configured session."""

    def __init__(
        self,
        name: str,
        events: list[tuple[Any, ...]],
        values: dict[MemoryFrameRef, object] | None = None,
        *,
        learn_values: dict[MemoryFrameRef, object] | None = None,
        commit_value: object = None,
        session_snapshot: object | None = None,
        open_error: BaseException | None = None,
        close_error: BaseException | None = None,
        restore_snapshot: bool = True,
    ) -> None:
        self.name = name
        self.events = events
        self.values = values or {}
        self.session_snapshot = session_snapshot
        self.open_error = open_error
        self.restore_snapshot = restore_snapshot
        self.session = _Session(
            name,
            events,
            self.values,
            learn_values=learn_values,
            commit_value=commit_value,
            snapshot=session_snapshot,
            close_error=close_error,
        )

    def open_run(self, *, snapshot=None) -> _Session:
        self.events.append(("open_run", self.name, snapshot))
        if self.open_error is not None:
            raise self.open_error
        if snapshot is not None and self.restore_snapshot:
            self.session._snapshot = snapshot
        return self.session

    def reset(self) -> None:
        self.events.append(("reset", self.name))


def _layout() -> tuple[
    tuple[MemoryFrameSpec, MemoryFrameSpec],
    dict[MemoryFrameRef, OriginRef | None],
]:
    """Return canonical outer-flow/local-agent specs under separate policies."""
    outer = MemoryFrameRef("root", ("root",), MemoryOwner.FLOW)
    local = MemoryFrameRef("root", ("root", "agent"), MemoryOwner.AGENT)
    specs = (
        MemoryFrameSpec(outer, "beta", {"limit": 2}),
        MemoryFrameSpec(local, "alpha", {}),
    )
    origins = {
        outer: OriginRef("outer-site"),
        local: OriginRef("local-site"),
    }
    return specs, origins


def _scope(layout: tuple[MemoryFrameSpec, ...]) -> MemoryScope:
    """Return the AGENT consumer scope inheriting both test frames."""
    return MemoryScope(
        "root",
        ("root", "agent"),
        tuple(spec.ref for spec in layout),
    )


def _context() -> RecallContext:
    """Return one exact visible AGENT recall context."""
    return RecallContext(
        "base system",
        "rendered prompt",
        {"question": "why"},
        ("root", "agent"),
    )


@pytest.mark.parametrize("capable", [(True, True), (True, False), (False, False)])
def test_conversation_selects_innermost_capable_frame_without_changing_recall(capable):
    layout, origins = _layout()
    policies = {name: _Memory(name, [], values={spec.ref: name for spec in layout})
                for name in ("alpha", "beta")}
    runtime = MemoryRuntime.open(layout=layout, origins=origins, memory=policies)
    calls = []

    def opener(ref):
        @contextmanager
        def open_turn(request, *, check_stopped):
            calls.append((ref, request))
            check_stopped()
            yield ref
        return open_turn

    try:
        for spec, enabled in zip(layout, capable):
            if enabled:
                policies[spec.policy_name].session.frames[spec.ref].open_turn = opener(spec.ref)
        for leaf in ("agent", "sibling"):
            # A sibling inherits the flow frame, never the first leaf's local frame.
            scope = _scope(layout) if leaf == "agent" else MemoryScope(
                "root", ("root", leaf), (layout[0].ref,),
            )
            request = ConversationRequest(scope.static_path, ConversationInvocation("run", 1), {})
            with runtime.open_conversation(scope, request, check_stopped=lambda: None) as selected:
                expected = next((spec.ref for spec, enabled in reversed(tuple(zip(layout, capable)))
                                 if enabled and spec.ref in scope.frame_stack), None)
                assert selected == expected
        assert [section.text for section in runtime.recall(_scope(layout), _context())] == ["beta", "alpha"]
        assert all(request.consumer_static_path[-1] in {"agent", "sibling"} for _, request in calls)
    finally:
        runtime.close()


def test_conversation_validates_scope_request_and_full_opened_coverage():
    layout, origins = _layout()
    runtime = MemoryRuntime.open(layout=layout[1:], origins={layout[1].ref: origins[layout[1].ref]},
                                 memory={"alpha": _Memory("alpha", [])})
    scope = _scope(layout)
    request = ConversationRequest(scope.static_path, ConversationInvocation("run", 1), {})
    try:
        with pytest.raises(TypeError, match="MemoryScope"):
            runtime.open_conversation(None, request, check_stopped=lambda: None)
        with pytest.raises(TypeError, match="ConversationRequest"):
            runtime.open_conversation(scope, None, check_stopped=lambda: None)
        with pytest.raises(TypeError, match="callable"):
            runtime.open_conversation(scope, request, check_stopped=None)
        wrong = ConversationRequest(("root", "other"), request.invocation, {})
        with pytest.raises(ValueError, match="scope"):
            runtime.open_conversation(scope, wrong, check_stopped=lambda: None)
        with pytest.raises(RuntimeError, match="opened layout"):
            runtime.open_conversation(scope, request, check_stopped=lambda: None)
    finally:
        runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.open_conversation(scope, request, check_stopped=lambda: None)


def _terminal_learning_state(
    runtime: MemoryRuntime,
) -> tuple[RunState, MemoryRunLog]:
    """Return a terminal primary run with empty eligible episode histories."""
    node = FuncNode("work", lambda _inputs: 7)
    flow = _graph(
        [node],
        [(START_ID, node.id), (node.id, END_ID)],
    )
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, {})
    pool.add_system("run_id", "memory-runtime-run")
    state = RunState(flow)
    log = MemoryRunLog("memory-runtime-run")
    commit_prepared(
        log,
        state,
        prepare_run_started(
            state,
            RunStartData(
                run_id="memory-runtime-run",
                flow_fingerprint="f" * 64,
                initial_pool=pool,
                start_wiring_override={},
                effective_llm_by_static_path={},
                num_workers=0,
                memory_layout=runtime.layout,
                memory_snapshots=runtime.snapshots,
                learnable=True,
            ),
        ),
    )
    for node_id, value in (
        (START_ID, {}),
        (node.id, 7),
        (END_ID, None),
    ):
        commit_prepared(log, state, prepare_node_started(state, node_id))
        commit_prepared(
            log,
            state,
            prepare_node_succeeded(state, node_id, value,
                                   forward_capture=_forward_capture(state.flow.nodes[node_id], {})),
        )
    assert isinstance(state.result, RunSucceeded)
    return state, log


def test_open_uses_policy_order_frame_order_and_captures_snapshots() -> None:
    layout, origins = _layout()
    events: list[tuple[Any, ...]] = []
    alpha = _Memory("alpha", events)
    beta = _Memory("beta", events)

    runtime = MemoryRuntime.open(
        layout=layout,
        origins=origins,
        memory={"beta": beta, "alpha": alpha},
    )

    assert events == [
        ("open_run", "alpha", None),
        ("open_run", "beta", None),
        ("open_frame", "beta", layout[0].ref),
        ("open_frame", "alpha", layout[1].ref),
        ("snapshot", "alpha"),
        ("snapshot", "beta"),
    ]
    assert runtime.layout == layout
    assert runtime.snapshots == (
        MemorySnapshotEnvelope("alpha", {"policy": "alpha"}),
        MemorySnapshotEnvelope("beta", {"policy": "beta"}),
    )

    runtime.close()
    runtime.close()
    assert events[-2:] == [("close", "beta"), ("close", "alpha")]


def test_open_restores_exact_recorded_snapshot_per_policy() -> None:
    layout, origins = _layout()
    events: list[tuple[Any, ...]] = []
    snapshots = (
        MemorySnapshotEnvelope("alpha", {"revision": 2}),
        MemorySnapshotEnvelope("beta", {"revision": 3}),
    )

    runtime = MemoryRuntime.open(
        layout=layout,
        origins=origins,
        memory={
            "alpha": _Memory("alpha", events),
            "beta": _Memory("beta", events),
        },
        snapshots=snapshots,
    )

    assert events[:2] == [
        ("open_run", "alpha", {"revision": 2}),
        ("open_run", "beta", {"revision": 3}),
    ]
    runtime.close()


def test_restore_rejects_noncanonical_snapshot_order_before_opening() -> None:
    layout, origins = _layout()
    events: list[tuple[Any, ...]] = []
    snapshots = (
        MemorySnapshotEnvelope("beta", {"revision": 3}),
        MemorySnapshotEnvelope("alpha", {"revision": 2}),
    )

    with pytest.raises(ValueError, match="canonical order"):
        MemoryRuntime.open(
            layout=layout,
            origins=origins,
            memory={
                "alpha": _Memory("alpha", events),
                "beta": _Memory("beta", events),
            },
            snapshots=snapshots,
        )

    assert events == []


def test_restore_rejects_a_session_that_changes_the_recorded_snapshot() -> None:
    layout, origins = _layout()
    events: list[tuple[Any, ...]] = []
    snapshots = (
        MemorySnapshotEnvelope("alpha", {"revision": 2}),
        MemorySnapshotEnvelope("beta", {"revision": 3}),
    )

    with pytest.raises(MemorySetupError, match="did not reproduce") as exc:
        MemoryRuntime.open(
            layout=layout,
            origins=origins,
            memory={
                "alpha": _Memory("alpha", events),
                "beta": _Memory(
                    "beta",
                    events,
                    restore_snapshot=False,
                ),
            },
            snapshots=snapshots,
        )

    assert exc.value.origin == origins[layout[0].ref]
    assert events[-2:] == [("close", "beta"), ("close", "alpha")]


def test_registry_rejects_aliases_and_attributes_missing_policy_origin() -> None:
    layout, origins = _layout()
    events: list[tuple[Any, ...]] = []
    shared = _Memory("shared", events)
    with pytest.raises(ValueError, match="multiple names"):
        MemoryRuntime.open(
            layout=layout,
            origins=origins,
            memory={"alpha": shared, "beta": shared},
        )

    with pytest.raises(MemorySetupError, match="'beta' was not supplied") as exc:
        MemoryRuntime.open(
            layout=layout,
            origins=origins,
            memory={"alpha": _Memory("alpha", events)},
        )
    assert exc.value.origin == origins[layout[0].ref]


def test_later_open_failure_closes_prior_session_and_preserves_origin() -> None:
    layout, origins = _layout()
    events: list[tuple[Any, ...]] = []
    failure = MemoryOperationError(
        "open_run",
        "beta.unavailable",
        "temporarily unavailable",
        True,
    )

    with pytest.raises(MemorySetupError, match="temporarily unavailable") as exc:
        MemoryRuntime.open(
            layout=layout,
            origins=origins,
            memory={
                "alpha": _Memory("alpha", events),
                "beta": _Memory("beta", events, open_error=failure),
            },
        )

    assert exc.value.origin == origins[layout[0].ref]
    assert events == [
        ("open_run", "alpha", None),
        ("open_run", "beta", None),
        ("close", "alpha"),
    ]


def test_invalid_opened_session_is_closed_before_setup_failure() -> None:
    layout, origins = _layout()
    events: list[tuple[Any, ...]] = []

    class _InvalidSession:
        def open_frame(self, spec: MemoryFrameSpec) -> _Frame:
            raise AssertionError(spec)

        def close(self) -> None:
            events.append(("close", "alpha-invalid"))

    class _InvalidMemory:
        def open_run(self, *, snapshot=None) -> _InvalidSession:
            events.append(("open_run", "alpha-invalid", snapshot))
            return _InvalidSession()

        def reset(self) -> None:
            pass

    with pytest.raises(MemorySetupError, match="invalid run session"):
        MemoryRuntime.open(
            layout=layout,
            origins=origins,
            memory={
                "alpha": _InvalidMemory(),
                "beta": _Memory("beta", events),
            },
        )

    assert events == [
        ("open_run", "alpha-invalid", None),
        ("close", "alpha-invalid"),
    ]


def test_recall_returns_nonempty_outer_to_local_sections_with_exact_context() -> None:
    layout, origins = _layout()
    events: list[tuple[Any, ...]] = []
    alpha = _Memory("alpha", events, {layout[1].ref: "local lesson"})
    beta = _Memory("beta", events, {layout[0].ref: ""})
    runtime = MemoryRuntime.open(
        layout=layout,
        origins=origins,
        memory={"alpha": alpha, "beta": beta},
    )
    context = _context()

    sections = runtime.recall(_scope(layout), context)

    # Empty outer recall is omitted but local identity remains explicit.
    assert len(sections) == 1
    assert sections[0].frame_ref == layout[1].ref
    assert sections[0].text == "local lesson"
    assert beta.session.calls[layout[0].ref] == [context]
    assert alpha.session.calls[layout[1].ref] == [context]
    runtime.close()


def test_recall_wraps_typed_policy_failure_with_exact_frame_origin() -> None:
    layout, origins = _layout()
    events: list[tuple[Any, ...]] = []
    failure = MemoryOperationError(
        "recall",
        "beta.search",
        "index unavailable",
        True,
    )
    runtime = MemoryRuntime.open(
        layout=layout,
        origins=origins,
        memory={
            "alpha": _Memory("alpha", events),
            "beta": _Memory("beta", events, {layout[0].ref: failure}),
        },
    )

    with pytest.raises(MemoryRecallError, match="index unavailable") as exc:
        runtime.recall(_scope(layout), _context())

    assert exc.value.origin == origins[layout[0].ref]
    assert exc.value.__cause__ is failure
    runtime.close()


def test_recall_wraps_noncanonical_policy_text_without_exposing_it() -> None:
    class _String(str):
        pass

    layout, origins = _layout()
    events: list[tuple[Any, ...]] = []
    runtime = MemoryRuntime.open(
        layout=layout,
        origins=origins,
        memory={
            "alpha": _Memory("alpha", events),
            "beta": _Memory("beta", events, {layout[0].ref: _String("secret")}),
        },
    )

    with pytest.raises(MemoryRecallError, match="invalid recall text") as exc:
        runtime.recall(_scope(layout), _context())

    assert "secret" not in str(exc.value)
    assert exc.value.origin == origins[layout[0].ref]
    runtime.close()
