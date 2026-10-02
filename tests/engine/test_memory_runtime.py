"""Run-scoped setup and recall for injected memory implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any
from types import SimpleNamespace

import pytest

from agent_composer.diagnostics import OriginRef
from agent_composer.memory import (
    ConversationInvocation,
    DeclSite,
    MemoryContent,
    MemoryContextRequest,
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
    MemorySpec,
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


def _enter_context(runtime, scope, request, *, check_stopped):
    with runtime.open_context(scope, request, check_stopped=check_stopped) as context:
        return context


def _recall_sections(runtime, scope, context):
    request = MemoryContextRequest(context, None, {})
    with runtime.open_context(scope, request, check_stopped=lambda: None) as admitted:
        return admitted.sections


@pytest.mark.parametrize("capable", [(True, True), (True, False), (False, True), (False, False)])
def test_context_recalls_every_frame_and_opens_only_innermost_contributed_source(capable):
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
                frame = policies[spec.policy_name].session.frames[spec.ref]
                source = SimpleNamespace(open_turn=opener(spec.ref))
                frame.value = MemoryContent(frame.value, source)
        for leaf in ("agent", "sibling"):
            # A sibling inherits the flow frame, never the first leaf's local frame.
            scope = _scope(layout) if leaf == "agent" else MemoryScope(
                "root", ("root", leaf), (layout[0].ref,),
            )
            request = MemoryContextRequest(
                RecallContext("system", "prompt", {}, scope.static_path),
                ConversationInvocation("run", 1), {},
            )
            with runtime.open_context(scope, request, check_stopped=lambda: None) as context:
                expected = next((spec.ref for spec, enabled in reversed(tuple(zip(layout, capable)))
                                 if enabled and spec.ref in scope.frame_stack), None)
                assert context.transcript == expected
                assert [section.text for section in context.sections] == (
                    ["beta", "alpha"] if leaf == "agent" else ["beta"]
                )
        assert len(calls) == sum(bool(any(enabled and spec.ref in scope.frame_stack
            for spec, enabled in zip(layout, capable)))
            for scope in (_scope(layout), MemoryScope("root", ("root", "sibling"), (layout[0].ref,))))
        assert len(policies["beta"].session.calls[layout[0].ref]) == 2
        assert len(policies["alpha"].session.calls[layout[1].ref]) == 1
        assert all(request.consumer_static_path[-1] in {"agent", "sibling"} for _, request in calls)
    finally:
        runtime.close()


def test_context_validates_scope_request_and_full_opened_coverage():
    layout, origins = _layout()
    runtime = MemoryRuntime.open(layout=layout[1:], origins={layout[1].ref: origins[layout[1].ref]},
                                 memory={"alpha": _Memory("alpha", [])})
    scope = _scope(layout)
    request = MemoryContextRequest(
                RecallContext("system", "prompt", {}, scope.static_path),
                ConversationInvocation("run", 1), {},
            )
    try:
        with pytest.raises(TypeError, match="MemoryScope"):
            _enter_context(runtime, None, request, check_stopped=lambda: None)
        with pytest.raises(TypeError, match="MemoryContextRequest"):
            _enter_context(runtime, scope, None, check_stopped=lambda: None)
        with pytest.raises(TypeError, match="callable"):
            _enter_context(runtime, scope, request, check_stopped=None)
        wrong = MemoryContextRequest(
            RecallContext("system", "prompt", {}, ("root", "other")), request.invocation, {},
        )
        with pytest.raises(ValueError, match="scope"):
            _enter_context(runtime, scope, wrong, check_stopped=lambda: None)
        with pytest.raises(RuntimeError, match="opened layout"):
            _enter_context(runtime, scope, request, check_stopped=lambda: None)
    finally:
        runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        _enter_context(runtime, scope, request, check_stopped=lambda: None)


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

    sections = _recall_sections(runtime, _scope(layout), context)

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
        _recall_sections(runtime, _scope(layout), _context())

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

    with pytest.raises(MemoryRecallError, match="invalid recall content") as exc:
        _recall_sections(runtime, _scope(layout), _context())

    assert "secret" not in str(exc.value)
    assert exc.value.origin == origins[layout[0].ref]
    runtime.close()


def test_context_ignores_uncontributed_frame_capabilities():
    layout, origins = _layout()
    alpha = _Memory("alpha", [], values={layout[1].ref: "local notes"})
    beta = _Memory("beta", [], values={layout[0].ref: "outer notes"})
    runtime = MemoryRuntime.open(layout=layout, origins=origins, memory={"alpha": alpha, "beta": beta})
    def forbidden(*args, **kwargs):
        pytest.fail("a frame's incidental open_turn is not a recall contribution")
    alpha.session.frames[layout[1].ref].open_turn = forbidden
    try:
        request = MemoryContextRequest(_context(), None, {})
        with runtime.open_context(_scope(layout), request, check_stopped=lambda: None) as context:
            assert context.transcript is None
            assert [section.text for section in context.sections] == ["outer notes", "local notes"]
    finally:
        runtime.close()


def test_transcript_admission_requires_invocation_before_opening_source():
    layout, origins = _layout()
    def forbidden(*args, **kwargs):
        pytest.fail("missing invocation must reject admission before the source opens")
    source = SimpleNamespace(open_turn=forbidden)
    alpha = _Memory("alpha", [], values={layout[1].ref: MemoryContent(transcript=source)})
    runtime = MemoryRuntime.open(layout=layout, origins=origins, memory={"alpha": alpha, "beta": _Memory("beta", [])})
    try:
        with pytest.raises(ValueError, match="accepted invocation"):
            with runtime.open_context(_scope(layout), MemoryContextRequest(_context(), None, {}), check_stopped=lambda: None):
                pytest.fail("missing invocation admitted")
    finally:
        runtime.close()


def test_stop_during_recall_prevents_transcript_source_acquisition():
    from threading import Event
    from agent_composer.nodes.base import _HostStop

    layout, origins = _layout()
    alpha, beta = _Memory("alpha", []), _Memory("beta", [])
    runtime = MemoryRuntime.open(
        layout=layout, origins=origins, memory={"alpha": alpha, "beta": beta},
    )
    stopped = Event()
    source_calls = []
    stop = _HostStop(primary=False)

    @contextmanager
    def open_turn(request, *, check_stopped):
        source_calls.append(request)
        yield object()

    def recall(context):
        stopped.set()
        return MemoryContent("notes", SimpleNamespace(open_turn=open_turn))

    def check_stopped():
        if stopped.is_set():
            raise stop

    alpha.session.frames[layout[1].ref].recall = recall
    request = MemoryContextRequest(_context(), ConversationInvocation("run", 1), {})
    try:
        with pytest.raises(_HostStop) as caught:
            with runtime.open_context(_scope(layout), request, check_stopped=check_stopped):
                check_stopped()
        assert caught.value is stop
        assert source_calls == []
    finally:
        runtime.close()


def _table() -> tuple[MemorySpec, ...]:
    """Return the key table matching ``_layout``: AGENT alpha, flow beta."""
    return (
        MemorySpec("alpha", "alpha", {}, (DeclSite((), ("root",), "agent"),), True),
        MemorySpec("beta", "beta", {"limit": 2}, (DeclSite((), ("root",), None),), False),
    )


@pytest.mark.parametrize("contributor", ["alpha", "beta"])
def test_context_admits_a_transcript_only_from_an_eligible_memory(contributor):
    layout, origins = _layout()
    policies = {name: _Memory(name, [], values={spec.ref: name for spec in layout})
                for name in ("alpha", "beta")}
    runtime = MemoryRuntime.open(layout=layout, origins=origins, memory=policies, memories=_table())
    opened = []

    @contextmanager
    def open_turn(request, *, check_stopped):
        opened.append(request)
        yield "transcript"

    try:
        spec = next(spec for spec in layout if spec.policy_name == contributor)
        frame = policies[contributor].session.frames[spec.ref]
        frame.value = MemoryContent(frame.value, SimpleNamespace(open_turn=open_turn))
        request = MemoryContextRequest(_context(), ConversationInvocation("run", 1), {})
        if contributor == "alpha":
            with runtime.open_context(_scope(layout), request, check_stopped=lambda: None) as context:
                assert context.transcript == "transcript"
            assert len(opened) == 1
        else:
            with pytest.raises(
                MemoryRecallError,
                match="memory 'beta' returned a transcript but is not conversation-eligible",
            ) as caught:
                _enter_context(runtime, _scope(layout), request, check_stopped=lambda: None)
            assert caught.value.origin == origins[layout[0].ref]
            assert opened == []
    finally:
        runtime.close()


def test_context_rejects_a_second_transcript() -> None:
    layout, origins = _layout()
    policies = {name: _Memory(name, [], values={spec.ref: name for spec in layout})
                for name in ("alpha", "beta")}
    # Artificial table: both keys eligible, so only the second-transcript rule can fire.
    table = (
        MemorySpec("alpha", "alpha", {}, (DeclSite((), ("root",), "agent"),), True),
        MemorySpec("beta", "beta", {"limit": 2}, (DeclSite((), ("root",), "other"),), True),
    )
    runtime = MemoryRuntime.open(layout=layout, origins=origins, memory=policies, memories=table)

    @contextmanager
    def open_turn(request, *, check_stopped):
        yield "transcript"

    try:
        for spec in layout:
            frame = policies[spec.policy_name].session.frames[spec.ref]
            frame.value = MemoryContent(frame.value, SimpleNamespace(open_turn=open_turn))
        request = MemoryContextRequest(_context(), ConversationInvocation("run", 1), {})
        with pytest.raises(
            MemoryRecallError,
            match="memory 'alpha' returned a second transcript for 'root/agent'",
        ) as caught:
            _enter_context(runtime, _scope(layout), request, check_stopped=lambda: None)
        assert caught.value.origin == origins[layout[1].ref]
    finally:
        runtime.close()
