"""Run-scoped setup, recall and learning for injected memory implementations."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from yian.diagnostics import OriginRef
from yian.memory import (
    ConversationInvocation,
    DeclSite,
    LearnContext,
    MemoryContent,
    MemoryContextRequest,
    MemoryMutation,
    MemoryOperationError,
    MemoryRecallError,
    MemoryScope,
    MemorySetupError,
    MemorySnapshotEnvelope,
    MemorySpec,
    RecallContext,
)
from yian.runtime.contracts import MemoryProposal
from yian.runtime.log import MemoryCommitRequestedRecord
from yian.runtime.memory_runtime import MemoryRuntime, _LearningProposalError


class _Session:
    """One observable fake session for one memory key."""

    def __init__(
        self,
        name: str,
        events: list[tuple[Any, ...]],
        value: object = "",
        *,
        learn_value: object = None,
        commit_value: object = None,
        snapshot: object | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.value = value
        self.learn_value = learn_value
        self.commit_value = commit_value
        self._snapshot = {"policy": name} if snapshot is None else snapshot
        self.close_error = close_error
        self.calls: list[RecallContext] = []
        self.learn_calls: list[LearnContext] = []

    @property
    def snapshot(self):
        self.events.append(("snapshot", self.name))
        return self._snapshot

    def recall(self, context: RecallContext | None = None) -> object:
        assert context is not None
        self.calls.append(context)
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        self.learn_calls.append(context)
        if isinstance(self.learn_value, BaseException):
            raise self.learn_value
        if self.learn_value is None:
            return []
        return self.learn_value  # type: ignore[return-value]

    def commit(self, commit_id: str, mutations: tuple[MemoryMutation, ...]) -> None:
        self.events.append(("commit", self.name, commit_id, mutations))
        if isinstance(self.commit_value, BaseException):
            raise self.commit_value
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
        value: object = "",
        *,
        learn_value: object = None,
        commit_value: object = None,
        session_snapshot: object | None = None,
        open_error: BaseException | None = None,
        close_error: BaseException | None = None,
        restore_snapshot: bool = True,
    ) -> None:
        self.name = name
        self.events = events
        self.open_error = open_error
        self.restore_snapshot = restore_snapshot
        self.session = _Session(
            name,
            events,
            value,
            learn_value=learn_value,
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


def _table() -> tuple[MemorySpec, ...]:
    """Return the recorded table: AGENT-owned alpha and flow-owned beta."""
    return (
        MemorySpec("alpha", "alpha", {}, (DeclSite((), ("root",), "agent"),), True),
        MemorySpec("beta", "beta", {"limit": 2}, (DeclSite((), ("root",), None),), False),
    )


def _origins() -> dict[str, OriginRef | None]:
    """Return one authored origin per key."""
    return {"alpha": OriginRef("local-site"), "beta": OriginRef("outer-site")}


def _open(memory: dict[str, Any], **kwargs: Any) -> MemoryRuntime:
    """Open the two-key test table over ``memory``."""
    return MemoryRuntime.open(memories=_table(), origins=_origins(), memory=memory, **kwargs)


def _scope() -> MemoryScope:
    """Return the AGENT consumer scope: flow key beta outside AGENT key alpha."""
    return MemoryScope(("agent",), ("beta", "alpha"))


def _context() -> RecallContext:
    """Return one exact visible AGENT recall context."""
    return RecallContext("base system", "rendered prompt", {"question": "why"}, ("agent",))


def _enter_context(runtime, scope, request, *, check_stopped):
    with runtime.open_context(scope, request, check_stopped=check_stopped) as context:
        return context


def _recall_sections(runtime, scope, context):
    request = MemoryContextRequest(context, None, {})
    with runtime.open_context(scope, request, check_stopped=lambda: None) as admitted:
        return admitted.sections


def test_context_recalls_every_key_in_stack_order_and_opens_the_eligible_transcript():
    policies = {name: _Memory(name, [], name) for name in ("alpha", "beta")}
    runtime = _open(policies)
    calls = []

    @contextmanager
    def open_turn(request, *, check_stopped):
        calls.append(request)
        check_stopped()
        yield "alpha transcript"

    policies["alpha"].session.value = MemoryContent("alpha", SimpleNamespace(open_turn=open_turn))
    try:
        for scope, transcript, texts in (
            (_scope(), "alpha transcript", ["beta", "alpha"]),
            # A sibling sees the flow key, never the first leaf's AGENT key.
            (MemoryScope(("sibling",), ("beta",)), None, ["beta"]),
        ):
            request = MemoryContextRequest(
                RecallContext("system", "prompt", {}, scope.static_path),
                ConversationInvocation("run", 1),
                {},
            )
            with runtime.open_context(scope, request, check_stopped=lambda: None) as context:
                assert context.transcript == transcript
                assert [section.text for section in context.sections] == texts
                assert [section.memory for section in context.sections] == texts
        assert len(policies["beta"].session.calls) == 2
        assert len(policies["alpha"].session.calls) == 1
        assert [request.consumer_static_path for request in calls] == [("agent",)]
    finally:
        runtime.close()


def test_context_validates_scope_request_and_full_opened_coverage():
    runtime = MemoryRuntime.open(
        memories=_table()[:1],
        origins={"alpha": OriginRef("local-site")},
        memory={"alpha": _Memory("alpha", [])},
    )
    scope = _scope()
    request = MemoryContextRequest(
        RecallContext("system", "prompt", {}, scope.static_path),
        ConversationInvocation("run", 1),
        {},
    )
    try:
        with pytest.raises(TypeError, match="MemoryScope"):
            _enter_context(runtime, None, request, check_stopped=lambda: None)
        with pytest.raises(TypeError, match="MemoryContextRequest"):
            _enter_context(runtime, scope, None, check_stopped=lambda: None)
        with pytest.raises(TypeError, match="callable"):
            _enter_context(runtime, scope, request, check_stopped=None)
        wrong = MemoryContextRequest(
            RecallContext("system", "prompt", {}, ("other",)), request.invocation, {},
        )
        with pytest.raises(ValueError, match="scope"):
            _enter_context(runtime, scope, wrong, check_stopped=lambda: None)
        with pytest.raises(RuntimeError, match="opened table"):
            _enter_context(runtime, scope, request, check_stopped=lambda: None)
    finally:
        runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        _enter_context(runtime, scope, request, check_stopped=lambda: None)


def test_open_validates_the_memory_table_and_its_origins() -> None:
    events: list[tuple[Any, ...]] = []
    memory = {"alpha": _Memory("alpha", events), "beta": _Memory("beta", events)}
    alpha, beta = _table()
    with pytest.raises(TypeError, match="MemorySpec"):
        MemoryRuntime.open(memories=list(_table()), origins=_origins(), memory=memory)
    with pytest.raises(ValueError, match="sorted by key and unique"):
        MemoryRuntime.open(memories=(beta, alpha), origins=_origins(), memory=memory)
    with pytest.raises(ValueError, match="sorted by key and unique"):
        MemoryRuntime.open(memories=(alpha, alpha), origins={"alpha": None}, memory=memory)
    with pytest.raises(ValueError, match="origins must exactly match"):
        MemoryRuntime.open(memories=_table(), origins={"alpha": None}, memory=memory)
    with pytest.raises(TypeError, match="OriginRef"):
        MemoryRuntime.open(
            memories=_table(), origins={"alpha": "site", "beta": None}, memory=memory,
        )
    assert events == []
    empty = MemoryRuntime.open(memories=(), origins={}, memory={})
    assert empty.memories == () and empty.snapshots == ()
    empty.close()


def test_open_uses_key_order_and_captures_snapshots() -> None:
    events: list[tuple[Any, ...]] = []
    runtime = _open({"beta": _Memory("beta", events), "alpha": _Memory("alpha", events)})

    assert events == [
        ("open_run", "alpha", None),
        ("open_run", "beta", None),
        ("snapshot", "alpha"),
        ("snapshot", "beta"),
    ]
    assert runtime.memories == _table()
    assert runtime.snapshots == (
        MemorySnapshotEnvelope("alpha", {"policy": "alpha"}),
        MemorySnapshotEnvelope("beta", {"policy": "beta"}),
    )

    runtime.close()
    runtime.close()
    assert events[-2:] == [("close", "beta"), ("close", "alpha")]


def test_open_restores_exact_recorded_snapshot_per_key() -> None:
    events: list[tuple[Any, ...]] = []
    snapshots = (
        MemorySnapshotEnvelope("alpha", {"revision": 2}),
        MemorySnapshotEnvelope("beta", {"revision": 3}),
    )

    runtime = _open(
        {"alpha": _Memory("alpha", events), "beta": _Memory("beta", events)},
        snapshots=snapshots,
    )

    assert events[:2] == [
        ("open_run", "alpha", {"revision": 2}),
        ("open_run", "beta", {"revision": 3}),
    ]
    runtime.close()


def test_restore_rejects_noncanonical_snapshot_order_before_opening() -> None:
    events: list[tuple[Any, ...]] = []
    snapshots = (
        MemorySnapshotEnvelope("beta", {"revision": 3}),
        MemorySnapshotEnvelope("alpha", {"revision": 2}),
    )

    with pytest.raises(ValueError, match="canonical order"):
        _open(
            {"alpha": _Memory("alpha", events), "beta": _Memory("beta", events)},
            snapshots=snapshots,
        )

    assert events == []


def test_restore_rejects_a_session_that_changes_the_recorded_snapshot() -> None:
    events: list[tuple[Any, ...]] = []
    snapshots = (
        MemorySnapshotEnvelope("alpha", {"revision": 2}),
        MemorySnapshotEnvelope("beta", {"revision": 3}),
    )

    with pytest.raises(MemorySetupError, match="did not reproduce") as exc:
        _open(
            {
                "alpha": _Memory("alpha", events),
                "beta": _Memory("beta", events, restore_snapshot=False),
            },
            snapshots=snapshots,
        )

    assert exc.value.origin == _origins()["beta"]
    assert events[-2:] == [("close", "beta"), ("close", "alpha")]


def test_registry_rejects_aliases_and_attributes_missing_key_origin() -> None:
    events: list[tuple[Any, ...]] = []
    shared = _Memory("shared", events)
    with pytest.raises(ValueError, match="multiple names"):
        _open({"alpha": shared, "beta": shared})

    with pytest.raises(MemorySetupError, match="memory 'beta' was not supplied") as exc:
        _open({"alpha": _Memory("alpha", events)})
    assert exc.value.origin == _origins()["beta"]


def test_later_open_failure_closes_prior_session_and_preserves_origin() -> None:
    events: list[tuple[Any, ...]] = []
    failure = MemoryOperationError("open_run", "beta.unavailable", "temporarily unavailable", True)

    with pytest.raises(MemorySetupError, match="temporarily unavailable") as exc:
        _open({"alpha": _Memory("alpha", events), "beta": _Memory("beta", events, open_error=failure)})

    assert exc.value.origin == _origins()["beta"]
    assert events == [
        ("open_run", "alpha", None),
        ("open_run", "beta", None),
        ("close", "alpha"),
    ]


def test_frame_era_session_is_closed_before_setup_failure() -> None:
    events: list[tuple[Any, ...]] = []

    class _FrameSession:
        """A session with the removed frame surface and no recall or learn."""

        snapshot: dict[str, Any] = {}

        def open_frame(self, spec: object) -> object:
            raise AssertionError(spec)

        def commit(self, commit_id: str, mutations: tuple[MemoryMutation, ...]) -> None:
            raise AssertionError(commit_id)

        def close(self) -> None:
            events.append(("close", "alpha-invalid"))

    class _FrameMemory:
        def open_run(self, *, snapshot=None) -> _FrameSession:
            events.append(("open_run", "alpha-invalid", snapshot))
            return _FrameSession()

        def reset(self) -> None:
            pass

    with pytest.raises(MemorySetupError, match="memory 'alpha' returned an invalid run session") as exc:
        _open({"alpha": _FrameMemory(), "beta": _Memory("beta", events)})

    assert exc.value.origin == _origins()["alpha"]
    assert events == [
        ("open_run", "alpha-invalid", None),
        ("close", "alpha-invalid"),
    ]


def test_recall_returns_nonempty_outer_to_local_sections_with_exact_context() -> None:
    events: list[tuple[Any, ...]] = []
    alpha = _Memory("alpha", events, "local lesson")
    beta = _Memory("beta", events, "")
    runtime = _open({"alpha": alpha, "beta": beta})
    context = _context()

    sections = _recall_sections(runtime, _scope(), context)

    # Empty outer recall is omitted but the local key stays explicit.
    assert len(sections) == 1
    assert sections[0].memory == "alpha"
    assert sections[0].text == "local lesson"
    assert beta.session.calls == [context]
    assert alpha.session.calls == [context]
    runtime.close()


def test_recall_wraps_typed_failure_with_the_key_origin() -> None:
    events: list[tuple[Any, ...]] = []
    failure = MemoryOperationError("recall", "beta.search", "index unavailable", True)
    runtime = _open({"alpha": _Memory("alpha", events), "beta": _Memory("beta", events, failure)})

    with pytest.raises(MemoryRecallError, match="memory 'beta' recall failed: index unavailable") as exc:
        _recall_sections(runtime, _scope(), _context())

    assert exc.value.origin == _origins()["beta"]
    assert exc.value.__cause__ is failure
    runtime.close()


def test_recall_wraps_noncanonical_text_without_exposing_it() -> None:
    class _String(str):
        pass

    events: list[tuple[Any, ...]] = []
    runtime = _open({
        "alpha": _Memory("alpha", events),
        "beta": _Memory("beta", events, _String("secret")),
    })

    with pytest.raises(MemoryRecallError, match="invalid recall content") as exc:
        _recall_sections(runtime, _scope(), _context())

    assert "secret" not in str(exc.value)
    assert exc.value.origin == _origins()["beta"]
    runtime.close()


def test_context_ignores_uncontributed_session_capabilities():
    alpha = _Memory("alpha", [], "local notes")
    beta = _Memory("beta", [], "outer notes")
    runtime = _open({"alpha": alpha, "beta": beta})

    def forbidden(*args, **kwargs):
        pytest.fail("a session's incidental open_turn is not a recall contribution")

    alpha.session.open_turn = forbidden
    try:
        request = MemoryContextRequest(_context(), None, {})
        with runtime.open_context(_scope(), request, check_stopped=lambda: None) as context:
            assert context.transcript is None
            assert [section.text for section in context.sections] == ["outer notes", "local notes"]
    finally:
        runtime.close()


def test_transcript_admission_requires_invocation_before_opening_source():
    def forbidden(*args, **kwargs):
        pytest.fail("missing invocation must reject admission before the source opens")

    source = SimpleNamespace(open_turn=forbidden)
    alpha = _Memory("alpha", [], MemoryContent(transcript=source))
    runtime = _open({"alpha": alpha, "beta": _Memory("beta", [])})
    try:
        with pytest.raises(ValueError, match="accepted invocation"):
            with runtime.open_context(
                _scope(), MemoryContextRequest(_context(), None, {}), check_stopped=lambda: None,
            ):
                pytest.fail("missing invocation admitted")
    finally:
        runtime.close()


def test_stop_during_recall_prevents_transcript_source_acquisition():
    from threading import Event

    from yian.nodes.base import _HostStop

    alpha, beta = _Memory("alpha", []), _Memory("beta", [])
    runtime = _open({"alpha": alpha, "beta": beta})
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

    alpha.session.recall = recall
    request = MemoryContextRequest(_context(), ConversationInvocation("run", 1), {})
    try:
        with pytest.raises(_HostStop) as caught:
            with runtime.open_context(_scope(), request, check_stopped=check_stopped):
                check_stopped()
        assert caught.value is stop
        assert source_calls == []
    finally:
        runtime.close()


@pytest.mark.parametrize("contributor", ["alpha", "beta"])
def test_context_admits_a_transcript_only_from_an_eligible_memory(contributor):
    policies = {name: _Memory(name, [], name) for name in ("alpha", "beta")}
    runtime = _open(policies)
    opened = []

    @contextmanager
    def open_turn(request, *, check_stopped):
        opened.append(request)
        yield "transcript"

    try:
        session = policies[contributor].session
        session.value = MemoryContent(contributor, SimpleNamespace(open_turn=open_turn))
        request = MemoryContextRequest(_context(), ConversationInvocation("run", 1), {})
        if contributor == "alpha":
            with runtime.open_context(_scope(), request, check_stopped=lambda: None) as context:
                assert context.transcript == "transcript"
            assert len(opened) == 1
        else:
            with pytest.raises(
                MemoryRecallError,
                match="memory 'beta' returned a transcript but is not conversation-eligible",
            ) as caught:
                _enter_context(runtime, _scope(), request, check_stopped=lambda: None)
            assert caught.value.origin == _origins()["beta"]
            assert opened == []
    finally:
        runtime.close()


def test_context_rejects_a_second_transcript_for_one_agent():
    table = (
        MemorySpec("alpha", "alpha", {}, (DeclSite((), ("root",), "agent"),), True),
        MemorySpec("gamma", "gamma", {}, (DeclSite((), ("root",), "agent"),), True),
    )
    source = SimpleNamespace(open_turn=lambda *a, **k: pytest.fail("no transcript may open"))
    policies = {name: _Memory(name, [], MemoryContent(name, source)) for name in ("alpha", "gamma")}
    runtime = MemoryRuntime.open(
        memories=table, origins={"alpha": None, "gamma": OriginRef("gamma-site")}, memory=policies,
    )
    request = MemoryContextRequest(_context(), ConversationInvocation("run", 1), {})
    try:
        with pytest.raises(MemoryRecallError, match="memory 'gamma' returned a second transcript") as caught:
            _enter_context(
                runtime, MemoryScope(("agent",), ("alpha", "gamma")), request, check_stopped=lambda: None,
            )
        assert caught.value.origin == OriginRef("gamma-site")
    finally:
        runtime.close()


def test_propose_learning_returns_one_proposal_per_key_in_table_order() -> None:
    mutation = MemoryMutation("m1", {"add": 1})
    alpha = _Memory("alpha", [], learn_value=[mutation])
    beta = _Memory("beta", [])
    runtime = _open({"alpha": alpha, "beta": beta})
    contexts = (LearnContext("run", "alpha", ()), LearnContext("run", "beta", (), feedback=("good",)))
    try:
        proposals = runtime.propose_learning(contexts)
        assert proposals == (MemoryProposal("alpha", (mutation,)), MemoryProposal("beta", ()))
        assert [context.memory for context in alpha.session.learn_calls] == ["alpha"]
        assert beta.session.learn_calls[0].feedback == ("good",)
        with pytest.raises(ValueError, match="complete recorded memory table in order"):
            runtime.propose_learning(tuple(reversed(contexts)))
        with pytest.raises(ValueError, match="complete recorded memory table in order"):
            runtime.propose_learning(contexts[:1])
        with pytest.raises(ValueError, match="one run"):
            runtime.propose_learning((contexts[0], LearnContext("other", "beta", ())))
    finally:
        runtime.close()


def test_propose_learning_attributes_a_failure_to_its_key() -> None:
    failure = MemoryOperationError("learn", "beta.down", "store down", False)
    runtime = _open({"alpha": _Memory("alpha", []), "beta": _Memory("beta", [], learn_value=failure)})
    contexts = (LearnContext("run", "alpha", ()), LearnContext("run", "beta", ()))
    try:
        with pytest.raises(_LearningProposalError) as caught:
            runtime.propose_learning(contexts)
        assert caught.value.memory == "beta"
        assert caught.value.failure.code == "beta.down"
        assert not hasattr(caught.value, "frame_ref")
    finally:
        runtime.close()


def test_commit_learning_commits_on_the_named_key_session() -> None:
    events: list[tuple[Any, ...]] = []
    runtime = _open({"alpha": _Memory("alpha", events), "beta": _Memory("beta", events)})
    mutation = MemoryMutation("m1", {"add": 1})
    try:
        request = MemoryCommitRequestedRecord("beta", "commit-1", (mutation,), attempt_id=1)
        assert runtime.commit_learning(request) is None
        assert events[-1] == ("commit", "beta", "commit-1", (mutation,))
        with pytest.raises(ValueError, match="enabled memory"):
            runtime.commit_learning(
                MemoryCommitRequestedRecord("gamma", "commit-2", (), attempt_id=1)
            )
    finally:
        runtime.close()
