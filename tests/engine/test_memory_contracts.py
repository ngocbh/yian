"""Contracts for source-neutral memory identities and policy seams."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
from types import TracebackType
from typing import Any

import pytest

from agent_composer.diagnostics import OriginRef
from agent_composer.memory import (
    JsonObject,
    LearnContext,
    Memory,
    MemoryBoundary,
    MemoryBoundaryKind,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryFrame,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOccurrenceKind,
    MemoryOccurrenceRef,
    MemoryOccurrenceSegment,
    MemoryOperationError,
    MemoryOwner,
    MemoryPolicySpec,
    MemoryRecallError,
    MemoryScope,
    MemorySession,
    MemorySetupError,
    RecallContext,
)


def _flow_ref() -> MemoryFrameRef:
    """Return one stable flow-frame identity for contract tests."""
    return MemoryFrameRef("root", ("root", "child"), MemoryOwner.FLOW)


def _agent_ref() -> MemoryFrameRef:
    """Return one stable agent-frame identity for contract tests."""
    return MemoryFrameRef("root", ("root", "child", "agent"), MemoryOwner.AGENT)


def test_identity_values_are_frozen_structured_products() -> None:
    """Static frame identity stays separate from dynamic occurrence identity."""
    frame = _agent_ref()
    segment = MemoryOccurrenceSegment(
        MemoryOccurrenceKind.MAP_ITEM,
        ("root", "mapped"),
        2,
    )
    occurrence = MemoryOccurrenceRef(frame, (segment,))
    boundary = MemoryBoundary(MemoryBoundaryKind.START, _flow_ref())
    scope = MemoryScope(
        root_flow_id="root",
        static_path=("root", "child", "agent"),
        frame_stack=(_flow_ref(), frame),
        occurrence_path=(segment,),
        boundary=boundary,
    )

    assert frame == MemoryFrameRef(
        "root", ("root", "child", "agent"), MemoryOwner.AGENT
    )
    assert occurrence == MemoryOccurrenceRef(frame, (segment,))
    assert scope.frame_stack == (_flow_ref(), frame)
    assert scope.occurrence_path == (segment,)
    assert scope.boundary == boundary
    with pytest.raises(FrozenInstanceError):
        frame.root_flow_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (
            lambda: MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, (), 0),
            ValueError,
            "static_path",
        ),
        (
            lambda: MemoryOccurrenceSegment(
                MemoryOccurrenceKind.MAP_ITEM, ["root"], 0  # type: ignore[arg-type]
            ),
            TypeError,
            "tuple",
        ),
        (
            lambda: MemoryOccurrenceSegment(
                MemoryOccurrenceKind.MAP_ITEM, ("root",), True
            ),
            TypeError,
            "index",
        ),
        (
            lambda: MemoryOccurrenceSegment(
                MemoryOccurrenceKind.LOOP_ITERATION, ("root",), -1
            ),
            ValueError,
            "index",
        ),
        (
            lambda: MemoryFrameRef("", ("root",), MemoryOwner.FLOW),
            ValueError,
            "root_flow_id",
        ),
        (
            lambda: MemoryFrameRef("root", ("",), MemoryOwner.FLOW),
            ValueError,
            "static_path",
        ),
        (
            lambda: MemoryOccurrenceRef(_agent_ref(), []),  # type: ignore[arg-type]
            TypeError,
            "occurrence_path",
        ),
        (
            lambda: MemoryBoundary(MemoryBoundaryKind.END, _agent_ref()),
            ValueError,
            "flow",
        ),
        (
            lambda: MemoryScope(
                "other",
                ("root",),
                frame_stack=(_flow_ref(),),
            ),
            ValueError,
            "root_flow_id",
        ),
    ],
)
def test_identity_values_reject_ambiguous_paths_and_indices(
    factory: Any,
    error: type[Exception],
    message: str,
) -> None:
    """Identities accept exact immutable paths and non-boolean indices only."""
    with pytest.raises(error, match=message):
        factory()


def test_policy_specs_preserve_origin_and_isolate_canonical_json() -> None:
    """Caller mutation cannot change a frozen policy specification."""
    origin = OriginRef("policy")
    source: JsonObject = {
        "z": [{"enabled": True}],
        "a": {"threshold": 0.5},
    }
    policy = MemoryPolicySpec("reflective", source, origin)

    source["z"][0]["enabled"] = False  # type: ignore[index]
    first = policy.config
    first["a"]["threshold"] = 1.0  # type: ignore[index]

    assert policy.name == "reflective"
    assert policy.origin is origin
    assert policy.config == {
        "a": {"threshold": 0.5},
        "z": [{"enabled": True}],
    }
    assert list(policy.config) == ["a", "z"]
    assert policy == MemoryPolicySpec(
        "reflective",
        {"a": {"threshold": 0.5}, "z": [{"enabled": True}]},
        origin,
    )


def test_json_scalar_identity_preserves_exact_types_at_every_depth() -> None:
    """Boolean, integer, and float JSON values remain distinct frozen values."""
    values = (True, 1, 1.0)
    policies = tuple(
        MemoryPolicySpec(
            "reflective",
            {"nested": [value, {"value": value}]},
        )
        for value in values
    )
    contexts = tuple(
        RecallContext("base", "prompt", {"value": value}, ("root", "agent"))
        for value in values
    )
    mutations = tuple(
        MemoryMutation("same", _agent_ref(), [value, {"value": value}])
        for value in values
    )

    assert len(set(policies)) == 3
    assert len({hash(policy) for policy in policies}) == 3
    assert len(set(contexts)) == 3
    assert len({hash(context) for context in contexts}) == 3
    assert len(set(mutations)) == 3
    assert len({hash(mutation) for mutation in mutations}) == 3

    for expected_type, policy, context, mutation in zip(
        (bool, int, float),
        policies,
        contexts,
        mutations,
    ):
        nested = policy.config["nested"]
        assert type(nested[0]) is expected_type  # type: ignore[index]
        assert type(nested[1]["value"]) is expected_type  # type: ignore[index]
        assert type(context.inputs["value"]) is expected_type
        payload = mutation.payload
        assert type(payload[0]) is expected_type  # type: ignore[index]
        assert type(payload[1]["value"]) is expected_type  # type: ignore[index]


class _DictSubclass(dict[str, object]):
    """A mapping subclass that is deliberately not an exact JSON object."""


class _ListSubclass(list[object]):
    """A list subclass that is deliberately not an exact JSON array."""


@pytest.mark.parametrize(
    "config",
    [
        {"tuple": (1, 2)},
        {"bytes": b"not-json"},
        {"nan": float("nan")},
        {"infinity": float("inf")},
        {"array": _ListSubclass([1])},
        _DictSubclass(value=1),
        {1: "non-string-key"},
    ],
)
def test_json_contracts_reject_non_exact_or_non_finite_values(config: Any) -> None:
    """Only exact, finite JSON shapes may cross the memory boundary."""
    with pytest.raises((TypeError, ValueError), match="JSON|json|finite|key"):
        MemoryPolicySpec("reflective", config)


def test_json_contracts_reject_cycles() -> None:
    """Circular Python containers are not JSON values."""
    config: JsonObject = {}
    config["self"] = config

    with pytest.raises(ValueError, match="circular"):
        MemoryPolicySpec("reflective", config)


def test_context_episode_and_mutation_keep_json_snapshots() -> None:
    """Every durable JSON-bearing value owns a defensive canonical snapshot."""
    inputs: JsonObject = {"items": [1, 2]}
    interaction: JsonObject = {"kind": "model", "value": {"text": "answer"}}
    output: JsonObject = {"answer": ["yes"]}
    payload: JsonObject = {"add": [{"id": "lesson", "text": "remember"}]}
    context = RecallContext("base", "prompt", inputs, ("root", "agent"))
    episode = MemoryEpisode(
        run_id="run",
        occurrence=MemoryOccurrenceRef(_agent_ref()),
        owner_static_path=("root", "child", "agent"),
        status=MemoryEpisodeStatus.SUCCEEDED,
        inputs=inputs,
        output=output,
        interactions=(interaction,),
    )
    mutation = MemoryMutation("m1", _agent_ref(), payload)
    learn = LearnContext(
        "run",
        _agent_ref(),
        (episode,),
        feedback=({"score": 1},),
    )

    inputs["items"].append(3)  # type: ignore[union-attr]
    interaction["value"]["text"] = "changed"  # type: ignore[index]
    output["answer"].append("no")  # type: ignore[union-attr]
    payload["add"].clear()  # type: ignore[union-attr]
    context.inputs["items"].append(4)  # type: ignore[union-attr]
    episode.interactions[0]["value"]["text"] = "changed again"  # type: ignore[index]
    mutation.payload["add"].append("changed")  # type: ignore[union-attr]
    learn.feedback[0]["score"] = 0  # type: ignore[index]

    assert context.inputs == {"items": [1, 2]}
    assert episode.inputs == {"items": [1, 2]}
    assert episode.output == {"answer": ["yes"]}
    assert episode.failure is None
    assert episode.interactions == ({"kind": "model", "value": {"text": "answer"}},)
    assert mutation.payload == {"add": [{"id": "lesson", "text": "remember"}]}
    assert learn.episodes == (episode,)
    assert learn.feedback == ({"score": 1},)


def test_episode_status_enforces_output_or_failure_shape() -> None:
    """Accepted success and failure episodes cannot carry conflicting outcomes."""
    failed = MemoryEpisode(
        run_id="run",
        occurrence=MemoryOccurrenceRef(_agent_ref()),
        owner_static_path=("root", "child", "agent"),
        status=MemoryEpisodeStatus.FAILED,
        inputs={},
        failure={"code": "node.execute", "message": "failed"},
    )

    assert failed.output is None
    assert failed.failure == {"code": "node.execute", "message": "failed"}
    with pytest.raises(ValueError, match="failure"):
        MemoryEpisode(
            run_id="run",
            occurrence=MemoryOccurrenceRef(_agent_ref()),
            owner_static_path=("root", "child", "agent"),
            status=MemoryEpisodeStatus.FAILED,
            inputs={},
        )
    with pytest.raises(ValueError, match="output"):
        MemoryEpisode(
            run_id="run",
            occurrence=MemoryOccurrenceRef(_agent_ref()),
            owner_static_path=("root", "child", "agent"),
            status=MemoryEpisodeStatus.FAILED,
            inputs={},
            output="conflict",
            failure={"code": "node.execute"},
        )
    with pytest.raises(ValueError, match="failure"):
        MemoryEpisode(
            run_id="run",
            occurrence=MemoryOccurrenceRef(_agent_ref()),
            owner_static_path=("root", "child", "agent"),
            status=MemoryEpisodeStatus.SUCCEEDED,
            inputs={},
            output=None,
            failure={"code": "unexpected"},
        )


class _FakeFrame:
    """A small structural implementation of the frame protocol."""

    def recall(self, context: RecallContext | None = None) -> str:
        """Return all memory for ``None`` and one query marker otherwise."""
        return "all" if context is None else context.prompt

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        """Return the protocol's intentional mutable proposal list."""
        return [MemoryMutation("learned", context.frame_ref, {"run": context.run_id})]


class _FakeSession:
    """A small structural implementation of the run-session protocol."""

    def __init__(self, frame: MemoryFrame) -> None:
        self._frame = frame
        self._snapshot: JsonObject = {"namespace": "test"}
        self.closed = False

    @property
    def snapshot(self) -> JsonObject:
        """Return a defensive snapshot as required by the protocol."""
        return {"namespace": self._snapshot["namespace"]}

    def __enter__(self) -> _FakeSession:
        """Enter this session."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close this session on context-manager exit."""
        self.close()

    def close(self) -> None:
        """Mark this fake session closed."""
        self.closed = True

    def open_frame(self, spec: MemoryFrameSpec) -> MemoryFrame:
        """Return the fake frame for any valid frame specification."""
        assert spec.policy_name == "reflective"
        return self._frame

    def commit(
        self,
        commit_id: str,
        mutations: tuple[MemoryMutation, ...],
    ) -> None:
        """Accept one immutable proposal batch."""
        assert commit_id
        assert isinstance(mutations, tuple)


class _FakeMemory:
    """A small structural implementation of the memory protocol."""

    def __init__(self) -> None:
        self.frame = _FakeFrame()
        self.reset_count = 0

    def open_run(self, *, snapshot: JsonObject | None = None) -> MemorySession:
        """Open one fake session, accepting a recorded snapshot if supplied."""
        assert snapshot is None or snapshot["namespace"] == "test"
        return _FakeSession(self.frame)

    def reset(self) -> None:
        """Record one reset request."""
        self.reset_count += 1


def test_protocols_accept_narrow_structural_implementations() -> None:
    """Backends implement source-neutral protocols without inheriting engine code."""
    frame = _FakeFrame()
    memory = _FakeMemory()
    session = memory.open_run()
    context = LearnContext("run", _agent_ref(), ())

    assert isinstance(frame, MemoryFrame)
    assert isinstance(session, MemorySession)
    assert isinstance(memory, Memory)
    assert frame.recall() == "all"
    assert frame.recall(None) == "all"
    assert isinstance(frame.learn(context), list)
    assert frame.learn(context)[0].payload == {"run": "run"}
    returned = session.snapshot
    returned["namespace"] = "changed"
    assert session.snapshot == {"namespace": "test"}
    with session as entered:
        assert entered is session
    assert session.closed is True  # type: ignore[attr-defined]


def test_memory_errors_expose_typed_fields_and_origin() -> None:
    """Ordinary policy failures and source-aware setup wrappers stay distinct."""
    origin = OriginRef("memory-policy")
    operation = MemoryOperationError(
        stage="commit",
        code="stale_revision",
        message="the session is stale",
        retryable=False,
    )
    setup = MemorySetupError("unknown policy", origin=origin)

    assert str(operation) == "the session is stale"
    assert operation.stage == "commit"
    assert operation.code == "stale_revision"
    assert operation.message == "the session is stale"
    assert operation.retryable is False
    assert isinstance(setup, ValueError)
    assert setup.origin is origin
    with pytest.raises(TypeError, match="retryable"):
        MemoryOperationError(
            "learn",
            "bad",
            "failed",
            retryable=1,  # type: ignore[arg-type]
        )


def test_memory_recall_error_preserves_an_explicit_exception_cause() -> None:
    """A caller can wrap an ordinary recall error without losing its cause."""
    origin = OriginRef("agent-policy")
    cause = RuntimeError("backend failed")

    with pytest.raises(MemoryRecallError) as caught:
        try:
            raise cause
        except RuntimeError as error:
            raise MemoryRecallError("memory recall failed", origin=origin) from error

    assert caught.value.origin is origin
    assert caught.value.__cause__ is cause


class _ForgedOrigin(OriginRef):
    """An OriginRef subclass that bypasses the base key validation."""

    def __post_init__(self) -> None:
        """Deliberately skip base validation to exercise contract hardening."""


def test_memory_origins_revalidate_subclass_keys_without_losing_identity() -> None:
    """Valid origin subclasses survive while forged invalid keys fail closed."""
    valid = _ForgedOrigin("valid")

    assert MemoryPolicySpec("reflective", {}, valid).origin is valid
    with pytest.raises(ValueError, match="origin key"):
        MemoryPolicySpec("reflective", {}, _ForgedOrigin(""))
    with pytest.raises(ValueError, match="origin key"):
        MemoryPolicySpec("reflective", {}, _ForgedOrigin(1))  # type: ignore[arg-type]


def test_importing_memory_contracts_does_not_load_heavy_backends() -> None:
    """The memory package facade stays a dependency-light source-neutral leaf."""
    project_root = Path(__file__).parents[2]
    script = """
import sys
before = set(sys.modules)
import agent_composer.memory
loaded = set(sys.modules) - before
for forbidden in (
    'langchain',
    'pydantic',
    'agent_composer.llm_clients',
    'agent_composer.nodes',
    'agent_composer.compile',
    'agent_composer.compose',
    'agent_composer.runtime',
    'agent_composer.run',
    'agent_composer.cli',
):
    assert not any(
        name == forbidden or name.startswith(forbidden + '.')
        for name in loaded
    ), (forbidden, sorted(loaded))
"""

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=True,
        text=True,
    )
