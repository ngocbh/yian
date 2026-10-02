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
    DeclSite,
    EvaluationEvidence,
    JsonObject,
    LearnContext,
    LearningFailure,
    LearningResult,
    Memory,
    MemoryDecl,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryMutation,
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryOperationError,
    MemoryOperationFailure,
    MemoryRecallError,
    MemoryScope,
    MemorySection,
    MemorySession,
    MemorySetupError,
    ModelTextInteraction,
    ModelValueInteraction,
    RecallContext,
    ToolCallInteraction,
)
from agent_composer.runtime.contracts import BackwardFailure, MemoryProposal


def _decl(config: Any, origin: OriginRef | None = None) -> MemoryDecl:
    """Build a custom-type declaration, so the built-in key allowlist does not apply."""
    return MemoryDecl("notes", "custom", config, DeclSite((), ("root",), None), origin)


_AGENT_PATH = ("root", "child", "agent")
"""One authored AGENT locator for episode contract tests."""


def test_identity_values_are_frozen_structured_products() -> None:
    """A scope holds its keys and occurrence path as frozen values."""
    scope_segment = MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ("child",), 2)
    scope = MemoryScope(("child", "agent"), ("outer", "local"), (scope_segment,))

    assert scope.memories == ("outer", "local")
    assert scope.occurrence_path == (scope_segment,)
    with pytest.raises(FrozenInstanceError):
        scope.memories = ("other",)  # type: ignore[misc]


def test_memory_scope_has_no_boundary() -> None:
    """START and END are marked by ExecutionScope; memory scopes carry no boundary."""
    import agent_composer.memory as memory
    import agent_composer.memory.contracts as contracts

    assert not hasattr(memory, "MemoryBoundary")
    assert not hasattr(memory, "MemoryBoundaryKind")
    assert not hasattr(contracts, "MemoryBoundary")
    assert not hasattr(contracts, "MemoryBoundaryKind")
    assert "boundary" not in MemoryScope.__dataclass_fields__
    with pytest.raises(TypeError):
        MemoryScope("root", ("root", "child"), (), (), None)  # type: ignore[call-arg]


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
            lambda: MemoryScope(("",)),
            ValueError,
            "static_path",
        ),
        (
            lambda: MemoryScope(("root",), (), []),  # type: ignore[arg-type]
            TypeError,
            "occurrence_path",
        ),
        (lambda: MemoryScope(("root",), ("a", "a")), ValueError, "unique"),
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


@pytest.mark.parametrize(
    "segment_path",
    [
        ("other",),
        ("child", "agent"),
        ("child", "agent", "inner"),
    ],
)
def test_scope_occurrence_path_must_strictly_contain_the_node(
    segment_path: tuple[str, ...],
) -> None:
    """The final occurrence callsite is an ancestor, never the scoped node itself."""
    segment = MemoryOccurrenceSegment(
        MemoryOccurrenceKind.LOOP_ITERATION,
        segment_path,
        0,
    )

    with pytest.raises(ValueError, match="innermost occurrence segment"):
        MemoryScope(("child", "agent"), occurrence_path=(segment,))


class _ForgedOccurrenceSegment(MemoryOccurrenceSegment):
    """An occurrence-segment subclass that bypasses base validation."""

    def __post_init__(self) -> None:
        """Deliberately skip validation for nested-carrier hardening tests."""


class _ForgedEpisode(MemoryEpisode):
    """An episode subclass whose fields may be forged without base validation."""


def test_nested_memory_carriers_reject_forged_subclass_values() -> None:
    """Nested memory products accept only exact, already-validated value types."""
    forged_segment = _ForgedOccurrenceSegment(
        MemoryOccurrenceKind.MAP_ITEM,
        ("root", "mapped"),
        -1,
    )
    forged_episode = object.__new__(_ForgedEpisode)
    object.__setattr__(forged_episode, "run_id", "")

    with pytest.raises(TypeError, match="occurrence_path"):
        MemoryScope(("mapped", "x"), occurrence_path=(forged_segment,))
    with pytest.raises(TypeError, match="occurrence_path"):
        MemoryEpisode("run", (forged_segment,), ("agent",), MemoryEpisodeStatus.SUCCEEDED, {})
    with pytest.raises(TypeError, match="episodes"):
        LearnContext("run", "notes", (forged_episode,))


def test_memory_decls_preserve_origin_and_isolate_canonical_json() -> None:
    """Caller mutation cannot change a frozen memory declaration."""
    origin = OriginRef("decl")
    source: JsonObject = {
        "z": [{"enabled": True}],
        "a": {"threshold": 0.5},
    }
    decl = _decl(source, origin)

    source["z"][0]["enabled"] = False  # type: ignore[index]
    first = decl.config
    first["a"]["threshold"] = 1.0  # type: ignore[index]

    assert decl.name == "notes"
    assert decl.origin is origin
    assert decl.config == {
        "a": {"threshold": 0.5},
        "z": [{"enabled": True}],
    }
    assert list(decl.config) == ["a", "z"]
    assert decl == _decl(
        {"a": {"threshold": 0.5}, "z": [{"enabled": True}]},
        origin,
    )


def test_json_scalar_identity_preserves_exact_types_at_every_depth() -> None:
    """Boolean, integer, and float JSON values remain distinct frozen values."""
    values = (True, 1, 1.0)
    decls = tuple(
        _decl({"nested": [value, {"value": value}]})
        for value in values
    )
    contexts = tuple(
        RecallContext("base", "prompt", {"value": value}, ("root", "agent"))
        for value in values
    )
    mutations = tuple(
        MemoryMutation("same", [value, {"value": value}])
        for value in values
    )

    assert len(set(decls)) == 3
    assert len({hash(decl) for decl in decls}) == 3
    assert len(set(contexts)) == 3
    assert len({hash(context) for context in contexts}) == 3
    assert len(set(mutations)) == 3
    assert len({hash(mutation) for mutation in mutations}) == 3

    for expected_type, decl, context, mutation in zip(
        (bool, int, float),
        decls,
        contexts,
        mutations,
    ):
        nested = decl.config["nested"]
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
        _decl(config)


def test_json_contracts_reject_cycles() -> None:
    """Circular Python containers are not JSON values."""
    config: JsonObject = {}
    config["self"] = config

    with pytest.raises(ValueError, match="circular"):
        _decl(config)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ModelTextInteraction(chr(0xD800)),
        lambda: _decl({"text": chr(0xD800)}),
        lambda: _decl({chr(0xD800): "text"}),
    ],
)
def test_memory_strings_must_have_a_canonical_utf8_representation(factory: Any) -> None:
    """Memory carriers reject strings the durable JSON boundary cannot encode."""
    with pytest.raises(ValueError, match="Unicode surrogate"):
        factory()


@pytest.mark.parametrize("carrier", ["value", "arguments"])
def test_memory_json_rejects_nesting_beyond_the_durable_wire_limit(
    carrier: str,
) -> None:
    """A conservative evidence bound stays below the schema-4 wire ceiling."""
    nested: JsonObject = {}
    for _ in range(300):
        nested = {"nested": nested}

    factory = (
        lambda: ModelValueInteraction(nested)
        if carrier == "value"
        else ToolCallInteraction("lookup", nested)
    )
    with pytest.raises(ValueError, match="nesting"):
        factory()


def test_context_episode_and_mutation_keep_json_snapshots() -> None:
    """Every durable JSON-bearing value owns a defensive canonical snapshot."""
    inputs: JsonObject = {"items": [1, 2]}
    interaction_value: JsonObject = {"text": "answer"}
    interaction = ModelValueInteraction(interaction_value)
    output: JsonObject = {"answer": ["yes"]}
    payload: JsonObject = {"add": [{"id": "lesson", "text": "remember"}]}
    context = RecallContext("base", "prompt", inputs, ("root", "agent"))
    episode = MemoryEpisode(
        run_id="run",
        occurrence_path=(),
        owner_static_path=("root", "child", "agent"),
        status=MemoryEpisodeStatus.SUCCEEDED,
        inputs=inputs,
        output=output,
        interactions=(interaction,),
    )
    mutation = MemoryMutation("m1", payload)
    learn = LearnContext(
        "run",
        "notes",
        (episode,),
        feedback=({"score": 1},),
    )

    inputs["items"].append(3)  # type: ignore[union-attr]
    interaction_value["text"] = "changed"
    output["answer"].append("no")  # type: ignore[union-attr]
    payload["add"].clear()  # type: ignore[union-attr]
    context.inputs["items"].append(4)  # type: ignore[union-attr]
    assert isinstance(episode.interactions[0], ModelValueInteraction)
    episode.interactions[0].value["text"] = "changed again"
    mutation.payload["add"].append("changed")  # type: ignore[union-attr]
    learn.feedback[0]["score"] = 0  # type: ignore[index]

    assert context.inputs == {"items": [1, 2]}
    assert episode.inputs == {"items": [1, 2]}
    assert episode.output == {"answer": ["yes"]}
    assert episode.failure is None
    assert episode.interactions == (ModelValueInteraction({"text": "answer"}),)
    assert mutation.payload == {"add": [{"id": "lesson", "text": "remember"}]}
    assert learn.episodes == (episode,)
    assert learn.feedback == ({"score": 1},)


def test_episode_status_enforces_output_or_failure_shape() -> None:
    """Accepted success and failure episodes cannot carry conflicting outcomes."""
    failed = MemoryEpisode(
        run_id="run",
        occurrence_path=(),
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
            occurrence_path=(),
            owner_static_path=("root", "child", "agent"),
            status=MemoryEpisodeStatus.FAILED,
            inputs={},
        )
    with pytest.raises(ValueError, match="output"):
        MemoryEpisode(
            run_id="run",
            occurrence_path=(),
            owner_static_path=("root", "child", "agent"),
            status=MemoryEpisodeStatus.FAILED,
            inputs={},
            output="conflict",
            failure={"code": "node.execute"},
        )
    with pytest.raises(ValueError, match="failure"):
        MemoryEpisode(
            run_id="run",
            occurrence_path=(),
            owner_static_path=("root", "child", "agent"),
            status=MemoryEpisodeStatus.SUCCEEDED,
            inputs={},
            output=None,
            failure={"code": "unexpected"},
        )


@pytest.mark.parametrize("sequence", [0, 17])
def test_episode_execution_sequence_is_optional_and_survives_context_copy(sequence) -> None:
    """Physical identity extends the existing positional host constructor."""
    legacy = MemoryEpisode("run", (), _AGENT_PATH,
                           MemoryEpisodeStatus.SUCCEEDED, {}, "answer", None, (), None, None, ())
    assert legacy.execution_seq is None
    inputs = {"items": [1]}
    episode = MemoryEpisode("run", (), _AGENT_PATH,
                            MemoryEpisodeStatus.SUCCEEDED, inputs, "answer", execution_seq=sequence)
    context = LearnContext("run", "notes", (episode,))
    inputs["items"].append(2)
    assert context.episodes[0].execution_seq == sequence
    assert context.episodes[0].inputs == {"items": [1]}
    object.__setattr__(episode, "execution_seq", True)
    with pytest.raises(TypeError, match="execution_seq"):
        LearnContext("run", "notes", (episode,))


@pytest.mark.parametrize("sequence", [False, True, -1, 1.0, "1"])
def test_episode_rejects_nonphysical_execution_sequences(sequence) -> None:
    with pytest.raises((TypeError, ValueError), match="execution_seq"):
        MemoryEpisode("run", (), _AGENT_PATH,
                      MemoryEpisodeStatus.SUCCEEDED, {}, execution_seq=sequence)


def test_episode_rejects_integer_subclass_execution_sequence() -> None:
    class Sequence(int):
        pass

    with pytest.raises(TypeError, match="execution_seq"):
        MemoryEpisode("run", (), _AGENT_PATH,
                      MemoryEpisodeStatus.SUCCEEDED, {}, execution_seq=Sequence(1))


def test_learn_context_requires_episodes_of_its_run() -> None:
    """A learning batch contains only episodes of its own run."""
    wrong_run = MemoryEpisode(
        run_id="other-run",
        occurrence_path=(),
        owner_static_path=("root", "child", "agent"),
        status=MemoryEpisodeStatus.SUCCEEDED,
        inputs={},
        output="answer",
    )
    with pytest.raises(ValueError, match="run_id"):
        LearnContext("run", "notes", (wrong_run,))


class _FakeSession:
    """A small structural implementation of the run-session protocol."""

    def __init__(self) -> None:
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

    def recall(self, context: RecallContext | None = None) -> str:
        """Return all memory for ``None`` and one query marker otherwise."""
        return "all" if context is None else context.prompt

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        """Return the protocol's intentional mutable proposal list."""
        return [MemoryMutation("learned", {"run": context.run_id})]

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
        self.reset_count = 0

    def open_run(self, *, snapshot: JsonObject | None = None) -> MemorySession:
        """Open one fake session, accepting a recorded snapshot if supplied."""
        assert snapshot is None or snapshot["namespace"] == "test"
        return _FakeSession()

    def reset(self) -> None:
        """Record one reset request."""
        self.reset_count += 1


def test_protocols_accept_narrow_structural_implementations() -> None:
    """Backends implement source-neutral protocols without inheriting engine code."""
    memory = _FakeMemory()
    session = memory.open_run()
    context = LearnContext("run", "notes", ())

    assert isinstance(session, MemorySession)
    assert isinstance(memory, Memory)
    assert session.recall() == "all"
    assert session.recall(None) == "all"
    assert isinstance(session.learn(context), list)
    assert session.learn(context)[0].payload == {"run": "run"}
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


def test_learning_result_keeps_primary_independent_failure_attribution() -> None:
    """Completion and failure attribution are independent of backend retry hints."""
    retryable = LearningFailure(
        "reflective",
        "learn",
        MemoryOperationFailure("learn", "busy", "try later", True),
    )
    final = LearningFailure(
        "profile",
        "commit",
        MemoryOperationFailure("commit", "rejected", "not applied", False),
    )

    assert LearningResult("succeeded").failures == ()
    assert LearningResult("failed", (final,)).status == "failed"
    result = LearningResult("incomplete", (retryable, final))
    assert result.failures == (retryable, final)
    assert LearningResult("failed", (retryable,)).failures == (retryable,)
    assert LearningResult("incomplete", (final,)).failures == (final,)


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

    assert _decl({}, valid).origin is valid
    with pytest.raises(ValueError, match="origin key"):
        _decl({}, _ForgedOrigin(""))
    with pytest.raises(ValueError, match="origin key"):
        _decl({}, _ForgedOrigin(1))  # type: ignore[arg-type]


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


@pytest.mark.parametrize("status,commit_id,has_failure", [
    ("unrequested", None, False), ("unknown", "commit-1", False),
    ("succeeded", "commit-1", False), ("failed", "commit-1", True),
])
def test_policy_commit_outcome_keeps_protocol_identity_and_backend_hints(status, commit_id, has_failure):
    from agent_composer.memory import PolicyCommitOutcome

    failure = MemoryOperationFailure("backend-stage", "retry", "Try later.", True) if has_failure else None
    outcome = PolicyCommitOutcome("history", status, commit_id, failure)
    assert (outcome.memory, outcome.status, outcome.commit_id, outcome.failure) == (
        "history", status, commit_id, failure,
    )
    with pytest.raises(FrozenInstanceError):
        outcome.status = "succeeded"


def test_learning_completion_is_independent_of_retryability_and_policy_confirmation():
    from agent_composer.memory import PolicyCommitOutcome

    failure = LearningFailure("history", "commit", MemoryOperationFailure("backend", "busy", "Retry.", True))
    succeeded = PolicyCommitOutcome("history", "succeeded", "commit-1")
    pending = PolicyCommitOutcome("other", "unrequested")
    assert LearningResult("failed", (failure,)).failures == (failure,)
    assert LearningResult("incomplete").policies == ()
    assert LearningResult("incomplete", policies=(succeeded, pending)).policies == (succeeded, pending)
    assert LearningResult("incomplete", policies=(succeeded,)).status == "incomplete"
    assert LearningResult("succeeded", policies=(succeeded,)).policies == (succeeded,)
    with pytest.raises(ValueError):
        LearningResult("succeeded", policies=(pending,))
    with pytest.raises(ValueError):
        LearningResult("failed")
    with pytest.raises(ValueError):
        LearningResult("incomplete", policies=(succeeded, succeeded))


@pytest.mark.parametrize("status, commit_id, with_failure", [
    ("unrequested", "commit-1", False), ("unrequested", None, True),
    ("unknown", None, False), ("unknown", "commit-1", True),
    ("succeeded", None, False), ("succeeded", "commit-1", True),
    ("failed", None, True), ("failed", "commit-1", False), ("other", None, False),
])
def test_policy_commit_outcome_rejects_inconsistent_optional_fields(status, commit_id, with_failure):
    from agent_composer.memory import PolicyCommitOutcome

    failure = MemoryOperationFailure("backend", "error", "Failed.", False) if with_failure else None
    with pytest.raises(ValueError):
        PolicyCommitOutcome("history", status, commit_id, failure)


@pytest.mark.parametrize("field", ["memory", "status", "commit_id"])
@pytest.mark.parametrize("invalid", ["", "\ud800", 1, True])
def test_policy_commit_outcome_requires_exact_valid_unicode_identities(field, invalid):
    from agent_composer.memory import PolicyCommitOutcome

    values = {"memory": "history", "status": "unknown", "commit_id": "commit-1"}
    values[field] = invalid
    with pytest.raises((TypeError, ValueError)):
        PolicyCommitOutcome(**values)


def test_learning_result_revalidates_exact_nested_policy_carriers():
    from agent_composer.memory import PolicyCommitOutcome

    first = PolicyCommitOutcome("second", "unknown", "commit-2")
    second = PolicyCommitOutcome("first", "unrequested")
    result = LearningResult("incomplete", (), (first, second))
    assert result.policies == (first, second)
    assert result.policies[0] is not first
    with pytest.raises(TypeError, match="tuple"):
        LearningResult("incomplete", policies=[first])

    class PolicySubclass(PolicyCommitOutcome):
        pass

    with pytest.raises(TypeError, match="PolicyCommitOutcome"):
        LearningResult("incomplete", policies=(PolicySubclass("history", "unrequested"),))
    object.__setattr__(first, "commit_id", None)
    with pytest.raises(ValueError, match="commit_id"):
        LearningResult("incomplete", policies=(first,))

    failure = MemoryOperationFailure("backend", "failed", "Failed.", True)
    object.__setattr__(failure, "retryable", 1)
    with pytest.raises(TypeError, match="retryable"):
        PolicyCommitOutcome("history", "failed", "commit-1", failure)


def test_scope_carries_a_root_relative_path_and_memory_keys() -> None:
    segment = MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ("child",), 2)
    scope = MemoryScope(("child", "agent"), ("notes", "chat"), (segment,))
    assert (scope.static_path, scope.memories, scope.occurrence_path) == (
        ("child", "agent"),
        ("notes", "chat"),
        (segment,),
    )
    assert MemoryScope(("plain",)).memories == ()
    with pytest.raises(ValueError, match="unique"):
        MemoryScope(("agent",), ("notes", "notes"))
    with pytest.raises(ValueError, match="memories"):
        MemoryScope(("agent",), ("",))
    with pytest.raises(TypeError, match="memories"):
        MemoryScope(("agent",), ["notes"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="innermost occurrence segment"):
        MemoryScope(("child",), (), (segment,))


def test_sections_follow_the_scope_key_order() -> None:
    segment = MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, ("loop",), 0)
    scope = MemoryScope(("loop", "agent"), ("notes", "chat"), (segment,))
    outer, inner = MemorySection("notes", "outer"), MemorySection("chat", "inner")
    evidence = EvaluationEvidence(scope, {}, "base", "prompt", (outer, inner))
    assert [section.memory for section in evidence.memory_sections] == ["notes", "chat"]
    with pytest.raises(ValueError, match="scope.memories order"):
        EvaluationEvidence(scope, {}, "base", "prompt", (inner, outer))
    with pytest.raises(ValueError, match="unique"):
        EvaluationEvidence(scope, {}, "base", "prompt", (outer, MemorySection("notes", "b")))
    with pytest.raises(ValueError, match="appear in scope.memories"):
        EvaluationEvidence(scope, {}, "base", "prompt", (MemorySection("other", "x"),))
    with pytest.raises(ValueError, match="memory"):
        MemorySection("", "text")


def test_episodes_contexts_mutations_and_failures_name_the_memory_key() -> None:
    segment = MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, ("loop",), 0)
    episode = MemoryEpisode(
        "run",
        (segment,),
        ("loop", "agent"),
        MemoryEpisodeStatus.SUCCEEDED,
        {},
        output="answer",
        memory_sections=(MemorySection("notes", "outer"),),
    )
    assert episode.occurrence_path == (segment,)
    with pytest.raises(ValueError, match="innermost occurrence segment"):
        MemoryEpisode("run", (segment,), ("other",), MemoryEpisodeStatus.SUCCEEDED, {})
    context = LearnContext("run", "notes", (episode,), feedback=("good",))
    assert (context.memory, context.episodes, context.feedback) == ("notes", (episode,), ("good",))
    with pytest.raises(ValueError, match="memory"):
        LearnContext("run", "", ())
    mutation = MemoryMutation("m1", {"add": 1})
    assert (mutation.mutation_id, mutation.payload) == ("m1", {"add": 1})
    assert not hasattr(mutation, "frame_ref")
    proposal = MemoryProposal("notes", (mutation,))
    assert proposal.memory == "notes" and proposal.mutations == (mutation,)
    with pytest.raises(ValueError, match="unique"):
        MemoryProposal("notes", (mutation, MemoryMutation("m1", {})))
    cause = MemoryOperationFailure("learn", "notes.failed", "boom", False)
    assert LearningFailure("notes", "learn", cause).memory == "notes"
    assert not hasattr(LearningFailure("notes", "commit", cause), "frame_ref")
    assert not hasattr(BackwardFailure("learn", "x", "y", memory="notes"), "frame_ref")


class _KeySession:
    """A structural run session that recalls and learns without frames."""

    snapshot: JsonObject = {}

    def __enter__(self) -> _KeySession:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release nothing."""

    def recall(self, context: RecallContext | None = None) -> str:
        return "all" if context is None else context.prompt

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        return [MemoryMutation("learned", {"memory": context.memory})]

    def commit(self, commit_id: str, mutations: tuple[MemoryMutation, ...]) -> None:
        assert commit_id and isinstance(mutations, tuple)


def test_session_protocol_recalls_and_learns_without_frames() -> None:
    session = _KeySession()
    assert isinstance(session, MemorySession)
    assert session.learn(LearnContext("run", "notes", ()))[0].payload == {"memory": "notes"}
    assert "open_frame" not in MemorySession.__dict__


def test_frame_types_are_gone() -> None:
    import agent_composer.memory as memory
    import agent_composer.memory.contracts as contracts
    import agent_composer.runtime.contracts as runtime_contracts

    for name in (
        "MemoryFrame",
        "MemoryFrameRef",
        "MemoryFrameSpec",
        "MemoryOccurrenceRef",
        "MemoryOwner",
    ):
        assert name not in memory.__all__
        assert not hasattr(memory, name)
        assert not hasattr(contracts, name)
    assert not hasattr(runtime_contracts, "FrameProposal")
    assert "open_frame" not in vars(contracts.MemorySession)
