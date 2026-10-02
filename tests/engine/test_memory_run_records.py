"""Run-record contracts for source-neutral memory evidence and learning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from typing import Any

import pytest

from agent_composer.diagnostics import Failure
from agent_composer.memory import (
    DeclSite,
    EvaluationEvidence,
    LearnContext,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryMutation,
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryOperationFailure,
    MemoryScope,
    MemorySection,
    MemorySnapshotEnvelope,
    MemorySpec,
    MemoryStack,
    ModelTextInteraction,
    ModelValueInteraction,
    ToolCallInteraction,
    ToolResultInteraction,
)
from agent_composer.memory.reflective import ReflectiveMemory
import agent_composer.memory.reflective as reflective
from agent_composer.runtime.log import (
    SCHEMA_VERSION,
    MemoryCommitFailedRecord,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    BackwardReadyRecord,
    NodeExpandedRecord,
    NodeFailedRecord,
    NodeRoutedRecord,
    NodeSucceededRecord,
    RecordEnvelope,
    RunStartedRecord,
    decode_record,
    encode_pool,
    encode_record,
)
from agent_composer.nodes.base import BackwardResult
from agent_composer.runtime.contracts import MemoryProposal
from agent_composer.typesys import VariablePool


_FINGERPRINT = "a" * 64
_DIGEST = "b" * 64
_ROOT = DeclSite((), ("root",), None)
_AGENT = DeclSite((), ("root",), "agent")


def _spec(
    key: str = "reflective",
    config: dict[str, Any] | None = None,
    owner: DeclSite = _AGENT,
) -> MemorySpec:
    """Return one key-table entry; an AGENT owner makes it conversation-eligible."""
    return MemorySpec(key, key, {} if config is None else config, (owner,), owner.node is not None)


_PLURAL_PRIVATE_CONFIG_CASES = (
    {"nested": {"secretTokens": ["private"]}},
    {"nested": {"apiKeys": ["private"]}},
    {"nested": {"backendPaths": ["/private/backend"]}},
    {"nested": {"pythonModules": ["package.policy"]}},
)


_PRIVATE_CONFIG_CASES = (
    {"nested": {"clientSecret": "private"}},
    {"nested": {"secretKey": "private"}},
    {"nested": {"PASSWORD": "private"}},
    {"nested": {"passphrase": "private"}},
    {"nested": {"passwd": "private"}},
    {"nested": {"pwd": "private"}},
    {"nested": {"access-token": "private"}},
    {"nested": {"refreshToken": "private"}},
    {"nested": {"session_key": "private"}},
    {"nested": {"subscriptionKey": "private"}},
    {"nested": {"authToken": "private"}},
    {"nested": {"bearer-key": "private"}},
    {"nested": {"api-key": "private"}},
    {"nested": {"apiKey": "private"}},
    {"nested": {"backend_token": "private"}},
    {"nested": {"Authorization": "private"}},
    {"nested": {"private-key": "private"}},
    {"nested": {"connectionString": "private"}},
    {"nested": {"credentials": "private"}},
    {"nested": {"webhookURL": "private"}},
    {"nested": {"webhookSignature": "private"}},
    {"webhook": {"signature": "private"}},
    {"nested": {"tenant_id": "private"}},
    {"nested": {"orgTenantId": "private"}},
    {"nested": {"pythonImport": "package.policy"}},
    {"nested": {"pythonImports": ["package.policy"]}},
    {"python": {"imports": ["package.policy"]}},
    {"nested": {"modulePath": "package.policy"}},
    {"nested": {"class-path": "package.Policy"}},
    {"nested": {"secretToken": "private"}},
    *_PLURAL_PRIVATE_CONFIG_CASES,
    {"nested": {"credentialFile": "/private/credentials"}},
    {"nested": {"apiKeyPath": "/private/api-key"}},
    {"nested": {"orgtenantid": "private"}},
    {"nested": {"backendDir": "/private/backend"}},
    {"nested": {"storageDirectory": "/private/storage"}},
    {"backend": {"path": "/private/backend"}},
    {"backend": {"directory": "/private/backend"}},
    {"backend": {"connection": "private-connection"}},
    {"storage": {"location": "/private/storage"}},
    {"database": {"dsn": "private-dsn"}},
    {"persistence": {"uri": "private://state"}},
)


def _segment(index: int = 0) -> MemoryOccurrenceSegment:
    return MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ("flow",), index)


def _evidence(*, bound_inputs: dict[str, Any] | None = None) -> EvaluationEvidence:
    return EvaluationEvidence(
        scope=MemoryScope(("flow", "agent"), ("notes", "reflective"), (_segment(),)),
        bound_inputs={"question": ["hello"]} if bound_inputs is None else bound_inputs,
        base_system_prompt="base",
        prompt="answer the question",
        memory_sections=(
            MemorySection("notes", "flow memory"),
            MemorySection("reflective", "agent memory"),
        ),
        interactions=(
            ModelTextInteraction("draft"),
            ModelValueInteraction({"answer": "done"}),
            ToolCallInteraction("lookup", {"key": "value"}),
            ToolResultInteraction("lookup", "result"),
        ),
    )


def _started(
    *,
    snapshots: tuple[MemorySnapshotEnvelope, ...] | None = None,
    learnable: bool = False,
    memories: tuple[MemorySpec, ...] | None = None,
    stacks: tuple[MemoryStack, ...] | None = None,
) -> RunStartedRecord:
    return RunStartedRecord(
        flow_fingerprint=_FINGERPRINT,
        initial_pool=encode_pool(VariablePool(start_id="start")),
        start_wiring_override={},
        effective_llm_by_static_path={},
        num_workers=0,
        memory_snapshots=snapshots,
        learnable=learnable,
        memories=memories,
        memory_stacks=stacks,
    )


def _round_trip(record: Any) -> Any:
    envelope = RecordEnvelope(SCHEMA_VERSION, "run", 0, record)
    encoded = encode_record(envelope)
    decoded = decode_record(encoded)
    assert encode_record(decoded) == encoded
    assert decoded == envelope
    return decoded.record


def test_source_neutral_carriers_are_defensive_and_frozen() -> None:
    snapshot_source = {"lessons": [{"text": "keep"}]}
    value_source = {"nested": [1]}
    arguments_source = {"query": ["x"]}
    inputs_source = {"question": ["hello"]}
    snapshot = MemorySnapshotEnvelope("reflective", snapshot_source)
    value = ModelValueInteraction(value_source)
    call = ToolCallInteraction("search", arguments_source)
    evidence = EvaluationEvidence(
        MemoryScope(("flow", "agent"), ("reflective",)),
        inputs_source,
        "base",
        "prompt",
        (MemorySection("reflective", "memory"),),
        (value, call),
    )

    snapshot_source["lessons"][0]["text"] = "changed"
    value_source["nested"].append(2)
    arguments_source["query"].append("y")
    inputs_source["question"].append("changed")
    snapshot.snapshot["lessons"].clear()
    value.value["nested"].append(3)
    call.arguments["query"].append("z")
    evidence.bound_inputs["question"].append("again")  # type: ignore[index]

    assert snapshot.snapshot == {"lessons": [{"text": "keep"}]}
    assert value.value == {"nested": [1]}
    assert call.arguments == {"query": ["x"]}
    assert evidence.bound_inputs == {"question": ["hello"]}
    with pytest.raises(FrozenInstanceError):
        evidence.prompt = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("config", _PRIVATE_CONFIG_CASES)
def test_memory_spec_config_rejects_private_keys(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="forbidden private key"):
        MemorySpec("reflective", "reflective", config, (_AGENT,), True)


@pytest.mark.parametrize("config", _PRIVATE_CONFIG_CASES)
def test_codec_revalidates_forged_private_memory_config(
    config: dict[str, Any],
) -> None:
    spec = _spec()
    started = _started(
        memories=(spec,),
        stacks=(),
        snapshots=(MemorySnapshotEnvelope("reflective", {"state": []}),),
    )
    object.__setattr__(spec, "_config", MemorySnapshotEnvelope("opaque", config)._snapshot)

    with pytest.raises(ValueError, match="forbidden private key"):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started))


def test_public_token_policy_options_round_trip_durably() -> None:
    config = {
        "stop_token": "semantic-stop-marker",
        "continuationToken": "semantic-continuation-marker",
    }
    started = _started(
        memories=(_spec(config=config),),
        stacks=(),
        snapshots=(MemorySnapshotEnvelope("reflective", {"state": []}),),
    )

    encoded = encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started))
    decoded = decode_record(encoded).record

    assert b"semantic-stop-marker" in encoded
    assert b"semantic-continuation-marker" in encoded
    assert type(decoded) is RunStartedRecord
    assert decoded.memories is not None
    assert decoded.memories[0].config == config


def test_snapshot_envelope_preserves_opaque_public_memory_state() -> None:
    public_state = {
        "lessons": [
            {
                "text": "Never place an api_key in durable memory configuration.",
                "tags": ["security", "configuration"],
            }
        ],
        "profile": {"preferred_runtime": "local"},
    }

    assert MemorySnapshotEnvelope("reflective", public_state).snapshot == public_state


def test_evidence_requires_coherent_prompts_and_ordered_scope_sections() -> None:
    scope = MemoryScope(("flow", "agent"), ("notes", "reflective"))

    with pytest.raises(ValueError, match="both"):
        EvaluationEvidence(scope, {}, "base", None)
    with pytest.raises(ValueError, match="appear in scope.memories"):
        EvaluationEvidence(
            scope,
            {},
            "base",
            "prompt",
            (MemorySection("other", "x"),),
        )
    with pytest.raises(ValueError, match="unique"):
        EvaluationEvidence(
            scope,
            {},
            "base",
            "prompt",
            (MemorySection("notes", "a"), MemorySection("notes", "b")),
        )
    with pytest.raises(ValueError, match="order"):
        EvaluationEvidence(
            scope,
            {},
            "base",
            "prompt",
            (MemorySection("reflective", "a"), MemorySection("notes", "b")),
        )
    with pytest.raises(TypeError, match="VisibleInteraction"):
        EvaluationEvidence(
            scope,
            {},
            "base",
            "prompt",
            interactions=({},),  # type: ignore[arg-type]
        )

    non_agent = EvaluationEvidence(scope, None, None, None)
    assert non_agent.bound_inputs is None
    assert non_agent.memory_sections == ()
    assert non_agent.interactions == ()


def test_memory_scope_requires_unique_keys_and_nested_occurrences() -> None:
    with pytest.raises(ValueError, match="unique"):
        MemoryScope(("flow", "agent"), ("notes", "notes"))
    assert EvaluationEvidence(
        MemoryScope(("neutral",)), {}, "base", "prompt",
    ).memory_sections == ()
    with pytest.raises(ValueError, match="lexical nesting"):
        MemoryScope(
            ("flow", "agent"),
            ("notes",),
            occurrence_path=(
                MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ("other",), 0),
                MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, ("flow",), 0),
            ),
        )
    with pytest.raises(ValueError, match="innermost occurrence segment"):
        MemoryScope(
            ("flow", "agent"),
            ("notes",),
            occurrence_path=(
                MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ("other",), 0),
            ),
        )


def test_episode_accepts_binding_failure_and_typed_visible_material() -> None:
    episode = MemoryEpisode(
        run_id="run",
        occurrence_path=(),
        owner_static_path=("flow", "agent"),
        status=MemoryEpisodeStatus.FAILED,
        inputs=None,
        failure={"code": "binding.evaluate"},
        base_system_prompt=None,
        prompt=None,
        memory_sections=(MemorySection("reflective", "memory"),),
        interactions=(ModelTextInteraction("visible"),),
    )

    assert episode.inputs is None
    assert episode.memory_sections == (MemorySection("reflective", "memory"),)
    assert episode.interactions == (ModelTextInteraction("visible"),)


def test_run_started_distinguishes_disabled_and_enabled_empty_memory() -> None:
    disabled = _round_trip(_started())
    enabled = _round_trip(_started(memories=(), stacks=(), snapshots=()))
    learning = _round_trip(_started(memories=(), stacks=(), snapshots=(), learnable=True))

    assert disabled.memories is None
    assert disabled.memory_stacks is None
    assert disabled.memory_snapshots is None
    assert enabled.memories == ()
    assert enabled.memory_stacks == ()
    assert enabled.memory_snapshots == ()
    assert learning.learnable is True
    assert not hasattr(enabled, "memory_layout")

    for partial in (
        dict(memories=(), stacks=(), snapshots=None),
        dict(memories=(), stacks=None, snapshots=()),
        dict(memories=None, stacks=(), snapshots=()),
    ):
        with pytest.raises(ValueError, match="all be recorded or all be absent"):
            _started(**partial)
    with pytest.raises(TypeError, match="tuples"):
        _started(memories=[], stacks=(), snapshots=())  # type: ignore[arg-type]
    assert _round_trip(_started(learnable=True)).learnable is True
    with pytest.raises(TypeError, match="boolean"):
        _started(memories=(), stacks=(), snapshots=(), learnable=1)  # type: ignore[arg-type]


def test_run_started_uses_learnable_as_its_only_wire_retention_field() -> None:
    started = _started(learnable=True)
    wire = json.loads(encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started)))

    assert wire["payload"]["learnable"] is True
    assert "learn" not in wire["payload"]

    wire["payload"]["learn"] = wire["payload"].pop("learnable")
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire).encode())


def test_run_started_requires_a_canonical_table_and_matching_snapshots() -> None:
    memories = (
        _spec("profile", {"mode": "flow"}, _ROOT),
        _spec("reflective", {"mode": "agent"}),
    )
    snapshots = (
        MemorySnapshotEnvelope("profile", {"state": 1}),
        MemorySnapshotEnvelope("reflective", {"state": 2}),
    )
    stacks = (
        MemoryStack(("agent",), ("profile", "reflective")),
        MemoryStack(("plain",), ("profile",)),
    )
    decoded = _round_trip(_started(memories=memories, stacks=stacks, snapshots=snapshots))

    assert decoded.memories == memories
    assert decoded.memory_stacks == stacks
    assert decoded.memory_snapshots == snapshots

    with pytest.raises(ValueError, match="memories keys must be sorted and unique"):
        _started(memories=tuple(reversed(memories)), stacks=stacks, snapshots=snapshots)
    with pytest.raises(ValueError, match="memories keys must be sorted and unique"):
        _started(memories=(memories[0], memories[0]), stacks=(), snapshots=snapshots[:1])
    with pytest.raises(ValueError, match="memory_snapshots memory keys must be sorted and unique"):
        _started(memories=memories, stacks=stacks, snapshots=tuple(reversed(snapshots)))
    with pytest.raises(ValueError, match="equal memories keys"):
        _started(memories=memories, stacks=stacks, snapshots=snapshots[:1])
    with pytest.raises(ValueError, match="locators must be sorted and unique"):
        _started(memories=memories, stacks=tuple(reversed(stacks)), snapshots=snapshots)
    with pytest.raises(ValueError, match="unknown memory keys"):
        _started(
            memories=memories,
            stacks=(MemoryStack(("agent",), ("missing",)),),
            snapshots=snapshots,
        )


def test_run_started_records_a_key_table_and_stacks() -> None:
    memories = (
        _spec("profile", {"mode": "flow"}, _ROOT),
        _spec("reflective", {"mode": "agent"}),
    )
    started = _started(
        memories=memories,
        stacks=(MemoryStack(("agent",), ("profile", "reflective")),),
        snapshots=(
            MemorySnapshotEnvelope("profile", {"state": 1}),
            MemorySnapshotEnvelope("reflective", {"state": 2}),
        ),
    )
    wire = json.loads(encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started)))

    assert "memory_layout" not in wire["payload"]
    assert wire["payload"]["memories"] == [
        {
            "key": "profile",
            "type": "profile",
            "config": {"mode": "flow"},
            "owners": [{"scope": [], "owner": ["root"], "node": None}],
            "conversation_eligible": False,
        },
        {
            "key": "reflective",
            "type": "reflective",
            "config": {"mode": "agent"},
            "owners": [{"scope": [], "owner": ["root"], "node": "agent"}],
            "conversation_eligible": True,
        },
    ]
    assert wire["payload"]["memory_stacks"] == [
        {"locator": ["agent"], "memories": ["profile", "reflective"]},
    ]
    disabled = json.loads(encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, _started())))
    assert disabled["payload"]["memories"] is None
    assert disabled["payload"]["memory_stacks"] is None
    assert disabled["payload"]["memory_snapshots"] is None

    for field in ("memory_stacks", "memories"):
        missing = json.loads(json.dumps(wire))
        missing["payload"].pop(field)
        with pytest.raises(ValueError):
            decode_record(json.dumps(missing).encode())
    legacy = json.loads(json.dumps(wire))
    legacy["payload"]["memory_layout"] = None
    with pytest.raises(ValueError):
        decode_record(json.dumps(legacy).encode())


@pytest.mark.parametrize(
    "record",
    [
        NodeSucceededRecord(
            "agent", "agent", {"kind": "str", "value": "done"}, _evidence()
        ),
        NodeRoutedRecord("agent", "yes", None, _evidence()),
        NodeExpandedRecord("agent", {}, (), _DIGEST, _evidence()),
        NodeFailedRecord(
            Failure("node.execute", "failed", "agent"),
            _evidence(),
        ),
    ],
    ids=lambda record: type(record).__name__,
)
def test_all_outcome_records_round_trip_closed_evidence(record: Any) -> None:
    assert _round_trip(record) == record


def test_existing_outcome_records_round_trip_with_no_evidence() -> None:
    records = (
        NodeSucceededRecord("agent", "agent", {"kind": "str", "value": "done"}),
        NodeRoutedRecord("agent", "yes"),
        NodeExpandedRecord("agent", {}, (), _DIGEST),
        NodeFailedRecord(Failure("node.execute", "failed", "agent")),
    )

    for record in records:
        decoded = _round_trip(record)
        assert decoded.evidence is None


def test_every_memory_commit_record_round_trips() -> None:
    mutations = (
        MemoryMutation("m1", {"action": "add"}),
        MemoryMutation("m2", {"action": "revise"}),
    )
    failure = MemoryOperationFailure("learn", "provider.error", "failed", True)
    records = (
        MemoryCommitRequestedRecord("reflective", "commit-1", mutations, attempt_id=1),
        MemoryCommitSucceededRecord("reflective", "commit-1", attempt_id=1),
        MemoryCommitFailedRecord("reflective", "commit-1", failure, attempt_id=1),
    )

    for record in records:
        assert _round_trip(record) == record


@pytest.mark.parametrize("attempt_id", [1, 2])
def test_memory_commit_wire_requires_an_explicit_positive_attempt_slot(attempt_id):
    failure = MemoryOperationFailure("backend", "failed", "Failed.", False)
    records = (
        MemoryCommitRequestedRecord("history", "commit-1", (), attempt_id=attempt_id),
        MemoryCommitSucceededRecord("history", "commit-1", attempt_id=attempt_id),
        MemoryCommitFailedRecord("history", "commit-1", failure, attempt_id=attempt_id),
    )
    for record in records:
        wire = json.loads(encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, record)))
        assert wire["payload"]["attempt_id"] == attempt_id
        del wire["payload"]["attempt_id"]
        with pytest.raises(ValueError, match="attempt_id"):
            decode_record(json.dumps(wire).encode())


def test_proposals_preserve_order_and_reject_duplicate_ids() -> None:
    ordered = (
        MemoryMutation("m2", {"sequence": 1}),
        MemoryMutation("m1", {"sequence": 2}),
    )
    assert MemoryProposal("reflective", ordered).mutations == ordered
    with pytest.raises(ValueError, match="unique"):
        MemoryCommitRequestedRecord(
            "reflective",
            "commit",
            (MemoryMutation("same", {}), MemoryMutation("same", {})),
            attempt_id=1,
        )


class _OrderedStructuredModel:
    """Return one deterministic ordered multi-action reflection."""

    def invoke(self, messages: list[object]) -> dict[str, object]:
        return {
            "actions": [
                {"action": "add", "lesson_id": "z", "text": "first"},
                {"action": "add", "lesson_id": "a", "text": "second"},
            ]
        }


class _OrderedModel:
    """Expose the structured-output seam used by ReflectiveMemory."""

    def with_structured_output(self, schema: type[object]) -> _OrderedStructuredModel:
        return _OrderedStructuredModel()


def test_reflective_multi_action_order_survives_records_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reflective, "model_from_config", lambda config: _OrderedModel())
    memory = ReflectiveMemory({"provider": "ollama", "model": "qwen"})
    episode = MemoryEpisode(
        "run",
        (),
        ("agent",),
        MemoryEpisodeStatus.SUCCEEDED,
        {"input": "value"},
        "output",
    )
    with memory.open_run() as session:
        proposals = tuple(session.learn(LearnContext("run", "reflective", (episode,))))

    assert [mutation.payload["lesson_id"] for mutation in proposals] == ["z", "a"]
    ready = _round_trip(
        BackwardReadyRecord(1, BackwardResult(), (), (MemoryProposal("reflective", proposals),))
    )
    requested = _round_trip(
        MemoryCommitRequestedRecord(
            "reflective", "commit", ready.proposals[0].mutations, attempt_id=1
        )
    )
    assert [mutation.payload["lesson_id"] for mutation in requested.mutations] == [
        "z",
        "a",
    ]

    with memory.open_run() as session:
        session.commit(requested.commit_id, requested.mutations)
    with memory.open_run() as session:
        assert session.recall() == "[z] first\n[a] second"


class _ForgedEvidence(EvaluationEvidence):
    """A valid-looking evidence subclass rejected at the durable boundary."""


class _ForgedMutation(MemoryMutation):
    """A valid-looking mutation subclass rejected by learning records."""


class _StringSubclass(str):
    """An equality-compatible string subclass rejected by exact contracts."""


def test_records_reject_forged_nested_carrier_subclasses() -> None:
    evidence = _evidence()
    forged_evidence = _ForgedEvidence(
        evidence.scope,
        evidence.bound_inputs,
        evidence.base_system_prompt,
        evidence.prompt,
        evidence.memory_sections,
        evidence.interactions,
    )
    forged_mutation = _ForgedMutation("m1", {})

    with pytest.raises(TypeError, match="evidence"):
        NodeSucceededRecord(
            "agent",
            "agent",
            {"kind": "str", "value": "done"},
            forged_evidence,
        )
    with pytest.raises(TypeError, match="MemoryMutation"):
        MemoryCommitRequestedRecord(
            "reflective",
            "commit",
            (forged_mutation,),
         attempt_id=1)


def test_nested_frozen_carriers_are_revalidated_before_recording() -> None:
    segment = _segment()
    scope = MemoryScope(("flow", "agent"), ("reflective",), (segment,))
    object.__setattr__(segment, "index", True)
    with pytest.raises(TypeError, match="index"):
        EvaluationEvidence(scope, {}, "base", "prompt")

    interaction = ModelTextInteraction("visible")
    object.__setattr__(interaction, "text", _StringSubclass("visible"))
    with pytest.raises(TypeError, match="text"):
        EvaluationEvidence(
            MemoryScope(("flow", "agent"), ("reflective",)),
            {},
            "base",
            "prompt",
            interactions=(interaction,),
        )

    mutation = MemoryMutation("m1", {})
    object.__setattr__(mutation, "mutation_id", _StringSubclass("m1"))
    with pytest.raises(TypeError, match="mutation_id"):
        MemoryCommitRequestedRecord("reflective", "commit", (mutation,), attempt_id=1)


def test_codec_revalidates_carriers_mutated_after_record_construction() -> None:
    evidence = _evidence()
    record = NodeSucceededRecord(
        "agent",
        "agent",
        {"kind": "str", "value": "done"},
        evidence,
    )
    object.__setattr__(evidence.scope, "memories", (_StringSubclass("notes"), "reflective"))

    with pytest.raises(TypeError, match="memories"):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, record))


def test_codec_replays_outer_record_invariants_after_forging() -> None:
    proposal = MemoryProposal(
        "reflective", (MemoryMutation("m1", {}), MemoryMutation("m2", {}))
    )
    ready = BackwardReadyRecord(1, BackwardResult(), (), (proposal,))
    object.__setattr__(ready.proposals[0].mutations[1], "mutation_id", "m1")
    with pytest.raises(ValueError, match="unique"):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, ready))

    commit = MemoryCommitRequestedRecord(
        "reflective",
        "commit",
        (MemoryMutation("m1", {}),),
     attempt_id=1)
    object.__setattr__(commit, "mutations", list(commit.mutations))
    with pytest.raises(TypeError, match="tuple"):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, commit))

    operation_failure = MemoryOperationFailure(
        "commit", "provider.error", "failed", False
    )
    failed = MemoryCommitFailedRecord("reflective", "commit", operation_failure, attempt_id=1)
    object.__setattr__(operation_failure, "retryable", 0)
    with pytest.raises(TypeError, match="boolean"):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, failed))

    failure = Failure("node.execute", "failed", "agent")
    node_failed = NodeFailedRecord(failure)
    object.__setattr__(failure, "code", _StringSubclass("node.execute"))
    with pytest.raises(TypeError, match="code"):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, node_failed))


def test_schema_four_rejects_old_schema_and_nested_extra_or_unknown_fields() -> None:
    evidence_record = NodeSucceededRecord(
        "agent",
        "agent",
        {"kind": "str", "value": "done"},
        _evidence(),
    )
    wire = json.loads(
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, evidence_record))
    )

    old = dict(wire)
    old["schema"] = 3
    with pytest.raises(ValueError):
        decode_record(json.dumps(old))

    extra = json.loads(json.dumps(wire))
    extra["payload"]["evidence"]["scope"]["extra"] = True
    with pytest.raises(ValueError):
        decode_record(json.dumps(extra))

    unknown = json.loads(json.dumps(wire))
    unknown["payload"]["evidence"]["interactions"][0]["kind"] = "provider_trace"
    with pytest.raises(ValueError):
        decode_record(json.dumps(unknown))

    interaction_extra = json.loads(json.dumps(wire))
    interaction_extra["payload"]["evidence"]["interactions"][0]["provider"] = "x"
    with pytest.raises(ValueError):
        decode_record(json.dumps(interaction_extra))


def test_memory_scope_wire_has_no_boundary_field() -> None:
    record = NodeSucceededRecord(
        "agent", "agent", {"kind": "str", "value": "done"}, _evidence(),
    )
    wire = json.loads(encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, record)))
    scope = wire["payload"]["evidence"]["scope"]
    assert sorted(scope) == ["memories", "occurrence_path", "static_path"]
    scope["boundary"] = None
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire))


def test_schema_four_rejects_origin_and_extra_memory_operation_fields() -> None:
    started = _started(
        memories=(_spec(owner=_ROOT),),
        stacks=(),
        snapshots=(MemorySnapshotEnvelope("reflective", {}),),
    )
    started_wire = json.loads(
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started))
    )
    started_wire["payload"]["memories"][0]["origin"] = {"key": "source"}
    with pytest.raises(ValueError):
        decode_record(json.dumps(started_wire))

    private_config = json.loads(
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started))
    )
    private_config["payload"]["memories"][0]["config"] = {
        "nested": {"api_key": "private"}
    }
    with pytest.raises(ValueError, match="forbidden private key"):
        decode_record(json.dumps(private_config))

    failed = MemoryCommitFailedRecord(
        "reflective",
        "commit",
        MemoryOperationFailure("commit", "provider.error", "sanitized", False),
     attempt_id=1)
    failed_wire = json.loads(
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, failed))
    )
    failed_wire["payload"]["failure"]["credentials"] = "secret"
    with pytest.raises(ValueError):
        decode_record(json.dumps(failed_wire))


@pytest.mark.parametrize("config", _PLURAL_PRIVATE_CONFIG_CASES)
def test_schema_four_rejects_plural_private_memory_config(
    config: dict[str, Any],
) -> None:
    started = _started(
        memories=(_spec(owner=_ROOT),),
        stacks=(),
        snapshots=(MemorySnapshotEnvelope("reflective", {}),),
    )
    wire = json.loads(
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started))
    )
    wire["payload"]["memories"][0]["config"] = config

    with pytest.raises(ValueError, match="forbidden private key"):
        decode_record(json.dumps(wire))


def test_run_log_copying_isolated_from_new_carrier_accessors() -> None:
    record = _round_trip(
        MemoryCommitRequestedRecord(
            "reflective",
            "commit",
            (MemoryMutation("m1", {"items": [1]}),),
         attempt_id=1)
    )

    record.mutations[0].payload["items"].append(2)
    assert record.mutations[0].payload == {"items": [1]}


def test_memory_payloads_name_keys_on_the_wire() -> None:
    record = NodeSucceededRecord(
        "agent", "agent", {"kind": "str", "value": "done"}, _evidence()
    )
    evidence = json.loads(
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, record))
    )["payload"]["evidence"]
    assert evidence["scope"] == {
        "static_path": ["flow", "agent"],
        "memories": ["notes", "reflective"],
        "occurrence_path": [{"kind": "map_item", "static_path": ["flow"], "index": 0}],
    }
    assert evidence["memory_sections"] == [
        {"memory": "notes", "text": "flow memory"},
        {"memory": "reflective", "text": "agent memory"},
    ]
    assert _round_trip(record) == record

    proposal = MemoryProposal("reflective", (MemoryMutation("m1", {"add": 1}),))
    ready = BackwardReadyRecord(1, BackwardResult(), (), (proposal,))
    ready_wire = json.loads(encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, ready)))
    assert ready_wire["payload"]["proposals"] == [
        {"memory": "reflective", "mutations": [{"mutation_id": "m1", "payload": {"add": 1}}]}
    ]
    assert "frame_proposals" not in ready_wire["payload"]
    assert _round_trip(ready).proposals == (proposal,)

    commit = MemoryCommitRequestedRecord("reflective", "c1", proposal.mutations, attempt_id=1)
    commit_wire = json.loads(encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, commit)))
    assert commit_wire["payload"]["mutations"] == [{"mutation_id": "m1", "payload": {"add": 1}}]

    ready_wire["payload"]["proposals"][0]["frame_ref"] = None
    with pytest.raises(ValueError):
        decode_record(json.dumps(ready_wire).encode())
