"""Run-record contracts for source-neutral memory evidence and learning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from typing import Any

import pytest

from agent_composer.diagnostics import Failure
from agent_composer.memory import (
    EvaluationEvidence,
    LearnContext,
    MemoryBoundary,
    MemoryBoundaryKind,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOccurrenceKind,
    MemoryOccurrenceRef,
    MemoryOccurrenceSegment,
    MemoryOperationFailure,
    MemoryOwner,
    MemoryPolicySpec,
    MemoryScope,
    MemorySection,
    MemorySnapshotEnvelope,
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
from agent_composer.runtime.contracts import FrameProposal
from agent_composer.typesys import VariablePool


_FINGERPRINT = "a" * 64
_DIGEST = "b" * 64
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


def _flow_ref() -> MemoryFrameRef:
    return MemoryFrameRef("root", ("root", "flow"), MemoryOwner.FLOW)


def _agent_ref() -> MemoryFrameRef:
    return MemoryFrameRef("root", ("root", "flow", "agent"), MemoryOwner.AGENT)


def _segment(index: int = 0) -> MemoryOccurrenceSegment:
    return MemoryOccurrenceSegment(
        MemoryOccurrenceKind.MAP_ITEM,
        ("root", "flow"),
        index,
    )


def _evidence(*, bound_inputs: dict[str, Any] | None = None) -> EvaluationEvidence:
    flow_ref = _flow_ref()
    agent_ref = _agent_ref()
    return EvaluationEvidence(
        scope=MemoryScope(
            root_flow_id="root",
            static_path=agent_ref.static_path,
            frame_stack=(flow_ref, agent_ref),
            occurrence_path=(_segment(),),
        ),
        bound_inputs={"question": ["hello"]} if bound_inputs is None else bound_inputs,
        base_system_prompt="base",
        prompt="answer the question",
        memory_sections=(
            MemorySection(flow_ref, "flow memory"),
            MemorySection(agent_ref, "agent memory"),
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
    layout: tuple[MemoryFrameSpec, ...] | None = None,
    snapshots: tuple[MemorySnapshotEnvelope, ...] | None = None,
    learnable: bool = False,
) -> RunStartedRecord:
    return RunStartedRecord(
        flow_fingerprint=_FINGERPRINT,
        initial_pool=encode_pool(VariablePool(start_id="start")),
        start_wiring_override={},
        effective_llm_by_static_path={},
        num_workers=0,
        memory_layout=layout,
        memory_snapshots=snapshots,
        learnable=learnable,
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
        MemoryScope("root", _agent_ref().static_path, (_agent_ref(),)),
        inputs_source,
        "base",
        "prompt",
        (MemorySection(_agent_ref(), "memory"),),
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
def test_frame_config_rejects_private_keys(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="forbidden private key"):
        MemoryFrameSpec(_agent_ref(), "reflective", config)


@pytest.mark.parametrize("config", _PRIVATE_CONFIG_CASES)
def test_codec_revalidates_forged_private_frame_config(
    config: dict[str, Any],
) -> None:
    frame = MemoryFrameSpec(_agent_ref(), "reflective", {})
    started = _started(
        layout=(frame,),
        snapshots=(MemorySnapshotEnvelope("reflective", {"state": []}),),
    )
    private_snapshot = MemorySnapshotEnvelope("opaque", config)
    object.__setattr__(frame, "_config", private_snapshot._snapshot)

    with pytest.raises(ValueError, match="forbidden private key"):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started))


def test_public_token_policy_options_round_trip_durably() -> None:
    config = {
        "stop_token": "semantic-stop-marker",
        "continuationToken": "semantic-continuation-marker",
    }
    started = _started(
        layout=(MemoryFrameSpec(_agent_ref(), "reflective", config),),
        snapshots=(MemorySnapshotEnvelope("reflective", {"state": []}),),
    )

    encoded = encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started))
    decoded = decode_record(encoded).record

    assert b"semantic-stop-marker" in encoded
    assert b"semantic-continuation-marker" in encoded
    assert type(decoded) is RunStartedRecord
    assert decoded.memory_layout is not None
    assert decoded.memory_layout[0].config == config


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
    scope = MemoryScope(
        "root",
        _agent_ref().static_path,
        (_flow_ref(), _agent_ref()),
    )

    with pytest.raises(ValueError, match="both"):
        EvaluationEvidence(scope, {}, "base", None)
    with pytest.raises(ValueError, match="scope.frame_stack"):
        EvaluationEvidence(
            scope,
            {},
            "base",
            "prompt",
            (
                MemorySection(
                    MemoryFrameRef("root", ("root", "other"), MemoryOwner.AGENT),
                    "x",
                ),
            ),
        )
    with pytest.raises(ValueError, match="unique"):
        EvaluationEvidence(
            scope,
            {},
            "base",
            "prompt",
            (MemorySection(_flow_ref(), "a"), MemorySection(_flow_ref(), "b")),
        )
    with pytest.raises(ValueError, match="order"):
        EvaluationEvidence(
            scope,
            {},
            "base",
            "prompt",
            (MemorySection(_agent_ref(), "a"), MemorySection(_flow_ref(), "b")),
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


def test_memory_scope_requires_one_lexical_unique_frame_chain() -> None:
    flow_ref = _flow_ref()
    agent_ref = _agent_ref()

    with pytest.raises(ValueError, match="unique"):
        MemoryScope(
            "root",
            agent_ref.static_path,
            (flow_ref, flow_ref),
        )
    sibling = MemoryFrameRef(
        "root",
        ("root", "sibling"),
        MemoryOwner.AGENT,
    )
    with pytest.raises(ValueError, match="lexical"):
        MemoryScope(
            "root",
            agent_ref.static_path,
            (flow_ref, sibling),
        )
    nested_flow = MemoryFrameRef(
        "root",
        ("root", "flow", "agent", "child"),
        MemoryOwner.FLOW,
    )
    with pytest.raises(ValueError, match="final local"):
        MemoryScope(
            "root",
            ("root", "flow", "agent", "child", "node"),
            (agent_ref, nested_flow),
        )
    with pytest.raises(ValueError, match="innermost"):
        MemoryScope(
            "root",
            agent_ref.static_path,
            (flow_ref, agent_ref),
            boundary=MemoryBoundary(MemoryBoundaryKind.START, flow_ref),
        )
    assert EvaluationEvidence(
        MemoryScope("root", ("root", "neutral")), {}, "base", "prompt",
    ).memory_sections == ()
    with pytest.raises(ValueError, match="lexical nesting"):
        MemoryScope(
            "root",
            agent_ref.static_path,
            (flow_ref, agent_ref),
            occurrence_path=(
                MemoryOccurrenceSegment(
                    MemoryOccurrenceKind.MAP_ITEM,
                    ("root", "other"),
                    0,
                ),
                MemoryOccurrenceSegment(
                    MemoryOccurrenceKind.LOOP_ITERATION,
                    ("root", "flow"),
                    0,
                ),
            ),
        )
    with pytest.raises(ValueError, match="begin"):
        MemoryFrameRef("root", ("other", "agent"), MemoryOwner.AGENT)


def test_episode_accepts_binding_failure_and_typed_visible_material() -> None:
    episode = MemoryEpisode(
        run_id="run",
        occurrence=MemoryOccurrenceRef(_agent_ref()),
        owner_static_path=_agent_ref().static_path,
        status=MemoryEpisodeStatus.FAILED,
        inputs=None,
        failure={"code": "binding.evaluate"},
        base_system_prompt=None,
        prompt=None,
        memory_sections=(MemorySection(_agent_ref(), "memory"),),
        interactions=(ModelTextInteraction("visible"),),
    )

    assert episode.inputs is None
    assert episode.memory_sections == (MemorySection(_agent_ref(), "memory"),)
    assert episode.interactions == (ModelTextInteraction("visible"),)


def test_run_started_distinguishes_disabled_and_enabled_empty_memory() -> None:
    disabled = _round_trip(_started())
    enabled = _round_trip(_started(layout=(), snapshots=()))
    learning = _round_trip(_started(layout=(), snapshots=(), learnable=True))

    assert disabled.memory_layout is None
    assert disabled.memory_snapshots is None
    assert enabled.memory_layout == ()
    assert enabled.memory_snapshots == ()
    assert learning.learnable is True

    with pytest.raises(ValueError, match="both"):
        _started(layout=(), snapshots=None)
    assert _round_trip(_started(learnable=True)).learnable is True
    with pytest.raises(TypeError, match="boolean"):
        _started(layout=(), snapshots=(), learnable=1)  # type: ignore[arg-type]


def test_run_started_uses_learnable_as_its_only_wire_retention_field() -> None:
    started = _started(learnable=True)
    wire = json.loads(encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started)))

    assert wire["payload"]["learnable"] is True
    assert "learn" not in wire["payload"]

    wire["payload"]["learn"] = wire["payload"].pop("learnable")
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire).encode())


def test_run_started_requires_canonical_layout_and_matching_snapshots() -> None:
    flow = MemoryFrameSpec(_flow_ref(), "profile", {"mode": "flow"})
    agent = MemoryFrameSpec(_agent_ref(), "reflective", {"mode": "agent"})
    snapshots = (
        MemorySnapshotEnvelope("profile", {"state": 1}),
        MemorySnapshotEnvelope("reflective", {"state": 2}),
    )
    decoded = _round_trip(_started(layout=(flow, agent), snapshots=snapshots))
    wire = json.loads(
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, decoded))
    )

    assert decoded.memory_layout == (flow, agent)
    assert decoded.memory_snapshots == snapshots
    assert "origin" not in json.dumps(wire["payload"]["memory_layout"])

    with pytest.raises(ValueError, match="sorted and unique"):
        _started(layout=(agent, flow), snapshots=snapshots)
    with pytest.raises(ValueError, match="sorted and unique"):
        _started(layout=(flow, flow), snapshots=(snapshots[0],))
    with pytest.raises(ValueError, match="sorted and unique"):
        _started(layout=(flow, agent), snapshots=tuple(reversed(snapshots)))
    with pytest.raises(ValueError, match="equal"):
        _started(layout=(flow, agent), snapshots=(snapshots[0],))
    other_root = MemoryFrameSpec(
        MemoryFrameRef("second", ("second",), MemoryOwner.FLOW),
        "profile",
        {},
    )
    with pytest.raises(ValueError, match="one root_flow_id"):
        _started(
            layout=(flow, other_root),
            snapshots=(snapshots[0],),
        )


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
    ref = _agent_ref()
    mutations = (
        MemoryMutation("m1", ref, {"action": "add"}),
        MemoryMutation("m2", ref, {"action": "revise"}),
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


def test_proposals_preserve_order_and_reject_wrong_ownership_or_duplicate_ids() -> None:
    ref = _agent_ref()
    other = MemoryFrameRef("root", ("root", "other"), MemoryOwner.AGENT)
    with pytest.raises(ValueError, match="frame_ref"):
        FrameProposal(ref, (MemoryMutation("m1", other, {}),))
    ordered = (
        MemoryMutation("m2", ref, {"sequence": 1}),
        MemoryMutation("m1", ref, {"sequence": 2}),
    )
    assert FrameProposal(ref, ordered).mutations == ordered
    with pytest.raises(ValueError, match="unique"):
        MemoryCommitRequestedRecord(
            "reflective",
            "commit",
            (MemoryMutation("same", ref, {}), MemoryMutation("same", other, {})),
         attempt_id=1)


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
    ref = _agent_ref()
    spec = MemoryFrameSpec(ref, "reflective", {})
    episode = MemoryEpisode(
        "run",
        MemoryOccurrenceRef(ref),
        ref.static_path,
        MemoryEpisodeStatus.SUCCEEDED,
        {"input": "value"},
        "output",
    )
    with memory.open_run() as session:
        proposals = tuple(
            session.open_frame(spec).learn(LearnContext("run", ref, (episode,)))
        )

    assert [mutation.payload["lesson_id"] for mutation in proposals] == ["z", "a"]
    ready = _round_trip(BackwardReadyRecord(1, BackwardResult(), (), (FrameProposal(ref, proposals),)))
    requested = _round_trip(
        MemoryCommitRequestedRecord("reflective", "commit", ready.frame_proposals[0].mutations, attempt_id=1)
    )
    assert [mutation.payload["lesson_id"] for mutation in requested.mutations] == [
        "z",
        "a",
    ]

    with memory.open_run() as session:
        session.open_frame(spec)
        session.commit(requested.commit_id, requested.mutations)
    with memory.open_run() as session:
        assert session.open_frame(spec).recall() == "[z] first\n[a] second"


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
    forged_mutation = _ForgedMutation("m1", _agent_ref(), {})

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
    ref = _agent_ref()
    segment = _segment()
    scope = MemoryScope(
        "root",
        ref.static_path,
        (ref,),
        (segment,),
    )
    object.__setattr__(segment, "index", True)
    with pytest.raises(TypeError, match="index"):
        EvaluationEvidence(scope, {}, "base", "prompt")

    interaction = ModelTextInteraction("visible")
    object.__setattr__(interaction, "text", _StringSubclass("visible"))
    with pytest.raises(TypeError, match="text"):
        EvaluationEvidence(
            MemoryScope("root", ref.static_path, (ref,)),
            {},
            "base",
            "prompt",
            interactions=(interaction,),
        )

    mutation_ref = MemoryFrameRef("root", ("root", "agent"), MemoryOwner.AGENT)
    mutation = MemoryMutation("m1", mutation_ref, {})
    object.__setattr__(mutation_ref, "root_flow_id", _StringSubclass("root"))
    with pytest.raises(TypeError, match="root_flow_id"):
        MemoryCommitRequestedRecord("reflective", "commit", (mutation,), attempt_id=1)


def test_codec_revalidates_carriers_mutated_after_record_construction() -> None:
    evidence = _evidence()
    record = NodeSucceededRecord(
        "agent",
        "agent",
        {"kind": "str", "value": "done"},
        evidence,
    )
    object.__setattr__(evidence.scope, "root_flow_id", _StringSubclass("root"))

    with pytest.raises(TypeError, match="root_flow_id"):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, record))


def test_codec_replays_outer_record_invariants_after_forging() -> None:
    proposal = FrameProposal(_agent_ref(), (MemoryMutation("m1", _agent_ref(), {}),))
    ready = BackwardReadyRecord(1, BackwardResult(), (), (proposal,))
    object.__setattr__(ready.frame_proposals[0].mutations[0], "frame_ref", _flow_ref())
    with pytest.raises(ValueError, match="frame_ref"):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, ready))

    commit = MemoryCommitRequestedRecord(
        "reflective",
        "commit",
        (MemoryMutation("m1", _agent_ref(), {}),),
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
    assert sorted(scope) == ["frame_stack", "occurrence_path", "root_flow_id", "static_path"]
    scope["boundary"] = None
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire))


def test_schema_four_rejects_origin_and_extra_memory_operation_fields() -> None:
    started = _started(
        layout=(MemoryFrameSpec(_flow_ref(), "reflective", {}),),
        snapshots=(MemorySnapshotEnvelope("reflective", {}),),
    )
    started_wire = json.loads(
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started))
    )
    started_wire["payload"]["memory_layout"][0]["origin"] = {"key": "source"}
    with pytest.raises(ValueError):
        decode_record(json.dumps(started_wire))

    private_config = json.loads(
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started))
    )
    private_config["payload"]["memory_layout"][0]["config"] = {
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
def test_schema_four_rejects_plural_private_frame_config(
    config: dict[str, Any],
) -> None:
    started = _started(
        layout=(MemoryFrameSpec(_flow_ref(), "reflective", {}),),
        snapshots=(MemorySnapshotEnvelope("reflective", {}),),
    )
    wire = json.loads(
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, started))
    )
    wire["payload"]["memory_layout"][0]["config"] = config

    with pytest.raises(ValueError, match="forbidden private key"):
        decode_record(json.dumps(wire))


def test_run_log_copying_isolated_from_new_carrier_accessors() -> None:
    record = _round_trip(
        MemoryCommitRequestedRecord(
            "reflective",
            "commit",
            (MemoryMutation("m1", _agent_ref(), {"items": [1]}),),
         attempt_id=1)
    )

    record.mutations[0].payload["items"].append(2)
    assert record.mutations[0].payload == {"items": [1]}
