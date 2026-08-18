"""Schema-4 run-record contracts for source-neutral memory evidence and learning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from typing import Any

import pytest

from agent_composer.diagnostics import Failure
from agent_composer.memory import (
    EvaluationEvidence,
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
    MemoryScope,
    MemorySection,
    MemorySnapshotEnvelope,
    ModelTextInteraction,
    ModelValueInteraction,
    ToolCallInteraction,
    ToolResultInteraction,
)
from agent_composer.runtime.log import (
    SCHEMA_VERSION,
    MemoryCommitFailedRecord,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    MemoryFrameLearnedRecord,
    MemoryFrameLearningFailedRecord,
    MemoryFrameLearningStartedRecord,
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
from agent_composer.typesys import VariablePool


_FINGERPRINT = "a" * 64
_DIGEST = "b" * 64


def _flow_ref() -> MemoryFrameRef:
    return MemoryFrameRef("root", ("root", "flow"), MemoryOwner.FLOW)


def _agent_ref() -> MemoryFrameRef:
    return MemoryFrameRef("root", ("root", "flow", "agent"), MemoryOwner.AGENT)


def _segment(index: int = 0) -> MemoryOccurrenceSegment:
    return MemoryOccurrenceSegment(
        MemoryOccurrenceKind.MAP_ITEM,
        ("root", "flow", "mapped"),
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
    learn: bool = False,
) -> RunStartedRecord:
    return RunStartedRecord(
        flow_fingerprint=_FINGERPRINT,
        initial_pool=encode_pool(VariablePool(start_id="start")),
        start_wiring_override={},
        effective_llm_by_static_path={},
        num_workers=0,
        memory_layout=layout,
        memory_snapshots=snapshots,
        learn=learn,
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
                    MemoryFrameRef("root", ("other",), MemoryOwner.AGENT),
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
    learning = _round_trip(_started(layout=(), snapshots=(), learn=True))

    assert disabled.memory_layout is None
    assert disabled.memory_snapshots is None
    assert enabled.memory_layout == ()
    assert enabled.memory_snapshots == ()
    assert learning.learn is True

    with pytest.raises(ValueError, match="both"):
        _started(layout=(), snapshots=None)
    with pytest.raises(ValueError, match="requires enabled"):
        _started(learn=True)
    with pytest.raises(TypeError, match="boolean"):
        _started(layout=(), snapshots=(), learn=1)  # type: ignore[arg-type]


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


def test_every_memory_learning_record_round_trips() -> None:
    ref = _agent_ref()
    occurrences = (
        MemoryOccurrenceRef(ref),
        MemoryOccurrenceRef(ref, (_segment(),)),
    )
    mutations = (
        MemoryMutation("m1", ref, {"action": "add"}),
        MemoryMutation("m2", ref, {"action": "revise"}),
    )
    failure = MemoryOperationFailure("learn", "provider.error", "failed", True)
    records = (
        MemoryFrameLearningStartedRecord("reflective", ref, occurrences),
        MemoryFrameLearnedRecord("reflective", ref, mutations),
        MemoryFrameLearningFailedRecord("reflective", ref, failure),
        MemoryCommitRequestedRecord("reflective", "commit-1", mutations),
        MemoryCommitSucceededRecord("reflective", "commit-1"),
        MemoryCommitFailedRecord("reflective", "commit-1", failure),
    )

    for record in records:
        assert _round_trip(record) == record


def test_learning_records_reject_wrong_ownership_order_and_duplicate_ids() -> None:
    ref = _agent_ref()
    other = MemoryFrameRef("root", ("root", "other"), MemoryOwner.AGENT)
    base = MemoryOccurrenceRef(ref)
    mapped = MemoryOccurrenceRef(ref, (_segment(),))

    with pytest.raises(ValueError, match="frame_ref"):
        MemoryFrameLearningStartedRecord(
            "reflective", ref, (MemoryOccurrenceRef(other),)
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        MemoryFrameLearningStartedRecord("reflective", ref, (mapped, base))
    with pytest.raises(ValueError, match="sorted and unique"):
        MemoryFrameLearningStartedRecord("reflective", ref, (base, base))
    with pytest.raises(ValueError, match="frame_ref"):
        MemoryFrameLearnedRecord(
            "reflective", ref, (MemoryMutation("m1", other, {}),)
        )
    with pytest.raises(ValueError, match="canonical"):
        MemoryFrameLearnedRecord(
            "reflective",
            ref,
            (MemoryMutation("m2", ref, {}), MemoryMutation("m1", ref, {})),
        )
    with pytest.raises(ValueError, match="unique"):
        MemoryCommitRequestedRecord(
            "reflective",
            "commit",
            (MemoryMutation("same", ref, {}), MemoryMutation("same", other, {})),
        )


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
        )


def test_nested_frozen_carriers_are_revalidated_before_recording() -> None:
    ref = MemoryFrameRef("root", ("root", "agent"), MemoryOwner.AGENT)
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
        MemoryCommitRequestedRecord("reflective", "commit", (mutation,))


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

    failed = MemoryCommitFailedRecord(
        "reflective",
        "commit",
        MemoryOperationFailure("commit", "provider.error", "sanitized", False),
    )
    failed_wire = json.loads(
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, failed))
    )
    failed_wire["payload"]["failure"]["credentials"] = "secret"
    with pytest.raises(ValueError):
        decode_record(json.dumps(failed_wire))


def test_run_log_copying_isolated_from_new_carrier_accessors() -> None:
    record = _round_trip(
        MemoryCommitRequestedRecord(
            "reflective",
            "commit",
            (MemoryMutation("m1", _agent_ref(), {"items": [1]}),),
        )
    )

    record.mutations[0].payload["items"].append(2)
    assert record.mutations[0].payload == {"items": [1]}
