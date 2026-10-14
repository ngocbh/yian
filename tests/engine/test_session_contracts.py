"""Detached source-neutral conversation identities and optional capabilities."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import inspect
import json
from typing import get_args, get_type_hints

import pytest

from yian.memory import contracts as c
from tests.engine._memory_fakes import _Memory


def test_memory_protocol_has_no_frames():
    memory = _Memory(recall_text="notes")
    session = memory.open_run()
    assert isinstance(memory, c.Memory)
    assert isinstance(session, c.MemorySession)
    assert {name for name in vars(c.MemorySession) if not name.startswith("_")} == {
        "snapshot",
        "close",
        "recall",
        "learn",
        "commit",
    }
    assert "__enter__" in vars(c.MemorySession)
    assert "__exit__" in vars(c.MemorySession)
    assert not hasattr(session, "open_frame")
    assert session.recall() == "notes"
    assert session.learn(c.LearnContext("run", "notes", ())) == []
    assert memory.recall_contexts == [None]
    assert [context.memory for context in memory.learn_contexts] == ["notes"]
    session.close()


@pytest.mark.parametrize("sequence", [True, False, 0, -1, 1.0, "1"])
def test_invocation_rejects_boolean_start_sequence(sequence):
    with pytest.raises((TypeError, ValueError)):
        c.ConversationInvocation("run", sequence)


def test_request_detaches_compatibility():
    descriptor = {"provider": "fake", "options": [True, 1, 1.0]}
    invocation = c.ConversationInvocation("run", 1)
    request = c.ConversationRequest(("root", "child", "agent"), invocation, descriptor)
    descriptor["options"].append("changed")
    request.compatibility["options"].append("changed")
    assert request.compatibility == {"provider": "fake", "options": [True, 1, 1.0]}
    assert request.invocation == invocation
    assert request.invocation is not invocation
    assert request != c.ConversationRequest(
        request.consumer_static_path,
        invocation,
        {"provider": "fake", "options": [1, 1, 1.0]},
    )


def _invocation():
    return c.ConversationInvocation("run", 1)


def _stamp(ordinal=0, revision=0):
    return c.ConversationStamp(_invocation(), "namespace", 0, revision, ordinal)


def test_item_payload_is_detached():
    payload = {"text": ["hello"]}
    item = c.ConversationItem("message-1", "api_message", payload)
    payload["text"].append("caller mutation")
    item.payload["text"].append("reader mutation")
    assert item.payload == {"text": ["hello"]}
    assert c.ConversationItem("x", "y", {"value": True}) != c.ConversationItem(
        "x", "y", {"value": 1}
    )


@pytest.mark.parametrize("field", ["generation", "expected_revision", "ordinal"])
@pytest.mark.parametrize("value", [True, -1, 1.0, "1"])
def test_stamp_requires_nonnegative_exact_counters(field, value):
    arguments = dict(
        invocation=_invocation(),
        namespace="ns",
        generation=0,
        expected_revision=0,
        ordinal=0,
    )
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        c.ConversationStamp(**arguments)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"v": float("nan")},
        {"v": float("inf")},
        {"v": ()},
        {"v": object()},
        {"v": "\ud800"},
        {1: "v"},
    ],
)
def test_conversation_json_rejects_noncanonical_values(payload):
    with pytest.raises((TypeError, ValueError)):
        c.ConversationItem("id", "format", payload)
    with pytest.raises((TypeError, ValueError)):
        c.ConversationRequest(("root", "agent"), _invocation(), payload)


def test_conversation_json_rejects_cycles():
    payload = {}
    payload["cycle"] = payload
    with pytest.raises(ValueError, match="circular"):
        c.ConversationItem("id", "format", payload)


def _failure():
    return c.MemoryOperationFailure("effect", "fake.failed", "Failed", False)


@pytest.mark.parametrize("kind", ["model", "tool"])
@pytest.mark.parametrize("status", ["pending", "completed", "failed", "interrupted"])
def test_effect_state_preserves_invocation_and_status(kind, status):
    invocation = _invocation()
    request = {"arguments": ["hello"]}
    failure = _failure() if status == "failed" else None
    effect = c.ConversationEffectState(
        invocation, 1, kind, request, status, (2, 3), failure
    )
    request["arguments"].clear()
    effect.request["arguments"].clear()
    assert effect.request == {"arguments": ["hello"]}
    assert effect.invocation == invocation and effect.invocation is not invocation
    assert effect.status == status
    assert effect.observation_ordinals == (2, 3)
    if failure is not None:
        assert effect.failure == failure and effect.failure is not failure


@pytest.mark.parametrize("ordinals", [[2], (True,), (1,), (3, 2), (2, 2)])
def test_effect_state_requires_ordered_later_observations(ordinals):
    with pytest.raises((TypeError, ValueError)):
        c.ConversationEffectState(
            _invocation(), 1, "model", {}, "pending", ordinals, None
        )


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted"])
def test_completion_preserves_actual_disposition(status):
    completed = c.ConversationCompletion(_invocation(), 3, status)
    assert completed.status == status
    assert completed.closed_ordinal == 3
    with pytest.raises(ValueError):
        c.ConversationCompletion(_invocation(), 3, "pending")


@pytest.mark.parametrize("field", ["event_digest", "before_digest", "after_digest"])
@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, "", None])
def test_applied_entry_validates_digest_shape(field, digest):
    values = dict(
        operation_id='["run",1,0]',
        event_digest="a" * 64,
        before_digest="b" * 64,
        after_digest="c" * 64,
    )
    assert c.ConversationApplied(**values).operation_id == '["run",1,0]'
    values[field] = digest
    with pytest.raises((TypeError, ValueError)):
        c.ConversationApplied(**values)


@pytest.mark.parametrize(
    "operation_id",
    [
        '["run",true,0]',
        '["run",0,0]',
        '["run",1,-1]',
        '["run",1,0.0]',
        '["run", 1, 0]',
        '["",1,0]',
        "{}",
        "not-json",
        '["run",1,0,4]',
    ],
)
def test_applied_entry_requires_canonical_operation_identity(operation_id):
    with pytest.raises((TypeError, ValueError)):
        c.ConversationApplied(operation_id, "a" * 64, "b" * 64, "c" * 64)


def _view(**changes):
    values = dict(
        namespace="namespace",
        generation=0,
        revision=0,
        compatibility={"provider": "fake"},
        items=(),
        effects=(),
        closed_invocations=(),
        applied=(),
    )
    values.update(changes)
    return c.ConversationView(**values)


def _applied(ordinal=0):
    return c.ConversationApplied(f'["run",1,{ordinal}]', "a" * 64, "b" * 64, "c" * 64)


def test_empty_view_has_exact_detached_shape():
    descriptor = {"options": [True]}
    view = _view(compatibility=descriptor)
    descriptor["options"].clear()
    view.compatibility["options"].clear()
    assert view.compatibility == {"options": [True]}
    assert (view.namespace, view.generation, view.revision) == ("namespace", 0, 0)
    assert (
        view.items,
        view.effects,
        view.closed_invocations,
        view.applied,
    ) == ((), (), (), ())
    with pytest.raises(FrozenInstanceError):
        view.revision = 1


def test_view_rejects_duplicate_item_ids():
    item = c.ConversationItem("id", "text", {})
    with pytest.raises(ValueError, match="unique"):
        _view(revision=1, applied=(_applied(),), items=(item, item))
    view = _view(revision=1, applied=(_applied(),), items=(item,))
    assert view.items[0] == item and view.items[0] is not item


@pytest.mark.parametrize("field", ["items", "effects", "closed_invocations", "applied"])
def test_view_requires_exact_tuple_collections(field):
    with pytest.raises(TypeError, match="tuple"):
        _view(**{field: []})


def test_view_requires_one_applied_entry_per_revision():
    with pytest.raises(ValueError, match="revision"):
        _view(revision=1)
    with pytest.raises(ValueError, match="revision"):
        _view(applied=(_applied(),))


def test_zero_revision_view_cannot_carry_history():
    with pytest.raises(ValueError, match="empty"):
        _view(items=(c.ConversationItem("id", "message", {}),))


@pytest.mark.parametrize("field", ["effects", "closed_invocations", "applied"])
def test_view_rejects_duplicate_ledger_identities(field):
    entry = {
        "effects": c.ConversationEffectState(
            _invocation(), 1, "model", {}, "pending", (), None
        ),
        "closed_invocations": c.ConversationCompletion(_invocation(), 2, "failed"),
        "applied": _applied(),
    }[field]
    changes = {"revision": 2, "applied": (_applied(), _applied(1))}
    changes[field] = (entry, entry)
    with pytest.raises(ValueError, match="unique"):
        _view(**changes)


def test_admitted_contains_detached_baseline():
    baseline = _view()
    stamp = _stamp()
    event = c.ConversationAdmitted(stamp, baseline)
    assert event.baseline == baseline and event.baseline is not baseline
    assert event.stamp == stamp and event.stamp is not stamp
    with pytest.raises(ValueError):
        c.ConversationAdmitted(_stamp(1), baseline)
    with pytest.raises(ValueError):
        c.ConversationAdmitted(_stamp(revision=1), baseline)


def test_requested_retains_only_new_items():
    item = c.ConversationItem("id", "text", {"text": "hello"})
    payload = {"args": [True]}
    event = c.ConversationEffectRequested(_stamp(1, 1), "model", payload, (item,))
    payload["args"].clear()
    event.payload["args"].clear()
    assert event.payload == {"args": [True]}
    assert event.items == (item,) and event.items[0] is not item
    with pytest.raises(ValueError, match="unique"):
        c.ConversationEffectRequested(_stamp(1, 1), "model", {}, (item, item))


@pytest.mark.parametrize(
    "status,has_failure",
    [("completed", True), ("pending", True), ("failed", False), ("unknown", False)],
)
def test_observed_rejects_invalid_terminal_failure(status, has_failure):
    failure = _failure() if has_failure else None
    with pytest.raises(ValueError):
        c.ConversationEffectObserved(_stamp(2, 2), 1, status, (), failure)
    with pytest.raises(ValueError):
        c.ConversationEffectState(_invocation(), 1, "model", {}, status, (2,), failure)


@pytest.mark.parametrize("status", ["pending", "completed", "failed", "interrupted"])
def test_observed_preserves_detached_payload_and_actual_status(status):
    failure = _failure() if status in ("failed", "interrupted") else None
    item = c.ConversationItem("id", "text", {})
    event = c.ConversationEffectObserved(_stamp(2, 2), 1, status, (item,), failure)
    assert event.status == status
    assert event.items == (item,) and event.items[0] is not item
    if failure is not None:
        assert event.failure == failure and event.failure is not failure


@pytest.mark.parametrize("ordinal", [True, 0, 2, 3])
def test_observed_requires_a_prior_effect(ordinal):
    with pytest.raises((TypeError, ValueError)):
        c.ConversationEffectObserved(_stamp(2, 2), ordinal, "completed", (), None)


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted"])
def test_closed_accepts_only_turn_dispositions(status):
    assert c.ConversationClosed(_stamp(3, 3), status).status == status
    with pytest.raises(ValueError):
        c.ConversationClosed(_stamp(3, 3), "pending")
    with pytest.raises(ValueError):
        c.ConversationClosed(_stamp(), status)


def test_conversation_capabilities_are_optional_and_exact():
    memory = _Memory()
    session = memory.open_run()
    assert not isinstance(session, c.TranscriptSource)
    assert set(get_args(c.ConversationCheckpoint)) == {
        c.ConversationAdmitted,
        c.ConversationEffectRequested,
        c.ConversationEffectObserved,
        c.ConversationClosed,
    }
    for protocol in (c.TranscriptSource, c.ConversationTurn):
        for member in vars(protocol).values():
            if inspect.isfunction(member):
                get_type_hints(member)
    turn_parameters = inspect.signature(c.TranscriptSource.open_turn).parameters
    assert "checkpoint" not in turn_parameters
    assert (
        inspect.signature(c.TranscriptSource.open_turn)
        .parameters["check_stopped"]
        .kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(c.ConversationTurn.request).parameters["items"].default == ()
    )


def test_conversation_contracts_are_exported_without_concrete_backend():
    import yian.memory as memory

    names = (
        "ConversationInvocation",
        "ConversationRequest",
        "ConversationItem",
        "ConversationStamp",
        "ConversationEffectKind",
        "ConversationEffectStatus",
        "ConversationEffectState",
        "ConversationCompletion",
        "ConversationApplied",
        "ConversationView",
        "ConversationAdmitted",
        "ConversationEffectRequested",
        "ConversationEffectObserved",
        "ConversationClosed",
        "ConversationCheckpoint",
        "CheckStopped",
        "TranscriptSource",
        "ConversationTurn",
    )
    for name in names:
        assert name in memory.__all__
        assert getattr(memory, name) is getattr(c, name)
    assert "SessionMemory" not in memory.__all__
    assert "SessionMemory" not in vars(memory)
    for name in (
        "ConversationAuthority",
        "CheckpointSink",
        "PreparedConversationApply",
        "ConversationAcceptance",
        "ConversationSession",
    ):
        assert name not in vars(c)
        assert name not in vars(memory)


@pytest.mark.parametrize("value", ["", "\ud800", 1, None])
@pytest.mark.parametrize(
    "factory",
    [
        lambda value: c.ConversationInvocation(value, 1),
        lambda value: c.ConversationItem(value, "text", {}),
        lambda value: c.ConversationItem("id", value, {}),
        lambda value: replace(_stamp(), namespace=value),
        lambda value: _view(namespace=value),
    ],
)
def test_conversation_identifiers_are_exact_nonempty_unicode(factory, value):
    with pytest.raises((TypeError, ValueError)):
        factory(value)


@pytest.mark.parametrize("kind", ["other", "", True])
def test_effect_kind_is_closed(kind):
    with pytest.raises((TypeError, ValueError)):
        c.ConversationEffectState(_invocation(), 1, kind, {}, "pending", (), None)
    with pytest.raises((TypeError, ValueError)):
        c.ConversationEffectRequested(_stamp(1, 1), kind, {}, ())


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted"])
def test_terminal_effect_requires_observation(status):
    with pytest.raises(ValueError, match="observation"):
        c.ConversationEffectState(
            _invocation(),
            1,
            "model",
            {},
            status,
            (),
            _failure() if status == "failed" else None,
        )


def test_pending_effect_can_have_no_observations():
    assert (
        c.ConversationEffectState(
            _invocation(), 1, "model", {}, "pending", (), None
        ).observation_ordinals
        == ()
    )


@pytest.mark.parametrize("field", ["generation", "revision"])
@pytest.mark.parametrize("value", [True, -1, 1.0])
def test_view_counters_are_exact_nonnegative_integers(field, value):
    with pytest.raises((TypeError, ValueError)):
        _view(**{field: value})


def test_nested_carrier_subclasses_are_rejected():
    class InvocationSubclass(c.ConversationInvocation):
        pass

    class ItemSubclass(c.ConversationItem):
        pass

    class ViewSubclass(c.ConversationView):
        pass

    with pytest.raises(TypeError, match="invocation"):
        c.ConversationRequest(("root", "agent"), InvocationSubclass("run", 1), {})
    with pytest.raises(TypeError, match="ConversationItem"):
        _view(
            revision=1, applied=(_applied(),), items=(ItemSubclass("id", "text", {}),)
        )
    baseline = _view()
    with pytest.raises(TypeError, match="ConversationView"):
        c.ConversationAdmitted(
            _stamp(),
            ViewSubclass(baseline.namespace, 0, 0, {}, (), (), (), ()),
        )


def test_nested_view_carriers_are_revalidated_and_detached():
    effect = c.ConversationEffectState(
        _invocation(), 1, "model", {}, "failed", (2,), _failure()
    )
    completion = c.ConversationCompletion(_invocation(), 3, "failed")
    item = c.ConversationItem("id", "text", {})
    entry = _applied()
    view = _view(
        revision=1,
        items=(item,),
        effects=(effect,),
        closed_invocations=(completion,),
        applied=(entry,),
    )
    assert view.effects[0] is not effect
    assert view.effects[0].invocation is not effect.invocation
    assert view.effects[0].failure is not effect.failure
    assert view.closed_invocations[0] is not completion
    assert view.applied[0] is not entry
    object.__setattr__(item, "item_id", "changed")
    object.__setattr__(effect, "status", "pending")
    object.__setattr__(completion, "status", "completed")
    object.__setattr__(entry, "event_digest", "d" * 64)
    assert view.items[0].item_id == "id"
    assert view.effects[0].status == "failed"
    assert view.closed_invocations[0].status == "failed"
    assert view.applied[0].event_digest == "a" * 64


def test_nested_corrupted_identity_is_revalidated():
    invocation = _invocation()
    object.__setattr__(invocation, "start_sequence", True)
    with pytest.raises(TypeError, match="start_sequence"):
        c.ConversationRequest(("root", "agent"), invocation, {})


def test_effect_checkpoints_reject_zero_ordinal_and_list_items():
    with pytest.raises(ValueError, match="ordinal"):
        c.ConversationEffectRequested(_stamp(), "model", {}, ())
    with pytest.raises(TypeError, match="tuple"):
        c.ConversationEffectRequested(_stamp(1, 1), "model", {}, [])
    with pytest.raises(TypeError, match="tuple"):
        c.ConversationEffectObserved(_stamp(2, 2), 1, "pending", [], None)


def test_content_digest_excludes_ledger():
    from yian.memory.conversation import conversation_content_digest

    first = _view(revision=1, applied=(_applied(),))
    second = _view(revision=1, applied=(replace(_applied(), event_digest="d" * 64),))
    assert conversation_content_digest(first) == conversation_content_digest(second)
    assert conversation_content_digest(first) != conversation_content_digest(_view())
    assert len(conversation_content_digest(first)) == 64
    variants = [_view(compatibility={"number": value}) for value in (True, 1, 1.0)]
    assert len({conversation_content_digest(value) for value in variants}) == 3


def test_admission_advances_revision_once():
    from yian.memory.conversation import (
        advance_conversation,
        checkpoint_digest,
        conversation_content_digest,
    )

    before = _view()
    checkpoint = c.ConversationAdmitted(_stamp(), before)
    after = advance_conversation(before, checkpoint)
    assert before == _view()
    assert after.revision == 1
    assert after.items == () and after.effects == () and after.closed_invocations == ()
    assert after.applied == (
        c.ConversationApplied(
            '["run",1,0]',
            checkpoint_digest(checkpoint),
            conversation_content_digest(before),
            conversation_content_digest(after),
        ),
    )
    with pytest.raises(c.MemoryOperationError) as caught:
        advance_conversation(after, checkpoint)
    assert caught.value.code == "session.conflict"


def _admitted():
    from yian.memory.conversation import advance_conversation

    baseline = _view()
    return advance_conversation(baseline, c.ConversationAdmitted(_stamp(), baseline))


def test_request_appends_new_items_and_pending_effect():
    from yian.memory.conversation import advance_conversation

    before = _admitted()
    item = c.ConversationItem("user-1", "message", {"role": "user", "text": "Hi"})
    event = c.ConversationEffectRequested(
        _stamp(1, 1), "model", {"model": "fake"}, (item,)
    )
    after = advance_conversation(before, event)
    assert before.items == () and before.effects == ()
    assert after.items == (item,)
    assert after.effects == (
        c.ConversationEffectState(
            _invocation(), 1, "model", {"model": "fake"}, "pending", (), None
        ),
    )
    assert after.revision == 2
    second = c.ConversationEffectRequested(_stamp(2, 2), "tool", {"arguments": [1]}, ())
    appended = advance_conversation(after, second)
    assert appended.items == after.items
    assert appended.effects[0] == after.effects[0]
    assert appended.effects[1].effect_ordinal == 2


def _requested():
    from yian.memory.conversation import advance_conversation

    return advance_conversation(
        _admitted(), c.ConversationEffectRequested(_stamp(1, 1), "model", {}, ())
    )


def test_observation_changes_only_matching_effect():
    from yian.memory.conversation import advance_conversation

    before = advance_conversation(
        _requested(), c.ConversationEffectRequested(_stamp(2, 2), "tool", {}, ())
    )
    item = c.ConversationItem("assistant-1", "message", {"text": "Hi"})
    after = advance_conversation(
        before,
        c.ConversationEffectObserved(_stamp(3, 3), 1, "pending", (item,), None),
    )
    assert before.items == () and before.effects[0].observation_ordinals == ()
    assert after.items == (item,)
    assert after.effects[0].observation_ordinals == (3,)
    assert after.effects[0].status == "pending"
    assert after.effects[1] == before.effects[1]
    complete = advance_conversation(
        after,
        c.ConversationEffectObserved(_stamp(4, 4), 1, "completed", (), None),
    )
    assert complete.effects[0].status == "completed"
    assert complete.effects[0].observation_ordinals == (3, 4)
    assert complete.items == (item,)
    assert complete.effects[1] == before.effects[1]


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_close_preserves_unresolved_effects(status):
    from yian.memory.conversation import advance_conversation

    before = _requested()
    after = advance_conversation(before, c.ConversationClosed(_stamp(2, 2), status))
    assert after.effects == before.effects
    assert after.items == before.items
    assert after.closed_invocations == (
        c.ConversationCompletion(_invocation(), 2, status),
    )
    with pytest.raises(c.MemoryOperationError) as caught:
        advance_conversation(before, c.ConversationClosed(_stamp(2, 2), "completed"))
    assert caught.value.code == "session.conflict"
    assert before.closed_invocations == () and before.revision == 2


def test_transition_rejects_skipped_ordinal():
    from yian.memory.conversation import advance_conversation

    before = _admitted()
    event = c.ConversationEffectRequested(_stamp(2, 1), "model", {}, ())
    original = c._copy_conversation_view(before)
    with pytest.raises(c.MemoryOperationError) as caught:
        advance_conversation(before, event)
    assert caught.value.code == "session.conflict" and not caught.value.retryable
    assert before == original and event.stamp.ordinal == 2


@pytest.mark.parametrize("status", [None, "pending", "completed", "closed_failed"])
def test_admission_preserves_unresolved_or_unclosed_effects(status):
    from yian.memory.conversation import advance_conversation

    before = _requested()
    if status in ("pending", "completed"):
        before = advance_conversation(
            before,
            c.ConversationEffectObserved(_stamp(2, 2), 1, status, (), None),
        )
    elif status == "closed_failed":
        before = advance_conversation(
            before, c.ConversationClosed(_stamp(2, 2), "failed")
        )
    event = c.ConversationAdmitted(
        replace(
            _stamp(revision=before.revision),
            invocation=c.ConversationInvocation("next", 1),
        ),
        before,
    )
    original = c._copy_conversation_view(before)
    after = advance_conversation(before, event)
    assert after.effects == before.effects
    assert after.closed_invocations == before.closed_invocations
    assert after.revision == before.revision + 1
    assert before == original


def test_admission_can_follow_effect_free_unclosed_invocation():
    from yian.memory.conversation import advance_conversation

    before = _admitted()
    event = c.ConversationAdmitted(
        replace(_stamp(revision=1), invocation=c.ConversationInvocation("next", 1)),
        before,
    )
    after = advance_conversation(before, event)
    assert after.revision == 2 and after.closed_invocations == ()


def _complete_history():
    from yian.memory.conversation import advance_conversation

    baseline = _view(compatibility={"provider": "fake", "options": [True, 1, 1.0]})
    events = (
        c.ConversationAdmitted(_stamp(), baseline),
        c.ConversationEffectRequested(
            _stamp(1, 1),
            "model",
            {"arguments": [True, 1, 1.0]},
            (c.ConversationItem("user", "message", {"text": "héllo"}),),
        ),
        c.ConversationEffectObserved(
            _stamp(2, 2),
            1,
            "completed",
            (c.ConversationItem("assistant", "message", {"text": "hello"}),),
            None,
        ),
        c.ConversationClosed(_stamp(3, 3), "completed"),
    )
    views = [baseline]
    for event in events:
        views.append(advance_conversation(views[-1], event))
    return views, events


def test_conversation_codecs_round_trip_complete_history_and_detach():
    from yian.memory.conversation import (
        checkpoint_digest,
        conversation_checkpoint_from_json,
        conversation_checkpoint_to_json,
        conversation_view_from_json,
        conversation_view_to_json,
    )

    views, events = _complete_history()
    for view in views:
        encoded = conversation_view_to_json(view)
        decoded = conversation_view_from_json(encoded)
        assert decoded == view and decoded is not view
        assert conversation_view_to_json(decoded) == encoded
        encoded["compatibility"]["options"].clear()
        assert decoded.compatibility["options"] == [True, 1, 1.0]
    for event in events:
        encoded = conversation_checkpoint_to_json(event)
        decoded = conversation_checkpoint_from_json(encoded)
        assert decoded == event and decoded is not event
        wire = json.dumps(
            encoded,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        assert checkpoint_digest(event) == hashlib.sha256(wire.encode()).hexdigest()


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "tuple", "bool", "key"]
)
def test_view_codec_rejects_wrong_closed_wire_shape(damage):
    from yian.memory.conversation import (
        conversation_view_from_json,
        conversation_view_to_json,
    )

    wire = conversation_view_to_json(_view())
    if damage == "missing":
        del wire["reference"]
    elif damage == "extra":
        wire["extra"] = None
    elif damage == "tuple":
        wire["items"] = ()
    elif damage == "bool":
        wire["revision"] = False
    else:
        wire["key"] = None
    with pytest.raises((TypeError, ValueError)):
        conversation_view_from_json(wire)


@pytest.mark.parametrize(
    "damage", ["final", "adjacency", "anchor", "ordinal", "length"]
)
def test_view_codec_rejects_broken_digest_chain(damage):
    from yian.memory.conversation import (
        conversation_view_from_json,
        conversation_view_to_json,
    )

    views, _ = _complete_history()
    wire = conversation_view_to_json(views[-1])
    if damage == "final":
        wire["applied"][-1]["after_digest"] = "0" * 64
    elif damage == "adjacency":
        wire["applied"][1]["before_digest"] = "0" * 64
    elif damage == "anchor":
        wire["applied"][0]["before_digest"] = "0" * 64
    elif damage == "ordinal":
        wire["applied"][1]["operation_id"] = '["run",1,8]'
    else:
        wire["applied"].pop()
    with pytest.raises(ValueError):
        conversation_view_from_json(wire)
    with pytest.raises(ValueError):
        conversation_view_to_json(_view(revision=1, applied=(_applied(),)))


def _changed_view(view, **changes):
    values = {
        name: getattr(view, name)
        for name in (
            "namespace",
            "generation",
            "revision",
            "compatibility",
            "items",
            "effects",
            "closed_invocations",
            "applied",
        )
    }
    values.update(changes)
    return c.ConversationView(**values)


def _reseal(view):
    from yian.memory.conversation import conversation_content_digest

    return _changed_view(
        view,
        applied=view.applied[:-1]
        + (replace(view.applied[-1], after_digest=conversation_content_digest(view)),),
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing_effect",
        "foreign_effect",
        "overlap",
        "early_close",
        "pending_success",
        "foreign_close",
    ],
)
def test_view_codec_rejects_impossible_history_even_with_matching_final_digest(damage):
    from yian.memory.conversation import conversation_view_to_json

    views, _ = _complete_history()
    view = views[-1]
    effect = view.effects[0]
    if damage == "missing_effect":
        view = _changed_view(view, effects=())
    elif damage == "foreign_effect":
        effect = c.ConversationEffectState(
            c.ConversationInvocation("foreign", 1),
            1,
            "model",
            {},
            "completed",
            (2,),
            None,
        )
        view = _changed_view(view, effects=(effect,))
    elif damage == "overlap":
        effect = c.ConversationEffectState(
            _invocation(), 1, "model", {}, "completed", (3,), None
        )
        view = _changed_view(view, effects=(effect,))
    elif damage == "early_close":
        view = _changed_view(
            view,
            closed_invocations=(
                c.ConversationCompletion(_invocation(), 2, "completed"),
            ),
        )
    elif damage == "pending_success":
        effect = c.ConversationEffectState(
            _invocation(), 1, "model", {}, "pending", (2,), None
        )
        view = _changed_view(view, effects=(effect,))
    else:
        view = _changed_view(
            view,
            closed_invocations=(
                c.ConversationCompletion(
                    c.ConversationInvocation("foreign", 1), 3, "completed"
                ),
            ),
        )
    view = _reseal(view)
    with pytest.raises(ValueError):
        conversation_view_to_json(view)


@pytest.mark.parametrize(
    "scenario",
    [
        "before_admit",
        "terminal_effect",
        "closed",
        "missing_request",
        "foreign_invocation",
        "namespace",
        "generation",
        "revision",
        "duplicate_request_item",
        "duplicate_observed_item",
        "baseline_descriptor",
    ],
)
def test_invalid_transitions_preserve_input_evidence(scenario):
    from yian.memory.conversation import advance_conversation

    views, _ = _complete_history()
    before = views[1]
    event = c.ConversationEffectRequested(_stamp(1, 1), "model", {}, ())
    if scenario == "before_admit":
        before = views[0]
        event = c.ConversationEffectRequested(_stamp(1, 0), "model", {}, ())
    elif scenario == "terminal_effect":
        before = views[3]
        event = c.ConversationEffectObserved(_stamp(3, 3), 1, "completed", (), None)
    elif scenario == "closed":
        before = views[4]
        event = c.ConversationEffectRequested(_stamp(4, 4), "model", {}, ())
    elif scenario == "missing_request":
        before = advance_conversation(
            _requested(),
            c.ConversationEffectObserved(_stamp(2, 2), 1, "pending", (), None),
        )
        event = c.ConversationEffectObserved(_stamp(3, 3), 2, "completed", (), None)
    elif scenario in ("duplicate_request_item", "duplicate_observed_item"):
        before = views[2]
        if scenario == "duplicate_request_item":
            event = c.ConversationEffectRequested(
                _stamp(2, 2), "tool", {}, before.items
            )
        else:
            event = c.ConversationEffectObserved(
                _stamp(2, 2), 1, "pending", before.items, None
            )
    elif scenario == "baseline_descriptor":
        before = _view(compatibility={"value": True})
        event = c.ConversationAdmitted(_stamp(), _view(compatibility={"value": 1}))
    else:
        field, value = {
            "foreign_invocation": (
                "invocation",
                c.ConversationInvocation("foreign", 1),
            ),
            "namespace": ("namespace", "foreign"),
            "generation": ("generation", 1),
            "revision": ("expected_revision", 2),
        }[scenario]
        event = c.ConversationEffectRequested(
            replace(event.stamp, **{field: value}), "model", {}, ()
        )
    original = c._copy_conversation_view(before)
    original_event = c._copy_conversation_checkpoint(event)
    with pytest.raises(c.MemoryOperationError) as caught:
        advance_conversation(before, event)
    assert (
        caught.value.stage == "conversation" and caught.value.code == "session.conflict"
    )
    assert before == original and event == original_event


def test_later_invocation_reuses_local_ordinals_without_replacing_prior_effects():
    from yian.memory.conversation import (
        advance_conversation,
        conversation_view_from_json,
        conversation_view_to_json,
    )

    views, _ = _complete_history()
    first = views[-1]
    invocation = c.ConversationInvocation("second", 2)
    stamps = [
        replace(_stamp(ordinal, 4 + ordinal), invocation=invocation)
        for ordinal in range(4)
    ]
    events = (
        c.ConversationAdmitted(stamps[0], first),
        c.ConversationEffectRequested(stamps[1], "tool", {"tool": "fake"}, ()),
        c.ConversationEffectObserved(stamps[2], 1, "failed", (), _failure()),
        c.ConversationClosed(stamps[3], "failed"),
    )
    after = first
    for event in events:
        after = advance_conversation(after, event)
    assert [effect.effect_ordinal for effect in after.effects] == [1, 1]
    assert after.effects[0] == first.effects[0]
    assert after.effects[1].invocation == invocation
    assert after.items == first.items
    assert conversation_view_from_json(conversation_view_to_json(after)) == after


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize(
    "damage", ["extra", "missing", "discriminator", "stamp", "bool", "array"]
)
def test_checkpoint_codec_rejects_malformed_closed_wire(index, damage):
    from yian.memory.conversation import (
        conversation_checkpoint_from_json,
        conversation_checkpoint_to_json,
    )

    _, events = _complete_history()
    wire = conversation_checkpoint_to_json(events[index])
    if damage == "extra":
        wire["extra"] = None
    elif damage == "missing":
        del wire["stamp"]
    elif damage == "discriminator":
        wire["type"] = "unknown"
    elif damage == "stamp":
        wire["stamp"]["invocation"]["extra"] = None
    elif damage == "bool":
        wire["stamp"]["ordinal"] = True
    else:
        wire["stamp"]["namespace"] = ["namespace"]
    with pytest.raises((TypeError, ValueError)):
        conversation_checkpoint_from_json(wire)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "\ud800", (), object()])
def test_codecs_reject_non_json_nested_values(bad):
    from yian.memory.conversation import (
        conversation_view_from_json,
        conversation_view_to_json,
    )

    wire = conversation_view_to_json(_view())
    wire["compatibility"]["bad"] = bad
    with pytest.raises((TypeError, ValueError)):
        conversation_view_from_json(wire)


def test_codecs_reject_cycles_and_non_string_keys():
    from yian.memory.conversation import (
        conversation_view_from_json,
        conversation_view_to_json,
    )

    wire = conversation_view_to_json(_view())
    wire["compatibility"]["cycle"] = wire
    with pytest.raises(ValueError, match="circular"):
        conversation_view_from_json(wire)
    wire = conversation_view_to_json(_view())
    wire["compatibility"][1] = "invalid"
    with pytest.raises(TypeError):
        conversation_view_from_json(wire)


@pytest.mark.parametrize("depth", [120, 123, 124, 125, 126, 127, 128])
def test_wire_nesting_is_checked_before_fold_and_encoder_return(depth):
    from yian.memory.conversation import (
        advance_conversation,
        conversation_checkpoint_from_json,
        conversation_checkpoint_to_json,
        conversation_view_from_json,
        conversation_view_to_json,
    )

    nested = "value"
    for _ in range(depth):
        nested = {"nested": nested}
    item = c.ConversationItem("deep", "message", nested)
    event = c.ConversationEffectRequested(_stamp(1, 1), "model", {}, (item,))
    before = _admitted()
    try:
        wire = conversation_checkpoint_to_json(event)
    except ValueError:
        with pytest.raises(ValueError):
            advance_conversation(before, event)
        assert before.revision == 1 and not before.items
    else:
        assert conversation_checkpoint_from_json(wire) == event
        if depth == 125:
            # A request can fit while its future admission baseline cannot.
            with pytest.raises(ValueError, match="nests deeper than the memory limit"):
                advance_conversation(before, event)
            assert before.revision == 1 and not before.items
        else:
            after = advance_conversation(before, event)
            assert (
                conversation_view_from_json(conversation_view_to_json(after)) == after
            )


@pytest.mark.parametrize("depth", [124, 125])
def test_publishable_view_can_complete_and_become_the_next_admission_baseline(depth):
    from yian.memory.conversation import (
        advance_conversation,
        conversation_checkpoint_from_json,
        conversation_checkpoint_to_json,
        conversation_view_from_json,
        conversation_view_to_json,
    )

    nested = "value"
    for _ in range(depth):
        nested = {"nested": nested}
    item = c.ConversationItem("deep", "message", nested)
    request = c.ConversationEffectRequested(_stamp(1, 1), "model", {}, (item,))
    before = _admitted()
    original = c._copy_conversation_view(before)
    try:
        requested = advance_conversation(before, request)
    except ValueError:
        assert depth == 125
        assert before == original
        return
    observed = advance_conversation(
        requested,
        c.ConversationEffectObserved(_stamp(2, 2), 1, "completed", (), None),
    )
    closed = advance_conversation(
        observed, c.ConversationClosed(_stamp(3, 3), "completed")
    )
    stamp = replace(_stamp(revision=4), invocation=c.ConversationInvocation("next", 1))
    admission = c.ConversationAdmitted(stamp, closed)
    wire = conversation_checkpoint_to_json(admission)
    assert conversation_checkpoint_from_json(wire) == admission
    resumed = advance_conversation(closed, admission)
    assert conversation_view_from_json(conversation_view_to_json(resumed)) == resumed
    assert before == original


@pytest.mark.parametrize("direction", ["encode", "decode"])
def test_view_codec_reserves_the_next_admission_baseline_wrapper(direction):
    from yian.memory.conversation import (
        checkpoint_digest,
        conversation_view_from_json,
        conversation_view_to_json,
    )

    nested = "value"
    for _ in range(125):
        nested = {"nested": nested}
    item = c.ConversationItem("deep", "message", nested)
    request = c.ConversationEffectRequested(_stamp(1, 1), "model", {}, (item,))
    shallow = _requested()
    entries = shallow.applied[:-1] + (
        replace(shallow.applied[-1], event_digest=checkpoint_digest(request)),
    )
    candidate = _reseal(_changed_view(shallow, items=(item,), applied=entries))
    wire = conversation_view_to_json(shallow)
    wire["items"] = [
        {"item_id": item.item_id, "format": item.format, "payload": nested}
    ]
    wire["applied"][-1]["event_digest"] = candidate.applied[-1].event_digest
    wire["applied"][-1]["after_digest"] = candidate.applied[-1].after_digest
    with pytest.raises(ValueError, match="nests deeper than the memory limit"):
        if direction == "encode":
            conversation_view_to_json(candidate)
        else:
            conversation_view_from_json(wire)


def test_checkpoint_digest_includes_complete_admission_ledger():
    from yian.memory.conversation import checkpoint_digest

    views, _ = _complete_history()
    baseline = views[-1]
    changed = _changed_view(
        baseline,
        applied=(replace(baseline.applied[0], event_digest="0" * 64),)
        + baseline.applied[1:],
    )
    stamp = replace(_stamp(revision=4), invocation=c.ConversationInvocation("next", 1))
    assert checkpoint_digest(
        c.ConversationAdmitted(stamp, baseline)
    ) != checkpoint_digest(c.ConversationAdmitted(stamp, changed))


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate_item",
        "duplicate_operation",
        "bad_status",
        "failure_type",
        "key_extra",
        "float_sequence",
    ],
)
def test_view_codec_rejects_invalid_nested_values(damage):
    from yian.memory.conversation import (
        conversation_view_from_json,
        conversation_view_to_json,
    )

    views, _ = _complete_history()
    wire = conversation_view_to_json(views[-1])
    if damage == "duplicate_item":
        wire["items"].append(wire["items"][0])
    elif damage == "duplicate_operation":
        wire["applied"][-1]["operation_id"] = wire["applied"][0]["operation_id"]
    elif damage == "bad_status":
        wire["effects"][0]["status"] = "unknown"
    elif damage == "failure_type":
        wire["effects"][0]["status"] = "failed"
        wire["effects"][0]["failure"] = {
            "stage": "effect",
            "code": "fake.failed",
            "message": "Failed",
            "retryable": 0,
        }
    elif damage == "key_extra":
        wire["key"] = {}
    else:
        wire["effects"][0]["invocation"]["start_sequence"] = 1.0
    original = json.dumps(wire, sort_keys=True)
    with pytest.raises((TypeError, ValueError)):
        conversation_view_from_json(wire)
    assert json.dumps(wire, sort_keys=True) == original


def test_view_codec_accepts_admission_after_unclosed_effectful_invocation():
    from yian.memory.conversation import (
        advance_conversation,
        conversation_view_from_json,
        conversation_view_to_json,
    )

    views, _ = _complete_history()
    before = views[3]
    admission = c.ConversationAdmitted(
        replace(
            _stamp(revision=before.revision),
            invocation=c.ConversationInvocation("next", 1),
        ),
        before,
    )
    after = advance_conversation(before, admission)
    assert conversation_view_from_json(conversation_view_to_json(after)) == after


def test_fold_and_codecs_revalidate_tampered_exact_carriers():
    from yian.memory.conversation import (
        advance_conversation,
        conversation_checkpoint_to_json,
        conversation_view_to_json,
    )

    before = _admitted()
    checkpoint = c.ConversationEffectRequested(_stamp(1, 1), "model", {}, ())
    object.__setattr__(checkpoint.stamp.invocation, "start_sequence", True)
    with pytest.raises(TypeError):
        advance_conversation(before, checkpoint)
    with pytest.raises(TypeError):
        conversation_checkpoint_to_json(checkpoint)
    assert before == _admitted()
    object.__setattr__(before, "generation", True)
    with pytest.raises(TypeError):
        conversation_view_to_json(before)


def test_fold_does_not_mask_unexpected_host_fault(monkeypatch):
    import yian.memory.conversation as conversation

    before = _admitted()
    checkpoint = c.ConversationEffectRequested(_stamp(1, 1), "model", {}, ())
    fault = RuntimeError("host digest fault")

    def fail(_value):
        raise fault

    monkeypatch.setattr(conversation.hashlib, "sha256", fail)
    with pytest.raises(RuntimeError) as caught:
        conversation.advance_conversation(before, checkpoint)
    assert caught.value is fault
    assert before.revision == 1 and before.effects == ()


@pytest.mark.parametrize("event_index", [0, 1, 2])
def test_checkpoint_codec_rejects_retired_native_state_without_rewriting(event_index):
    from copy import deepcopy

    from yian.memory.conversation import (
        conversation_checkpoint_from_json,
        conversation_checkpoint_to_json,
    )

    _, events = _complete_history()
    wire = conversation_checkpoint_to_json(events[event_index])
    if event_index == 0:
        wire["baseline"]["reference"] = {"provider": "codex", "session_id": "thread"}
    elif event_index == 1:
        wire["kind"] = "native_session"
    else:
        wire["reference"] = {"provider": "codex", "session_id": "thread"}
    original = deepcopy(wire)
    with pytest.raises(
        c.MemoryOperationError, match="use fresh SessionMemory"
    ) as caught:
        conversation_checkpoint_from_json(wire)
    assert caught.value.code == "session.unsupported_history"
    assert wire == original


def test_memory_context_request_detaches_every_value_boundary():
    inputs = {"nested": ["original"]}
    compatibility = {"provider": "test", "format": {"version": 1}}
    recall = c.RecallContext("system", "prompt", inputs, ("root", "agent"))
    invocation = c.ConversationInvocation("run", 3)
    request = c.MemoryContextRequest(recall, invocation, compatibility)
    inputs["nested"].append("outside")
    request.recall.inputs["nested"].append("returned")
    compatibility["format"]["version"] = 2
    request.compatibility["format"]["version"] = 3
    object.__setattr__(recall, "prompt", "mutated")
    object.__setattr__(invocation, "run_id", "mutated")
    assert request.recall.inputs == {"nested": ["original"]}
    assert request.recall.prompt == "prompt"
    assert request.compatibility == {"provider": "test", "format": {"version": 1}}
    assert request.invocation == c.ConversationInvocation("run", 3)
    with pytest.raises(FrozenInstanceError):
        request.recall = recall
    assert c.MemoryContextRequest(request.recall, None, {}).invocation is None


@pytest.mark.parametrize("field,value", [
    ("recall", None), ("recall", {}),
    ("invocation", {}), ("invocation", "run"),
    ("compatibility", []), ("compatibility", {"value": float("nan")}),
])
def test_memory_context_request_rejects_invalid_carriers(field, value):
    args = {
        "recall": c.RecallContext("", "", {}, ("root", "agent")),
        "invocation": c.ConversationInvocation("run", 1),
        "compatibility": {},
    }
    args[field] = value
    with pytest.raises((TypeError, ValueError)):
        c.MemoryContextRequest(**args)


def test_memory_content_validates_explicit_transcript_sources():
    from types import SimpleNamespace

    source = SimpleNamespace(open_turn=lambda request, **kwargs: None)
    content = c.MemoryContent("notes", source)
    assert content.text == "notes"
    assert content.transcript is source
    assert c.MemoryContent() == c.MemoryContent("", None)
    for text in (None, 7, "\ud800"):
        with pytest.raises((TypeError, ValueError)):
            c.MemoryContent(text)
    for source in (object(), SimpleNamespace(open_turn=None)):
        with pytest.raises(TypeError, match="open_turn"):
            c.MemoryContent(transcript=source)
    with pytest.raises(FrozenInstanceError):
        content.text = "changed"


def test_memory_context_carriers_are_contracts_only_exports():
    import yian.memory as memory

    for name in ("MemoryContent", "MemoryContext", "MemoryContextRequest", "TranscriptSource"):
        assert name in memory.__all__
        assert getattr(memory, name) is getattr(c, name)
    assert "ConversationFrame" not in vars(memory)
    assert "ConversationFrame" not in vars(c)


def test_conversation_carriers_have_no_owner_key():
    import yian.memory as memory

    assert not hasattr(c, "ConversationKey")
    assert "ConversationKey" not in memory.__all__
    assert [field.name for field in fields(c.ConversationStamp)] == [
        "invocation",
        "namespace",
        "generation",
        "expected_revision",
        "ordinal",
    ]
    assert "key" not in {field.name for field in fields(c.ConversationView)}
