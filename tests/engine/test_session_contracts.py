"""Detached source-neutral conversation identities and optional capabilities."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import inspect
from typing import get_args, get_type_hints

import pytest

from agent_composer.memory import contracts as c
from tests.engine._memory_fakes import _Memory


def _frame_ref(owner=c.MemoryOwner.FLOW):
    return c.MemoryFrameRef("root", ("root", "child"), owner)


def test_legacy_memory_protocol_is_unchanged():
    memory = _Memory()
    session = memory.open_run()
    frame = session.open_frame(c.MemoryFrameSpec(_frame_ref(), "session", {}))
    assert isinstance(memory, c.Memory)
    assert isinstance(frame, c.MemoryFrame)
    assert {name for name in vars(c.MemorySession) if not name.startswith("_")} == {
        "snapshot",
        "close",
        "open_frame",
        "commit",
    }
    assert "__enter__" in vars(c.MemorySession)
    assert "__exit__" in vars(c.MemorySession)
    assert frame.recall() == ""
    session.close()


def test_flow_conversation_key_accepts_descendant():
    ref = _frame_ref()
    key = c.ConversationKey(ref, ("root", "child", "agent"))
    assert key.frame_ref == ref
    assert key.frame_ref is not ref
    assert key.consumer_static_path == ("root", "child", "agent")
    with pytest.raises(FrozenInstanceError):
        key.consumer_static_path = ("changed",)


@pytest.mark.parametrize(
    "path", [("other", "agent"), ("root", "other"), ("root", "child")]
)
def test_conversation_key_rejects_foreign_root(path):
    with pytest.raises(ValueError):
        c.ConversationKey(_frame_ref(), path)
    agent = _frame_ref(c.MemoryOwner.AGENT)
    assert c.ConversationKey(agent, agent.static_path).frame_ref == agent
    with pytest.raises(ValueError):
        c.ConversationKey(agent, agent.static_path + ("descendant",))


@pytest.mark.parametrize(
    "path", [["root", "child", "agent"], (), ("root", ""), ("root", "\ud800")]
)
def test_conversation_key_rejects_invalid_path(path):
    with pytest.raises((TypeError, ValueError)):
        c.ConversationKey(_frame_ref(), path)


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


def _key():
    return c.ConversationKey(_frame_ref(), ("root", "child", "agent"))


def _invocation():
    return c.ConversationInvocation("run", 1)


def _stamp(ordinal=0, revision=0):
    return c.ConversationStamp(
        "session", _key(), _invocation(), "namespace", 0, revision, ordinal
    )


def test_item_payload_is_detached():
    payload = {"text": ["hello"]}
    item = c.ConversationItem("message-1", "api_message", payload)
    payload["text"].append("caller mutation")
    item.payload["text"].append("reader mutation")
    assert item.payload == {"text": ["hello"]}
    assert c.ConversationItem("x", "y", {"value": True}) != c.ConversationItem(
        "x", "y", {"value": 1}
    )


@pytest.mark.parametrize(
    "provider,session_id",
    [("", "id"), ("fake", ""), ("\ud800", "id"), (None, "id"), ("fake", 1)],
)
def test_reference_requires_nonempty_strings(provider, session_id):
    with pytest.raises((TypeError, ValueError)):
        c.ConversationReference(provider, session_id)
    assert c.ConversationReference("fake", "id").session_id == "id"


@pytest.mark.parametrize("field", ["generation", "expected_revision", "ordinal"])
@pytest.mark.parametrize("value", [True, -1, 1.0, "1"])
def test_stamp_requires_nonnegative_exact_counters(field, value):
    arguments = dict(
        policy_name="session",
        key=_key(),
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


@pytest.mark.parametrize("kind", ["model", "tool", "native_session"])
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
        key=_key(),
        compatibility={"provider": "fake"},
        items=(),
        reference=None,
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
    key = _key()
    view = _view(key=key, compatibility=descriptor)
    descriptor["options"].clear()
    view.compatibility["options"].clear()
    assert view.compatibility == {"options": [True]}
    assert view.key == key and view.key is not key
    assert (view.namespace, view.generation, view.revision) == ("namespace", 0, 0)
    assert (
        view.items,
        view.reference,
        view.effects,
        view.closed_invocations,
        view.applied,
    ) == ((), None, (), (), ())
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
        _view(reference=c.ConversationReference("fake", "session"))


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
        c.ConversationEffectObserved(_stamp(2, 2), 1, status, (), None, failure)
    with pytest.raises(ValueError):
        c.ConversationEffectState(_invocation(), 1, "model", {}, status, (2,), failure)


@pytest.mark.parametrize("status", ["pending", "completed", "failed", "interrupted"])
def test_observed_preserves_detached_payload_and_actual_status(status):
    failure = _failure() if status in ("failed", "interrupted") else None
    item = c.ConversationItem("id", "text", {})
    reference = c.ConversationReference("fake", "thread")
    event = c.ConversationEffectObserved(
        _stamp(2, 2), 1, status, (item,), reference, failure
    )
    assert event.status == status
    assert event.reference == reference and event.reference is not reference
    assert event.items == (item,) and event.items[0] is not item
    if failure is not None:
        assert event.failure == failure and event.failure is not failure


@pytest.mark.parametrize("ordinal", [True, 0, 2, 3])
def test_observed_requires_a_prior_effect(ordinal):
    with pytest.raises((TypeError, ValueError)):
        c.ConversationEffectObserved(_stamp(2, 2), ordinal, "completed", (), None, None)


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted"])
def test_closed_accepts_only_turn_dispositions(status):
    assert c.ConversationClosed(_stamp(3, 3), status).status == status
    with pytest.raises(ValueError):
        c.ConversationClosed(_stamp(3, 3), "pending")
    with pytest.raises(ValueError):
        c.ConversationClosed(_stamp(), status)


def test_authority_detaches_baseline_and_checkpoints():
    baseline = _view()
    events = (
        c.ConversationAdmitted(_stamp(), baseline),
        c.ConversationClosed(_stamp(1, 1), "completed"),
    )
    authority = c.ConversationAuthority(baseline, events)
    assert authority.baseline == baseline and authority.baseline is not baseline
    assert authority.checkpoints == events
    assert all(
        copy is not original for copy, original in zip(authority.checkpoints, events)
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "empty",
        "list",
        "wrong_first",
        "gap",
        "revision",
        "post_close",
        "second_admit",
        "baseline",
        "invocation",
        "policy",
    ],
)
def test_authority_rejects_noncontiguous_or_mismatched_segment(scenario):
    baseline = _view()
    first = c.ConversationAdmitted(_stamp(), baseline)
    close = c.ConversationClosed(_stamp(1, 1), "completed")
    events = (first, close)
    if scenario == "empty":
        events = ()
    elif scenario == "list":
        events = list(events)
    elif scenario == "wrong_first":
        events = (close,)
    elif scenario == "gap":
        events = (first, c.ConversationClosed(_stamp(2, 1), "completed"))
    elif scenario == "revision":
        events = (first, c.ConversationClosed(_stamp(1, 2), "completed"))
    elif scenario == "post_close":
        events += (c.ConversationClosed(_stamp(2, 2), "completed"),)
    elif scenario == "second_admit":
        events += (first,)
    elif scenario == "baseline":
        baseline = _view(compatibility={"provider": "different"})
    elif scenario == "invocation":
        events = (
            first,
            replace(
                close,
                stamp=replace(
                    close.stamp, invocation=c.ConversationInvocation("other", 1)
                ),
            ),
        )
    elif scenario == "policy":
        events = (
            first,
            replace(close, stamp=replace(close.stamp, policy_name="other")),
        )
    with pytest.raises((TypeError, ValueError)):
        c.ConversationAuthority(baseline, events)


def test_conversation_capabilities_are_optional_and_exact():
    memory = _Memory()
    session = memory.open_run()
    frame = session.open_frame(c.MemoryFrameSpec(_frame_ref(), "session", {}))
    assert not isinstance(session, c.ConversationSession)
    assert not isinstance(frame, c.ConversationFrame)
    assert set(get_args(c.ConversationAcceptance)) == {
        "unaccepted",
        "accepted",
        "uncertain",
    }
    assert set(get_args(c.ConversationCheckpoint)) == {
        c.ConversationAdmitted,
        c.ConversationEffectRequested,
        c.ConversationEffectObserved,
        c.ConversationClosed,
    }
    for protocol in (c.ConversationFrame, c.ConversationSession, c.ConversationTurn):
        for member in vars(protocol).values():
            if inspect.isfunction(member):
                get_type_hints(member)
    prepare = inspect.signature(c.ConversationSession.prepare_checkpoint)
    assert prepare.parameters["before"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(c.ConversationFrame.open_turn)
        .parameters["check_stopped"]
        .kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(c.ConversationTurn.request).parameters["items"].default == ()
    )


def test_conversation_contracts_are_exported_without_concrete_backend():
    import agent_composer.memory as memory

    names = (
        "ConversationKey",
        "ConversationInvocation",
        "ConversationRequest",
        "ConversationItem",
        "ConversationReference",
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
        "ConversationAuthority",
        "CheckpointSink",
        "CheckStopped",
        "PreparedConversationApply",
        "ConversationAcceptance",
        "ConversationFrame",
        "ConversationSession",
        "ConversationTurn",
    )
    for name in names:
        assert name in memory.__all__
        assert getattr(memory, name) is getattr(c, name)
    assert "SessionMemory" not in memory.__all__
    assert "SessionMemory" not in vars(memory)


@pytest.mark.parametrize("value", ["", "\ud800", 1, None])
@pytest.mark.parametrize(
    "factory",
    [
        lambda value: c.ConversationInvocation(value, 1),
        lambda value: c.ConversationItem(value, "text", {}),
        lambda value: c.ConversationItem("id", value, {}),
        lambda value: replace(_stamp(), policy_name=value),
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
        c.ConversationAuthority(
            ViewSubclass(
                baseline.namespace, 0, 0, baseline.key, {}, (), None, (), (), ()
            ),
            (),
        )


def test_nested_view_carriers_are_revalidated_and_detached():
    effect = c.ConversationEffectState(
        _invocation(), 1, "model", {}, "failed", (2,), _failure()
    )
    completion = c.ConversationCompletion(_invocation(), 3, "failed")
    item = c.ConversationItem("id", "text", {})
    reference = c.ConversationReference("fake", "thread")
    entry = _applied()
    view = _view(
        revision=1,
        items=(item,),
        reference=reference,
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
    object.__setattr__(reference, "session_id", "changed")
    object.__setattr__(entry, "event_digest", "d" * 64)
    assert view.items[0].item_id == "id"
    assert view.effects[0].status == "failed"
    assert view.closed_invocations[0].status == "failed"
    assert view.reference.session_id == "thread"
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
        c.ConversationEffectObserved(_stamp(2, 2), 1, "pending", [], None, None)
