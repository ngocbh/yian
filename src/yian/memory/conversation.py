"""Pure conversation transitions, exact JSON codecs, and content evidence.

This leaf owns no backend state or external effects. Both live application and
recovery can fold the same closed checkpoint vocabulary through these rules.
"""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json

import yian.memory.contracts as c


def _canonical(value: c.JsonValue) -> str:
    """Encode exact JSON without normalizing distinct scalar types."""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _invocation_json(value: c.ConversationInvocation) -> c.JsonObject:
    return {"run_id": value.run_id, "start_sequence": value.start_sequence}


def _item_json(value: c.ConversationItem) -> c.JsonObject:
    return {"item_id": value.item_id, "format": value.format, "payload": value.payload}


def _failure_json(value: c.MemoryOperationFailure | None) -> c.JsonObject | None:
    if value is None:
        return None
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _effect_json(value: c.ConversationEffectState) -> c.JsonObject:
    return {
        "invocation": _invocation_json(value.invocation),
        "effect_ordinal": value.effect_ordinal,
        "kind": value.kind,
        "request": value.request,
        "status": value.status,
        "observation_ordinals": list(value.observation_ordinals),
        "failure": _failure_json(value.failure),
    }


def _completion_json(value: c.ConversationCompletion) -> c.JsonObject:
    return {
        "invocation": _invocation_json(value.invocation),
        "closed_ordinal": value.closed_ordinal,
        "status": value.status,
    }


def _view_json(value: c.ConversationView, *, ledger: bool = True) -> c.JsonObject:
    result = {
        "namespace": value.namespace,
        "generation": value.generation,
        "revision": value.revision,
        "compatibility": value.compatibility,
        "items": [_item_json(item) for item in value.items],
        # Retain this inert field so message-only snapshot digests stay valid.
        "reference": None,
        "effects": [_effect_json(effect) for effect in value.effects],
        "closed_invocations": [
            _completion_json(completion) for completion in value.closed_invocations
        ],
    }
    if ledger:
        result["applied"] = [
            {field.name: getattr(entry, field.name) for field in fields(entry)}
            for entry in value.applied
        ]
    return result


def _wire(value: object) -> c.JsonObject:
    """Detach the complete wire, including its structural nesting overhead."""
    return c._thaw_json_object(c._freeze_json_object(value, "conversation wire"))


def _view_wire(value: object) -> c.JsonObject:
    """Reserve the wrapper needed when this view becomes an admission baseline."""
    if type(value) is not dict:
        raise TypeError("conversation view wire must be an exact JSON object")
    return _wire({"baseline": value})["baseline"]


def _object(value: object, names: str) -> c.JsonObject:
    if type(value) is not dict:
        raise TypeError("conversation wire record must be an exact JSON object")
    if set(value) != set(names.split()):
        raise ValueError(f"conversation wire record requires exactly: {names}")
    return value


def _array(value: object) -> list:
    if type(value) is not list:
        raise TypeError("conversation wire array must be an exact list")
    return value


def _invocation_from_json(value: object) -> c.ConversationInvocation:
    return c.ConversationInvocation(**_object(value, "run_id start_sequence"))


def _items_from_json(value: object) -> tuple[c.ConversationItem, ...]:
    return tuple(
        c.ConversationItem(**_object(item, "item_id format payload"))
        for item in _array(value)
    )


def _reject_native_history(data: c.JsonObject) -> None:
    """Reject retired wire state without adopting or rewriting its history."""
    if data.get("reference") is not None or data.get("kind") == "native_session":
        raise c.MemoryOperationError(
            "snapshot",
            "session.unsupported_history",
            "Native session history is unsupported; use fresh SessionMemory.",
            False,
        )


def _failure_from_json(value: object) -> c.MemoryOperationFailure | None:
    if value is None:
        return None
    return c.MemoryOperationFailure(**_object(value, "stage code message retryable"))


def _effect_from_json(value: object) -> c.ConversationEffectState:
    data = _object(
        value,
        "invocation effect_ordinal kind request status observation_ordinals failure",
    )
    _reject_native_history(data)
    return c.ConversationEffectState(
        _invocation_from_json(data["invocation"]),
        data["effect_ordinal"],
        data["kind"],
        data["request"],
        data["status"],
        tuple(_array(data["observation_ordinals"])),
        _failure_from_json(data["failure"]),
    )


def _completion_from_json(value: object) -> c.ConversationCompletion:
    data = _object(value, "invocation closed_ordinal status")
    return c.ConversationCompletion(
        _invocation_from_json(data["invocation"]),
        data["closed_ordinal"],
        data["status"],
    )


def conversation_view_to_json(view: c.ConversationView) -> c.JsonObject:
    """Encode a fully validated conversation as detached exact JSON.

    Args:
        view (`ConversationView`): Exact view with complete valid ledger evidence.

    Returns:
        `JsonObject`: Closed wire object with detached payloads and enough nesting
            capacity to be used as a later admission's baseline.

    Raises:
        TypeError: A carrier or nested value has the wrong exact type.
        ValueError: History, digest evidence, or wire nesting is invalid.

    Example:
        `conversation_view_from_json(conversation_view_to_json(view)) == view`.
    """
    value = c._copy_conversation_view(view)
    _validate_view(value)
    return _view_wire(_view_json(value))


def conversation_view_from_json(value: c.JsonObject) -> c.ConversationView:
    """Decode exact JSON and validate all retained conversation evidence.

    Args:
        value (`JsonObject`): Complete closed view wire; arrays must be lists and
            nesting must also fit within a later admission's baseline wrapper.

    Returns:
        `ConversationView`: Fully detached immutable conversation state.

    Raises:
        TypeError: A wire value has the wrong exact type.
        ValueError: Fields, identity, history, digest evidence, or nesting is invalid.
        MemoryOperationError: Retired provider-native history requires fresh memory.

    Example:
        `view = conversation_view_from_json(saved_view)` validates before use.
    """
    data = _object(
        _view_wire(value),
        (
            "namespace generation revision compatibility items reference effects "
            "closed_invocations applied"
        ),
    )
    _reject_native_history(data)
    view = c.ConversationView(
        data["namespace"],
        data["generation"],
        data["revision"],
        data["compatibility"],
        _items_from_json(data["items"]),
        tuple(_effect_from_json(item) for item in _array(data["effects"])),
        tuple(
            _completion_from_json(item) for item in _array(data["closed_invocations"])
        ),
        tuple(
            c.ConversationApplied(
                **_object(entry, "operation_id event_digest before_digest after_digest")
            )
            for entry in _array(data["applied"])
        ),
    )
    _validate_view(view)
    return view


def conversation_content_digest(view: c.ConversationView) -> str:
    """Hash complete conversation content except its application ledger.

    Args:
        view (`ConversationView`): Exact shaped carrier, including its revision.
            Ledger shape is checked, but its digest evidence is not interpreted.

    Returns:
        `str`: Lowercase SHA-256 of canonical JSON without the `applied` field.

    Raises:
        TypeError: A carrier or nested value has the wrong exact type.
        ValueError: A carrier or JSON value has invalid shape.

    Example:
        `conversation_content_digest(view)` remains unchanged when only a validly
        shaped ledger digest changes.
    """
    detached = c._copy_conversation_view(view)
    return hashlib.sha256(
        _canonical(_view_json(detached, ledger=False)).encode()
    ).hexdigest()


def _stamp_json(value: c.ConversationStamp) -> c.JsonObject:
    return {
        "invocation": _invocation_json(value.invocation),
        "namespace": value.namespace,
        "generation": value.generation,
        "expected_revision": value.expected_revision,
        "ordinal": value.ordinal,
    }


def _checkpoint_json(value: c.ConversationCheckpoint) -> c.JsonObject:
    if type(value) is c.ConversationAdmitted:
        _validate_view(value.baseline)
        return {
            "type": "admitted",
            "stamp": _stamp_json(value.stamp),
            "baseline": _view_json(value.baseline),
        }
    if type(value) is c.ConversationEffectRequested:
        return {
            "type": "effect_requested",
            "stamp": _stamp_json(value.stamp),
            "kind": value.kind,
            "payload": value.payload,
            "items": [_item_json(item) for item in value.items],
        }
    if type(value) is c.ConversationEffectObserved:
        return {
            "type": "effect_observed",
            "stamp": _stamp_json(value.stamp),
            "effect_ordinal": value.effect_ordinal,
            "status": value.status,
            "items": [_item_json(item) for item in value.items],
            # The original checkpoint digest includes this inert wire field.
            "reference": None,
            "failure": _failure_json(value.failure),
        }
    if type(value) is c.ConversationClosed:
        return {
            "type": "closed",
            "stamp": _stamp_json(value.stamp),
            "status": value.status,
        }
    raise TypeError("unsupported conversation checkpoint")


def _stamp_from_json(value: object) -> c.ConversationStamp:
    data = _object(
        value,
        "invocation namespace generation expected_revision ordinal",
    )
    return c.ConversationStamp(
        _invocation_from_json(data["invocation"]),
        data["namespace"],
        data["generation"],
        data["expected_revision"],
        data["ordinal"],
    )


def conversation_checkpoint_to_json(
    checkpoint: c.ConversationCheckpoint,
) -> c.JsonObject:
    """Encode a checkpoint with its closed discriminator and complete payload.

    Args:
        checkpoint (`ConversationCheckpoint`): One exact checkpoint carrier.

    Returns:
        `JsonObject`: Detached closed wire; admission includes the complete baseline.

    Raises:
        TypeError: A carrier or nested value has the wrong exact type.
        ValueError: Shape, baseline evidence, or wire nesting is invalid.

    Example:
        `wire = conversation_checkpoint_to_json(admission)` retains baseline evidence.
    """
    return _wire(_checkpoint_json(c._copy_conversation_checkpoint(checkpoint)))


def conversation_checkpoint_from_json(value: c.JsonObject) -> c.ConversationCheckpoint:
    """Decode one strict member of the closed conversation checkpoint vocabulary.

    Args:
        value (`JsonObject`): Exact checkpoint wire with a supported `type` field.

    Returns:
        `ConversationCheckpoint`: Detached immutable checkpoint carrier.

    Raises:
        TypeError: A wire value has the wrong exact type.
        ValueError: Fields, discriminator, shape, baseline evidence, or nesting is invalid.
        MemoryOperationError: Retired provider-native history requires fresh memory.

    Example:
        `checkpoint = conversation_checkpoint_from_json(saved_checkpoint)`.
    """
    data = _wire(value)
    kind = data.get("type")
    c._require_string(kind, "checkpoint type")
    names = {
        "admitted": "type stamp baseline",
        "effect_requested": "type stamp kind payload items",
        "effect_observed": "type stamp effect_ordinal status items reference failure",
        "closed": "type stamp status",
    }
    if kind not in names:
        raise ValueError("unsupported conversation checkpoint type")
    _object(data, names[kind])
    _reject_native_history(data)
    stamp = _stamp_from_json(data["stamp"])
    if kind == "admitted":
        return c.ConversationAdmitted(
            stamp, conversation_view_from_json(data["baseline"])
        )
    if kind == "effect_requested":
        return c.ConversationEffectRequested(
            stamp, data["kind"], data["payload"], _items_from_json(data["items"])
        )
    if kind == "effect_observed":
        return c.ConversationEffectObserved(
            stamp,
            data["effect_ordinal"],
            data["status"],
            _items_from_json(data["items"]),
            _failure_from_json(data["failure"]),
        )
    return c.ConversationClosed(stamp, data["status"])


def checkpoint_digest(checkpoint: c.ConversationCheckpoint) -> str:
    """Hash one complete checkpoint, including an admission's complete baseline.

    Args:
        checkpoint (`ConversationCheckpoint`): One exact closed checkpoint carrier.

    Returns:
        `str`: Lowercase SHA-256 of the complete canonical checkpoint JSON.

    Raises:
        TypeError: A carrier or nested value has the wrong exact type.
        ValueError: Shape or retained baseline evidence is invalid.

    Example:
        `checkpoint_digest(admission)` includes the baseline's application ledger.
    """
    value = conversation_checkpoint_to_json(checkpoint)
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _zero_view(view: c.ConversationView) -> c.ConversationView:
    return c.ConversationView(
        view.namespace,
        view.generation,
        0,
        view.compatibility,
        (),
        (),
        (),
        (),
    )


def _operation(entry: c.ConversationApplied) -> tuple[c.ConversationInvocation, int]:
    run_id, start_sequence, ordinal = json.loads(entry.operation_id)
    return c.ConversationInvocation(run_id, start_sequence), ordinal


def _validate_view(view: c.ConversationView) -> None:
    """Prove the supplied ledger's ordering, anchor, adjacency, and final content."""
    expected_digest = conversation_content_digest(_zero_view(view))
    previous_invocation = None
    previous_ordinal = -1
    seen = set()
    operations = {}
    last_ordinals = {}
    for index, entry in enumerate(view.applied):
        invocation, ordinal = _operation(entry)
        if invocation != previous_invocation:
            if invocation in seen or ordinal != 0:
                raise ValueError(
                    "ledger must have contiguous admission-led invocations"
                )
            seen.add(invocation)
        elif ordinal != previous_ordinal + 1:
            raise ValueError("ledger operation ordinals must be contiguous")
        if entry.before_digest != expected_digest:
            raise ValueError("ledger digest chain does not match its predecessor")
        expected_digest = entry.after_digest
        previous_invocation, previous_ordinal = invocation, ordinal
        operations[invocation, ordinal] = index
        last_ordinals[invocation] = ordinal
    if expected_digest != conversation_content_digest(view):
        raise ValueError("ledger final digest does not match conversation content")
    _validate_history(view, operations, last_ordinals)


def _validate_history(
    view: c.ConversationView,
    operations: dict[tuple[c.ConversationInvocation, int], int],
    last_ordinals: dict[c.ConversationInvocation, int],
) -> None:
    """Account for every non-admission operation using retained actual evidence."""
    claimed = set()

    def claim(invocation: c.ConversationInvocation, ordinal: int) -> int:
        identity = invocation, ordinal
        if identity not in operations or identity in claimed:
            raise ValueError("history must name unique accepted operation ordinals")
        claimed.add(identity)
        return operations[identity]

    effects_by_invocation: dict[
        c.ConversationInvocation, list[c.ConversationEffectState]
    ] = {}
    previous = -1
    for effect in view.effects:
        position = claim(effect.invocation, effect.effect_ordinal)
        if position <= previous:
            raise ValueError("effects must retain accepted request order")
        previous = position
        effects_by_invocation.setdefault(effect.invocation, []).append(effect)
        for ordinal in effect.observation_ordinals:
            claim(effect.invocation, ordinal)

    previous = -1
    for completion in view.closed_invocations:
        position = claim(completion.invocation, completion.closed_ordinal)
        if (
            position <= previous
            or last_ordinals[completion.invocation] != completion.closed_ordinal
        ):
            raise ValueError("closure must be the invocation's last accepted operation")
        previous = position
        if completion.status == "completed" and any(
            effect.status == "pending"
            for effect in effects_by_invocation.get(completion.invocation, ())
        ):
            raise ValueError("successful closure cannot retain pending effects")

    expected = {identity for identity in operations if identity[1] > 0}
    if claimed != expected:
        raise ValueError(
            "every positive operation must have retained effect or closure evidence"
        )
    if view.items and not view.effects:
        raise ValueError("retained items require an accepted effect request")


def _conflict(message: str) -> c.MemoryOperationError:
    return c.MemoryOperationError("conversation", "session.conflict", message, False)


def advance_conversation(
    before: c.ConversationView, checkpoint: c.ConversationCheckpoint
) -> c.ConversationView:
    """Apply exactly one new checkpoint without mutating either input.

    Args:
        before (`ConversationView`): Complete validated predecessor content and ledger.
        checkpoint (`ConversationCheckpoint`): New checkpoint naming that predecessor.

    Returns:
        `ConversationView`: Detached successor with one new revision and ledger entry.

    Raises:
        TypeError: An input contains a wrong exact carrier or JSON type.
        ValueError: An input's shape or retained history evidence is invalid.
        MemoryOperationError: The operation conflicts with its predecessor.

    Example:
        `after = advance_conversation(before, admission)` advances revision once;
        folding the same admission again is a conflict, not another revision.
    """
    before = c._copy_conversation_view(before)
    checkpoint = c._copy_conversation_checkpoint(checkpoint)
    _validate_view(before)
    stamp = checkpoint.stamp
    if (
        stamp.namespace != before.namespace
        or stamp.generation != before.generation
        or stamp.expected_revision != before.revision
    ):
        raise _conflict("checkpoint does not name the current predecessor")
    operation_id = _canonical(
        [stamp.invocation.run_id, stamp.invocation.start_sequence, stamp.ordinal]
    )
    if any(entry.operation_id == operation_id for entry in before.applied):
        raise _conflict("checkpoint operation has already been applied")
    items, effects, completions = (
        before.items,
        before.effects,
        before.closed_invocations,
    )
    if type(checkpoint) is c.ConversationAdmitted:
        _validate_view(checkpoint.baseline)
        if _canonical(_view_json(checkpoint.baseline)) != _canonical(
            _view_json(before)
        ):
            raise _conflict(
                "admission baseline does not match the complete predecessor"
            )
        if any(_operation(entry)[0] == stamp.invocation for entry in before.applied):
            raise _conflict("invocation was already admitted")
    else:
        if not before.applied:
            raise _conflict("checkpoint requires an admitted invocation")
        invocation, ordinal = _operation(before.applied[-1])
        if invocation != stamp.invocation or stamp.ordinal != ordinal + 1:
            raise _conflict("checkpoint must be the current invocation's next ordinal")
        if any(value.invocation == invocation for value in completions):
            raise _conflict("invocation is already closed")
        if type(checkpoint) in (
            c.ConversationEffectRequested,
            c.ConversationEffectObserved,
        ):
            existing_ids = {item.item_id for item in items}
            if any(item.item_id in existing_ids for item in checkpoint.items):
                raise _conflict("checkpoint repeats an existing conversation item ID")
            items += checkpoint.items
        if type(checkpoint) is c.ConversationEffectRequested:
            effects += (
                c.ConversationEffectState(
                    invocation,
                    stamp.ordinal,
                    checkpoint.kind,
                    checkpoint.payload,
                    "pending",
                    (),
                    None,
                ),
            )
        elif type(checkpoint) is c.ConversationEffectObserved:
            index = next(
                (
                    index
                    for index, effect in enumerate(effects)
                    if effect.invocation == invocation
                    and effect.effect_ordinal == checkpoint.effect_ordinal
                ),
                None,
            )
            if index is None:
                raise _conflict("observation has no matching requested effect")
            effect = effects[index]
            if effect.status != "pending":
                raise _conflict("a terminal effect cannot receive more observations")
            observed = c.ConversationEffectState(
                invocation,
                effect.effect_ordinal,
                effect.kind,
                effect.request,
                checkpoint.status,
                effect.observation_ordinals + (stamp.ordinal,),
                checkpoint.failure,
            )
            effects = effects[:index] + (observed,) + effects[index + 1 :]
        elif type(checkpoint) is c.ConversationClosed:
            if checkpoint.status == "completed" and any(
                effect.invocation == invocation and effect.status == "pending"
                for effect in effects
            ):
                raise _conflict("successful closure cannot leave pending effects")
            completions += (
                c.ConversationCompletion(
                    invocation,
                    stamp.ordinal,
                    checkpoint.status,
                ),
            )
        else:
            raise TypeError("unsupported conversation checkpoint")
    entry = c.ConversationApplied(
        operation_id,
        checkpoint_digest(checkpoint),
        conversation_content_digest(before),
        "0" * 64,
    )
    after = c.ConversationView(
        before.namespace,
        before.generation,
        before.revision + 1,
        before.compatibility,
        items,
        effects,
        completions,
        before.applied + (entry,),
    )
    entry = c.ConversationApplied(
        entry.operation_id,
        entry.event_digest,
        entry.before_digest,
        conversation_content_digest(after),
    )
    result = c.ConversationView(
        after.namespace,
        after.generation,
        after.revision,
        after.compatibility,
        after.items,
        after.effects,
        after.closed_invocations,
        before.applied + (entry,),
    )
    conversation_view_to_json(result)
    return result
