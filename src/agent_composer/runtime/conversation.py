"""Pure run-local conversation projections over the lower memory fold."""

from dataclasses import dataclass, replace

from agent_composer.memory.contracts import (
    ConversationAdmitted,
    ConversationAuthority,
    ConversationCheckpoint,
    ConversationInvocation,
    ConversationView,
)
from agent_composer.memory.conversation import (
    advance_conversation,
    conversation_checkpoint_from_json,
    conversation_checkpoint_to_json,
    conversation_view_from_json,
    conversation_view_to_json,
)


@dataclass(frozen=True, slots=True)
class RuntimeConversationState:
    """Accepted admission-led segments and views, including retired invocations.

    Args:
        segments (`tuple[ConversationAuthority, ...]`, *optional*, defaults to `()`):
            One accepted admission-led segment per invocation, in admission order,
            preserved after graph pruning. Validated and detached on construction.
        views (`tuple[ConversationView, ...]`, *optional*, defaults to `()`):
            Corresponding resulting views, one per segment. Each must equal the
            lower fold of that segment and is detached on construction.
        retired_invocations (`frozenset[ConversationInvocation]`, *optional*, defaults to `frozenset()`):
            Admitted effect-free invocations superseded by an accepted recovery
            frontier. Each identity must belong to a retained segment.

    Example:
        `current = RuntimeConversationState()` creates an empty projection before
        `prepare_conversation_checkpoint(current, admission)` prepares a successor.
    """

    segments: tuple[ConversationAuthority, ...] = ()
    views: tuple[ConversationView, ...] = ()
    retired_invocations: frozenset[ConversationInvocation] = frozenset()

    def __post_init__(self) -> None:
        """Detach supplied carriers and prove each view from its accepted segment."""
        if type(self.segments) is not tuple or type(self.views) is not tuple:
            raise TypeError("conversation segments and views must be tuples")
        if len(self.segments) != len(self.views):
            raise ValueError("conversation segments and views must be paired")
        invocations = set()
        segments = []
        views = []
        for segment, view in zip(self.segments, self.views):
            if (
                type(segment) is not ConversationAuthority
                or type(view) is not ConversationView
            ):
                raise TypeError("conversation projection requires exact carriers")
            segment = ConversationAuthority(segment.baseline, segment.checkpoints)
            view_wire = conversation_view_to_json(view)
            expected = segment.baseline
            for checkpoint in segment.checkpoints:
                expected = advance_conversation(expected, checkpoint)
            if conversation_view_to_json(expected) != view_wire:
                raise ValueError("conversation view must match its accepted segment")
            segments.append(segment)
            views.append(conversation_view_from_json(view_wire))
            invocation = segment.checkpoints[0].stamp.invocation
            if invocation in invocations:
                raise ValueError("conversation invocation was already admitted")
            invocations.add(invocation)
        if type(self.retired_invocations) is not frozenset:
            raise TypeError("retired invocations must be a frozenset")
        retired_invocations = []
        for invocation in self.retired_invocations:
            if type(invocation) is not ConversationInvocation:
                raise TypeError("retired invocations require exact identities")
            retired_invocations.append(replace(invocation))
        retired = frozenset(retired_invocations)
        if not retired <= invocations:
            raise ValueError("only admitted invocations can be retired")
        object.__setattr__(self, "segments", tuple(segments))
        object.__setattr__(self, "views", tuple(views))
        object.__setattr__(self, "retired_invocations", retired)

    def index(self, invocation: ConversationInvocation) -> int | None:
        """Find an admitted invocation, including one retired during recovery.

        Args:
            invocation (`ConversationInvocation`): Accepted start identity to find.

        Returns:
            `int | None`: Admission-order index, or `None` when not admitted.
        """
        return next(
            (
                index
                for index, segment in enumerate(self.segments)
                if segment.checkpoints[0].stamp.invocation == invocation
            ),
            None,
        )


def _checked_projection(
    segments: tuple[ConversationAuthority, ...],
    views: tuple[ConversationView, ...],
    retired_invocations: frozenset[ConversationInvocation],
) -> RuntimeConversationState:
    """Retain already-proved immutable entries without replaying older segments."""
    result = object.__new__(RuntimeConversationState)
    object.__setattr__(result, "segments", segments)
    object.__setattr__(result, "views", views)
    object.__setattr__(result, "retired_invocations", retired_invocations)
    return result


def _require_prepared_projection(
    current: RuntimeConversationState,
    prepared: RuntimeConversationState,
    expected: RuntimeConversationState,
) -> None:
    """Require exact unchanged roots and canonical newly prepared entry values."""
    if type(prepared) is not RuntimeConversationState:
        raise TypeError("conversation plan requires an exact next state")
    if (
        type(prepared.segments) is not tuple
        or type(prepared.views) is not tuple
        or prepared.retired_invocations is not current.retired_invocations
        or len(prepared.segments) != len(expected.segments)
        or len(prepared.views) != len(expected.views)
    ):
        raise ValueError("conversation plan has invalid projection containers")
    for index, (segment, view) in enumerate(zip(expected.segments, expected.views)):
        supplied_segment, supplied_view = (
            prepared.segments[index],
            prepared.views[index],
        )
        if index < len(current.segments) and segment is current.segments[index]:
            if supplied_segment is not segment or supplied_view is not view:
                raise ValueError("conversation plan changed an unrelated invocation")
            continue
        if type(supplied_segment) is not ConversationAuthority:
            raise TypeError("conversation plan requires an exact accepted segment")
        detached = ConversationAuthority(
            supplied_segment.baseline, supplied_segment.checkpoints
        )
        if detached != segment or conversation_view_to_json(
            supplied_view
        ) != conversation_view_to_json(view):
            raise ValueError(
                "conversation plan disagrees with its canonical checkpoint"
            )


def prepare_conversation_checkpoint(
    current: RuntimeConversationState,
    checkpoint: ConversationCheckpoint,
) -> RuntimeConversationState:
    """Prepare one detached successor without backend calls or state mutation.

    Args:
        current (`RuntimeConversationState`): Exact accepted run-local projection.
        checkpoint (`ConversationCheckpoint`): One new exact checkpoint carrier.

    Returns:
        `RuntimeConversationState`: Projection with the accepted segment advanced.

    Raises:
        `TypeError`: A projection or checkpoint has an invalid exact type.
        `ValueError`: The invocation is retired, missing, or already admitted.
        `MemoryOperationError`: The lower fold rejects the checkpoint's predecessor
            or effect ordering.

    Example:
        `next_state = prepare_conversation_checkpoint(current, admission)` leaves
        `current` unchanged and retains the admission's complete baseline.
    """
    if type(current) is not RuntimeConversationState:
        raise TypeError("current must be a RuntimeConversationState")
    checkpoint = conversation_checkpoint_from_json(
        conversation_checkpoint_to_json(checkpoint)
    )
    invocation = checkpoint.stamp.invocation
    if invocation in current.retired_invocations:
        raise ValueError("conversation invocation is retired")
    index = current.index(invocation)
    if type(checkpoint) is ConversationAdmitted:
        if index is not None:
            raise ValueError("conversation invocation was already admitted")
        view = advance_conversation(checkpoint.baseline, checkpoint)
        segment = ConversationAuthority(checkpoint.baseline, (checkpoint,))
        return _checked_projection(
            current.segments + (segment,),
            current.views + (view,),
            current.retired_invocations,
        )
    if index is None:
        raise ValueError("conversation checkpoint requires an admitted invocation")
    previous = current.segments[index]
    segment = ConversationAuthority(
        previous.baseline, previous.checkpoints + (checkpoint,)
    )
    view = advance_conversation(current.views[index], checkpoint)
    return _checked_projection(
        current.segments[:index] + (segment,) + current.segments[index + 1 :],
        current.views[:index] + (view,) + current.views[index + 1 :],
        current.retired_invocations,
    )


__all__ = ["RuntimeConversationState", "prepare_conversation_checkpoint"]
