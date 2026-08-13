"""PauseReason — a discriminated union describing *what a suspended node awaits*.

This is the typed payload on `PauseRequested` and the thing a checkpoint persists
so a scheduler/watcher knows when to wake the run. Adding a new way to wait is a
new subclass, nothing else.

Maps to the temporary generic compatibility surface:
- a legacy/artificial human reason -> HumanInputRequired
- WATCH composite's WAIT            -> EventAwaited (watcher fires -> resume)
- a plain timed wait                -> ScheduledPause

Authorable HUMAN_INPUT and agent ``ask_user`` calls no longer produce a pause;
they block inside their live node execution. The human reason remains wire/test
compatibility until the subsequent run-log cutover removes this package.
"""

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class HumanInputRequired(BaseModel):
    """
    A legacy/artificial paused leaf awaits a human-shaped answer.

    The host re-prompts on resume and delivers the answer as the parked leaf's
    Output. Authorable HUMAN_INPUT and agent ``ask_user`` do not emit this reason.

    Attributes:
        type (`Literal["human_input_required"]`):
            The discriminator tag for the `PauseReason` union.
        prompt (`str`):
            The text to show the human.
        answer_schema (`list[dict]`, *optional*, defaults to `[]`):
            IOField-shaped description of the expected answer; the CLI renders against it.
        questions (`list[dict]`, *optional*, defaults to `[]`):
            Structured questions for the host to render. Empty (default) = the legacy
            prompt-only form. Non-empty = a list of serialized `QuestionSpec` dicts, each
            shaped `{question, header, options, multi_select}`.
        node_title (`str`, *optional*, defaults to `None`):
            A human-friendly node title for display, if any.
        node_id (`str`, *optional*, defaults to `None`):
            Logical resumption identity — the node the answer is delivered to as its Output.
        slot (`str`, *optional*, defaults to `None`):
            Vestigial compatibility field (always `None`); removed with the
            subsequent run-log cutover.
    """

    type: Literal["human_input_required"] = "human_input_required"
    prompt: str
    # IOField-shaped expected answer (loose dicts; the CLI renders against it).
    answer_schema: list[dict[str, Any]] = Field(default_factory=list)
    # Serialized QuestionSpec dicts ({question, header, options, multi_select}) for the host
    # to render. Empty (default) => legacy prompt-only form; non-empty => structured questions.
    questions: list[dict[str, Any]] = Field(default_factory=list)
    node_title: Optional[str] = None
    # LOGICAL resumption identity: the artificial/legacy node that paused. The generic
    # compatibility path delivers the answer as that leaf's Output keyed on `node_id`.
    # `slot` is vestigial and remains only for the temporary wire shape.
    node_id: Optional[str] = None
    slot: Optional[str] = None


class EventAwaited(BaseModel):
    """
    A suspended node awaits an external event (a `WATCH` composite's `WAIT`).

    A watcher re-checks the condition on a poll cadence and, when it fires, resumes the
    run by delivering the watcher payload as the node's Output.

    Attributes:
        type (`Literal["event_awaited"]`):
            The discriminator tag for the `PauseReason` union.
        event_spec (`dict`, *optional*, defaults to `{}`):
            What to watch — a condition over a source.
        poll (`dict`, *optional*, defaults to `{}`):
            How often to re-check the condition.
        node_id (`str`, *optional*, defaults to `None`):
            Logical resumption identity. `None` means not host-resumable (a bare event an
            external watcher satisfies; `resume_command` rejects it).
    """

    type: Literal["event_awaited"] = "event_awaited"
    # what to watch (a condition over a source) + how often to re-check.
    event_spec: dict[str, Any] = Field(default_factory=dict)
    poll: dict[str, Any] = Field(default_factory=dict)
    # LOGICAL resumption identity: the WAIT node whose watcher payload the host
    # delivers as the node's Output. None (default) => not host-resumable (a bare
    # EventAwaited() an external watcher satisfies; resume_command rejects it).
    node_id: Optional[str] = None


class ScheduledPause(BaseModel):
    """
    A suspended node awaits a wall-clock time (a plain timed wait).

    Attributes:
        type (`Literal["scheduled_pause"]`):
            The discriminator tag for the `PauseReason` union.
        resume_at (`str`, *optional*, defaults to `None`):
            ISO timestamp to wake at; `None` means "until poked".
        node_id (`str`, *optional*, defaults to `None`):
            Logical resumption identity — the node whose timed wait the host releases.
    """

    type: Literal["scheduled_pause"] = "scheduled_pause"
    resume_at: Optional[str] = None  # ISO timestamp; None = "until poked"
    # LOGICAL resumption identity: the node whose timed wait the host releases.
    node_id: Optional[str] = None


PauseReason = Annotated[
    Union[HumanInputRequired, EventAwaited, ScheduledPause],
    Field(discriminator="type"),
]
