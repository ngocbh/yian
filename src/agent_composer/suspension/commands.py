"""Commands — external control delivered to a (possibly suspended) run.

`DeliverAnswerCommand` is the temporary generic vehicle an external watcher/host
uses to satisfy a pause: it delivers the awaited value (for example an event
payload) to the parked leaf as that leaf's `Output` (deliver-as-Output).
`AbortCommand` terminates a run.
"""

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class CommandType(str, Enum):
    ABORT = "abort"
    DELIVER_ANSWER = "deliver_answer"


class AbortCommand(BaseModel):
    type: Literal[CommandType.ABORT] = CommandType.ABORT
    reason: str = ""


class DeliverAnswerCommand(BaseModel):
    """Deliver an answer to a parked leaf as its `Output`.

    The engine writes `value` to `pool[node_id]` (deliver-as-Output) and fires the
    leaf's existing out-edges — no scratch, no re-run. `node_id` is resolved against
    the LIVE graph (`engine.flow.nodes`), so it may be a runtime-namespaced id. A
    synthetic scheduled release conventionally delivers `value=None`; authorable
    WAIT does not use this command."""

    type: Literal[CommandType.DELIVER_ANSWER] = CommandType.DELIVER_ANSWER
    node_id: str
    value: Any = None


Command = Annotated[
    Union[AbortCommand, DeliverAnswerCommand],
    Field(discriminator="type"),
]
