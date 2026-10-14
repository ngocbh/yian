"""Transient engine observer vocabulary shared by the runtime and callers.

Node observers describe evaluation after their corresponding durable transition
has been applied: ``NodeStarted`` followed by ``NodeSucceeded``, ``NodeRouted``,
``NodeExpanded``, or ``NodeFailed``. ``StreamChunk`` is intentionally transient,
as are the in-loop AGENT observers (``ModelCallStarted``/``ModelCallFinished`` with
``TokenUsage``, ``ToolCallStarted``/``ToolCallFinished``): they reach ``on_event``
between a node's start and outcome but are never yielded, recorded, or replayed,
and ``NodeSkipped`` is derived from a route's durable skip closure. A node only
describes its result; the reducer owns graph state and variable-pool writes.

Run observers bracket a fresh or recovered execution segment with ``RunStarted``
or ``RunResumed``. ``RunSucceeded`` is derived at successful quiescence;
``RunFailed`` is derived from a durable node failure or from a reduced transition
that makes the terminal output unreachable. Plain dataclasses keep this
low-level observer module separate from the serialized record vocabulary in
``runtime.log``; canonical semantic failures come from ``diagnostics``.
"""

from dataclasses import dataclass
from typing import Any, Optional

from yian.diagnostics import Failure


# --------------------------------------------------------------------------- #
# Node-level signals and reducer observer notifications
# --------------------------------------------------------------------------- #


@dataclass
class NodeStarted:
    """A node began executing."""

    node_id: str


@dataclass
class StreamChunk:
    """A streamed fragment of a node's output (e.g. an LLM token)."""

    node_id: str
    key: str
    chunk: str
    final: bool = False
    # AGENT text chunks name the model exchange they belong to; None for other streams.
    exchange_id: Optional[str] = None


@dataclass
class NodeSucceeded:
    """A node finished and produced its single output value (the engine does the pool write).

    `commit_as` exposes the effective redirect retained by the success record: when set,
    the reducer wrote the value under that target and fired its out-edges instead of those
    of `node_id`. The dispatcher resolves an explicit `Output.commit_as` before the node's
    baked target; `None` means the value was committed under `node_id`.
    """

    node_id: str
    output: Any = None  # the node's single produced value
    commit_as: Optional[str] = None


@dataclass
class NodeRouted:
    """A router (CASE) selected an out-edge handle; the engine takes it and skip-floods the
    siblings. Stores no value.

    `handle` is the taken branch's handle: a `case`'s `then:` target id, or `"#else"`
    (`ELSE_HANDLE`) for its `else:` route, also taken when no `when:` matches.
    """

    node_id: str
    handle: str


@dataclass
class NodeSkipped:
    """A post-durable observer that a route skip-flood made a node unreachable."""

    node_id: str


@dataclass
class NodeFailed:
    """A node failed with one canonical semantic failure."""

    failure: Failure


@dataclass
class NodeExpanded:
    """A spawner returned a seed-first `Grow`; the dispatcher builds and splices its fragment.

    The node ran; its eventual value is committed by the fragment terminal via `commit_as`.
    """

    node_id: str
    grow: Any = None


# --------------------------------------------------------------------------- #
# In-loop AGENT observers (transient: callback-only, never yielded or recorded)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported token counts for one model call.

    Counts come from the reply's ``usage_metadata``; a provider that reports no
    usage yields no ``TokenUsage``. Optional details are ``None`` when unreported.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


@dataclass(frozen=True)
class ModelCallStarted:
    """An AGENT node is about to call its model.

    ``purpose`` is ``"chat"``, ``"structured_native"``, ``"structured_fallback"``, or
    ``"history_summary"``. ``provider`` and ``model`` echo the effective config.
    """

    node_id: str
    exchange_id: str
    purpose: str
    provider: Optional[str] = None
    model: Optional[str] = None


@dataclass(frozen=True)
class ModelCallFinished:
    """One model call returned or raised.

    ``usage`` is the provider's own count, or ``None`` when it reported none. A
    call that raised has no ``usage`` unless the provider replied first (a native
    structured reply that failed to parse). A streamed reply cut by a host
    interrupt has no ``error`` and the usage of the chunks already received.
    ``error`` is the exception type name only, never provider text.
    """

    node_id: str
    exchange_id: str
    purpose: str
    provider: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[TokenUsage] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ToolCallStarted:
    """An AGENT node is about to run one granted tool call from a model reply."""

    node_id: str
    exchange_id: str
    call_index: int
    tool_call_id: str
    name: str
    arguments: Any = None


@dataclass(frozen=True)
class ToolCallFinished:
    """One tool call returned (``result`` text) or raised (``error`` type name).

    A call that a host interrupt skipped is reported as started and then finished
    with ``error="interrupted"``; the tool did not run.
    """

    node_id: str
    exchange_id: str
    call_index: int
    tool_call_id: str
    name: str
    result: Optional[str] = None
    error: Optional[str] = None


# Hosts sharing one ``on_event`` callback can separate these from post-durable observers.
IN_LOOP_EVENTS = (ModelCallStarted, ModelCallFinished, ToolCallStarted, ToolCallFinished)


# --------------------------------------------------------------------------- #
# Run-level events
# --------------------------------------------------------------------------- #


@dataclass
class RunStarted:
    """The lead event of a fresh run."""

    pass


@dataclass
class RunResumed:
    """A recovered run segment began after its unfinished nodes were requeued."""

    pass


@dataclass
class RunSucceeded:
    """The run reached its terminal; `output` is the flow's single committed value."""

    output: Any = None  # the flow's single (possibly object) terminal value


@dataclass
class RunFailed:
    """The run ended on a node failure or an unreachable terminal output."""

    failure: Failure
