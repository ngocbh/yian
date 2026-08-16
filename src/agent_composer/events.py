"""Transient engine observer vocabulary shared by the runtime and callers.

Node observers describe evaluation after their corresponding durable transition
has been applied: ``NodeStarted`` followed by ``NodeSucceeded``, ``NodeRouted``,
``NodeExpanded``, or ``NodeFailed``. ``StreamChunk`` is intentionally transient,
and ``NodeSkipped`` is derived from a route's durable skip closure. A node only
describes its result; the reducer owns graph state and variable-pool writes.

Run observers bracket a fresh or recovered execution segment with ``RunStarted``
or ``RunResumed``. ``RunSucceeded`` is derived at successful quiescence;
``RunFailed`` is derived from a durable node failure or from a reduced transition
that makes the terminal output unreachable. Plain dataclasses keep this
low-level observer module separate from the serialized record vocabulary in
``runtime.log``; source-neutral location values come from ``diagnostics``.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

from agent_composer.diagnostics import Failure, SourceRange

# --------------------------------------------------------------------------- #
# Source location (for precise error framing)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceSpan:
    """A pointer to a sub-location in a flow's YAML, for precise error framing.

    Produced at the site a runtime failure originates from (binding knows the input
    key; an assert carries its author-facing display) and retained beside the
    canonical `Failure`; the CLI resolves it through the authored source maps.

    Attributes:
        node (`str | None`):
            The node id the failure belongs to; `None` for a flow-level location
            (a boundary/post assert, a flow input).
        kind (`Literal["input", "assert", "field"]`):
            Which family of sub-line to resolve against — an input parameter, an
            assertion display, or a node field (the kind fallback).
        key (`str`):
            The discriminator within that family — the input name, the assertion's
            stable display text, or the field name.
    """

    node: Optional[str]
    kind: Literal["input", "assert", "field"]
    key: str


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
    siblings. Stores no value."""

    node_id: str
    handle: str


@dataclass
class NodeSkipped:
    """A post-durable observer that a route skip-flood made a node unreachable."""

    node_id: str


@dataclass
class NodeFailed:
    """A node failed with one canonical semantic diagnostic."""

    failure: Failure
    # Temporary YAML presentation bridge; None for source-free failures.
    locator: Optional[SourceSpan] = None
    # Concrete range resolved at the failure site when available.
    range: Optional[SourceRange] = None


@dataclass
class NodeExpanded:
    """A spawner returned a seed-first `Grow`; the dispatcher builds and splices its fragment.

    The node ran; its eventual value is committed by the fragment terminal via `commit_as`.
    """

    node_id: str
    grow: Any = None


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
    # Same temporary presentation bridge as the originating NodeFailedRecord.
    locator: Optional[SourceSpan] = None
    # Concrete authored range, or None when only an opaque origin is known.
    range: Optional[SourceRange] = None
