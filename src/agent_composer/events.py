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

from agent_composer.diagnostics import SourceRange

# --------------------------------------------------------------------------- #
# Source location (for precise error framing)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceSpan:
    """A pointer to a sub-location in a flow's YAML, for precise error framing.

    Produced at the site a runtime failure originates from (binding knows the input
    key; an assert knows its expression) and carried on `NodeFailed`/`RunFailed`; the
    CLI resolves it to a 1-based source line via the parser's sub-line maps.

    Attributes:
        node (`str | None`):
            The node id the failure belongs to; `None` for a flow-level location
            (a boundary/post assert, an input declaration).
        kind (`Literal["input", "assert", "input_decl", "field"]`):
            Which family of sub-line to resolve against — a node input binding, an
            assert expression, a flow input declaration, or a node field (the kind
            fallback).
        key (`str`):
            The discriminator within that family — the input name, the assert
            expression string, the input-decl name, or the field name.
    """

    node: Optional[str]
    kind: Literal["input", "assert", "input_decl", "field"]
    key: str


@dataclass(frozen=True)
class Diagnostic:
    """A renderable compile/run error: one message + a primary range + secondary notes.

    `severity` is "error" | "warning"; `primary` is the SourceRange the caret points
    at; `notes` is a tuple of secondary annotation strings (e.g. outer call frames of a
    nested-call traceback, or an anchor definition site for a merged key). The single
    `render_diagnostic` renderer (CLI) consumes this — no kind-specific rendering.
    """

    severity: str
    message: str
    primary: SourceRange | None
    notes: tuple[str, ...] = ()


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
    """A node raised; the engine boundary captured the error message and type."""

    node_id: str
    error: str
    error_type: str = ""
    # The precise YAML sub-location the failure originates from (an input binding,
    # an assert expr); None when no sub-line is determinable (a code raise) -> the
    # CLI falls back to the node's best sub-line / header.
    locator: Optional[SourceSpan] = None
    # The concrete source span the funnel resolves ALONGSIDE the symbolic locator, by looking
    # up the node's recorded `field_ranges[(locator.kind, locator.key)]` (kind-blind — keyed on
    # the LOCATOR, never on `node.kind`; the kind tag is what keeps a field from colliding with
    # a like-named input). None when the node records no range for that (kind, key). The CLI
    # prefers this over resolving the symbolic `locator`. Additive/coexistent with `locator`.
    range: Optional[SourceRange] = None
    # The formatted Python traceback of the raising call, captured at the node-failure
    # boundary. None for failures synthesized without an exception (e.g. an assert that did
    # not hold). Surfaced by the CLI only under `--engine-trace`.
    traceback: Optional[str] = None


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

    error: str
    error_type: str = ""
    # Flow-level precise location (a boundary assert expr, a failed input decl);
    # None when not determinable -> the CLI falls back to a plain message.
    locator: Optional[SourceSpan] = None
    # The concrete source span resolved ALONGSIDE the symbolic locator (the flow-boundary twin
    # of NodeFailed.range). A failing input decl's declaration line is looked up on the
    # StartNode's `field_ranges[("input_decl", name)]`; a false boundary assert instead carries
    # its own range, resolved BY POSITION off `pre_assert_ranges`/`post_assert_ranges` (assert
    # text is not a unique identity, so asserts are deliberately not in `field_ranges`). None
    # when neither is recorded. The CLI prefers this over resolving the locator.
    range: Optional[SourceRange] = None
    # The formatted Python traceback of the originating node failure, if one was captured
    # (carried up from the aborting `NodeFailed`). None for flow-level failures with no
    # exception behind them (boundary assert / input coercion). CLI shows it under
    # `--engine-trace`.
    traceback: Optional[str] = None
