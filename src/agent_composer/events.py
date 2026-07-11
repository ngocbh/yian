"""Engine event vocabulary — shared by nodes, the runtime, and callers.

Two tiers (graphon's "two-tier event model"):

- **Node events** are produced by `Node.run()`. A node emits `NodeStarted`, then
  optionally `StreamChunk`s (token streaming) and/or a `PauseRequested`, and
  terminates with `NodeSucceeded` or `NodeFailed`. The node only *describes* its
  one output value on `NodeSucceeded.output`; the engine — not the node — writes
  it into the variable pool. That split keeps nodes pure and testable.

- **Run events** are produced by `FlowEngine.run()` and streamed to the caller
  (the CLI): `RunStarted` then one terminal of
  `RunSucceeded | RunFailed | RunPaused | RunAborted`.

Plain dataclasses, not pydantic: these are transient in-process signals, never
serialized (what *is* serialized is the checkpoint, which captures pause
*reasons*, not events).

`PauseReason` is typed in `suspension.pause`; referenced here as `Any` to keep
`events` a dependency-free leaf alongside `typesys`.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

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
class SourceRange:
    """A concrete 1-based source span: [start, end) with an exclusive end column.

    Coordinates are 1-based (line 1, col 1 = first char); `end_col` is one past the
    last covered column (half-open) so a zero-width caret is start == end. Produced by
    the compose loader from PyYAML marks and by the funnel that resolves a symbolic
    `SourceSpan` key to the node's recorded range.
    """

    start_line: int
    start_col: int
    end_line: int
    end_col: int


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
# Node-level events
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

    `commit_as` is the commit redirect the engine's `_on_success` reads: when set, the value
    is written under `commit_as` (and that target's out-edges fire) instead of `node_id`.
    Folded by `eval_node` from `result.commit_as or node.commit_as` (node-chosen wins over the
    engine-baked node field); `None` for an ordinary node (commit under `node_id`).
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
class NodeFailed:
    """A node raised; the engine boundary captured the error message and type."""

    node_id: str
    error: str
    error_type: str = ""
    # The precise YAML sub-location the failure originates from (an input binding,
    # an assert expr); None when no sub-line is determinable (a code raise) -> the
    # CLI falls back to the node's best sub-line / header.
    locator: Optional[SourceSpan] = None
    # The concrete source span the funnel resolves ALONGSIDE the symbolic locator, by
    # looking up the node's recorded `field_ranges[locator.key]` (kind-blind — keyed only
    # on the locator key). None when the node records no range for that key. The CLI
    # prefers this over resolving the symbolic `locator`. Additive/coexistent with `locator`.
    range: Optional[SourceRange] = None
    # The formatted Python traceback of the raising call, captured at the node-failure
    # boundary. None for failures synthesized without an exception (e.g. an assert that did
    # not hold). Surfaced by the CLI only under `--engine-trace`.
    traceback: Optional[str] = None


@dataclass
class NodeExpanded:
    """A spawner (CALL/MAP/AGENT/LOOP) returned a `Grow`; the dispatcher's `_apply_grow` splices
    the subgraph into the live graph. The node ran (its value is committed by the subgraph
    terminal via `commit_as`)."""

    node_id: str
    grow: Any = None


@dataclass
class PauseRequested:
    """A node cannot proceed until an external signal arrives.

    `reason` is a `suspension.pause.PauseReason` (HumanInputRequired /
    EventAwaited / ...). The engine resets the node and suspends the run.
    """

    node_id: str
    reason: Any


# Anything a node's `_run` generator may yield before returning its result.
NodeStreamEvent = (StreamChunk, PauseRequested)


# --------------------------------------------------------------------------- #
# Run-level events
# --------------------------------------------------------------------------- #


@dataclass
class RunStarted:
    """The lead event of a fresh run."""

    pass


@dataclass
class RunResumed:
    """A resumed run's lead event (the `engine.resume()` twin of `RunStarted`)."""

    pass


@dataclass
class RunSucceeded:
    """The run reached its terminal; `output` is the flow's single committed value."""

    output: Any = None  # the flow's single (possibly object) terminal value


@dataclass
class RunFailed:
    """The run ended on an unrecovered node failure."""

    error: str
    error_type: str = ""
    # Flow-level precise location (a boundary assert expr, a failed input decl);
    # None when not determinable -> the CLI falls back to a plain message.
    locator: Optional[SourceSpan] = None
    # The formatted Python traceback of the originating node failure, if one was captured
    # (carried up from the aborting `NodeFailed`). None for flow-level failures with no
    # exception behind them (boundary assert / input coercion). CLI shows it under
    # `--engine-trace`.
    traceback: Optional[str] = None


@dataclass
class RunPaused:
    """The run suspended; `reasons` carry what each paused node is waiting for."""

    reasons: list[Any] = field(default_factory=list)


@dataclass
class RunAborted:
    """The run was aborted by an `AbortCommand` rather than completing or failing."""

    pass
