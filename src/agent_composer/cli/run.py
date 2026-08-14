"""`ac run` — load a flow, gather inputs, drive it to a terminal, render the output.

Fresh inputs come from flags (`--input k=v`, repeatable; `--inputs file.json`); any
declared input still missing is prompted for interactively (required ones are
starred). Fresh runs use a reported durable log under
``~/.agent_composer/runs`` unless ``--log`` selects a new path. ``--resume``
recovers an existing path without gathering fresh inputs or model configuration.

The engine runs behind a main-thread terminal broker: blocking HUMAN_INPUT and
agent ``ask_user`` calls wait for that broker and then continue inside the same
live node call. WAIT stays live in its worker until its deadline. Answer types
are enforced at the engine boundary; an invalid one fails the run.

`--provider`/`--model` feed the outermost layer of the `llm_config` cascade (fill-the-gap,
not a hard override): they fill only the fields an agent and its enclosing flow leave unset.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError
from contextlib import contextmanager
from functools import partial
import itertools
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import typer
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from agent_composer.compile.model import END_ID, START_ID
from agent_composer.cli.terminal import PromptRequest, TerminalBroker
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.loader import load_flow, top_frame_ranges
from agent_composer.run import (
    RunResult,
    _resume_flow_controlled,
    _run_flow_controlled,
)
from agent_composer.events import Diagnostic, NodeFailed, SourceRange, SourceSpan
from agent_composer.nodes.base import _HostStop
from agent_composer.tools import install_ask_user
from agent_composer.typesys.seeding import default_run_id
from agent_composer.typesys.values import ValueKind

console = Console()
err_console = Console(stderr=True)
_DEFAULT_BROKER_EOF = object()


def _ensure_cwd_importable() -> None:
    """Put the current working directory on `sys.path` so a flow's `code: module:function`
    refs to repo-local packages resolve without the caller exporting `PYTHONPATH=$PWD`.

    Mirrors how `python script.py` makes things importable from the invocation dir, and
    matches the documented "run from the repo root" convention (e.g. a seed's
    `tests.seeds.fns:fn` or `examples.trading_fns:fn`). Prepended and idempotent. Trade-off:
    a flow can then import any package under the cwd — acceptable for a local dev CLI."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _default_run_log_path(run_id: str) -> Path:
    """Create the private default CLI directories and return one fresh log path.

    The file itself is deliberately not created here. ``JsonlRunLog.create``
    retains exclusive creation, owner-only file mode, locking, and durability.

    Args:
        run_id (`str`):
            Fresh identity used as the JSONL basename.

    Returns:
        `Path`:
            ``~/.agent_composer/runs/<run_id>.jsonl`` with both CLI-owned
            directories forced to mode ``0700``.
    """
    state_dir = Path.home() / ".agent_composer"
    runs_dir = state_dir / "runs"
    for directory in (state_dir, runs_dir):
        directory.mkdir(mode=0o700, exist_ok=True)
        directory.chmod(0o700)
    return runs_dir / f"{run_id}.jsonl"


def _report_run_log(path: Path) -> None:
    """Print the durable authority path even when progress is quiet."""
    message = Text("Run log: ", style="dim")
    message.append(str(path), style="dim")
    err_console.print(message, soft_wrap=True)


def _render_host_run_error(
    action: str,
    error: BaseException,
    *,
    engine_trace: bool,
) -> None:
    """Render an expected run-boundary host failure without a default traceback.

    Args:
        action (`str`):
            User-facing operation name, either ``"run"`` or ``"resume"``.
        error (`BaseException`):
            Documented path, codec, fingerprint, argument, or locking failure.
        engine_trace (`bool`):
            Whether to append the active Python traceback for engine debugging.
    """
    message = Text(f"{action} failed: ", style="red bold")
    message.append(str(error), style="red")
    err_console.print(message)
    if engine_trace:
        err_console.print_exception()


# Lines of `.yaml` context shown above and below the offending line in the error panel
# (a window, so a large flow doesn't dump in full — mirrors a Python traceback's code frame).
_ERR_CONTEXT = 5

# When a failure carries no precise locator, fall back to the best sub-field range for the
# node's kind before the node header. These are field NAMES; the lookup tags each `("field", name)`
# to match how `field_ranges` is keyed (a CODE node -> its `code:` range). First field present wins.
_KIND_FALLBACK_FIELDS = ("code",)

# The engine's `field` locator uses the SINGULAR write-boundary vocabulary (`output`/`input`),
# but a node's `field_ranges` is keyed by the descriptor's canonical PLURAL field names
# (`outputs`/`inputs`). Translate the locator key to the descriptor key before lookup.
_FIELD_KEY_ALIAS = {"output": "outputs", "input": "inputs"}


def _field_range(fields: dict, key: str) -> Optional[SourceRange]:
    """Look up a `field` locator key in a node's `field_ranges` (keyed by `(kind, key)`),
    honoring the singular alias. `fields` is one node's per-field map; the lookup tags `("field",
    …)` so a field never collides with a like-named input (`("input", …)`)."""
    return fields.get(("field", key)) or fields.get(("field", _FIELD_KEY_ALIAS.get(key, key)))


def _top_ranges(loaded, text: str):
    """The top flow's `(node_ranges, field_ranges)` for run-error framing.

    Prefers the loaded IR's render-only `SourceFrame` (`loaded.source`) when the caller holds
    it; otherwise recomputes the maps from `text` alone (best-effort). Both map a node id to
    its header `SourceRange` and to its per-field `SourceRange`s — the source the single-frame
    renderer boxes a node/field range from."""
    src = getattr(loaded, "source", None) if loaded is not None else None
    if src is not None:
        return src.node_ranges, src.field_ranges
    return top_frame_ranges(text)


def _render_frame_box(text: str, marks, title: str) -> bool:
    """Print ONE boxed `.yaml` source frame (no message). Returns whether a box was drawn.

    `marks` are the 1-based offending lines; the panel shows the source around them with line
    numbers and each mark highlighted, titled `title:line[,line...]`. Returns `False` (nothing
    printed) when no mark is in range, so the caller can fall back to a plain line. Shared by
    the single-frame render and the multi-frame call-traceback stack."""
    lines = text.splitlines()
    marks = sorted({m for m in marks if 1 <= m <= len(lines)})
    if not marks:
        return False
    start = max(1, marks[0] - _ERR_CONTEXT)
    end = min(len(lines), marks[-1] + _ERR_CONTEXT)
    frame = Syntax(
        text,
        "yaml",
        line_numbers=True,
        line_range=(start, end),
        highlight_lines=set(marks),
        word_wrap=True,
    )
    panel_title = f"{title}:{','.join(str(m) for m in marks)}"
    err_console.print(Panel(frame, title=panel_title, title_align="left", border_style="red"))
    return True


def _render_source_frame(text: str, marks, title: str, message: str) -> None:
    """Print an error as a boxed `.yaml` source frame + the message below (red).

    Shared by compile and runtime single-frame errors: boxes the source around `marks`
    (like a Python traceback's code box), then prints `message` under it. When no mark is in
    range, prints a plain `title: message` line (no box)."""
    if not _render_frame_box(text, marks, title):
        err_console.print(Text(f"{title}: ", style="red bold") + Text(message, style="red"))
        return
    err_console.print(Text(message, style="red"))


def _render_load_error(err: LoadError, flow: Path, text: str) -> None:
    """Print a located compile error for the author as a boxed `.yaml` source frame.

    Surfaces `LoadError.line`/`.lines` (the loader's source-line tracking) at the CLI boundary
    so a failed compile points at WHERE in the `.yaml` it broke. A multi-line error (e.g. a cycle,
    which implicates several nodes) highlights ALL of them. When no line is known (or all are out
    of range), prints `file: <message>` with no frame. The "why" legend (`.notes`) follows.
    """
    marks = err.lines or ([err.line] if err.line is not None else [])
    if len(marks) > 1:
        # A multi-node error (a cycle implicates several nodes) highlights EVERY stuck line —
        # `.range` anchors only the first, so the multi-mark frame wins here to box them all.
        _render_source_frame(text, marks, flow.name, str(err))
        _render_notes(err)
        return
    if err.range is not None:
        # Span-aware path (Step 8): the loader resolved a precise `SourceRange`, so box it
        # directly instead of re-deriving a line. The `.notes` "why" legend still follows.
        render_diagnostic(Diagnostic("error", str(err), err.range), text, flow.name)
        _render_notes(err)
        return
    _render_source_frame(text, marks, flow.name, str(err))
    _render_notes(err)


def _node_range(node_id: str, node_ranges: dict, field_ranges: dict) -> Optional[SourceRange]:
    """Best `SourceRange` for a (possibly namespaced/clone) node id, or `None`.

    Tries the leaf segment first, then the owning top-level segment (so a runtime-namespaced
    `gate/approve` or a loop-driver clone `polish~5` resolves to the authored `gate:`/`polish:`
    node). For each, prefers the node's kind fallback sub-field (a code node's `code:` range),
    else its header range. `None` when neither segment is an authored top-level node."""
    leaf = _last_segment(node_id)
    owning = _strip_clone_suffix(node_id.split("/", 1)[0])
    for key in (leaf, owning):
        if key is None:
            continue
        fields = field_ranges.get(key, {})
        for fname in _KIND_FALLBACK_FIELDS:
            if ("field", fname) in fields:
                return fields[("field", fname)]
        if key in node_ranges:
            return node_ranges[key]
    return None


def _resolve_range(
    span: Optional[SourceSpan], node_id: Optional[str], loaded, text: str
) -> Optional[SourceRange]:
    """Resolve a run failure to a source `SourceRange` for the single-frame box, or `None`.

    A `kind=field` locator (e.g. an `output:` coercion mismatch) resolves against the failing
    node's `field_ranges` (honoring the `output`/`outputs` alias); otherwise the node's own
    range via `_node_range` (kind fallback, else header). `None` when nothing resolves, so the
    caller degrades to a plain message line."""
    node_ranges, field_ranges = _top_ranges(loaded, text)
    if span is not None and span.kind == "field" and span.node is not None:
        seg = _last_segment(span.node) or span.node
        rng = _field_range(field_ranges.get(seg, {}), span.key)
        if rng is not None:
            return rng
    if node_id:
        return _node_range(node_id, node_ranges, field_ranges)
    return None


def _strip_clone_suffix(seg: str) -> str:
    """Drop a runtime clone suffix from one id segment, mapping it back to its authored name:
    a map element's `#<n>` (`gate#0` -> `gate`) or a loop driver clone's `~<k>` (`polish~5` ->
    `polish`). Both are runtime-only decorations the parser never indexes."""
    return seg.split("#", 1)[0].split("~", 1)[0]


def _last_segment(node_id: Optional[str]) -> Optional[str]:
    """The final `/`-segment of a (possibly namespaced) id, with any runtime `#<n>`/`~<k>`
    suffix stripped (`gate#0/approve` -> `approve`, `polish~5` -> `polish`). `None` for a
    `None` id."""
    if node_id is None:
        return None
    return _strip_clone_suffix(node_id.split("/")[-1])


def _walk_call_frames(loaded, node_id: str, top_text: str, top_label: str,
                      span: Optional[SourceSpan]):
    """Walk a namespaced runtime node id through the baked IR, one source frame per segment.

    A runtime failure inside a called child surfaces with a NAMESPACED id (`gate/approve` =
    node `approve` inside the child the top-level `call` node `gate` invokes; `gate#0/inner`
    for a map element). This walks the id segment-by-segment — frame 0 the top flow, then each
    child a `call`/`map` descends into (a `defs:` callable or an external `uses:` file via the
    node's render-only `child_source` `SourceFrame`) — down to the failing leaf.

    Returns `list[(label, text, range)]`, most-recent-call-last (ready to render stacked). Each
    frame's `label` names WHERE it lives: a top/external file frame is the filename alone; a
    `defs:` child frame is filename-qualified (`<file> defs:<name>`) since its nodes physically
    live in that file. A segment with NO recorded range (a synthetic `__start__`/`__end__`/`__ask…`
    runtime segment, or a collapsed compact single-node def) is SKIPPED — its frame is dropped, the
    frames collected so far kept. Descent stops at the first segment that is not a call/map
    with a baked child (the leaf), or when a segment can't be resolved. For the LAST segment a
    `kind=field` locator (e.g. an `output:` coercion) is preferred, else the node's kind
    fallback field (e.g. a code node's `code:`), else its body range (the node header). Never
    raises — the caller treats any failure as "no multi-frame stack" and falls back to the
    single-frame path."""
    segments = [_strip_clone_suffix(seg) for seg in node_id.split("/")]
    nodes = loaded.compiled.nodes
    # Both the top frame and each descended child read ranges off a render-only `SourceFrame`:
    # frame 0 from the top flow's own `loaded.source`, then each child from its `child_source`.
    # (The baked call/loop/map nodes a walk descends THROUGH carry no `.range` — only leaf kind
    # builds stamp it — so the SourceFrame, populated by a span parse, is the uniform source.)
    top_src = getattr(loaded, "source", None)
    node_ranges: dict = dict(top_src.node_ranges) if top_src is not None else {}
    field_ranges: dict = dict(top_src.field_ranges) if top_src is not None else {}
    text, label = top_text, top_label
    current_file = top_label  # the display filename of the file the current frame lives in
    frames: list[tuple[str, str, SourceRange]] = []
    span_leaf = _last_segment(span.node) if span is not None else None

    for i, seg in enumerate(segments):
        is_last = i == len(segments) - 1
        rng: Optional[SourceRange] = None
        fields = field_ranges.get(seg, {})
        if is_last:
            # precise field locator (e.g. `output:` for a coercion / type mismatch)
            if span is not None and span.kind == "field" and span_leaf == seg:
                rng = _field_range(fields, span.key)
            if rng is None:  # else the node's kind fallback (e.g. a code node's `code:`)
                for fname in _KIND_FALLBACK_FIELDS:
                    if ("field", fname) in fields:
                        rng = fields[("field", fname)]
                        break
        if rng is None:
            rng = node_ranges.get(seg)  # the node body range (None -> synthetic/unauthored: skip)
        if rng is not None:
            frames.append((label, text, rng))

        # Descend into the child this segment calls, if any (else the leaf is reached).
        node = nodes.get(seg)
        if node is None:
            break
        child = getattr(node, "child", None)
        src = getattr(node, "child_source", None)
        if child is None or src is None:
            break
        nodes = child.nodes
        text = src.text
        node_ranges, field_ranges = src.node_ranges, src.field_ranges
        # A `defs:` child physically lives in the CURRENT file -> qualify its frame title with
        # that filename (`<file> defs:<name>`); an external `uses:` child is a different file,
        # so its own filename becomes the current file and stands alone as the title.
        if src.label.startswith("defs:"):
            label = f"{current_file} {src.label}"
        else:
            current_file = src.label
            label = src.label
    return frames


def _render_frame_stack(frames, message: str) -> None:
    """Render a multi-frame call traceback (most-recent-call-last) + the message below.

    Mirrors a Python traceback: a header, then one boxed `.yaml` frame per descended level
    (outermost `call` first, the failing leaf last), then the error message in red. Each
    frame carries a `SourceRange`; the box anchors on its `start_line` (one line per frame,
    like a Python traceback), which for a call-chain node is its header line and for the failing
    leaf its precise field / kind-fallback line."""
    err_console.print(Text("call traceback (most recent call last):", style="red bold"))
    for label, ftext, rng in frames:
        _render_frame_box(ftext, [rng.start_line], label)
    err_console.print(Text(message, style="red"))


def render_diagnostic(diag: Diagnostic, source_text: str, title: str = "") -> None:
    """Render a `Diagnostic` as the CLI's one located error frame (the Step-8 funnel).

    Boxes `diag.primary` — the lines it COVERS highlighted in `source_text` — and prints
    `diag.message` below in red, the same boxed frame a compile/run error used to get, now fed
    by a resolved `SourceRange` rather than a line lookup. The range is half-open, so which
    lines it covers is the range's own business (`SourceRange.covered_lines`) — reading
    `end_line` here would highlight one line too many for a block construct, whose end mark
    lands on the next token's line. When `diag.notes` is non-empty (the outer call-traceback
    frames of a nested `call`/`map` failure, innermost last) they render as a stacked "call
    traceback" header ABOVE the primary, mirroring a Python traceback. Falls back to a plain
    `title: message` line when there is no primary range (or it falls outside the source).
    Writes to `err_console`; `title` labels the primary box (the flow filename)."""
    if diag.notes:
        err_console.print(Text("call traceback (most recent call last):", style="red bold"))
        for note in diag.notes:
            err_console.print(Text(f"  {note}", style="dim"))
    primary = diag.primary
    marks = primary.covered_lines(source_text) if primary else []
    _render_source_frame(source_text, marks, title, diag.message)


def _render_traceback(tb: str) -> None:
    """Box the captured Python traceback below the error frame (only under `--engine-trace`).

    The default error display stays terse (frame + message); this is the opt-in stack for
    debugging a node's own code (a code/tool raise) or the engine."""
    err_console.print(
        Panel(
            Text(tb.rstrip(), style="dim"),
            title="python traceback",
            title_align="left",
            border_style="red",
        )
    )


def _render_run_error(
    result: RunResult, flow: Path, text: str, loaded=None, engine_trace: bool = False
) -> None:
    """Print a runtime failure as a located `.yaml` frame at its PRECISE originating line.

    The failure's `SourceSpan` locator (from the last `NodeFailed`, or the flow-level
    `RunResult.locator` when no node is behind it) names exactly where the run broke — an
    input binding, an assert expr, an input decl, an `output:` coercion.

    When the failing node id is RUNTIME-NAMESPACED (a node inside a called child, e.g.
    `gate/approve`) and the loaded IR is available, the error renders a **call traceback**: a
    boxed frame per descended level — the top-level `call` node, then each child (`defs:` or
    external `uses:` file) — down to the failing leaf, most-recent-call-last
    (`_walk_call_frames`). With fewer than two frames it falls back to the single-frame box,
    whose `SourceRange` is resolved by, in order:

    1. the `NodeFailed`'s resolved `range` (the funnel's key->range twin of its locator);
    2. else the flow-level `RunResult.range` (a node-less boundary input-decl / assert failure);
    3. else `_resolve_range`: a `kind=field` locator's field range, else the node's own range
       (its kind fallback `code:` range, else the header), resolving a namespaced/clone id to
       its owning top-level node;
    4. else a plain `run <status>: <message>` line (no frame).

    When `engine_trace` is set and a Python traceback was captured (a code/tool/agent raise),
    it is boxed below for debugging."""
    failed = [e for e in result.events if isinstance(e, NodeFailed)]
    nf = failed[-1] if failed else None
    span = nf.locator if nf is not None else getattr(result, "locator", None)
    node_id = nf.node_id if nf is not None else (span.node if span is not None else None)
    message = result.error or "(no detail)"
    tb = getattr(result, "traceback", None)

    # Multi-frame call traceback: when the failing id is namespaced (a node inside a called
    # child) and we have the loaded IR, walk into the child(ren) and box a frame per level.
    # The walk must never break error rendering -> any failure degrades to the single frame.
    frames: list = []
    if loaded is not None and node_id and "/" in node_id:
        try:
            frames = _walk_call_frames(loaded, node_id, text, flow.name, span)
        except Exception:
            frames = []
    if len(frames) >= 2:
        _render_frame_stack(frames, message)
        if engine_trace and tb:
            _render_traceback(tb)
        return

    # Span-aware single frame (Step 8): a concrete `SourceRange` the funnel already resolved —
    # on the failing `NodeFailed` (a node error), else on the flow-level `RunResult` (a node-less
    # boundary input-decl / assert failure). Box it directly.
    rng = nf.range if nf is not None else None
    if rng is None:
        rng = getattr(result, "range", None)
    # else resolve from the locator/node against the source ranges (kind=field precision, then
    # the node's kind fallback / header), recomputed from the loaded IR or the text.
    if rng is None:
        rng = _resolve_range(span, node_id, loaded, text)

    if rng is None:
        err_console.print(f"[red]run {result.status}: {message}[/red]")
    else:
        render_diagnostic(Diagnostic("error", message, rng), text, flow.name)
    if engine_trace and tb:
        _render_traceback(tb)


def _render_notes(err: LoadError) -> None:
    """Print the error's "why" legend (`LoadError.notes`) under the message, if any.

    Each note is an explanatory line the source frame can't show — e.g. the dependency
    edges that close a cycle — rendered indented and dim so it reads as context, not a
    second error.
    """
    for note in err.notes or []:
        err_console.print(Text(f"  ↳ {note}", style="yellow"))


class _ProgressReporter:
    """Render per-node progress for `ac run` as the engine streams node events.

    A node shows a live spinner while running, then is rewritten in place as a green
    `✓ <node>` on success or a red `✗ <node>` (with the error on the next line) on
    failure. With `verbose`, each node's output is printed under its check.

    `on_event` is invoked on a single thread, but several nodes can be *running* at the
    same time (a fan-out), so `_running` is a map and the live region shows one spinner
    per member. On a real terminal the spinners animate in a Rich `Live` region and the
    finished lines scroll above it; off a terminal (a pipe, CI, the test runner) there
    is no spinner — only the final `✓`/`✗` line per node is printed.

    Progress goes to stderr so the flow's actual output on stdout stays pipeable.
    """

    def __init__(self, console: Console, verbose: bool) -> None:
        self._console = console
        self._verbose = verbose
        # node_id -> its live Spinner, for every node currently running.
        self._running: Dict[str, Spinner] = {}
        self._live: Any = None  # rich.live.Live while active on a terminal, else None

    @property
    def is_live(self) -> bool:
        return self._live is not None

    def start(self) -> None:
        """Begin a live spinner region (terminal only). Idempotent."""
        if self._live is not None or not self._console.is_terminal:
            return
        from rich.live import Live

        self._live = Live(console=self._console, refresh_per_second=12, transient=True)
        self._live.start()
        self._refresh()

    def stop(self) -> None:
        """Tear down the live region (e.g. before a questionary prompt). Idempotent."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _refresh(self) -> None:
        """Redraw the live region with one spinner line per running node."""
        if self._live is not None:
            self._live.update(Group(*self._running.values()))

    def _emit(self, renderable: Any) -> None:
        """Print a permanent line; under a live region it scrolls above the spinners."""
        self._console.print(renderable)

    def handle(self, event: Any) -> None:
        """Fold one engine event into the display. Boundary nodes are ignored."""
        node_id = getattr(event, "node_id", None)
        # Skip synthesized boundary nodes at any depth: the top-level `__start__`/`__end__`
        # AND the per-iteration/per-element namespaced ones (`polish#0/__start__`), which carry
        # no authored meaning and only clutter the progress stream.
        if _last_segment(node_id) in (START_ID, END_ID):
            return
        name = type(event).__name__
        if name == "NodeStarted":
            self._running[node_id] = Spinner("dots", text=Text(node_id, style="cyan"))
            self._refresh()
        elif name == "NodeSucceeded":
            self._running.pop(node_id, None)
            self._emit(Text(f"✓ {node_id}", style="green"))
            if self._verbose:
                self._emit_output(event.output)
            self._refresh()
        elif name == "NodeRouted":
            # A router (CASE) produces no value — it selected an out-edge handle.
            self._running.pop(node_id, None)
            self._emit(Text(f"✓ {node_id} → {event.handle}", style="green"))
            self._refresh()
        elif name == "NodeExpanded":
            # A spawner (CALL/MAP/LOOP) grew the graph instead of returning a value: a
            # loop driver continuing to the next iteration, a call/map splicing its child. It has
            # no terminal of its own (its value is committed by the spliced subgraph's terminal),
            # so just clear its spinner — no ✓ line. Without this the driver clones (`polish`,
            # `polish~1`, …) would spin forever, never leaving `_running`.
            self._running.pop(node_id, None)
            self._refresh()
        elif name == "NodeFailed":
            self._running.pop(node_id, None)
            self._emit(Text(f"✗ {node_id}", style="red bold"))
            self._emit(Text(f"    {event.error}", style="red"))
            self._refresh()

    def _emit_output(self, output: Any) -> None:
        """Print a node's produced value, indented under its check (verbose only)."""
        body = output if isinstance(output, str) else repr(output)
        for line in body.splitlines() or [""]:
            self._emit(Text(f"    {line}", style="dim"))


class _ControllerStopEvent(threading.Event):
    """One transient stop bit coupled to the broker's prompt-wakeup operation.

    The runtime knows only the ordinary ``threading.Event`` interface. Coupling
    ``set`` here keeps terminal ownership in the CLI while guaranteeing that a
    runtime failure wakes an active or queued ``ask_user`` before worker join.
    """

    def __init__(self, cancel_prompts: Callable[[], None]) -> None:
        super().__init__()
        self._cancel_prompts = cancel_prompts
        self._linked_lock = threading.Lock()
        self._linked_children: set[threading.Event] = set()

    def set(self) -> None:
        """Signal this lifecycle, its active child runs, and broker prompt waiters."""
        super().set()
        with self._linked_lock:
            linked_children = tuple(self._linked_children)
        for child in linked_children:
            child.set()
        self._cancel_prompts()

    @contextmanager
    def linked_child(self):
        """Yield a one-way child event for one nested engine lifecycle.

        Parent cancellation sets the child. A nested failure sets only the child,
        so a failed flow invoked as a tool can return its ordinary failure result
        without cancelling the enclosing chat run.
        """
        child = threading.Event()
        with self._linked_lock:
            if self.is_set():
                child.set()
            else:
                self._linked_children.add(child)
        try:
            yield child
        finally:
            with self._linked_lock:
                self._linked_children.discard(child)


def _run_with_terminal_controller(
    controller: Callable[
        [
            threading.Event,
            Optional[Callable[[Any], None]],
            Callable[..., Any],
            Callable[[Callable[[], None]], None],
        ],
        RunResult,
    ],
    *,
    prompt_handler: Callable[[PromptRequest], Any],
    reporter: Optional[_ProgressReporter],
    eof_value: Any = _DEFAULT_BROKER_EOF,
) -> RunResult:
    """Run one complete CLI lifecycle behind a main-thread terminal broker.

    ``controller`` owns one run attempt. It receives the shared stop event, a
    brokered observer, the installed
    ``ask_user`` handler, and the broker's render enqueue operation. The main
    thread alone serves the FIFO and joins the controller before registry cleanup.
    ``eof_value`` optionally normalizes prompt EOF to a caller-owned value; omitted
    EOF keeps the broker's exception policy.
    """

    broker: TerminalBroker
    stop_event = _ControllerStopEvent(lambda: broker.cancel_all())

    def prompt_on_main(request: PromptRequest) -> Any:
        """Suspend Rich Live while the main thread owns one terminal prompt."""
        if reporter is not None:
            reporter.stop()
        try:
            return prompt_handler(request)
        finally:
            if reporter is not None and not stop_event.is_set() and not request.cancelled:
                reporter.start()

    broker_kwargs: dict[str, Any] = {"stop_event": stop_event}
    if eof_value is not _DEFAULT_BROKER_EOF:
        broker_kwargs["eof_value"] = eof_value
    broker = TerminalBroker(prompt_on_main, **broker_kwargs)

    def ask_user(**kwargs: Any) -> Any:
        """Translate broker lifecycle cancellation into the runtime stop signal."""
        try:
            return broker.ask_user(**kwargs)
        except _HostStop:
            raise
        except EOFError as exc:
            raise _HostStop("terminal interaction stopped") from exc
        except CancelledError as exc:
            raise _HostStop("terminal interaction stopped", primary=False) from exc

    on_event: Optional[Callable[[Any], None]]
    if reporter is None:
        on_event = None
    else:
        on_event = lambda event: broker.render(partial(reporter.handle, event))

    controller_finished = threading.Event()
    controller_signal_mask: Any = None

    def drive() -> None:
        """Publish exactly one result or exception from the controller thread."""
        try:
            # A new POSIX thread inherits the launcher's temporary SIGINT block. Restore the
            # caller's original mask before this controller can spawn engine workers/processes.
            if controller_signal_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, controller_signal_mask)
            result = controller(stop_event, on_event, ask_user, broker.render)
        except BaseException as exc:
            stop_event.set()
            broker.controller_complete(exception=exc)
        else:
            broker.controller_complete(result=result)
        finally:
            # The main thread waits on this explicit target-lifecycle bit before joining.
            # A SIGINT-interrupted Thread.join can report a still-running thread as stopped.
            controller_finished.set()

    controller_thread = threading.Thread(target=drive, name="ac-controller")
    completion = None
    host_error: Optional[tuple[BaseException, Any]] = None
    started = False

    def remember_host_error(exc: BaseException) -> None:
        """Keep the first host cause while repeated interrupts only reinforce stop."""
        nonlocal host_error
        if host_error is None:
            host_error = (exc, exc.__traceback__)
        stop_event.set()

    def start_controller() -> None:
        """Launch with SIGINT deferred until ``started`` records the OS thread.

        CPython's ``Thread.start`` can be interrupted after creating the OS thread but before
        publishing ``ident``. Blocking SIGINT across that short POSIX launch handshake closes
        the otherwise unobservable join-before-registry-cleanup race.
        """
        nonlocal controller_signal_mask, started
        pthread_sigmask = getattr(signal, "pthread_sigmask", None)
        if threading.current_thread() is threading.main_thread() and pthread_sigmask is not None:
            previous_mask = pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
            controller_signal_mask = previous_mask
            try:
                controller_thread.start()
                started = True
            finally:
                # A pending SIGINT is delivered here, after `started` became authoritative.
                pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            return
        controller_thread.start()
        started = True

    with install_ask_user(ask_user):
        if reporter is not None:
            reporter.start()
        try:
            start_controller()
            completion = broker.serve()
        except BaseException as exc:
            remember_host_error(exc)
        finally:
            if started:
                # Keep registry ownership until the controller target has finished. Repeated
                # Ctrl-C during a slow cooperative drain is deferred; it must not interrupt
                # cleanup or replace the first host cause.
                while not controller_finished.is_set():
                    try:
                        controller_finished.wait()
                    except BaseException as exc:
                        remember_host_error(exc)
                while True:
                    try:
                        controller_thread.join()
                        break
                    except BaseException as exc:
                        remember_host_error(exc)
            if reporter is not None:
                try:
                    reporter.stop()
                except BaseException as exc:
                    remember_host_error(exc)

    if host_error is not None:
        error, tb = host_error
        raise error.with_traceback(tb)
    if completion is None:
        raise RuntimeError("terminal controller exited without a completion")
    if completion.exception is not None:
        if isinstance(completion.exception, _HostStop):
            cause = completion.exception.__cause__
            if isinstance(cause, (CancelledError, EOFError)):
                raise cause
            raise RuntimeError("terminal controller stopped") from completion.exception
        raise completion.exception
    return completion.result


def _parse_kv(pairs: List[str]) -> Dict[str, Any]:
    """Parse repeated `--input k=v` flags into a dict (values stay strings; the engine
    coerces them against each input's declared type at the run boundary)."""
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(f"--input must be k=v (got {pair!r})")
        key, value = pair.split("=", 1)
        out[key.strip()] = value
    return out


# Example value for the scalars whose accepted string form isn't self-evident from the
# type name — the ISO-8601 date/datetime, which the engine parses with
# `date.fromisoformat` / `datetime.fromisoformat` (a bare date is not a valid datetime).
_FORMAT_EXAMPLE: Dict[Any, str] = {
    ValueKind.DATE: "2026-05-21",
    ValueKind.DATETIME: "2026-05-21T14:30",
}


def _format_hint(typ: Any) -> Optional[str]:
    """An `e.g. <value>` example for an input whose string format isn't obvious, or
    `None` when none is needed. Keyed off the typ's scalar `kind`, so it fires for
    `Optional[date]` too (the resolved typ stays `DATE`, just nullable)."""
    example = _FORMAT_EXAMPLE.get(getattr(typ, "kind", None))
    return f"e.g. {example}" if example else None


def _input_label(decl: Any) -> str:
    """The questionary prompt label for one declared input.

    Carries the input's name, declared `type`, a required (`*`) / `optional` mark,
    any default, and — for an ISO-8601 scalar with no default — an example value (a
    default already shows the format, so the example is dropped there). So an author at
    the prompt sees what is expected without reading the `.yaml`. Examples:
        `topic (str) *`                                      (required)
        `as_of (Optional[date]) [optional] e.g. 2026-05-21`  (date: example shown)
        `as_of (Optional[date]) [default: 2026-05-21]`       (default shows the format)
        `window (int) [default: 30]`                         (optional, has a default)
    """
    parts = [decl.name]
    if getattr(decl, "type_str", None):
        parts.append(f"({decl.type_str})")
    hint = _format_hint(decl.type)
    if decl.required:
        parts.append("*")
        if hint:
            parts.append(hint)
    elif decl.default is not None:
        # The default value itself shows the accepted format, so skip the example.
        parts.append(f"[default: {decl.default}]")
    else:
        parts.append("[optional]")
        if hint:
            parts.append(hint)
    return " ".join(parts)


def _flow_banner(
    name: Optional[str], description: Optional[str], version: Optional[str] = None
) -> Optional[Panel]:
    """The "what am I running" banner printed at the start of a run (to stderr).

    A boxed panel of the flow's identity — `name`/`description`/`version` — so the
    author sees what the run is before inputs/progress. Returns `None` when there is no
    metadata at all (nothing to show)."""
    if not (name or description or version):
        return None
    body = Text()
    body.append("Running flow: ", style="bold")
    body.append(name or "(unnamed)")
    if version:
        body.append(f"  (version: {version})", style="dim")
    if description:
        body.append("\nDescription: ", style="bold")
        body.append(description)
    return Panel(body, border_style="cyan", expand=False)


def _prompt_missing(decls: List[Any], have: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Prompt for each declared input not already supplied. Returns the gathered values,
    or None if the user cancels (Ctrl-C / Esc). The widget follows the declared type: a
    boolean is a confirm, a `Literal[...]` enum is a select, everything else is free text.

    Each prompt's label carries the input's type + a required/optional mark + default
    (see `_input_label`)."""
    import questionary

    style = _q_style()
    gathered: Dict[str, Any] = {}
    for decl in decls:
        if decl.name in have:
            continue
        label = _input_label(decl)
        typ = decl.type
        if typ.kind == ValueKind.BOOLEAN:
            value = questionary.confirm(
                label, default=bool(decl.default), qmark="?", style=style
            ).ask()
        elif typ.tags:  # a Literal[...] enum
            value = questionary.select(
                label, choices=sorted(typ.tags), qmark="?", style=style
            ).ask()
        else:
            default = "" if decl.default is None else str(decl.default)
            value = questionary.text(label, default=default, qmark="?", style=style).ask()

        if value is None:  # Ctrl-C / Esc
            return None
        if isinstance(value, str) and value.strip() == "" and not decl.required:
            continue
        gathered[decl.name] = value
    return gathered


# sentinel choice appended to every question's option list — selecting it routes to a
# free-text prompt so the human can answer outside the offered labels. The arrow marks it
# visually as the escape hatch rather than a literal option.
_OTHER = "✎ Other (write your own)"

# cached questionary Style so every interactive prompt (missing inputs + human_input
# questions) shares one palette: a cyan accent for the marker/pointer/answer, a green tick
# for multi-select, dimmed instructions. Built lazily because `questionary` is imported
# inside the prompting functions (it pulls in prompt_toolkit, kept off the import path of
# non-interactive runs). `None` until first built.
_Q_STYLE: Any = None


def _q_style() -> Any:
    """Return the shared `questionary.Style`, building (and caching) it on first use."""
    global _Q_STYLE
    if _Q_STYLE is None:
        import questionary

        _Q_STYLE = questionary.Style(
            [
                ("qmark", "fg:#00afff bold"),
                ("question", "bold"),
                ("answer", "fg:#00afff bold"),
                ("pointer", "fg:#00afff bold"),
                ("highlighted", "fg:#00afff bold"),
                ("selected", "fg:#5faf5f"),
                ("separator", "fg:#6c6c6c"),
                ("instruction", "fg:#6c6c6c italic"),
                ("text", ""),
            ]
        )
    return _Q_STYLE


def assemble_question_answers(questions, ask):
    """Build a human_input answer record from a question list + an `ask` callable.

    `questions` is the host request's question list (each a dict {question, header, options,
    multi_select}). `ask(question) -> answer` is the per-question elicitation (the real
    one wraps questionary; tests pass a scripted callable). Returns a record keyed by
    each question's `header`: a single label `str` for a single-select question, a
    `list[str]` for a `multi_select` question (the `ask` callable returns whichever form).
    """
    record = {}
    for q in questions:
        record[q["header"]] = ask(q)
    return record


def _exit_questionary_prompt(application: Any) -> None:
    """Wake one active prompt-toolkit application for a controller stop.

    ``Application.exit`` must execute on the application's asyncio loop. The
    broker's cancellation callback schedules this function thread-safely; a
    late callback may race with normal application shutdown, so an already
    closed prompt is deliberately a no-op.
    """
    try:
        application.exit(
            exception=_HostStop("terminal interaction stopped", primary=False)
        )
    except BaseException:
        return


def _ask_widget(widget: Any, request: Optional[PromptRequest] = None) -> Any:
    """Run one questionary widget, optionally coupled to broker cancellation.

    The ordinary pre-run input prompts have no broker request and keep
    questionary's legacy ``ask`` behavior. Brokered prompts use the underlying
    prompt-toolkit application so Ctrl-C remains visible and an engine stop can
    wake the main-thread widget from a controller worker.
    """
    if request is None:
        return widget.ask()
    if widget.should_skip_question:
        return widget.default

    application = widget.application
    loop_holder: list[Any] = []

    def cancel() -> None:
        """Schedule prompt exit from any engine/controller worker thread."""
        if not loop_holder:
            return
        loop = loop_holder[0]
        try:
            loop.call_soon_threadsafe(_exit_questionary_prompt, application)
        except RuntimeError:
            # The prompt loop won the race and has already closed.
            return

    def pre_run() -> None:
        """Publish the active prompt loop before accepting controller cancellation."""
        loop_holder.append(asyncio.get_running_loop())
        request.set_cancel_callback(cancel)

    try:
        return application.run(pre_run=pre_run)
    finally:
        request.set_cancel_callback(None)


def _ask_question(question, index=None, total=None, request=None):
    """Elicit one question via questionary; return its answer (str, list[str], or None).

    `index`/`total` (1-based, optional) drive a styled "Question N of M · <header>" rule
    printed above the widget so a multi-question request reads as a numbered sequence; omit
    them (the default) for a bare single prompt. Decorated choices ("label — description")
    are rendered for display but mapped back to the bare `label` so the returned value is
    always the bare label, never the hint string. An "Other" escape is always offered;
    choosing it routes to a free-text prompt. Returns `None` on cancel (questionary
    `.ask()` yields None on Ctrl-C/Esc) for the caller to interpret."""
    import questionary

    text = question["question"]
    options = question.get("options") or []

    # A styled separator naming the question's position + answer key, so the human can map
    # each prompt back to the record it fills (the answer is keyed by `header`).
    if index is not None and total is not None:
        header = question.get("header") or ""
        title = f"[bold]Question {index} of {total}[/bold]"
        if header:
            title += f" [dim]· {header}[/dim]"
        err_console.print()
        err_console.print(Rule(title, style="cyan", align="left"))

    style = _q_style()

    # free-text-only question: no choices, just elicit a string.
    if not options:
        return _ask_widget(
            questionary.text(text, qmark="?", style=style),
            request,
        )

    # map each displayed choice back to its bare label so the return value is the label.
    # The em-dash keeps the label visually distinct from its hint in the choice list.
    display_to_label = {}
    choices = []
    for opt in options:
        label = opt["label"]
        desc = opt.get("description") or ""
        display = f"{label}  —  {desc}" if desc else label
        display_to_label[display] = label
        choices.append(display)
    choices.append(_OTHER)

    if question.get("multi_select"):
        picked = _ask_widget(
            questionary.checkbox(text, choices=choices, qmark="?", style=style),
            request,
        )
        if picked is None:  # cancelled
            return None
        labels = []
        for chosen in picked:
            if chosen == _OTHER:
                other = _ask_widget(
                    questionary.text("Your answer", qmark="✎", style=style),
                    request,
                )
                if other is None:
                    return None
                labels.append(other)
            else:
                labels.append(display_to_label[chosen])
        return labels

    chosen = _ask_widget(
        questionary.select(
            text, choices=choices, qmark="?", style=style, use_indicator=True
        ),
        request,
    )
    if chosen is None:  # cancelled
        return None
    if chosen == _OTHER:
        return _ask_widget(
            questionary.text("Your answer", qmark="✎", style=style),
            request,
        )
    return display_to_label[chosen]


def _prompt_request(request: PromptRequest) -> Any:
    """Render and answer one broker request on the process main thread."""
    import questionary

    if request.questions:
        title = request.title or request.node_id or "this step"
        count = len(request.questions)
        intro = Text()
        intro.append("Your input is needed for ", style="bold")
        intro.append(str(title), style="bold cyan")
        if request.question:
            intro.append("\n")
            intro.append(request.question)
        plural = "question" if count == 1 else "questions"
        intro.append(f"\n\n{count} {plural} to answer.", style="dim")
        err_console.print(
            Panel(intro, border_style="cyan", expand=False, title="human input")
        )
        sequence = itertools.count(1)
        return assemble_question_answers(
            request.questions,
            lambda question: _ask_question(
                question,
                index=next(sequence),
                total=count,
                request=request,
            ),
        )

    label = request.question or f"input for {request.node_id}"
    return _ask_widget(
        questionary.text(label, qmark="?", style=_q_style()),
        request,
    )


def _agent_providers(loaded, cli_cfg: Dict[str, Any]) -> List[str]:
    """Collect the distinct effective LLM providers of every agent node in `loaded`.

    Snapshots the process provider/model defaults once each, then resolves the
    `llm_config` cascade (with `cli_cfg` as the outermost gap-fill layer) on a throwaway
    copy of the compiled flow. The returned effective map covers the whole static call
    tree, and every provider in it is concrete.

    Args:
        loaded (`ComposedFlow`):
            The compiled flow whose agent providers to enumerate.
        cli_cfg (`dict[str, Any]`):
            The `--provider`/`--model` CLI layer (may be empty) — the cascade's
            outermost authored gap-fill layer.

    Returns:
        `List[str]`:
            The distinct provider names, in first-seen order.
    """
    import copy

    from agent_composer._settings import default_llm_model, default_llm_provider
    from agent_composer.compile.llm_cascade import resolve_llm_cascade

    probe = copy.deepcopy(loaded.compiled)
    defaults = {
        "provider": default_llm_provider().lower(),
        "model": default_llm_model(),
    }
    effective = resolve_llm_cascade(probe, dict(cli_cfg), defaults=defaults)

    seen: List[str] = []
    for config in effective.values():
        provider = config["provider"]
        if provider not in seen:
            seen.append(provider)
    return seen


def _ensure_provider_keys(loaded, cli_cfg: Dict[str, Any]) -> None:
    """Prompt for any API key the flow's agents will need, before the run starts.

    Turns a cryptic mid-run "missing key" API failure into an upfront interactive
    prompt. Keyless (`ollama`) and unknown providers are skipped inside
    [`ensure_api_key`][agent_composer.cli.utils.ensure_api_key]; a required-but-missing
    key with no TTY re-raises the `RuntimeError` for `run` to surface cleanly.

    Args:
        loaded (`ComposedFlow`):
            The compiled flow whose agents' providers determine the keys to check.
        cli_cfg (`dict[str, Any]`):
            The `--provider`/`--model` CLI layer, threaded into provider resolution.
    """
    from agent_composer.cli.utils import ensure_api_key

    for provider in _agent_providers(loaded, cli_cfg):
        ensure_api_key(provider)


def _validate_run_log_options(
    *,
    input_values: List[str],
    inputs: Optional[Path],
    provider: Optional[str],
    model: Optional[str],
    log: Optional[Path],
    resume: Optional[Path],
) -> None:
    """Reject fresh-only data on recovery and existing fresh-log paths.

    This check runs before input files are read, prompts are shown, or provider
    configuration is resolved.

    Args:
        input_values (`List[str]`):
            Repeated fresh ``--input`` assignments, empty when omitted.
        inputs (`Path | None`):
            Fresh ``--inputs`` JSON path, or ``None``.
        provider (`str | None`):
            Fresh provider cascade value, or ``None``.
        model (`str | None`):
            Fresh model cascade value, or ``None``.
        log (`Path | None`):
            Explicit new authority path, or ``None`` for the default.
        resume (`Path | None`):
            Existing authority path to recover, or ``None`` for a fresh run.

    Raises:
        `typer.BadParameter`:
            Recovery is combined with fresh-only data, or ``--log`` already
            exists and therefore cannot be exclusively created.
    """
    if resume is not None:
        conflicts = []
        if input_values:
            conflicts.append("--input")
        if inputs is not None:
            conflicts.append("--inputs")
        if provider is not None:
            conflicts.append("--provider")
        if model is not None:
            conflicts.append("--model")
        if log is not None:
            conflicts.append("--log")
        if conflicts:
            raise typer.BadParameter(
                "--resume cannot be combined with " + ", ".join(conflicts)
            )
        return

    if log is not None and os.path.lexists(log):
        raise typer.BadParameter(
            f"--log path already exists: {log}",
            param_hint="--log",
        )


def run(
    flow: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Path to a flow .yaml"),
    input: List[str] = typer.Option(  # noqa: A002 - matches the user-facing flag name
        None, "--input", "-i", help="An input as k=v (repeatable)."
    ),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", exists=True, dir_okay=False, readable=True, help="A JSON file of inputs."
    ),
    log: Optional[Path] = typer.Option(
        None,
        "--log",
        dir_okay=False,
        help="Create a fresh durable run log at PATH instead of the default location.",
    ),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        exists=True,
        dir_okay=False,
        readable=True,
        writable=True,
        help="Recover the run recorded at PATH; fresh input, provider/model, and "
        "log flags are forbidden.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress per-node progress."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Also print each node's output as it finishes."
    ),
    num_workers: Optional[int] = typer.Option(
        None,
        "--num-workers",
        "-w",
        min=0,
        help="Worker pool size. Omitted means 0 for a fresh run and preserves "
        "the recorded value on resume.",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Override the LLM provider for agents that set none (cascade)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Override the LLM model for agents that set none (cascade)."
    ),
    engine_trace: bool = typer.Option(
        False,
        "--engine-trace",
        help="On a compile, run-log, or runtime node failure, also print the engine "
        "Python traceback; by default only the concise or located error is shown.",
    ),
) -> None:
    """Start or recover a durable flow run and print its terminal output."""
    _validate_run_log_options(
        input_values=input or [],
        inputs=inputs,
        provider=provider,
        model=model,
        log=log,
        resume=resume,
    )
    _ensure_cwd_importable()  # so `code: pkg.mod:fn` refs to repo-local packages resolve
    text = flow.read_text()
    try:
        loaded = load_flow(text, search_paths=[flow.parent])
    except LoadError as err:
        # An author's flow failed to compile: point at WHERE in the `.yaml` it broke, not at
        # the engine internals. `--engine-trace` adds the Python traceback for engine debugging.
        _render_load_error(err, flow, text)
        if engine_trace:
            err_console.print_exception()
        raise typer.Exit(code=1)

    # The "what am I running" banner (flow name/description/version) — stderr, like
    # progress; suppressed under --quiet.
    if not quiet:
        banner = _flow_banner(loaded.name, loaded.description, loaded.version)
        if banner is not None:
            err_console.print(banner)

    # `--quiet` silences progress entirely; otherwise the main-thread broker streams
    # node events to this reporter. `--verbose` adds each node's output.
    reporter = _ProgressReporter(err_console, verbose=verbose)

    if resume is not None:
        selected_log = resume
        _report_run_log(selected_log)

        def controller(
            stop_event: threading.Event,
            on_event: Optional[Callable[[Any], None]],
            _ask_user: Callable[..., Any],
            _render: Callable[[Callable[[], None]], None],
        ) -> RunResult:
            """Recover one run while the helper owns its terminal broker."""
            return _resume_flow_controlled(
                loaded,
                log=selected_log,
                num_workers=num_workers,
                on_event=on_event,
                stop_event=stop_event,
            )

    else:
        supplied: Dict[str, Any] = {}
        if inputs is not None:
            supplied.update(json.loads(inputs.read_text()))
        if input:
            supplied.update(_parse_kv(input))

        prompted = _prompt_missing(loaded.input, supplied)
        if prompted is None:
            err_console.print("[yellow]run cancelled[/yellow]")
            raise typer.Exit(code=1)
        supplied.update(prompted)

        # The CLI flags supply the OUTERMOST cascade layer (fill-the-gap), not a hard
        # override: an agent's own llm_config and a flow-level llm_config still win.
        cli_cfg = {
            key: value
            for key, value in {"provider": provider, "model": model}.items()
            if value
        }
        # Prompt for any required API key before the run creates its durable file.
        try:
            _ensure_provider_keys(loaded, cli_cfg)
        except RuntimeError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

        run_id = default_run_id()
        try:
            selected_log = log if log is not None else _default_run_log_path(run_id)
        except (OSError, TypeError, ValueError) as exc:
            _render_host_run_error("run", exc, engine_trace=engine_trace)
            raise typer.Exit(code=1)
        selected_workers = 0 if num_workers is None else num_workers
        _report_run_log(selected_log)

        def controller(
            stop_event: threading.Event,
            on_event: Optional[Callable[[Any], None]],
            _ask_user: Callable[..., Any],
            _render: Callable[[Callable[[], None]], None],
        ) -> RunResult:
            """Drive one fresh run while the helper owns its terminal broker."""
            return _run_flow_controlled(
                loaded,
                supplied,
                log=selected_log,
                run_id=run_id,
                on_event=on_event,
                llm_config=cli_cfg or None,
                num_workers=selected_workers,
                stop_event=stop_event,
            )

    try:
        result = _run_with_terminal_controller(
            controller,
            prompt_handler=_prompt_request,
            reporter=None if quiet else reporter,
        )
    except KeyboardInterrupt:
        # The controller helper has already stopped progress, signalled the shared
        # event, woken every prompt, and joined all engine/controller workers.
        err_console.print("[yellow]cancelled by user[/yellow]")
        raise typer.Exit(code=130)
    except (CancelledError, EOFError):
        # A closed input stream or lifecycle cancellation is not Ctrl-C. The
        # controller boundary has already joined its workers; report the same
        # ordinary cancellation outcome used by pre-run prompts.
        err_console.print("[yellow]run cancelled[/yellow]")
        raise typer.Exit(code=1)
    except (OSError, TypeError, ValueError) as exc:
        action = "resume" if resume is not None else "run"
        _render_host_run_error(action, exc, engine_trace=engine_trace)
        raise typer.Exit(code=1)

    if result.status == "succeeded":
        out = result.output
        if isinstance(out, str) and out.strip():
            console.print(Markdown(out))
        else:
            console.print(out)
    else:
        # A node failure points at WHERE in the `.yaml` it raised (boxed frame), mirroring a
        # compile error; a node-less failure (assert/input-coercion) prints the plain message.
        _render_run_error(result, flow, text, loaded=loaded, engine_trace=engine_trace)
        raise typer.Exit(code=1)
