"""The Node base contract — the single most portable idea from graphon.

A node is a **pure function of its bound input record**: it implements
`run(inputs, **caps) -> NodeResult` and returns ONE of the closed sum
`Output | Route | Pause | Grow` (the node-result sum type) — **or**, for a streaming
kind, a generator that yields `StreamChunk` and then *returns* a `NodeResult`.
The node never receives the pool: the engine's `runtime.eval_node` seam binds its
inputs (the read boundary) and hands it a record. The one effectful kind that still
needs a narrow capability is a mapped `call` (`bind_item`, a keyword-only arg); every other kind
takes only `inputs`. Failure is **not** a variant — a failing node `raise`s and the
engine boundary turns it into `NodeFailed`. A returned `Pause` becomes one
`PauseRequested`; the engine delivers the answer as the parked leaf's `Output`
(deliver-as-Output — the node never re-runs). A streaming kind is a generator that
yields `StreamChunk` and *returns* its `NodeResult` (drained by `_drain_node_generator`).

Invariant: a node never writes the pool. It *describes* its one output value as
`Output(value)`; the engine performs the write under `node_id`. Keeps nodes pure.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

from agent_composer.expr import ExpressionError
from agent_composer.expr.expressions import evaluate_when_record
from agent_composer.nodes.binding import ParamDecl
from agent_composer.typesys.values import Type

if TYPE_CHECKING:
    # Import-cycle guard: `compile.model` imports `nodes.base`, so `Flow` is only pulled in for
    # type checking. `Grow.subgraph`/`replay_grow` annotate it as a forward-ref string at runtime.
    from agent_composer.compile.model import Flow
    from agent_composer.events import SourceRange


class NodeKind(str, Enum):
    """Closed vocabulary. Dispatch is an explicit `match`, never a registry."""

    AGENT = "agent"
    CODE = "code"
    MODEL = "model"
    TOOL = "tool"
    CASE = "case"
    HUMAN_INPUT = "human_input"  # suspend for a person
    WAIT = "wait"  # internal-only: suspend for an external poke (WATCH uses it)
    LOOP = "loop"  # internal-only: iterate a callable to a fixpoint (`('a -> 'a) -> 'a -> 'a`)
    START = "start"  # internal-only: loader-synthesized input boundary (parameter binding)
    END = "end"      # internal-only: loader-synthesized return boundary (record + list modes)
    CALL = "call"    # internal-only: consult another flow once (REF — `kind: call`)
    MAP = "map"      # internal-only: map a callable over a list (`kind: map` + `over:`)


# --- the node's return type is a closed sum ----------------------------------------
# A pure node returns ONE of these (or a generator that yields StreamChunk and returns one).


@dataclass(frozen=True)
class Output:
    """A produced value. The engine writes `value` into the pool under the node id.

    `commit_as` is the node-CHOSEN commit redirect: when set, the engine writes `value`
    under `commit_as` instead of the node's own id (and fires that target's out-edges).
    A `str` target id, or `None` (the common case) to commit under the node's own id.
    No `run()` sets it today — it is the roadmap seam for a node that names its own commit
    target; the engine-baked `Node.commit_as` covers the current subflow-terminal redirects.
    """

    value: Any = None
    commit_as: Optional[str] = None


@dataclass(frozen=True)
class Route:
    """A routing-only outcome (a terminal): carries only the chosen case handle — no value,
    no post-asserts. Returned by a router (CASE); the engine takes the handle's out-edge and
    skip-floods the siblings."""

    handle: str


@dataclass(frozen=True)
class Pause:
    """A leaf wait (HUMAN_INPUT / WAIT / an agent mid-loop control-pause). `reason` is a
    `suspension.pause.PauseReason`. The engine emits `PauseRequested` and suspends."""

    reason: Any


@dataclass(frozen=True)
class Grow:
    """A spawner expands into a subgraph. The engine splices `subgraph` generically and
    applies `prune`. Fields:
      - `subgraph`: the `Flow` to splice — the builder's self-describing graph fragment. The engine
        schedules its `start_id` (one node) and fans out downstream via `_advance`; a `Flow` whose
        terminal carries a baked `commit_as=<spawner id>` commits under the spawner on success.
      - `prune`: ids to retire in the same step (∅ for call/map; a self-respawn loop's finished
        iteration ids). Defaults to ∅.
      - `seed`: the PURE BUILDER INPUT the engine persists so a resumed run can rebuild the spliced
        subgraph without re-running non-deterministic nodes (e.g. an agent's LLM turn). Kind-shaped
        but OPAQUE to the engine (a call record dict, a map records list, agent segments, a loop
        (record, index)); it is durability data, NOT a per-kind policy switch."""

    subgraph: "Flow"
    prune: "frozenset[str]" = field(default_factory=frozenset)
    seed: Any = None


# The closed sum a pure `run(inputs)` returns.
NodeResult = Union[Output, Route, Pause, Grow]


class Node(ABC):
    """
    The base contract every node kind implements: a pure function of its input record.

    A node implements [`run`][agent_composer.nodes.base.Node.run] and returns one of the
    closed sum `Output | Route | Pause | Grow` (or, for a streaming kind, a generator that yields
    `StreamChunk` and returns a `NodeResult`). The node never receives the pool — the engine's
    `eval_node` seam binds its inputs and hands it a record — and never writes the pool; it
    *describes* its one output as `Output(value)` and the engine performs the write under the
    node id. Failure is a `raise`, not a variant.

    Attributes:
        kind (`NodeKind`):
            The closed-vocabulary tag the engine dispatches on. Set per subclass.
        id (`str`):
            The node's unique id within its (possibly namespaced) flow.
        title (`str`, *optional*):
            A human-friendly display title, or `None`.
        output_type (`Type`, *optional*):
            The declared output Type (one value); `None` leaves the write unenforced.
        params (`list[ParamDecl]`, *optional*):
            The node-side declared params (no source — the flow owns the wiring); `None`
            for a node that declares no inputs.
        pre_asserts (`list[str]`):
            Node-local `asserts:` checked against the bound record before `run`.
        post_asserts (`list[str]`):
            Node-local `asserts:` (reading `${output}`) checked after `run`.
        commit_as (`str`, *optional*):
            Engine-baked commit redirect: write this node's Output under `commit_as`
            instead of its own id (and fire that target's out-edges). Set by the
            `_grow_*` expanders on a subflow terminal (a CALL/MAP child END filler,
            a MAP END-list filler, an agent resume continuation) to point back at the
            spawner id; `None` for an ordinary node (commit under its own id).
        origin_id (`str`, *optional*):
            "Attribute my grows and self-commit to this origin node." `None` on every
            ordinary node. Set only on a self-respawning loop driver: the compiled loop
            `L` sets `origin_id = L` (== self) and each fresh per-iteration clone `L~k`
            sets `origin_id = L` too, so the engine can attribute every iteration's
            `Grow`/`Output` to the single durable origin `L`. A driver is the origin iff
            `self.id == self.origin_id`, which is what gates "do not self-prune".
        env (`dict[str, Any]`):
            Author-supplied static per-node config (`{str -> literal}`), baked at build
            time from the flow-level `env:` default merged with the node's own `env:`
            (node override wins). Opaque to the engine core — kind-blind: only a node's
            own `run()` reads the keys it understands (AGENT: `max_tool_iterations`).
            Unknown keys are carried but ignored. `{}` when none is authored.
    """

    kind: ClassVar[NodeKind]
    # Declares "I may grow the graph" (a spawner returns a `Grow`). Overridden True by
    # the spawner kinds (CALL/MAP/AGENT/LOOP); the `eval_node` seam gates the grow path on it.
    is_spawner: ClassVar[bool] = False
    # Declares "I bind my inputs PER ELEMENT via a `bind_item` cap rather than once up front"
    # (MAP). When True, the read seam (`eval_node`) starts the record empty and supplies
    # `caps['bind_item']`; when False (the common case) it binds `params` once from the pool.
    # Overridden True by MapNode; every other kind binds up front.
    binds_per_item: ClassVar[bool] = False
    # REF-depth policy for the engine's growth core. The depth stamped on a grow's subgraph
    # spawners + its terminal is `parent_depth + grow_depth_delta`; a positive delta is also
    # bounded by MAX_REF_DEPTH. Possible values:
    #   None  — this grow is NOT REF recursion; the core does NO depth work (LOOP: bounded by
    #           max_iters + MAX_TOTAL_NODES, not depth).
    #   0     — carry the parent depth UNCHANGED, no bound (AGENT: a K-pause chain is one call).
    #   1     — a nested call, +1 and bounded (CALL/MAP: each child is one deeper level).
    # Overridden by CALL/MAP (1) and AGENT (0); the default None fits LOOP + any non-spawner.
    grow_depth_delta: ClassVar[Optional[int]] = None
    # Declares "on a grow, also stamp `_spawner_expansion` at MY OWN bare id" (AGENT). An agent
    # that pauses, resumes, and pauses AGAIN grows twice at the SAME spawner id, so its record must
    # be findable under its own id for the re-pause to nest under it (re-pause idempotency). Every
    # other spawner grows at a FRESH namespaced id per instance, so it needs no self-stamp.
    # Overridden True by AgentNode.
    grow_restamps_self: ClassVar[bool] = False
    # Declares "I am the self-respawning fixpoint-iteration driver" (LOOP). The growth core
    # (`_apply_grow`) gates the origin-keyed single-live-record ledger invariant under this trait:
    # each iteration is a fresh driver clone whose grow is attributed to `origin_id` (the compiled
    # loop) and supersedes the prior iteration's GrowRecord, so the ledger stays bounded to one
    # record per loop. A plain trait check suffices — the core never dispatches on the kind.
    # Overridden True by LoopNode; False for every other kind.
    is_loop: ClassVar[bool] = False
    # Declares "I am LLM-backed" (AGENT). When True, the read seam (`eval_node`) builds the
    # `caps['llm']` capability — a `model_from_config`-shaped factory the engine owns — and
    # passes it to `run`; when False (the common case) no LLM cap is built. Mirrors
    # `binds_per_item` gating. Overridden True by AgentNode.
    needs_llm: ClassVar[bool] = False

    def __init__(
        self,
        node_id: str,
        *,
        title: Optional[str] = None,
        output_type: Optional[Type] = None,
    ) -> None:
        self.id = node_id
        self.title = title
        # The node's declared output Type (one value). Threaded by the compiler;
        # None for fakes / nodes that declare none (then the write is unenforced).
        self.output_type: Optional[Type] = output_type
        # The node-side signature (the node/flow split): declared params with NO source — the flow owns
        # the wiring in `CompiledFlow.wiring[node_id][param]`. Stamped by the compiler
        # (`build_*`/case desugar); the engine's `eval_node` binds via `params` + `flow.wiring`.
        # `None` for a fake / directly-constructed node that declares no inputs (== no params).
        self.params: Optional[list[ParamDecl]] = None
        # Node-local `asserts:` (a per-node contract), classified + stamped by the loader and
        # enforced by the engine's `eval_node` seam: PRE checked against the bound input record
        # before `run`; POST (reads `${output}`) after `run` against `{**inputs, output}`.
        # Empty for most nodes.
        self.pre_asserts: list[str] = []
        self.post_asserts: list[str] = []
        # Engine-baked commit redirect (see the class docstring): a subflow terminal
        # commits its Output under this spawner id instead of its own id. Stamped by the
        # `_grow_*` expanders; `None` for an ordinary node.
        self.commit_as: Optional[str] = None
        # "Attribute my grows/self-commit to this origin" (see the class docstring). `None`
        # on ordinary nodes; set to the compiled loop id on a loop driver (origin == self on
        # the compiled `L`, == `L` on each fresh iteration clone `L~k`). A driver is the
        # origin iff `self.id == self.origin_id` (which gates "do not self-prune").
        self.origin_id: Optional[str] = None
        # Author-supplied static per-node config, opaque to the engine core (kind-blind): a
        # plain `{str -> literal}` map baked at build time from the flow-level `env:` default
        # merged with this node's own `env:` (node override wins; see the compose builder). The
        # engine NEVER reads it — only a node's own `run()` interprets the keys it understands
        # (e.g. AGENT reads `max_tool_iterations`); unknown keys are carried but ignored. `{}`
        # for a node that declares no `env:` and inherits no flow default.
        self.env: dict[str, Any] = {}
        # Render-only source-location metadata, populated by the compose builder (never by hand)
        # and NEVER serialized: a checkpoint stores no topology, so `.range`/`.field_ranges` are
        # rebuilt from the flow spec on restore, never round-tripped. Purely for diagnostics.
        # `self.range`: this node's own `SourceRange` in the flow YAML (the whole node block); None
        #   for a fake / directly-constructed node with no known source.
        # `self.field_ranges`: `dict[field_name -> SourceRange]` for the node's individual authored
        #   fields (a bad input binding, an assert expr); `{}` when no per-field spans are recorded.
        self.range: Optional["SourceRange"] = None
        self.field_ranges: dict[str, "SourceRange"] = {}

    @abstractmethod
    def run(self, inputs: dict[str, Any], **caps: Any) -> "NodeResult":
        """Execute the node as a pure function of its bound input record.

        Returns a `NodeResult` (`Output | Route | Pause | Grow`), or — for a streaming kind —
        a generator that yields `StreamChunk` and *returns* a `NodeResult`. The one effectful
        cap left is a mapped `call`'s `bind_item` (keyword-only); every other kind takes only `inputs`. A
        failure is a `raise`, not a variant — the engine boundary turns it into `NodeFailed`."""

    def on_failure(self, exc: Exception, inputs: dict[str, Any], **caps: Any) -> "NodeResult":
        """Recovery seam: called by the engine wrapper when run() raises. Default: re-raise."""
        raise exc

    def bind_reserved(self, node_wiring: dict, pool) -> dict[str, Any]:
        """Reserved input keys the read seam (`eval_node`) must pre-resolve from
        `node_wiring` + `pool` before `run`, merged into the bound record.

        Possible keys (per kind): a timed WAIT returns `{"until": <ISO ts>}`; a MAP returns
        `{"over": <list>}`. Default: `{}` (an ordinary node reserves no keys). `node_wiring`
        is the flow-owned wiring for this node (`flow.wiring[id]`); `pool` is the live
        `VariablePool` the seam reads sources from."""
        return {}

    def reserved_wiring_keys(self) -> set[str]:
        """The reserved AUTHOR-wiring key names this node owns beyond its declared `params`
        — the keys `check_wiring_parity` must allow, and validation treats specially.

        These are the static NAMES of the same keys `bind_reserved` resolves at run (a timed
        WAIT: `until`; a MAP: `over`), but returned with NO pool — load-time only. Default:
        the empty set (an ordinary node reserves no keys)."""
        return set()

    def iter_boundary_records(self, seed: Any) -> "list[tuple[dict, str]]":
        """The `(record, label)` pairs the engine's growth core boundary-checks EAGERLY, before it
        attaches the grow to the ledger (so a boundary failure leaves no orphan expansion).

        Each pair names one input record whose EFFECTIVE inputs (coerce + default) are checked
        against this node's child BOUNDARY asserts, plus a human `label` the failure message
        formats as `f"{label} boundary assert failed: {bad}"`. A subflow spawner returns one pair
        per child instance: CALL -> one pair from its call-arg seed; MAP -> one per element (label
        carries the element index). Default: `[]` (a node with no boundary check — AGENT, LOOP)."""
        return []

    def replay_grow(self, seed: Any) -> "Flow":
        """Durable-replay seam (the READ half of `Grow`): rebuild THIS spawner's subgraph from the
        persisted `seed` (the pure builder input captured on the live `Grow.seed`). The engine's
        `_replay_expansions` calls it on restore to re-grow a paused run's expanded subgraphs in a
        fresh process, WITHOUT re-running the non-deterministic body that first produced the seed
        (an agent's LLM turn, a resolved `over:` list). It is the pure inverse of the node's own
        live grow: `run()` builds the subgraph AND returns the seed; `replay_grow(seed)` rebuilds
        the SAME subgraph from that seed alone. Only spawners (`is_spawner`) ever appear in the
        durable ledger, so the base default is a loud error for a non-spawner."""
        raise NotImplementedError(
            f"node {self.id!r} ({type(self).__name__}) is not a spawner and cannot replay_grow"
        )

    @staticmethod
    def _assert_holds(expr: str, record: dict) -> bool:
        """Evaluate a node assert against `record`; a raising assert (ordered/arith over a
        non-scalar / None `${output}`) is treated as NOT holding (-> a clean NodeFailed)."""
        try:
            return bool(evaluate_when_record(expr, record))
        except ExpressionError:
            return False

    def _drain_node_generator(self, gen: Generator) -> Generator[Any, Any, "NodeResult"]:
        """Forward a streaming node's yielded `StreamChunk`s and capture its RETURNED
        `NodeResult`. A pause is now a *returned* `Pause` (not a yielded event), so a
        generator only ever yields `StreamChunk`; the dispatch happens in `eval_node`."""
        try:
            event = next(gen)
            while True:
                yield event  # StreamChunk
                event = next(gen)
        except StopIteration as stop:
            result = stop.value
            if not isinstance(result, (Output, Route, Pause, Grow)):
                raise RuntimeError(f"node {self.id!r} generator did not return a NodeResult")
            return result
