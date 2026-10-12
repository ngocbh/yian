"""LoopNode — the self-respawning `while:`/`until:`/`times:` driver.

Each loop iteration is a FRESH `LoopNode` driver clone whose `run` owns the WHOLE loop policy —
the continue-vs-stop decision, the iteration index, the runaway guard, and the self-respawn grow.
There is no engine-side loop step: the engine core stays kind-blind (splice `Grow`, apply
`prune`, commit `Output`, fold the durable ledger).

Names:
  L      the COMPILED loop node id (iteration-0 driver, `origin_id == L == self.id`).
  L~k    a fresh driver clone for iteration `k >= 1` (`iteration=k`, `origin_id=L`, id `f"{L}~{k}"`).
  L#k/…  the cloned body for iteration `k` at callsite `map_callsite(L, k)` == `f"{L}#{k}"`.

Per-iteration (driver@k = `L` when k==0 else `L~k`, run on the carried record):
  STOP → `Output(carried, commit_as=origin)` — generic commit-and-advance under `L`.
  CONT → `Grow(seed=(carried, k),
         prune={body_{k-1} ids} ∪ ({self.id} if self.id != origin))`.

The compile-owned injected builder constructs body/driver at the ORIGIN callsite
(`self.origin_id`), never `self.id`, so live and durable-replay windows are byte-identical.
"""
from dataclasses import replace
from typing import Any, Callable, ClassVar, Iterator, Optional

from yian.diagnostics import OriginRef
from yian.expr.expressions import evaluate_when_record
from yian.expr.lineage import FeedbackValue
from yian.nodes.base import (
    BackwardContext, BackwardResult, Grow, Node, NodeKind, Output, _project_structural_feedback,
)
from yian.nodes.contracts import (
    ExecutionPosition,
    ExecutionScope,
    Parameter,
    _iter_origin_carrier,
    _rewrite_origin_carrier,
)
from yian.nodes.ids import map_callsite, ns


class LoopMaxExceeded(RuntimeError):
    """The while/until runaway guard tripped (the loop would start more than `max` body runs).

    Raised only when the loop sets `max:`; a while/until loop without one has no limit.

    Raised inside `LoopNode.run` so `eval_node` records a located `Failure` whose
    `cause_type` is `"LoopMaxExceeded"` and whose message names the exceeded limit."""


class LoopNode(Node):
    """A higher-order `('a -> 'a) -> 'a -> 'a` driver over a carried record.

    Fields (baked at load by `compose.nodes.loop.build`; copied onto every clone by `respawn`):
      child          the compiled body child flow (`'a -> 'a`) used by the injected growth builder.
      child_inputs   the body's canonical parameters (subset of the carried record).
      predicate_kind one of "while" | "until" | "times".
      predicate      the boolean source for `while`/`until`, e.g. "not ${exited}" (bare =
                     carried-record scope); None for `times` (no predicate).
      times          the fixed run count when `predicate_kind == "times"`; None otherwise.
      max_iters      the runaway guard for `while`/`until` (None = no limit); for `times`, equals
                     the count N.
      iteration      THIS driver's iteration index `k` (0 on the compiled `L`, `k` on a clone `L~k`).
      origin_id      the origin loop id `L` (== self.id on the compiled node; == `L` on every clone).
    """

    kind = NodeKind.LOOP
    is_spawner: ClassVar[bool] = True  # each continuing iteration returns a seed-first Grow
    is_loop: ClassVar[bool] = True  # the fixpoint-iteration driver: the ledger stays single-record
    grow_depth_delta: ClassVar[int] = 0  # inherit enclosing call depth without consuming a level

    def __init__(
        self,
        node_id: str,
        *,
        flow_id: str,
        flow_version: Optional[str] = None,
        child: Any = None,
        child_inputs: Optional[list[Parameter]] = None,
        predicate_kind: str = "while",
        predicate: Optional[str] = None,
        times: Optional[int] = None,
        max_iters: Optional[int] = None,
        title: Optional[str] = None,
        iteration: int = 0,
        origin_id: Optional[str] = None,
    ) -> None:
        super().__init__(node_id, title=title)
        self.flow_id = flow_id
        self.flow_version = flow_version
        self.child = child
        self.child_inputs = child_inputs
        self.predicate_kind = predicate_kind
        self.predicate = predicate
        self.times = times
        self.max_iters = max_iters
        self.iteration = iteration
        # The compiled `L` sets origin_id == self.id (uniform); a clone `L~k` carries the origin.
        self.origin_id: str = origin_id or node_id

    def skipped_output_target(self) -> str:
        """Return the compiled loop origin that receives a stopped driver output."""
        return self.origin_id

    def iter_local_origins(
        self,
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> Iterator[OriginRef]:
        """Yield common origins followed by the cached body signature."""
        seen = _seen_carriers if _seen_carriers is not None else set()
        yield from super().iter_local_origins(_seen_carriers=seen)
        if self.child_inputs is not None:
            for parameter in self.child_inputs:
                yield from _iter_origin_carrier(parameter, seen)

    def rewrite_local_origins(
        self,
        mapper: Callable[[OriginRef], OriginRef],
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> None:
        """Rewrite common and cached-signature origins on a copied LOOP."""
        seen = _seen_carriers if _seen_carriers is not None else set()
        super().rewrite_local_origins(mapper, _seen_carriers=seen)
        if self.child_inputs is not None:
            for parameter in self.child_inputs:
                _rewrite_origin_carrier(parameter, mapper, seen)

    def origin_children(self) -> tuple[Any, ...]:
        """Return the baked body flow, or no child for an unbaked LOOP."""
        return () if self.child is None else (self.child,)

    def validate_execution_setup(self) -> None:
        """Require the body flow that Compose must bake before execution."""
        super().validate_execution_setup()
        if self.child is None:
            raise RuntimeError(f"loop node {self.id!r} was not baked with a body child")

    def forward(self, inputs: dict[str, Any], **caps: Any):
        """Own the WHOLE loop policy (no engine `_loop_step`): decide continue-vs-stop on the
        carried record, and on continue emit a self-respawning `Grow` (body_k + a fresh `L~(k+1)`
        driver) that self-prunes this driver + the previous body.

        Stop rule (`k = self.iteration`, the index of the body about to run / that just fed this
        driver):
          while  — STOP when the predicate is FALSE on the carried record.
          until  — DO-WHILE: STOP when `k >= 1` AND the predicate is TRUE (`k == 0` always continues).
          times  — STOP when `k >= self.max_iters` (`times N` baked as `max_iters`).
        On STOP: `Output(carried, commit_as=self.origin_id)` — generic commit-and-advance under `L`.
        On CONTINUE: runaway guard, only when `max:` is set (while/until raise `LoopMaxExceeded`
        if `should_stop(k)`), then a `Grow` splicing body_k + `respawn(k+1)`, pruning body_{k-1} +
        self (except origin)."""
        carried = dict(inputs)
        k = self.iteration
        if self._should_stop_now(carried, k):
            return Output(carried, commit_as=self.origin_id)
        # CONTINUE: runaway guard for predicate loops, only when `max:` is set (times stops via
        # _should_stop_now above; `should_stop` is False without a max).
        # Guard on `should_stop(k)` (NOT k+1): driver@k grows body_k, so body_k is the (k+1)-th
        # body. It is within budget iff `k < max_iters`; raise once `k >= max_iters`. This makes
        # `max: M` permit exactly M body runs (the loop build documents "max must permit at
        # least one body run" — `max: 1` runs one body then fails on the second driver).
        if self.predicate_kind in ("while", "until") and self.should_stop(k):
            raise LoopMaxExceeded(
                f"loop {self.origin_id!r} exceeded max ({self.max_iters})")
        prune = {ns(map_callsite(self.origin_id, k - 1), cid) for cid in self.child.nodes} \
            if k >= 1 else set()
        if self.id != self.origin_id:
            prune.add(self.id)                       # self-prune (never the origin L)
        return Grow(seed=(carried, k), prune=frozenset(prune))

    def _should_stop_now(self, carried: dict[str, Any], k: int) -> bool:
        """Continue-vs-stop on the carried record at index `k`. Pure. See `run` for the stop rule."""
        if self.predicate_kind == "times":
            return self.should_stop(k)               # k >= N
        truth = evaluate_when_record(self.predicate, carried)
        if self.predicate_kind == "while":
            return not truth                         # while: stop on FALSE
        return k >= 1 and truth                      # until (do-while): stop on TRUE, k >= 1

    def capture_backward_context(self) -> dict:
        """Capture every carried input, including fields the body does not consume.

        Returns:
            `dict`: The `input_output_paths` mapping from each carried input name
                to its same-named field in a stopped driver's record output.
        """
        return {"input_output_paths": {name: [name] for name in self.backward_input_names()}}

    def backward(
        self, context: BackwardContext, feedback: tuple[FeedbackValue, ...],
    ) -> BackwardResult:
        """Project a stopped driver's recorded carried output without testing its predicate.

        Args:
            context (`BackwardContext`): Recorded carried inputs, accepted output
                or failure, full input inventory, and captured named paths.
            feedback (`tuple[FeedbackValue, ...]`): Ordered text or structural
                selections over the accepted carried record.

        Returns:
            `BackwardResult`: Input messages for a successful stopping driver.
                Failed or continuing drivers stop; growth connections handle the
                separate continuation boundary.

        Raises:
            `TypeError`: If a message batch or captured path has an invalid type.
            `ValueError`: If feedback or a captured path does not match the
                recorded output, or the required structural layout is absent.
        """
        if context.failure is not None or type(context.output) is not dict:
            return BackwardResult()
        return _project_structural_feedback(context, feedback)

    def respawn(self, k: int) -> "LoopNode":
        """A fresh driver clone for iteration `k` at id `f"{origin}~{k}"`, carrying the SAME baked
        fields and the ORIGIN id. Pure: no engine state. `origin_id` threads forward so the clone
        attributes its grow/commit to the compiled loop and self-prunes (`id != origin`).

        `params` must be copied for `eval_node` to bind the clone's carried inputs off the CONTINUE
        wiring. Generated `origin_context` is copied unchanged while the authored loop origin
        remains local, so each iteration's failure trace contains that origin once. The complete
        `output_contract` and asserts are copied for clone self-consistency only — the STOP commit
        validates against the COMPILED origin `L`'s contract via its commit target, not the clone's."""
        clone = LoopNode(
            f"{self.origin_id}~{k}", flow_id=self.flow_id, flow_version=self.flow_version,
            child=self.child, child_inputs=self.child_inputs,
            predicate_kind=self.predicate_kind, predicate=self.predicate,
            times=self.times, max_iters=self.max_iters,
            title=self.title, iteration=k, origin_id=self.origin_id)
        clone.origin = self.origin
        # Keep only the enclosing ancestry. ``failure_trace()`` appends this driver's
        # authored loop origin, so copying that origin into the context would duplicate it
        # on every later iteration.
        clone.origin_context = self.origin_context
        clone.memory_scope = self.memory_scope
        if self.execution_scope is not None:
            scope = self.execution_scope
            if type(scope) is not ExecutionScope:
                raise TypeError("loop execution scope must be an ExecutionScope")
            ExecutionScope(scope.static_path, scope.flow_path, scope.occurrence_path, scope.boundary)
            current = ExecutionPosition("loop_driver", scope.static_path, self.iteration)
            if not scope.occurrence_path or scope.occurrence_path[-1] != current:
                raise ValueError("loop scope must end in its current driver position")
            clone.execution_scope = replace(scope, occurrence_path=(
                *scope.occurrence_path[:-1], ExecutionPosition("loop_driver", scope.static_path, k),
            ))
        clone.params = self.params
        clone.output_contract = self.output_contract
        clone.pre_asserts = list(self.pre_asserts)
        clone.post_asserts = list(self.post_asserts)
        return clone

    def should_stop(self, iteration: int) -> bool:
        """The loop's hard iteration budget: True once `iteration` has reached `max_iters`.

        Pure, node-owned budget policy (`iteration >= self.max_iters`); always False when there is
        no max (`max_iters is None`), since then there is no budget. `run` consults this both as
        the `times` stop-count and as the `while`/`until` runaway guard."""
        if self.max_iters is None:
            return False
        return iteration >= self.max_iters
