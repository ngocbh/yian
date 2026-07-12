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
  CONT → `Grow(subgraph = {body_k (NO commit_as), L~(k+1) driver, edge body_k.END→L~(k+1),
         driver wiring}, prune = {body_{k-1} ids} ∪ ({self.id} if self.id != origin), seed=(carried, k))`.

BOTH `run` (on `L~k`) and `replay_grow` (on `L`) build body/driver at the ORIGIN callsite
(`self.origin_id`), never `self.id`, so live and durable-replay windows are byte-identical.
"""
from typing import Any, ClassVar, Optional

from agent_composer.compile.expand import loop_continue_subgraph, map_callsite, ns
from agent_composer.expr.expressions import evaluate_when_record
from agent_composer.nodes.base import Grow, Node, NodeKind, Output


class LoopMaxExceeded(RuntimeError):
    """The while/until runaway guard tripped (the next iteration would reach `max_iters`).

    Raised inside `LoopNode.run` so `eval_node`'s boundary makes it a located `NodeFailed`
    carrying this type name (`error_type == "LoopMaxExceeded"`) and the `"exceeded max (N)"`
    message on the `RunFailed` event."""


class LoopNode(Node):
    """A higher-order `('a -> 'a) -> 'a -> 'a` driver over a carried record.

    Fields (baked at load by `compose.nodes.loop.build`; copied onto every clone by `respawn`):
      child          the compiled body subflow (the `'a -> 'a` callable); the Grow subgraph body.
      child_inputs   the body's declared input decls (subset of the carried record).
      predicate_kind one of "while" | "until" | "times".
      predicate      the boolean source for `while`/`until`, e.g. "not ${exited}" (bare =
                     carried-record scope); None for `times` (no predicate).
      times          the fixed run count when `predicate_kind == "times"`; None otherwise.
      max_iters      the runaway guard for `while`/`until`; for `times`, equals the count N.
      iteration      THIS driver's iteration index `k` (0 on the compiled `L`, `k` on a clone `L~k`).
      origin_id      the origin loop id `L` (== self.id on the compiled node; == `L` on every clone).
    """

    kind = NodeKind.LOOP
    is_spawner: ClassVar[bool] = True  # grows the graph: each iteration splices a Grow(Flow)
    is_loop: ClassVar[bool] = True  # the fixpoint-iteration driver: the ledger stays single-record

    def __init__(
        self,
        node_id: str,
        *,
        flow_id: str,
        flow_version: Optional[str] = None,
        child: Any = None,
        child_inputs: Any = None,
        child_source: Any = None,
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
        self.child_source = child_source
        self.predicate_kind = predicate_kind
        self.predicate = predicate
        self.times = times
        self.max_iters = max_iters
        self.iteration = iteration
        # The compiled `L` sets origin_id == self.id (uniform); a clone `L~k` carries the origin.
        self.origin_id = origin_id or node_id

    def run(self, inputs: dict[str, Any], **caps: Any):
        """Own the WHOLE loop policy (no engine `_loop_step`): decide continue-vs-stop on the
        carried record, and on continue emit a self-respawning `Grow` (body_k + a fresh `L~(k+1)`
        driver) that self-prunes this driver + the previous body.

        Stop rule (`k = self.iteration`, the index of the body about to run / that just fed this
        driver):
          while  — STOP when the predicate is FALSE on the carried record.
          until  — DO-WHILE: STOP when `k >= 1` AND the predicate is TRUE (`k == 0` always continues).
          times  — STOP when `k >= self.max_iters` (`times N` baked as `max_iters`).
        On STOP: `Output(carried, commit_as=self.origin_id)` — generic commit-and-advance under `L`.
        On CONTINUE: runaway guard (while/until raise `LoopMaxExceeded` if `should_stop(k+1)`),
        then a `Grow` splicing body_k + `respawn(k+1)`, pruning body_{k-1} + self (except origin)."""
        if self.child is None:
            raise RuntimeError(f"loop node {self.id!r} was not baked with a body child")
        carried = dict(inputs)
        k = self.iteration
        if self._should_stop_now(carried, k):
            return Output(carried, commit_as=self.origin_id)
        # CONTINUE: runaway guard for predicate loops (times stops via _should_stop_now above).
        # Guard on `should_stop(k)` (NOT k+1): driver@k grows body_k, so body_k is the (k+1)-th
        # body. It is within budget iff `k < max_iters`; raise once `k >= max_iters`. This makes
        # `max: M` permit exactly M body runs (the loop build documents "max must permit at
        # least one body run" — `max: 1` runs one body then fails on the second driver).
        if self.predicate_kind in ("while", "until") and self.should_stop(k):
            raise LoopMaxExceeded(
                f"loop {self.origin_id!r} exceeded max ({self.max_iters})")
        driver = self.respawn(k + 1)
        prune = {ns(map_callsite(self.origin_id, k - 1), cid) for cid in self.child.nodes} \
            if k >= 1 else set()
        if self.id != self.origin_id:
            prune.add(self.id)                       # self-prune (never the origin L)
        sg = loop_continue_subgraph(self.child, self.origin_id, carried, k, driver)
        return Grow(sg, prune=frozenset(prune), seed=(carried, k))

    def _should_stop_now(self, carried: dict[str, Any], k: int) -> bool:
        """Continue-vs-stop on the carried record at index `k`. Pure. See `run` for the stop rule."""
        if self.predicate_kind == "times":
            return self.should_stop(k)               # k >= N
        truth = evaluate_when_record(self.predicate, carried)
        if self.predicate_kind == "while":
            return not truth                         # while: stop on FALSE
        return k >= 1 and truth                      # until (do-while): stop on TRUE, k >= 1

    def respawn(self, k: int) -> "LoopNode":
        """A fresh driver clone for iteration `k` at id `f"{origin}~{k}"`, carrying the SAME baked
        fields and the ORIGIN id. Pure: no engine state. `origin_id` threads forward so the clone
        attributes its grow/commit to the compiled loop and self-prunes (`id != origin`).

        `params` must be copied for `eval_node` to bind the clone's carried inputs off the CONTINUE
        wiring. `output_type`/asserts are copied for clone self-consistency only — the STOP commit
        reads the COMPILED origin `L`'s type via `_on_success`, not the clone's."""
        clone = LoopNode(
            f"{self.origin_id}~{k}", flow_id=self.flow_id, flow_version=self.flow_version,
            child=self.child, child_inputs=self.child_inputs,
            child_source=self.child_source, predicate_kind=self.predicate_kind,
            predicate=self.predicate, times=self.times, max_iters=self.max_iters,
            title=self.title, iteration=k, origin_id=self.origin_id)
        clone.params = self.params
        clone.output_type = self.output_type
        clone.pre_asserts = list(self.pre_asserts)
        clone.post_asserts = list(self.post_asserts)
        return clone

    def should_stop(self, iteration: int) -> bool:
        """The loop's hard iteration budget: True once `iteration` has reached `max_iters`.

        Pure, node-owned budget policy (`iteration >= self.max_iters`); assumes `max_iters` was
        baked at load (`compose.nodes.loop.build`). `run` consults this both as the `times`
        stop-count and as the `while`/`until` runaway guard."""
        return iteration >= self.max_iters

    def replay_grow(self, seed: Any):
        """Durable-replay inverse: rebuild the LIVE window `{body_k, L~(k+1)}` from the persisted
        `(carried, k)` seed via the SAME CONTINUE builder — pure, no body re-run. Runs on the
        COMPILED `L` (`self.id == self.origin_id`), so bodies land at `L#k/…` (origin-keyed) exactly
        as the live grow. The self-pruned `L~k` and pruned `body_{k-1}` are NOT rebuilt (they were
        gone at snapshot); only the resident window is reproduced. The seed round-trips through JSON
        as a 2-element list, so unpacking covers tuple and list."""
        carried, k = seed
        driver = self.respawn(k + 1)
        return loop_continue_subgraph(self.child, self.origin_id, dict(carried), k, driver)
