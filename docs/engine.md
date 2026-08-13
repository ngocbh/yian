# The engine — running a `CompiledFlow`

> Phase 2 of the [lifecycle](architecture.md). The [front end](compose.md) produced a
> typed graph; the engine *interprets* it. The core (`runtime/engine.py` +
> `runtime/eval_node.py`) is **kind-blind**: it branches on the closed `Outcome` sum
> and on node-owned traits, never on a node's `NodeKind`. The node side of the same
> contract is in [Nodes](nodes.md).

## Two actors, one narrow contract

There are exactly two kinds of thing, and they meet at one method.

```
   ┌──────────────── ENGINE ────────────────┐        node.run(inputs) → Outcome
   │ owns the mutable world:                 │  ───────────────────────────────▶
   │   • the pool   (every node's output)    │        ┌──────── NODE ────────┐
   │   • the queue  (what can run next)       │        │ a PURE function of   │
   │   • the graph  (boxes + arrows; GROWs)   │  ◀──── │ its inputs. Never    │
   │   • events + dormant checkpoints          │ inputs │ touches the pool.    │
   └──────────────────────────────────────────┘        └──────────────────────┘
```

The engine owns *state and scheduling*; the node owns *computation*. The node is
**pure at the graph boundary** — handed a plain dict plus declared capabilities,
it returns a result and cannot read or write shared memory. An LLM/human/clock leaf
may perform an external effect, but all graph state still lives in one place the
engine controls.

## The loop: read → eval → apply

Running a graph is three questions asked over and over: *what can run now? what does it
produce? what changes?* Each node goes through the same three steps — the engine does
READ and APPLY, while the base `Node.run` template owns the uniform EVAL wrapper:

```python
def run(flow, inputs):
    state = StateManager(flow.types); state.seed(inputs)   # the typed memory (the "pool")
    queue = ReadyQueue(flow.graph);   queue.add(flow.start_id)

    while (node := queue.pop_ready(state)) is not None:
        inputs  = state.bind(node.params, flow.wiring[node.id])  # READ : memory → plain dict
        outcome = node.run(inputs, **caps)                       # EVAL : shared template
        apply(outcome, node, state, queue, flow.graph)           # APPLY: write / grow / pause
    return state.result()                                        # whatever END produced

def Node.run(self, inputs, **caps):      # the generic template — IDENTICAL for every kind
    inputs = apply_defaults(self.params, inputs)
    check(self.pre_asserts, inputs)
    out = self._run_impl(inputs, **caps)                # the ONLY kind-specific step
    if isinstance(out, Output):
        check(self.post_asserts, {**inputs, "output": out.value})
    return out
```

This pseudocode is illustrative; the implementation also preserves pristine
post-assert inputs, source ranges, and dormant generator plumbing.

`Node.run` is a fixed recipe applied to *every* node. The read boundary is kind-blind
but **node-configured** — a node declares *how* to bind rather than the engine branching
on kind (e.g. `binds_per_item` on `map`, `bind_reserved` for a `wait`'s `until` or a
`map`'s `over`). The only kind-specific step is `_run_impl`.

A flow's own top-level `asserts:` are not a separate engine phase: at build they are
lowered onto the synthesized `__start__`/`__end__` boundary nodes. A boundary assert
(`${input.X}` only) rides `__start__` and fires *inside* `StartNode.run`, after input
coercion — a dedicated slot the generic `check(node.pre_asserts)` seam leaves alone, since
it must see the coerced values. A post assert (`${node.output}`) rides `__end__` as an
ordinary `node.post_asserts` and fires through the `Node.run` recipe above. So `run.py`
enforces no asserts itself; a false assert comes back as a `NodeFailed`/`RunFailed` event
like any other node assert.

## `Outcome` — the whole vocabulary between the two actors

A node returns exactly one of four things, and this closed set is the *only* thing the
engine branches on:

```python
Outcome =
  | Output(value, commit_as=None)   # "here is my result"    → bind it, unlock dependents
  | Route(handle)                   # "route only, no value" → take this branch, skip siblings
  | Pause(reason)                   # artificial compatibility → checkpoint and stop
  | Grow(seed, prune=∅)             # "I'm really a subflow"  → build + splice more nodes
```

```python
def apply(outcome, node, state, queue, graph, grow_builder):
    match outcome:                        # the ONLY match in the engine — never on kind
        case Output(value, commit_as):
            target = commit_as or node.id
            state.set(target, value); queue.done(target)     # write-once, unlock dependents
        case Route(handle):
            queue.route(node, handle)                         # take edge, skip-flood siblings
        case Pause(reason):
            checkpoint(state, queue); raise Suspended(node, reason)
        case Grow(seed, prune):
            subgraph = grow_builder(node, seed)               # injected by run.py
            graph.splice(subgraph); graph.prune(prune)        # add boxes; retire spent ones
            queue.add(subgraph.start)
```

`Output` covers ordinary steps. `Route` covers a pure router (`case`): it stores nothing
and only selects a live out-edge. `Pause` is dormant custom/artificial compatibility;
authorable AGENT `ask_user`, HUMAN_INPUT, and WAIT calls block in-process and return
`Output`. `Grow` is how a node that *is really a subflow* expands.

## Growth: a node that is really a subflow

`call`, `map`, and `loop` are not single steps — each *is a smaller flow*. A spawner
returns only a canonical, kind-shaped seed; no node imports the compile layer or builds
topology. `compile.expand.build_grow` is the sole kind-aware interpreter that turns the
spawner and seed into a self-describing `Flow`. Package-root `run.py` injects that
callable into the kind-blind runtime for live growth and checkpoint replay.

One convention unifies all three constructed fragments: **the subgraph is a
well-formed sub-flow with a single `__start__` and `__end__`**. The engine schedules
the `start`, and the builder bakes `commit_as=<spawner id>` onto the terminal — so
however many inner nodes run, downstream sees one clean output committed under the
spawner's id. Runtime may consume `compile.model` topology but never imports
`compile.expand`.

The splice carries a few extra concerns (stamp recursion depth, loop bookkeeping),
but the growth core reads **node-owned traits**
(`grow_depth_delta`, `is_loop`) — never
`node.kind`. A child's asserts are **not** an engine-side splice concern: a called flow's
`asserts:` are lowered onto its own `__start__`/`__end__` at build (just like the
top-level flow, below), and those boundary nodes are cloned into the parent along with the
rest of the child — so the child's boundary asserts fire through `StartNode.run` and its
post asserts through the same base template, on the ordinary node path, identically to
the top level.
`prune` is growth's generic inverse: a self-respawning loop retires the
iteration it just finished in the same step, so it grows without leaking. Adding a new
spawner kind means setting traits on its node; the core never gains a branch. Details:
[Nodes](nodes.md).

## Failure and durability

Node evaluation may raise (a failed assert, a bad LLM response); the engine catches the
exception and emits a `NodeFailed` event carrying a source locator, so the CLI can point
at the offending YAML line. The dormant `Node.on_failure` method is reserved for a future
error-strategy seam but is not driven today.

Because graph-visible values are write-once and scheduling state is centralized,
the dormant `Pause` compatibility variant can serialize a custom/artificial run
to a `RunCheckpoint`;
`resume_flow()` rebuilds the graph from the flow spec and continues. Authorable
AGENT, HUMAN_INPUT, and WAIT never take this path. The topology is *not* stored —
it is rebuilt from source, while compatibility run state and dynamic-growth
descriptors are serialized.
See [`suspension/`](reference.md) and the [Python API](api.md).

## Serial and parallel, one contract

The pseudocode above is the serial reference engine. A parallel engine drains several
ready nodes at once through the *same* read → eval → apply contract — safe precisely
because nodes are pure and every write happens write-once at `apply` (which stays
serialized). Only the number of nodes in flight differs. A pooled drive owns its workers
through a full join: abnormal exit during dispatch first signals cooperative stop, while
closing the stream after a terminal `RunPaused` leaves its resume event untouched.

## The entire interface the engine sees

| The engine sees | Meaning |
|---|---|
| `node.id`, `node.params` | where to store output; input *names* (the flow owns their sources) |
| `node.pre_asserts` / `node.post_asserts` | conditions the base `run` template checks around the body |
| `node.run(inputs, **caps) → Outcome` | the shared template the engine drives |
| `node._run_impl(inputs, **caps) → Outcome` | the one kind-specific body step |
| `node.is_spawner` | may it return `Grow`? |
| read-boundary traits | `binds_per_item`, `bind_reserved`, `reserved_wiring_keys`, `needs_stop_event` |
| growth traits | `grow_depth_delta`, `is_loop` |

That is the complete list — a small set of traits/hooks plus the run/body pair. No `NodeKind` is
visible to the engine.

## Why a closed match, on the right axis

An earlier engine branched on node *kind* in many places. That dispatch is now fully
removed from the core; a ratchet test (`tests/engine/test_kind_census.py`) holds the
kind-branch count at **zero**. The design keeps the value of an explicit closed `match`
— but on the **`Outcome`** (four arms), not the kind (a dozen and growing). New kinds
extend the node side; the engine never changes.
