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
   │   • the event log + checkpoints          │ inputs │ touches the pool.    │
   └──────────────────────────────────────────┘        └──────────────────────┘
```

The engine owns *state and scheduling*; the node owns *computation*. The node is
**pure** — handed a plain dict, it returns a value; it cannot read or write shared
memory. That purity is what makes a run reproducible and resumable: all state lives in
one place the engine controls.

## The loop: read → eval → apply

Running a graph is three questions asked over and over: *what can run now? what does it
produce? what changes?* Each node goes through the same three steps — the engine does
READ and APPLY, and wraps EVAL:

```python
def run(flow, inputs):
    state = StateManager(flow.types); state.seed(inputs)   # the typed memory (the "pool")
    queue = ReadyQueue(flow.graph);   queue.add(flow.start_id)

    while (node := queue.pop_ready(state)) is not None:
        inputs  = state.bind(node.params, flow.wiring[node.id])  # READ : memory → plain dict
        outcome = run_node(node, inputs, caps)                   # EVAL : asserts + run (generic)
        apply(outcome, node, state, queue, flow.graph)           # APPLY: write / grow / pause
    return state.result()                                        # whatever END produced

def run_node(node, inputs, caps):        # the generic wrapper — IDENTICAL for every kind
    check(node.pre_asserts, inputs)
    try:
        out = node.run(inputs, **caps)                 # the ONLY kind-specific step
    except Exception as exc:
        out = node.on_failure(exc, inputs, **caps)     # recovery hook (default: re-raise)
    if isinstance(out, Output):
        check(node.post_asserts, {**inputs, "output": out.value})
    return out
```

`run_node` is a fixed recipe applied to *every* node. The read boundary is kind-blind
but **node-configured** — a node declares *how* to bind rather than the engine branching
on kind (e.g. `binds_per_item` on `map`, `bind_reserved` for a `wait`'s `until` or a
`map`'s `over`). The only kind-specific step is `run` itself.

## `Outcome` — the whole vocabulary between the two actors

A node returns exactly one of four things, and this closed set is the *only* thing the
engine branches on:

```python
Outcome =
  | Output(value, commit_as=None)   # "here is my result"    → bind it, unlock dependents
  | Route(handle)                   # "route only, no value" → take this branch, skip siblings
  | Pause(reason)                   # "I need to wait/ask"    → checkpoint and stop
  | Grow(subgraph, prune=∅)         # "I'm really a subflow"  → splice in more nodes
```

```python
def apply(outcome, node, state, queue, graph):
    match outcome:                        # the ONLY match in the engine — never on kind
        case Output(value, commit_as):
            target = commit_as or node.id
            state.set(target, value); queue.done(target)     # write-once, unlock dependents
        case Route(handle):
            queue.route(node, handle)                         # take edge, skip-flood siblings
        case Pause(reason):
            checkpoint(state, queue); raise Suspended(node, reason)
        case Grow(subgraph, prune):
            graph.splice(subgraph); graph.prune(prune)        # add boxes; retire spent ones
            queue.add(subgraph.start)
```

`Output` covers ordinary steps. `Route` covers a pure router (`case`): it stores nothing
and only selects a live out-edge. `Pause` suspends for a human or a timer. `Grow` is how
a node that *is really a subflow* expands.

## Growth: a node that is really a subflow

`call`, `map`, and `loop` are not single steps — each *is a smaller flow*. Rather than
hard-coding them, such a node returns a self-describing `Flow` subgraph and the engine
splices it in. One convention unifies all three: **the subgraph is a well-formed sub-flow
with a single `__start__` and `__end__`**. The engine schedules the `start`, and the
spawner bakes `commit_as=<its own id>` onto the terminal — so however many inner nodes
run, downstream sees one clean output committed under the spawner's id.

The splice carries a few extra concerns (check the child's boundary asserts, stamp
recursion depth, loop bookkeeping), but the growth core reads **node-owned traits**
(`iter_boundary_records`, `grow_depth_delta`, `grow_restamps_self`, `is_loop`) — never
`node.kind`. `prune` is growth's generic inverse: a self-respawning loop retires the
iteration it just finished in the same step, so it grows without leaking. Adding a new
spawner kind means setting traits on its node; the core never gains a branch. Details:
[Nodes](nodes.md).

## Failure and durability

Failures never crash the loop. `run_node` may raise (a failed assert, a bad LLM
response); the engine turns any exception into a `NodeFailed` event carrying a source
locator, so the CLI can point at the offending YAML line. `on_failure` gives the node
first refusal (default: re-raise) — the future home of retry/fallback policy.

Because state is write-once and all in the pool, `Pause` serializes the whole run to a
`RunCheckpoint`; `resume_flow()` rebuilds the graph from the flow spec and continues.
The topology is *not* stored — only the pool and queue — because the graph is
reconstructable from the source. See [`suspension/`](reference.md) and the
[Python API](api.md).

## Serial and parallel, one contract

The pseudocode above is the serial reference engine. A parallel engine drains several
ready nodes at once through the *same* read → eval → apply contract — safe precisely
because nodes are pure and every write happens write-once at `apply` (which stays
serialized). Only the number of nodes in flight differs.

## The entire interface the engine sees

| The engine sees | Meaning |
|---|---|
| `node.id`, `node.params` | where to store output; input *names* (the flow owns their sources) |
| `node.pre_asserts` / `node.post_asserts` | conditions `run_node` checks purely, before / after `run` |
| `node.run(inputs, **caps) → Outcome` | the one kind-specific step |
| `node.on_failure(...)` | recovery seam (default: re-raise) |
| `node.is_spawner` | may it return `Grow`? |
| read-boundary traits | `binds_per_item`, `bind_reserved`, `reserved_wiring_keys`, `needs_llm` |
| growth traits | `iter_boundary_records`, `grow_depth_delta`, `grow_restamps_self`, `is_loop` |

That is the complete list — a small set of traits/hooks plus `run`. No `NodeKind` is
visible to the engine.

## Why a closed match, on the right axis

An earlier engine branched on node *kind* in many places. That dispatch is now fully
removed from the core; a ratchet test (`tests/engine/test_kind_census.py`) holds the
kind-branch count at **zero**. The design keeps the value of an explicit closed `match`
— but on the **`Outcome`** (four arms), not the kind (a dozen and growing). New kinds
extend the node side; the engine never changes.
