# Nodes — the unit of work

> The node side of the [engine contract](engine.md). Each kind supplies `_run_impl`
> plus a few static traits and inherits the shared `run` template; the engine reads
> only those and the returned `NodeResult`, never the kind.

## A node is a pure function

A flow is a graph of steps; **a node is one step** — "ask an LLM", "run a Python
function", "repeat until good enough". The engine reads the node's inputs out of the pool
and hands them over as a plain dict; the node computes and returns a `NodeResult`. It
**cannot** touch the pool, the queue, or the graph.

```
   inputs (plain dict) ──▶  ┌── NODE ──┐  ──▶ NodeResult
                            │  pure    │
                            └──────────┘
```

Purity here is a boundary rule: the graph state is reproducible and can be reconstructed
from its authoritative records because no node mutates it directly. An effectful leaf may
still consult an LLM, a human, or the wall clock, but its only graph-visible result is the
`NodeResult` it returns.

## The contract

Each kind implements exactly one body method and declares a few static fields. The base
`Node.run` template wraps `_run_impl` with default filling and pre/post assertions; the
engine always drives that shared wrapper.

```python
class Node:
    id: NodeId                       # where my output is stored
    params: list[Parameter]          # formal inputs only — the flow owns their Binding sources
    pre_asserts: list[Assertion]      # ordered contracts checked before the body
    post_asserts: list[Assertion]     # ordered contracts checked after an Output
    is_spawner: bool = False         # may I return Grow (expand the graph)?

    def run(self, inputs, **caps) -> NodeResult:   # shared template; do not override
        inputs = apply_defaults(self.params, inputs)
        check_assertions(self.pre_asserts, inputs)  # evaluate expression; report display
        outcome = self._run_impl(inputs, **caps)   # the ONE kind-specific step
        if isinstance(outcome, Output):
            check_assertions(self.post_asserts, {**inputs, "output": outcome.value})
        return outcome

    def _run_impl(self, inputs, **caps) -> NodeResult:
        ...
```

`run` receives two things:

- **`inputs`** — the params, already resolved from the pool into a plain dict (never the
  pool itself). Each source-neutral `Parameter` declares a name plus optional
  contract/default/coercion policy; the **flow** owns where it reads from, and the
  engine binds the two before calling.
- **`caps`** — narrow capabilities a plain input dict cannot carry, such as a mapped
  call's `caps["bind_item"]` or a cooperative `stop_event`. The read boundary supplies
  only traits a node explicitly requests. The AGENT node is the documented exception for
  model access: its mode builds the configured model directly through `llm_clients`.

**Why only `_run_impl`?** Default filling and assertion checks are identical for every
kind — so they are protocol, not kind behavior, and live in the base template. See
[the engine loop](engine.md#the-loop-read-eval-record-reduce).

## Two families

Every kind is one of two shapes — a *concept*, not a class hierarchy, decided purely by
what `run` returns:

```
   LEAF  (is_spawner=False)                 SPAWNER  (is_spawner=True)
   run → Output | Route                     run → Grow(seed) | Output
   one unit of work:                        may expand into __start__ → children → __end__
   agent, code, model, tool, case,           call, map, loop
   human_input, wait, start, end
```

- An ordinary **value-producing leaf** does one thing and returns `Output`. AGENT
  `ask_user`, HUMAN_INPUT, and WAIT may block inside the live invocation, but
  still return `Output`. WAIT returns `None` after its absolute UTC deadline.
- A **router** (`case`) is a leaf that returns `Route(handle)` — no value, it just selects
  the live out-edge (siblings skip-flood). Still `is_spawner=False`; routing isn't growth.
- A **spawner** returns `Grow` with a canonical seed describing the requested work.
  Compile-owned `build_grow` turns it into a subgraph with one `__start__` and
  `__end__` (whose result becomes this node's result via a baked `commit_as`). See
  [growth](engine.md#growth-a-node-that-is-really-a-subflow).

## A kind is one `_run_impl`

```python
class CodeNode(Node):                  # LEAF — run a Python function
    def _run_impl(self, inputs, **caps):
        return Output(self.fn(inputs))

class AgentNode(Node):                 # LEAF — one live model/tool loop
    def _run_impl(self, inputs, **caps):
        return Output(self.run_model_loop(inputs))

class CaseNode(Node):                  # ROUTER — pick a branch, produce no value
    def _run_impl(self, inputs, **caps):
        return Route(self.select(inputs))

class MapNode(Node):                   # SUBFLOW — run a child once per item
    is_spawner = True
    def _run_impl(self, inputs, **caps):
        records = [caps["bind_item"](item) for item in inputs["over"]]
        return Grow(seed=records)       # compile.build_grow clones + wires the child flow
```

Notice what is **not** here: no node reads the pool, emits an event, or checks its own
success/failure plumbing. A spawner also never imports compile or constructs graph
topology. Each kind expresses only its own logic.

## Traits, briefly

Most kinds set none of these. They exist so the engine's read boundary and growth core
stay kind-blind — a node *declares* how it wants to be driven instead of the engine
branching on kind.

| Trait / hook | Set by | Purpose |
|---|---|---|
| `binds_per_item` | map | bind inputs per element via `caps["bind_item"]`, not once up front |
| `bind_reserved` / `reserved_wiring_keys` | wait, map | reserved keys resolved before `run` (`until`, `over`); the names for load-time checks |
| `grow_depth_delta` / `is_loop` | spawners | recursion-depth stamp; loop bookkeeping |
| `needs_stop_event` | agent, wait | receive the run-owned cooperative stop event |

## Why this shape

- **One method to add a kind** — implement `_run_impl`; the wrapper is supplied unchanged.
- **Structural purity** — nodes never mutate shared graph state, so scheduling and
  run-record replay remain reproducible even when a leaf performs an external effect.
- **The engine stays kind-blind** — all that varies is the `NodeResult` a `run` returns.

For the authoring surface of each kind (`kind:`, its fields, examples), see
[Flow syntax](syntax.md). For each node class, see the [API reference](reference.md).
