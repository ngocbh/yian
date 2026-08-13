# Nodes — the unit of work

> The node side of the [engine contract](engine.md). Each kind supplies `run` plus a few
> static traits; the engine reads only those and the returned `Outcome`, never the kind.

## A node is a pure function

A flow is a graph of steps; **a node is one step** — "ask an LLM", "run a Python
function", "repeat until good enough". The engine reads the node's inputs out of the pool
and hands them over as a plain dict; the node computes and returns an `Outcome`. It
**cannot** touch the pool, the queue, or the graph.

```
   inputs (plain dict) ──▶  ┌── NODE ──┐  ──▶ Outcome
                            │  pure    │
                            └──────────┘
```

Purity is the whole point: same inputs → same result (**reproducible**), and you can stop
after any node and continue (**resumable**), because no state hides inside a node.

## The contract

Each kind implements exactly one method and declares a few static fields. The engine
wraps `run` in a generic recipe (`run_node`, in the engine) — the node never writes that
wrapper.

```python
class NodeBase(ABC):
    id: NodeId                       # where my output is stored
    params: list[ParamDecl]          # my input NAMES only — the flow owns their sources (wiring)
    pre_asserts / post_asserts       # conditions the engine checks (purely) before / after run
    is_spawner: bool = False         # may I return Grow (expand the graph)?

    @abstractmethod
    def run(self, inputs, **caps) -> Outcome:      # the ONE thing each kind implements
        ...
    def on_failure(self, exc, inputs, **caps):     # recovery seam (default: re-raise)
        raise exc
```

`run` receives two things:

- **`inputs`** — the params, already resolved from the pool into a plain dict (never the
  pool itself). The node declares only param *names*; the **flow** owns where each reads
  from, and the engine binds the two before calling.
- **`caps`** — narrow capabilities a plain input dict cannot carry, such as a mapped
  call's `caps["bind_item"]` or a cooperative `stop_event`. The read boundary supplies
  only traits a node explicitly requests. The AGENT node is the documented exception for
  model access: its mode builds the configured model directly through `llm_clients`.

**Why only `run`?** The wrapper that checks assertions and funnels errors is identical for
every kind — so it is *protocol*, not node behavior, and lives on the engine. See
[the engine loop](engine.md#the-loop-read-eval-apply).

## Two families

Every kind is one of two shapes — a *concept*, not a class hierarchy, decided purely by
what `run` returns:

```
   LEAF  (is_spawner=False)                 SPAWNER  (is_spawner=True)
   run → Output | Pause | Route             run → Grow(seed) | Output
   one unit of work:                        may expand into __start__ → children → __end__
   agent, code, model, tool, case,           call, map, loop
   human_input, wait, start, end
```

- A **leaf** does one thing and returns `Output` (or, for a custom/artificial
  compatibility node, temporary generic `Pause`). AGENT `ask_user`, HUMAN_INPUT,
  and WAIT block inside the live invocation and return `Output`; authorable nodes
  do not pause the graph. WAIT returns `None` after its absolute UTC deadline.
- A **router** (`case`) is a leaf that returns `Route(handle)` — no value, it just selects
  the live out-edge (siblings skip-flood). Still `is_spawner=False`; routing isn't growth.
- A **spawner** returns `Grow` with a canonical seed describing the requested work.
  Compile-owned `build_grow` turns it into a subgraph with one `__start__` and
  `__end__` (whose result becomes this node's result via a baked `commit_as`). See
  [growth](engine.md#growth-a-node-that-is-really-a-subflow).

## A kind is just a `run`

```python
class CodeNode(NodeBase):              # LEAF — run a Python function
    def run(self, inputs, **caps):
        return Output(self.fn(inputs))

class AgentNode(NodeBase):             # LEAF — one live model/tool loop
    def run(self, inputs, **caps):
        return Output(self.run_model_loop(inputs))

class CaseNode(NodeBase):              # ROUTER — pick a branch, produce no value
    def run(self, inputs, **caps):
        return Route(self.select(inputs))

class MapNode(NodeBase):              # SUBFLOW — run a child once per item
    is_spawner = True
    def run(self, inputs, **caps):
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

## Why this shape

- **One method to add a kind** — implement `run`; the wrapper is supplied unchanged.
- **Purity end to end** — nodes never see shared state, so runs are reproducible/resumable.
- **The engine stays kind-blind** — all that varies is the `Outcome` a `run` returns.

For the authoring surface of each kind (`kind:`, its fields, examples), see
[Flow syntax](syntax.md). For each node class, see the [API reference](reference.md).
