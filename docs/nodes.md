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
- **`caps`** — *capabilities*: the side-effecting helpers a pure dict can't carry (the LLM
  client `caps["llm"]`, a mapped call's `caps["bind_item"]`). The engine builds them per
  call and owns their lifecycle; a node that needs none ignores `caps`.

**Why only `run`?** The wrapper that checks assertions and funnels errors is identical for
every kind — so it is *protocol*, not node behavior, and lives on the engine. See
[the engine loop](engine.md#the-loop-read-eval-apply).

## Two families

Every kind is one of two shapes — a *concept*, not a class hierarchy, decided purely by
what `run` returns:

```
   LEAF  (is_spawner=False)                 SUBFLOW  (is_spawner=True)
   run → Output | Pause | Route             run → Grow(subgraph)
   one unit of work:                        "a node that IS a smaller flow":
   agent, code, model, tool, case,          expands into __start__ → children → __end__
   human_input, wait, start, end            call, map, loop
```

- A **leaf** does one thing and returns `Output` (or `Pause` if it must wait).
- A **router** (`case`) is a leaf that returns `Route(handle)` — no value, it just selects
  the live out-edge (siblings skip-flood). Still `is_spawner=False`; routing isn't growth.
- A **subflow** returns `Grow` describing the boxes to splice in. By convention the
  subgraph has a single `__start__` and `__end__` (whose result becomes this node's result
  via a baked `commit_as`). See [growth](engine.md#growth-a-node-that-is-really-a-subflow).

## A kind is just a `run`

```python
class CodeNode(NodeBase):              # LEAF — run a Python function
    def run(self, inputs, **caps):
        return Output(self.fn(inputs))

class AgentNode(NodeBase):             # LEAF — one LLM step
    def run(self, inputs, **caps):     # self.llm_config = WHICH model (serializable);
        return Output(caps["llm"](self.prompt.format(inputs)))   # caps["llm"] = live client

class CaseNode(NodeBase):              # ROUTER — pick a branch, produce no value
    def run(self, inputs, **caps):
        return Route(self.select(inputs))

class MapNode(NodeBase):              # SUBFLOW — run a child once per item
    is_spawner = True
    def run(self, inputs, **caps):
        children = [clone(self.body, i, bake(item)) for i, item in enumerate(inputs["over"])]
        end = ListEnd(n=len(children), commit_as=self.id)   # gathers N results; commits under me
        return Grow(Flow(nodes=[Start(), *children, end], edges=..., start_id=..., end_id=...))
```

Notice what is **not** here: no node reads the pool, emits an event, or checks its own
success/failure plumbing. Each kind expresses only its own logic.

## Traits, briefly

Most kinds set none of these. They exist so the engine's read boundary and growth core
stay kind-blind — a node *declares* how it wants to be driven instead of the engine
branching on kind.

| Trait / hook | Set by | Purpose |
|---|---|---|
| `binds_per_item` | map | bind inputs per element via `caps["bind_item"]`, not once up front |
| `bind_reserved` / `reserved_wiring_keys` | wait, map | reserved keys resolved before `run` (`until`, `over`); the names for load-time checks |
| `needs_llm` | agent | build the `caps["llm"]` model factory |
| `grow_depth_delta` / `grow_restamps_self` / `is_loop` | spawners | recursion-depth stamp; agent re-pause nesting; loop bookkeeping |

## Why this shape

- **One method to add a kind** — implement `run`; the wrapper is supplied unchanged.
- **Purity end to end** — nodes never see shared state, so runs are reproducible/resumable.
- **The engine stays kind-blind** — all that varies is the `Outcome` a `run` returns.

For the authoring surface of each kind (`kind:`, its fields, examples), see
[Flow syntax](syntax.md). For each node class, see the [API reference](reference.md).
