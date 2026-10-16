# Runtime nodes — the unit of work

> The node side of the [engine contract](engine.md). Each kind supplies `forward`
> plus a few static traits and inherits the shared `run` template; the engine reads
> only those and the returned `NodeResult`, never the kind.

## Pure at the graph-state boundary

A flow is a graph of steps; **a node is one step** — "ask an LLM", "run a Python
function", "repeat until good enough". The engine reads the node's inputs out of the pool
and hands them over as a plain dict; the node computes and returns a `NodeResult`. It
**cannot** touch the pool, the queue, or the graph.

```
   inputs (plain dict) ──▶  ┌── NODE ──┐  ──▶ NodeResult
                            │  pure    │
                            └──────────┘
```

This is a boundary rule, not a claim that every node is mathematically pure. The
engine can deterministically apply and replay recorded graph-state transitions
because nodes do not mutate that state directly. An effectful leaf may still
consult an LLM, a human, or the wall clock, but its only graph-visible outcome
is the `NodeResult` it returns. See [Agent](agent.md) for how nodes and flows fit
the same conceptual model.

## The contract

Each kind implements exactly one body method and declares a few static fields. The base
`Node.run` template wraps `forward` with default filling and pre/post assertions; the
engine always drives that shared wrapper.

```python
class Node:
    id: NodeId                       # physical execution identity
    commit_as: NodeId | None         # optional baked logical-output redirect
    params: list[Parameter]          # formal inputs only — the flow owns their Binding sources
    pre_asserts: list[Assertion]      # ordered contracts checked before the body
    post_asserts: list[Assertion]     # ordered contracts checked after an Output
    is_spawner: bool = False         # may I return Grow (expand the graph)?

    def run(self, inputs, **caps) -> NodeResult:   # shared template; do not override
        inputs = apply_defaults(self.params, inputs)
        check_assertions(self.pre_asserts, inputs)  # evaluate expression; report display
        outcome = self.forward(inputs, **caps)      # the ONE kind-specific step
        if isinstance(outcome, Output):
            check_assertions(self.post_asserts, {**inputs, "output": outcome.value})
        return outcome

    def forward(self, inputs, **caps) -> NodeResult:
        ...

    def backward(
        self, context: BackwardContext, feedback: tuple[FeedbackValue, ...],
    ) -> BackwardResult:
        return BackwardResult()  # default behavioral stop: all channels empty

    def skipped_output_target(self) -> NodeId:
        return self.id if self.commit_as is None else self.commit_as
```

`run` receives two things:

- **`inputs`** — the params, already resolved from the pool into a plain dict (never the
  pool itself). Each source-neutral `Parameter` declares a name plus optional
  contract/default/coercion policy; the **flow** owns where it reads from, and the
  engine binds the two before calling. A flow-owned `Binding` is a template string, a
  literal, or a `ValueTemplate` (an authored mapping or list whose nested `${...}`
  strings and tagged parts are evaluated in place). A tagged binding's value is checked
  against its tag before `run`, without conversion; a mismatch fails with `binding.type`.
- **`caps`** — narrow capabilities a plain input dict cannot carry, such as a mapped
  call's `caps["bind_item"]` or a cooperative `stop_event`. The read boundary supplies
  only traits a node explicitly requests. The AGENT node is the documented exception for
  model access: its loop (`run_agent`) builds the configured model directly through
  `llm_clients`.

**Why only `forward`?** Default filling and assertion checks are identical for every
kind — so they are protocol, not kind behavior, and live in the base template. See
[the engine loop](engine.md#the-loop-read-eval-record-reduce).

The runtime calls `backward` only during the explicit reverse fold started by
[`RunResult.backward(...)`](api.md#explicit-backward-feedback), never during forward
execution. A reached accepted occurrence receives its detached `BackwardContext`
and an ordered tuple of ordinary feedback messages. The default returns empty
channels and stops propagation.

An override returns a `BackwardResult` with three channels: `input_feedback`,
`self_feedback`, and `variable_feedback`. The runtime validates the complete result
against the captured declaration names and recorded value shapes before routing
any input feedback through retained expression routes. Self and variable feedback
remain attributed to this occurrence; only input feedback propagates upstream.

`skipped_output_target()` is a pure, deterministic topology hook, not another
effectful body method. It must derive only from execution state represented in
the flow manifest and return an existing node id. The base implementation mirrors
the baked `Node.commit_as`; LOOP overrides it with manifest-backed `origin_id` to
mirror its dynamic success redirect. A result-carried `Output.commit_as` affects
success only unless the hook explicitly names the same target for absence.

## Two families

Every kind is one of two shapes — a *concept*, not a class hierarchy, decided purely by
what `run` returns:

```
   LEAF  (is_spawner=False)                 SPAWNER  (is_spawner=True)
   run → Output | Route                     run → Grow(seed) | Output
   one unit of work:                        may expand into __start__ → children → __end__
   agent, code, tool, case,                  call, map, loop
   human_input, wait, start, end
```

- An ordinary **value-producing leaf** does one thing and returns `Output`.
  `ask_user` on an AGENT node, HUMAN_INPUT, and WAIT may block inside the live invocation, but
  still return `Output`. WAIT returns `None` after its absolute UTC deadline.
- A **router** (`case`) is a leaf that returns `Route(handle)` — no value, it just selects
  the live out-edges with that handle (the target and any node the loader lifted out of
  it; siblings skip-flood); on no match a `case` routes to the `#else`
  handle. Still `is_spawner=False`; routing isn't growth.
  A `depends_on:`/`runs_after:` edge from a router is not a branch: the router settled by
  routing, so the dependent runs whichever branch was taken.
- A **spawner** returns `Grow` with a canonical seed describing the requested work.
  Compile-owned `build_grow` turns it into a subgraph with one `__start__` and
  `__end__` (whose output and absence cross this node's boundary via baked
  `commit_as` and the inherited skip-target hook). See
  [growth](engine.md#growth-expanding-a-child-flow).

## A kind is one `forward`

```python
class CodeNode(Node):                  # LEAF — run a Python function
    def forward(self, inputs, **caps):
        return Output(self.fn(inputs))

class AgentNode(Node):                 # AGENT node — one live model/tool loop
    def forward(self, inputs, **caps):
        return Output(self.run_model_loop(inputs))

class CaseNode(Node):                  # ROUTER — pick a branch, produce no value
    def forward(self, inputs, **caps):
        return Route(self.select(inputs))

class MapNode(Node):                   # SPAWNER — apply a child flow per item
    is_spawner = True
    def forward(self, inputs, **caps):
        records = [caps["bind_item"](item) for item in inputs["over"]]
        return Grow(seed=records)       # compile.build_grow clones + wires the child flow
```

A value entry in `nodes:` (`total: !int ${a.output + b.output}`, see
[Values in `nodes:`](syntax.md#values-in-nodes)) is not a kind of its own: it lowers at
load to a CODE node with the entry's id whose code returns its one input.

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
| `needs_stop_event` | agent, wait, tool | receive the run-owned cooperative stop event |
| `needs_tools` | tool, human_input, agent | receive the run's `ToolSet` (its tools, tool groups, and `ask_user` handler) |
| `granted_tools()` | base; agent overrides | tool and group ids the run boundary checks against the run's tools before the run starts |
| `needs_steps` | tool | receive `caps["steps"]`, a `NodeSteps`: the node's start count, the steps its earlier starts recorded, and `record(step)`, which appends a step to the run log and returns once it is applied |
| `needs_live_events` | agent | receive transient `emit_stream`/`emit_event` sinks for the host callbacks that exist |
| `skipped_output_target()` | base; loop overrides | logical target whose out-edges receive this physical node's absence |

A redirected target is an exclusive, inert logical anchor: it has exactly one
unfinished physical redirect owner and is either an expanded spawner or an
unknown placeholder with no incoming work. It cannot also be ready, in flight, or
already committed, and all of its outgoing edges must still be unknown;
supporting several possible producers would require a separate
OR-join/multi-writer contract.

## Why this shape

- **One method to add a kind** — implement `forward`; the wrapper is supplied unchanged.
- **Structural purity** — nodes never mutate shared graph state, so scheduling and
  recorded engine-state replay stay deterministic even when a leaf performs an
  external effect. Replaying that effect itself is a separate concern.
- **The engine stays kind-blind** — all that varies is the `NodeResult` a `run` returns.

For the authoring surface of each kind (`kind:`, its fields, examples), see
[Flow syntax](syntax.md). For each node class, see the [API reference](reference.md).
