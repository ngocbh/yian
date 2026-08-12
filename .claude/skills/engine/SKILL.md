---
name: engine
description: >-
  Use when implementing, extending, or changing the Agent Composer flow engine —
  anything under src/agent_composer/ (node kinds, runtime/scheduling, typed state,
  suspend/resume, compile/loader, expressions) or its contract. Enforces the
  workflow: design from the functional model (OCaml as reference), fit the engine
  invariants, plan, then implement in small tested steps. Trigger on requests like
  "add a … node kind", "implement … in the engine", "change how the runtime …".
---

# Engine development workflow

The goal is a *functional language of agents* — a small, general language for
data transformation whose leaf computations may be LLM agents. Every engine
change follows the same loop — **never jump straight to code.**

**Companions:** [`reference.md`](reference.md) — the node contract
(`Output | Route | Pause | Grow`), the `NodeKind` table, the OCaml-analogue map, the
non-negotiable invariants, the layer ladder, and a design-note template.
[`templates/node_kind/`](templates/node_kind/) — a copy-paste skeleton + a
file-by-file [`WIRING.md`](templates/node_kind/WIRING.md) for adding a new
authorable node kind.

## The mental model: you're extending a functional language

The engine is a **small functional language for data transformation** (full
framing + OCaml concept map in [`src/agent_composer/README.md`](../../../src/agent_composer/README.md)).
Hold this model whenever you design anything:

- A **flow is a function** — typed inputs in, typed outputs out. An **agent is
  just a flow** whose leaf is an LLM loop. Same contract, same composability.
- Flows **compose** (a node can *be* a flow, via a `call:`/`uses:` reference) and
  nest to any depth.
- Computation is **pure at the boundary**: a node *returns* outputs, the engine
  *binds* them under `(node_id, key)` like `let`-bindings — never a mutation.
- The **author fixes the structure** (the call graph); the LLM fills leaf boxes,
  it does not rewrite the graph at runtime.

So a new construct isn't "a workflow feature" — it's "a language construct," held
to a language's bar: **does it have a clear type, does it compose, is it pure
where it should be?** If a proposed primitive only works at the top level, or
only for one flow shape, or needs to mutate the pool — stop; that's a smell.

## 0. Orient
Read what's relevant to the change:
- [`src/agent_composer/README.md`](../../../src/agent_composer/README.md) — what the engine is, the layout, + the layer ladder.
- [`docs/syntax.md`](../../../docs/syntax.md) — the flow-authoring surface (YAML shape, `${...}` refs, node kinds).
- GitHub issues (`gh issue list`) — the TODO tracker (keep it current per CLAUDE.md "Zeroth rule").

## 1. Design from the functional model — look to OCaml first
Our north star is a *functional language of agents*. Design the feature as a
**language designer** would, not by porting a mechanism. Reach for OCaml's
*design philosophy and conventions*, then adapt them to our engine:

- **What's the type?** Every construct has a clear type. A flow is `'a -> 'b`; a
  new primitive must say what it consumes and what it produces. If you can't write
  its signature, the design isn't done.
- **Does it compose?** OCaml builds everything from small composable pieces
  (functions, modules, variants). Prefer the general combinator that composes over
  a one-off feature. A primitive that only works at the top level is a smell.
- **Is it pure / explicit about effects?** OCaml keeps evaluation referentially
  transparent and makes effects explicit (effect handlers, `option`/`result`
  instead of hidden failure). Map our pause/resume, failure, and binding onto
  those shapes rather than inventing ad-hoc control flow.
- **Sum types + exhaustive match.** OCaml models a closed set of cases as a
  variant and forces an exhaustive `match`. Our closed sets are `NodeKind` (kept
  closed, no registry/metaclass) and the node result `Outcome`. Note the *core*
  matches on `Outcome`, not on `NodeKind` — any kind-specific `match` belongs in a
  node's own `run` or is expressed via node-owned traits/hooks, never in the engine.
- **Modules with explicit signatures (`.mli`).** A package exposes a narrow,
  declared interface and hides the rest — which is just the `structure` skill's
  package charters. Honor the boundary.

Write down the OCaml analogue you're borrowing and how it maps to nodes / pool /
runtime. If there's an established functional name for the construct (combinator,
fold, continuation, effect, functor), use it.

When borrowing runtime mechanics from any prior worker engine, **borrow** the
correctness-critical parts (single-writer dispatcher, 3-state edge join,
outputs-before-successors, recursive skip-flood, layered checkpoint,
discriminated pause reasons) and **drop** scale/framework baggage (external DBs,
dynamic worker scaling, plugin registries/metaclasses, heavy layering,
multi-tenancy) — and anything that fights the functional model above.

## 2. Fit it to the engine invariants
Decide keep / simplify / drop against these **non-negotiable engine invariants**.
Most are just the functional model (above) made enforceable:

- **Deterministic workflow engine** *(referential transparency at the structure
  level)* — the author fixes the call graph; the LLM fills leaf boxes. A flow
  never rewrites itself at runtime. No agentic routing.
- **A flow is a function — clear typed inputs and outputs** — every flow has an
  explicit input/output signature. An agent is one kind of flow; it gets no
  special contract.
- **Hierarchical / composable** *(function composition)* — a node can *be*
  another flow (a `call:`/`uses:` reference → a child subgraph seeded from the
  parent's pool), nestable to arbitrary depth. This is how an agent calls another
  agent and how composite nodes are built. Never assume a node is a leaf; preserve
  recursion through compile + run + checkpoint.
- **General / expressive** — the engine **privileges no output type** (report,
  classification, decision, number, structured record — whatever the terminal
  node emits) and no domain. When adding a primitive, prefer the most general form
  that composes over a use-case-specific feature. If a real flow can't be
  expressed, that's an engine gap — fix the primitive, don't special-case the flow.
- **Node never writes the pool** *(purity)* — it returns outputs; the engine binds
  them under `(node_id, key)`, like an immutable `let`. No node mutates shared state.
- **Typed, losslessly-serializable state** *(the type system)* —
  `TypedValue` / `VariablePool`; the basis for durable checkpoints and `${...}`
  references.
- **Durable suspend/resume** *(algebraic effects + handlers)* — a node *performs*
  a pause via `PauseRequested`; the run serializes to a `RunCheckpoint`; an
  external scheduler is the *handler* that resumes it (re-run-on-resume).
  `HUMAN_INPUT`/`WAIT` are the effect-performing nodes.
- **Dependency-light core (with a deliberate exception)** — keep the engine free
  of a DB and heavy frameworks; external capabilities enter through **injected
  seams** (plain callables). **Exception (by decision):** the AGENT node is *not*
  SDK-free — it decomposes into a `mode` (the loop, in `nodes/agent/modes/`) +
  skills (tools), and a mode imports langchain + `agent_composer.llm_clients` and
  builds its own model via `model_from_config`. Prefer seams for *new* external
  deps, but model access in agent modes is a direct call.
- **Domain-agnostic** — no domain-specific field names or types in the engine;
  domain lives at the contract/boundary (the loader/builder). The `system`
  namespace is generic.
- **Closed `NodeKind` vocabulary; the core is kind-BLIND** — `NodeKind` stays a closed enum (no
  registry/metaclass), but the engine core (`runtime/engine.py` + `runtime/eval_node.py`) does NOT
  branch on it. Any kind-specific `match` lives in a node's own `_run_impl`; the core dispatches only on
  the closed `Outcome` sum and on node-owned traits/hooks (`binds_per_item`, `bind_reserved`,
  `grow_depth_delta`, `grow_restamps_self`, `is_loop`, `needs_llm`). A
  ratchet test (`tests/engine/test_kind_census.py`) holds the core's `NodeKind`/`*Expansion`
  dispatch count at 0.
- **`Node.env` is engine-opaque** — a per-node `dict[str,Any]` of static config the compiler
  bakes at build (`{**flow_env, **node_env}`, node wins, flow-local). The engine NEVER reads it;
  only a node's own `_run_impl()` interprets known keys (e.g. AGENT's `max_tool_iterations`). It is the
  kind-blind knob for "tune this node" — add a key by having the node read it, never by an engine
  branch. A resumed AGENT keeps its cap by threading `env` onto the rebuilt resume node.
- **Single-writer invariant** — workers are pure executors; the dispatcher is the
  only state mutator.
- **Single-process CLI target** — favor the simplest thing that works in-process;
  durable/cross-process impls are injected from outside, not baked into the core.

## 3. Plan
Write a short plan: the **OCaml analogue** you're borrowing and its type; what you
simplify/drop and why; where it lands in the layer map
(`events ← typesys ← nodes ← compile ← compose`, with `expr` feeding nodes and
`suspension` feeding runtime); and the **seam** if it touches an external
dependency. Flag any non-obvious choice and confirm before coding (CLAUDE.md "ask
when uncertain").

## 4. Implement in small, tested steps
Per CLAUDE.md: bottom-up by dependency, **code + at least one test + run green**
before moving on. Tests live in `tests/engine/`; run with
`PYTHONPATH=src pytest tests/engine` (or `pip install -e '.[all]'` first). Keep
the engine import-clean (no new heavy deps in the core). Close the tracking
GitHub issue when the work lands (reference the commit/PR). Commit at green.

### Package layout
Follow the **`structure` skill** for where files and folders go — package charters
in `__init__.py`, `common.py` vs `utils.py`, one-way imports, no cycles. The
engine's layer ladder (`events ← typesys ← nodes ← compile ← compose`, with `expr`
and `suspension` feeding in) is an instance of that framework.
