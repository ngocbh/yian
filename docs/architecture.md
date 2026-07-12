# Architecture — from a YAML file to a running flow

> A map for contributors. If [Home](index.md) tells you *why* Agent Composer looks
> the way it does, this page tells you *how* it works end to end — the journey one
> `flow.yaml` takes to become a result. Every box below links to a deeper page.

## The 10-second model

There are only two big moves, and one long-lived object between them:

```
   flow.yaml  ──load──▶  LoadedFlow  ──run──▶  RunResult
   (text)               (a typed graph)       (output | error | paused)
```

- **Load** is a *compiler*. It reads the text once, checks it, and lowers it to a
  frozen, typed graph. All the YAML-awareness lives here and nowhere else.
- **Run** is an *interpreter*. It walks that graph, handing each box its inputs and
  collecting what it returns, until there is nothing left to do.

Everything else on this page is just naming the pieces of those two moves.

```python
from agent_composer.compose.loader import load_flow
from agent_composer.compose.run import run_flow

loaded = load_flow(open("flow.yaml").read())      # text  -> LoadedFlow
result = run_flow(loaded, {"question": "..."})    # graph -> RunResult
print(result.output)
```

That is the whole public API. The two functions are the seam between the two halves
of the codebase.

---

## Phase 1 — Load: text becomes a typed graph

`load_flow()` (in `compose/loader.py`) is the front door. It runs a fixed pipeline of
small passes, each one taking the output of the last. Think of it as a compiler's
stages: **parse → desugar → wire → check → freeze.**

```
 text
  │  parse_file            YAML text -> a span-tagged tree (every value knows its line/col)
  │  parse_nodes           tree      -> one frozen *descriptor* per node (kind-specific)
  │  desugar_*             rewrite sugar into core nodes:
  │                          inline ${f(x)} calls, adaptive questions, case targets, cases
  │  build (per kind)      each descriptor -> a runtime Node  (+ its flow-owned wiring;
  │                          a case lowers in its own build, emitting its branch edges)
  │  infer_data_edges      read every ${...} -> draw the edges NOBODY drew by hand
  │                          (+ append each case's own data+control edges)
  │  validate_references   every ${...} points at something real & type-compatible
  │  reject_cycles         the graph must be a DAG
  │  check_case_handles    every branch has a home
  ▼
 CompiledFlow  (wrapped, with input decls + asserts + source frame, as a LoadedFlow)
```

Two ideas make this phase tractable:

**Edges are inferred, never authored.** You write `input: {brief: ${frame.output}}`;
`infer_data_edges` reads that reference and adds the `frame → this` edge. The graph is
a *consequence* of the data references, so it can never disagree with them.

**The front end is split on two axes (issue #70).** Per-node-kind logic lives in
`compose/nodes/<kind>.py` — each kind module owns its descriptor, its `parse()`, and
its `build()`, and registers them into the `KINDS` table. Cross-node graph passes live
in `compose/stages/`. The engine core sees none of it.

The output is a **`LoadedFlow`**: a `CompiledFlow` (the immutable topology) plus the
typed `input:` declarations, the `asserts`, and a render-only *source frame* used to
draw error tracebacks that point back at your YAML.

→ **[compose.md](compose.md)** is the full tour of this phase: the pipeline pass by
pass, the two-axis layout, descriptors vs. nodes, and how a `${...}` becomes an edge.

---

## Phase 2 — Run: the interpreter walks the graph

`run_flow()` hands the `CompiledFlow` to a **`FlowEngine`** (`runtime/engine.py`). The
engine owns everything mutable — the memory, the ready-queue, the growing graph, the
event log — and it runs one small loop:

1. **What can run now?** A node whose every input edge is satisfied.
2. **What does it produce?** Bind its inputs from memory, call `node.run(inputs)`, get
   back an *outcome*.
3. **What changes?** Store the outputs, unlock the successors, repeat.

The node is a **pure function of its inputs**. It is handed a plain dict and returns
one of four outcomes — it never touches the engine's memory:

| Outcome | Meaning |
|---------|---------|
| `Output` | here are my typed results — bind them and continue |
| `Route`  | a `case` decision — take this branch, skip the others |
| `Pause`  | I need something from outside (a human, a timer) — checkpoint the run |
| `Grow`   | splice this sub-graph in and continue (how `call`/`map`/`loop` expand) |

The engine's defining rule: **it never asks what *kind* of node this is.** It reads
inputs, calls `run`, and branches only on the *shape of the outcome* and a few
node-owned traits (`is_spawner`, `binds_per_item`, …). Adding a brand-new node kind
requires zero engine changes — a ratchet test (`tests/engine/test_kind_census.py`)
keeps the engine's kind-branch count pinned at 0.

Because all state lives in one place the engine controls, a run can stop after *any*
node and resume later: `Pause` serializes the world to a `RunCheckpoint`, and
`resume_flow()` rebuilds and continues it.

→ **[engine.md](engine.md)** covers the scheduler, the edge-join, growth, and durable
suspend/resume. **[nodes.md](nodes.md)** covers the node contract and each kind's
`run()`.

---

## The stuff that flows through the pipes

Two subsystems are used by *both* phases, so they sit underneath:

- **Typed values & the pool** — every input, output, and stored value is a typed,
  losslessly-serializable `TypedValue`; the engine's memory is a `VariablePool` of
  them. That serializability is what makes checkpoints possible.
  → **[typing.md](typing.md)**
- **The `${...}` grammar** — one small expression language: parse to an AST, then
  either *evaluate* it against the pool (at run time) or *ref-walk* it to find its
  dependencies (at load time, to infer edges). One grammar, two readers.
  → **[expressions.md](expressions.md)**

---

## Four properties, and where they come from

Everything in the codebase is in service of four guarantees — each is a *consequence*
of a design choice above, not a feature bolted on:

| Property | Falls out of… |
|----------|---------------|
| **Reproducible** — same flow, same run | the author fixes the graph; the LLM only fills leaf boxes |
| **Checkpointable / resumable** | pure nodes + all state in one serializable pool |
| **Extensible** — new kinds don't touch the core | the kind-blind engine + the per-kind `KINDS` table |
| **Readable / debuggable** | the flow *is* the plan; source spans point errors back at the YAML |

---

## Where the code lives

The packages form an acyclic ladder — each layer imports only from those below it:

```
events        run + node event vocabulary, source spans          [leaf, no deps]
  ▲
typesys       typed values + the VariablePool
  ▲
nodes         the Node contract + one package per kind      (expr feeds this)
  ▲
compile       CompiledFlow IR + representation-neutral validation
  ▲
compose       the YAML front end: text -> CompiledFlow       ← all YAML lives here
```

Off to the side: `runtime/` (the engine + worker pool, fed by `suspension/`),
`llm_clients/` (lazy provider adapters), `tools/` (the tool registry), and `cli/`
(the `ac` entry point). The rule that keeps the picture honest: **never add a
back-edge**, and **never teach `runtime/` about a `NodeKind`**.

---

## Where to go next

<div class="grid cards" markdown>

- :material-cog: **[compose.md](compose.md)** — Phase 1 in depth: how YAML is lowered to a `CompiledFlow`.
- :material-play: **[engine.md](engine.md)** — Phase 2 in depth: the scheduler and the node contract.
- :material-cube: **[nodes.md](nodes.md)** — every node kind and what its `run()` returns.
- :material-shape: **[typing.md](typing.md)** — the typed value system and the variable pool.
- :material-code-braces: **[expressions.md](expressions.md)** — the `${...}` grammar.

</div>
