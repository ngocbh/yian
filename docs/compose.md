# compose — how YAML becomes a `CompiledFlow`

> The front end. `compose/` is the **only** package that knows YAML exists. It takes
> flow text and lowers it to a `CompiledFlow` — the typed, frozen graph the
> [engine](engine.md) runs. Everything domain- or syntax-specific lives here so the
> layers below stay generic. For the authoring surface (what you *write*), see
> [Flow syntax](syntax.md); this page is how that surface is *compiled*.

## The shape of the phase

One function, `load_flow()`, drives a fixed pipeline. Read it like a compiler:
**parse → desugar → build → wire → check → freeze.**

```
 text
  │
  ├─ parse_file            → a span-tagged tree (every scalar remembers its line/col)
  ├─ read_typedefs         → the type registry (typedefs:)
  ├─ read_flow_inputs      → the typed input: declarations
  ├─ parse_nodes           → one frozen DESCRIPTOR per node (per-kind, source-ordered)
  │
  ├─ desugar_inline_calls        ┐  rewrite author sugar into core nodes,
  ├─ desugar_adaptive_questions  │  minting synthetic nodes where needed.
  ├─ desugar_case_call_targets   │  After this, only core kinds remain.
  ├─ expand_case_outputs         ┘
  │
  ├─ build (per kind, via KINDS) → a runtime NODE + its flow-owned WIRING
  │
  ├─ infer_data_edges       → read every ${...} → draw the edges (nobody draws them by hand)
  ├─ reconcile_case_edges   → branch edges agree with the case graph
  ├─ validate_references    → every ${...} resolves & type-checks; classify asserts
  ├─ reject_cycles          → the graph is a DAG
  ├─ check_case_handles     → every branch has a home
  ▼
 CompiledFlow   (returned inside a LoadedFlow: + input decls, asserts, source frame)
```

Everything below just explains three of those words: **descriptor**, **build**, and
**edge inference**.

---

## Descriptors vs. nodes

The pipeline has two representations of a node, and the split matters:

| | **Descriptor** | **Node** |
|---|---|---|
| what | a frozen record of *what you wrote* | the runtime object the engine calls |
| where | `compose/kinds/<kind>.py` | `nodes/<kind>/` |
| carries | raw fields + `SourceRange`s | `run()`, typed `output_type`, params |
| lifetime | load-time only | lives through the whole run |

`parse_nodes` produces descriptors; the desugar passes rewrite descriptors; `build`
turns each surviving descriptor into a node. Keeping them separate means the messy,
YAML-shaped, source-located data stays in `compose/`, and the node the engine sees is
clean and generic.

---

## The two-axis layout (#70)

`compose/` is split along two axes so a change stays local:

- **Per node kind** → `compose/kinds/<kind>.py`. Each module owns three things and
  registers them at import into the `KINDS` table:
  ```python
  KINDS["code"] = KindSpec(descriptor_cls=CodeDescriptor, parse=parse, build=build)
  ```
  - a frozen **descriptor** dataclass (`CodeDescriptor`),
  - **`parse(node_id, reader)`** — read one node body, recording each field's `SourceRange`,
  - **`build(desc, *, registry, resolver, …)`** — construct the runtime `Node` + wiring.

  Adding or changing a kind is one file. The shared dispatch sites
  (`stages/parse.py`, `build.build_leaf_node`) consult `KINDS` — they never grow an
  `if kind == …` ladder.

- **Cross-node graph passes** → `compose/stages/`. These operate on the *whole* set of
  descriptors: `parse_nodes`, the desugars, `infer_ordering_edges`,
  `validate_references`, `reject_cycles`, wiring-parity, and the rest.

A sibling table, `BINDING_FIELDS`, records which descriptor fields carry `${...}`
strings (a code node's `inputs`, a tool's `args`, a map's `over`). One kind-blind walk
in `bindings.py` reads it, so the desugar passes rewrite bindings without an isinstance
ladder.

---

## Build: descriptor → node + wiring

`build()` produces **two** things, and the split is deliberate:

```python
node, wiring = KINDS[desc.kind].build(desc, registry=registry, resolver=resolver)
```

- the **node** carries only what it needs to run: its `output_type` (resolved against
  the type registry), its sink `params` (names, mostly untyped — the type travels with
  the source), and render-only `range`/`field_ranges` for tracebacks.
- the **wiring** is *flow-owned*: `{param → "${source}"}`. The node knows its parameter
  *names*; the flow knows where each one is *sourced from*. This is why a node is
  reusable and why the flow, not the node, owns the graph.

Leaf kinds (`agent`/`code`/`model`/`tool`) ignore `resolver`; `call`/`map`/`loop` use
it to resolve the child flow they expand into.

---

## Edges are inferred, never drawn

You never write an edge. You write a reference:

```yaml
verdict:
  input:
    for_case: ${for_case.output}      # <- this line IS the edge for_case -> verdict
    against_case: ${against_case.output}
```

`infer_data_edges` (in `build.py`) ref-walks every binding, reads out the node ids each
`${...}` depends on, and adds exactly those edges. The graph is a
*consequence* of the data references — it cannot drift out of sync with them, and
parallelism is automatic: two nodes that don't reference each other simply have no edge
between them, so the engine runs them concurrently. (`depends_on`/`runs_after` add
ordering-only edges for the rare case with no data dependency.)

The same `${...}` grammar is *evaluated* at run time and *ref-walked* here at load
time — one grammar, two readers. See [Expressions](expressions.md).

---

## Desugaring: sugar in, core out

Several authoring conveniences are lowered to core nodes before `build`, so the rest of
the pipeline only ever sees a small set of kinds:

| You write | Lowered to |
|---|---|
| inline `${ f(x) }` in a binding | a synthetic `call` node; the binding rewritten to `${synth.output}` |
| `human_input` with `adaptive_questions:` | a synthetic composer `agent` + the rewritten gate |
| `case` branch `then: ${child}` | a synthetic `call` node as the branch target |
| `${some_case.output}` | a coalesce over the case's branch targets |

Because desugaring is a descriptor-level rewrite, the synthetic nodes flow through
`build`, edge inference, validation, and the DAG check exactly like hand-written ones —
no special path downstream.

---

## What you get back

`load_flow()` returns a **`LoadedFlow`**:

- `compiled` — the `CompiledFlow` (immutable topology: nodes, inferred edges, wiring,
  synthesized `start`/`end` boundaries),
- `input` — the typed `InputDecl`s (validated against caller inputs at run time),
- `asserts` — flow-level pre/post checks,
- `source` — a render-only `SourceFrame` (the text + per-node ranges) that lets a
  runtime failure draw a caret back at your YAML, even across nested flow calls.

Hand it to [`run_flow()`](engine.md) and the interpreter takes over.

---

## Errors: one type, always located

Every load failure is a single `LoadError` carrying a `SourceRange`. Each pass raises it
at the most precise anchor it has — a bad field at that field's range, an unresolved
`${...}` at the binding, a cycle at the node header. That is why the CLI can box the
offending line instead of dumping a stack trace. When a pass genuinely lacks an interior
line (e.g. a type error from the line-less `typesys` layer), it anchors at the enclosing
section header rather than guessing.

---

## Where to look in the code

| You want to change… | Go to |
|---|---|
| a node kind's fields or how it builds | `compose/kinds/<kind>.py` |
| a whole-graph pass (edges, validation, cycles) | `compose/stages/` |
| the YAML → span-tree parse | `parser.py` + `spanloader.py` |
| the located cursor over the tree | `reader.py` (`NodeReader`) |
| the pipeline wiring itself | `loader.py` (`_assemble`) |
| running the compiled flow | [engine.md](engine.md) |
